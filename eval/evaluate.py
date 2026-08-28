"""
eval/evaluate.py

Evaluation loop: runs N episodes with the current policy (no gradient),
reports success rate per task type and overall.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from envs.alfworld_env import AlfworldTextEnv
from utils.action_parser import parse_action, match_admissible
from utils.stopping_criteria import build_action_stop_criteria
from models.encoder import sample_z

logger = logging.getLogger(__name__)

# 无放回 eval 用的固定顺序种子：eval 开始时只打乱一次 game 顺序，之后连续
# reset 按序遍历整个 game 池（不重复不遗漏），固定值保证每次 eval 可复现。
EVAL_ORDER_SEED = 1234


def _build_eval_prompt(
    obs: str,
    info: Dict,
    history: List,
    steps: int,
    skill_lib,
    skill_text: str,
) -> str:
    """
    构造单个 eval 步的 prompt（串行与 batch 路径共用）。

    从原串行内联逻辑抽出，行为不变。steps 是本 episode 的累计步数
    （≥1），step_count = steps-1 是这步之前已走的步数。
    """
    from training.rollout import _STEP_PROMPT
    hist_lines = [
        f"Step {h_step}: Obs: {h_obs} → Action: {h_act}"
        for h_step, h_obs, h_act in history
    ]
    history_str = "\n".join(hist_lines) if hist_lines else "(none yet)"
    gen_skills, task_skills = skill_lib.retrieve(
        info["task_description"], task_type=info["task_type"]
    )
    skill_guidance = skill_lib.format_for_prompt(gen_skills, task_skills)
    step_count = steps - 1             # steps already taken before this one
    current_step = steps               # this step's number
    return _STEP_PROMPT.format(
        task_description=info["task_description"],
        skill_guidance=skill_guidance or f"- {skill_text}",
        step_count=step_count,
        history_len=len(history),      # most recent N shown (window-truncated)
        history=history_str,
        current_step=current_step,
        obs=obs,
        admissible=", ".join(info["admissible_commands"]),
    )


def run_eval(
    trainer,               # InfoskillTrainer instance
    env_factory: Callable, # () → AlfworldTextEnv
    n_episodes: int = 64,
    eval_logger = None,    # Optional logger for eval episodes
) -> Dict[str, float]:
    """
    Evaluate the current policy for n_episodes episodes (no gradient).

    Args:
        trainer:     InfoskillTrainer (holds model, encoder, projector, etc.)
        env_factory: Callable with no args → a single fresh AlfworldTextEnv.
        n_episodes:  Number of eval episodes.

    Returns:
        Dict with 'overall_success' and per-task-type success rates.
    """
    model      = trainer._base_model  # 解包模型（DDP 对象没有 get_input_embeddings/generate）
    tokenizer  = trainer.tokenizer
    encoder    = trainer.encoder
    projector  = trainer.projector
    skill_lib  = trainer.skill_lib
    device     = trainer.device
    rcfg       = trainer.cfg.get("rollout", {})

    max_steps      = rcfg.get("max_steps", 50)
    max_new_tokens = rcfg.get("max_new_tokens", 128)  # 从配置读取，默认 128
    max_prompt_len = rcfg.get("max_prompt_len", 8192)
    history_len    = rcfg.get("history_len", 3)

    # DDP 分片：所有 rank 共同评估，各自只跑一段连续切片，最后 all_gather 合并。
    # 若不拆，只有 rank0 评估全部 n_episodes（140 局约 3h），其他 rank 在训练循环
    # 的 barrier 上空等 → NCCL barrier 600s 超时被杀（run 20260824）。独立单卡
    # eval 脚本（scripts/eval_checkpoint.py）的 trainer 代理没有这些字段，
    # 用 getattr 给单机默认值，行为与原先一致。
    is_ddp     = bool(getattr(trainer, "is_ddp", False))
    rank       = int(getattr(trainer, "rank", 0))
    world_size = int(getattr(trainer, "world_size", 1))

    # eval_batch_size > 1 → 多 episode 并行（滑动窗口），走独立 batch 路径；
    # =1（默认）→ 现有串行路径（完全不动）。两种路径产生的每局决策序列相同
    # （同一 greedy、同一 prompt、同一 stopping），只是 GPU 利用率不同。
    eval_batch_size = int(trainer.cfg.get("training", {}).get("eval_batch_size", 1))
    if eval_batch_size > 1:
        return _run_eval_batched(
            trainer, env_factory, n_episodes, eval_batch_size, eval_logger,
        )

    # Put modules in eval mode
    model.eval()
    encoder.eval()
    projector.eval()

    results: Dict[str, List[bool]] = defaultdict(list)
    eval_start = time.time()
    # exp4: encoder 直接吃原始文本，不再需要预先算 embedding 的缓存
    # （旧 skill_emb_cache 删除）。

    # Reuse one env across episodes: AlfworldTextEnv.reseed switches the task
    # in-memory, avoiding re-initialising ALFWorld (re-scanning game dirs)
    # every episode.
    env = env_factory()
    reuse_env = hasattr(env, "reseed")

    # ── 无放回全量覆盖 ──────────────────────────────────────────────────────
    # TextWorld 的 shuffled_cycle 迭代器第一轮按 seed 洗牌后的顺序逐个 yield、
    # 不重复，只有耗尽后才重洗。所以 eval 开始时用固定 seed 打乱一次顺序，
    # 之后每 episode 直接连续 reset()，即天然无放回遍历整个 game 池。
    # 旧实现每 episode reseed(ep_idx) 是"有放回随机抽样"（每次从新洗牌的列表
    # 取第一个），会重复、无法保证覆盖全部 game。
    pool_size = getattr(env, "num_games", None)
    if pool_size and n_episodes > pool_size:
        logger.warning(
            "n_episodes=%d 超过 game 池大小 %d，已截断为 pool 大小（无放回上限）",
            n_episodes, pool_size,
        )
        n_episodes = pool_size
    if reuse_env:
        # 先把全局 np RNG 固定到 EVAL_ORDER_SEED，再让底层 seed() 洗牌。底层
        # TextworldBatchGymEnv.seed() 用 np.random.shuffle(game_files) 洗牌；它
        # 内部是否先 np.random.seed(seed) 不确定，而训练按 seed+rank 播种后各
        # rank 全局 RNG 状态不同——若它直接消费全局状态，DDP 两 rank 会洗出不同
        # 顺序，分片切片就错乱（重复/遗漏）。先固定全局 RNG 则无论其内部行为都
        # 得到同一顺序，同时让单卡 eval 顺序也变成 EVAL_ORDER_SEED 的纯函数
        # （不再受训练进度影响）。
        np.random.seed(EVAL_ORDER_SEED)
        env.reseed(EVAL_ORDER_SEED)   # 固定打乱一次顺序，保证可复现

    # ── DDP 分片：本 rank 评估的绝对 episode 区间 [slice_start, slice_end) ───
    # 两 rank 基于同一个 EVAL_ORDER_SEED 洗牌（env.reseed 内部按 seed 洗 game_files，
    # seed 确定则洗牌结果确定），各自切片互不重叠、合并后恰好覆盖全部 n_episodes，
    # 无放回语义保持。rank r 用连续 reset() 跳过它之前的 r*n_per_rank 个 game，
    # 从自己切片起点开始评估。
    if is_ddp and world_size > 1:
        n_per_rank  = -(-n_episodes // world_size)          # ceil
        slice_start = rank * n_per_rank
        slice_end   = min(slice_start + n_per_rank, n_episodes)
        if not reuse_env:
            logger.warning(
                "rank=%d DDP eval 分片依赖 reuse_env（env.reseed），当前 env 不支持，"
                "切片定位可能不准确", rank,
            )
    else:
        slice_start, slice_end = 0, n_episodes
    if slice_start > 0:
        for _ in range(slice_start):
            env.reset()   # 丢弃，仅推进底层 game 指针到本 rank 切片起点
        logger.info("rank=%d eval slice: episodes [%d, %d)", rank, slice_start, slice_end)

    with torch.no_grad():
        pbar = tqdm(range(slice_start, slice_end), desc="Eval", unit="ep", dynamic_ncols=True)
        for ep_idx in pbar:
            ep_start = time.time()
            if ep_idx > 0 and not reuse_env:
                env.close()
                env = env_factory()
            obs, info = env.reset()
            task_type = info["task_type"]
            history: List = []
            won = False
            steps = 0

            # 记录 episode 开始信息
            logger.info(
                "\n=== Episode %d/%d START ===\ntask_type=%s\ntask_desc=%s\ninitial_obs=%s",
                ep_idx + 1, n_episodes, task_type, info["task_description"], obs[:200]
            )
            if eval_logger is not None:
                eval_logger.info(
                    "\n=== Episode %d/%d START ===\ntask_type=%s\ntask_desc=%s\ninitial_obs=%s",
                    ep_idx + 1, n_episodes, task_type, info["task_description"], obs
                )

            for step in range(max_steps):
                steps += 1
                # Retrieve skill
                skill = skill_lib.retrieve_for_encoder(
                    info["task_description"], task_type=info["task_type"]
                )

                # exp4: encoder 直接吃原始文本（与 training/rollout.py 的
                # state_text 格式一致），不再预先用 Qwen mean-pool 出向量。
                state_text = f"Task: {info['task_description']}\nObservation: {obs}"

                # Encode → soft prefix
                mu, log_var, _pooled_state = encoder([state_text], [skill.grounding_text])
                z_tilde     = sample_z(mu, log_var)             # [1, L]
                soft_prefix = projector(z_tilde)                # [1, m, H]

                # Build prompt
                prompt = _build_eval_prompt(
                    obs, info, history, steps, skill_lib, skill.grounding_text
                )
                msg = [{"role": "user", "content": prompt}]
                text = tokenizer.apply_chat_template(
                    msg, tokenize=False, add_generation_prompt=True
                )
                enc = tokenizer(
                    text, return_tensors="pt", truncation=True, max_length=max_prompt_len
                ).to(device)

                embed_layer  = model.get_input_embeddings()
                input_embeds = embed_layer(enc.input_ids)           # [1, seq, H]
                inputs_embeds = torch.cat([soft_prefix.to(input_embeds.dtype), input_embeds], dim=1)
                prefix_mask   = torch.ones(
                    1, soft_prefix.size(1),
                    dtype=enc.attention_mask.dtype, device=device
                )
                attention_mask = torch.cat([prefix_mask, enc.attention_mask], dim=1)

                output_ids = model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,            # greedy decoding for eval
                    pad_token_id=tokenizer.eos_token_id,
                    # 出现 </action>（动作已给出）立即停止，省掉后续多余的
                    # token（eval 提速，单序列无连带截断问题）。
                    stopping_criteria=build_action_stop_criteria(tokenizer),
                )

                # generate() only returns newly generated tokens when called with inputs_embeds
                raw_output = tokenizer.decode(
                    output_ids[0], skip_special_tokens=True
                )
                action_text, _ = parse_action(raw_output)
                matched_action = match_admissible(
                    action_text, info["admissible_commands"]
                )

                # 每步输出 LLM 生成详情（用于调试）
                logger.info(
                    "  ep=%d step=%d | raw_output: %s",
                    ep_idx + 1, step + 1, raw_output[:200]
                )
                logger.info(
                    "  ep=%d step=%d | parsed_action: %s | matched: %s",
                    ep_idx + 1, step + 1, action_text, matched_action
                )
                if eval_logger is not None:
                    eval_logger.info(
                        "  ep=%d step=%d | raw_output: %s",
                        ep_idx + 1, step + 1, raw_output
                    )
                    eval_logger.info(
                        "  ep=%d step=%d | parsed_action: %s | matched: %s",
                        ep_idx + 1, step + 1, action_text, matched_action
                    )

                obs, reward, done, info = env.step(matched_action)

                # 记录 step 结果
                logger.info(
                    "  ep=%d step=%d | reward=%.2f done=%s won=%s | obs: %s",
                    ep_idx + 1, step + 1, reward, done, info["won"], obs[:150]
                )
                if eval_logger is not None:
                    eval_logger.info(
                        "  ep=%d step=%d | reward=%.2f done=%s won=%s | obs: %s",
                        ep_idx + 1, step + 1, reward, done, info["won"], obs
                    )

                history.append((steps, obs, matched_action))
                if len(history) > history_len:
                    history.pop(0)

                if done:
                    won = info["won"]
                    logger.info(
                        "=== Episode %d/%d DONE === won=%s steps=%d reward=%.2f\n",
                        ep_idx + 1, n_episodes, won, steps, reward
                    )
                    if eval_logger is not None:
                        eval_logger.info(
                            "=== Episode %d/%d DONE === won=%s steps=%d reward=%.2f\n",
                            ep_idx + 1, n_episodes, won, steps, reward
                        )
                    break

            results[task_type].append(won)

            ep_time = time.time() - ep_start
            step_time = ep_time / max(steps, 1)
            so_far = sum(sum(v) for v in results.values())
            total_so_far = sum(len(v) for v in results.values())
            running_sr = so_far / total_so_far if total_so_far else 0.0
            elapsed_total = time.time() - eval_start
            # ETA 按本 rank 切片剩余局数算（不是全局 n_episodes）
            eta = elapsed_total / total_so_far * (slice_end - ep_idx - 1) if total_so_far else 0.0

            pbar.set_postfix(task=task_type[:12], won=won, steps=steps, sr=f"{running_sr:.2f}", s_ep=f"{ep_time:.1f}s")
            logger.info(
                "ep=%d/%d  task_type=%-25s  won=%-5s  steps=%2d  "
                "ep_time=%.1fs  step_time=%.1fs  running_sr=%.2f  elapsed=%.0fs  eta=%.0fs",
                ep_idx + 1, n_episodes, task_type, won, steps,
                ep_time, step_time, running_sr, elapsed_total, eta,
            )

    env.close()

    metrics = _aggregate_eval_metrics(results, is_ddp, world_size, device, rank)

    # Restore training mode
    model.train()
    encoder.train()
    projector.train()

    return metrics


# 按照 6 个任务类型的顺序输出（串行聚合与 batch 聚合共用）
_TASK_ORDER = [
    "pick_and_place",
    "look_at_obj_in_light",
    "clean",
    "heat",
    "cool",
    "examine",
]


def _aggregate_eval_metrics(
    results: Dict[str, List[bool]],
    is_ddp: bool,
    world_size: int,
    device,
    rank: int,
) -> Dict[str, float]:
    """
    把每个任务类型的 wins 计数聚合成 success 率，并汇总打印。

    从原 run_eval 内联逻辑抽出，行为不变。DDP 分片下各 rank 只评估一部分
    game，需要 all_gather 合并成全局计数；单卡（独立 eval 脚本）直接用本
    rank 结果。汇总打印只在 rank0（或单卡）执行。
    """
    metrics: Dict[str, float] = {}

    # 每个任务类型聚合 (wins, total)。
    if is_ddp and world_size > 1:
        import torch.distributed as dist
        local_counts = torch.zeros(len(_TASK_ORDER), 2, dtype=torch.long, device=device)
        for ti, tt in enumerate(_TASK_ORDER):
            wins = results.get(tt, [])
            local_counts[ti, 0] = sum(wins)
            local_counts[ti, 1] = len(wins)
        gathered = [torch.zeros_like(local_counts) for _ in range(world_size)]
        dist.all_gather(gathered, local_counts)
        final_counts = torch.stack(gathered).sum(dim=0).cpu()
    else:
        final_counts = torch.zeros(len(_TASK_ORDER), 2, dtype=torch.long)
        for ti, tt in enumerate(_TASK_ORDER):
            wins = results.get(tt, [])
            final_counts[ti, 0] = sum(wins)
            final_counts[ti, 1] = len(wins)

    total_episodes = int(final_counts[:, 1].sum().item())
    for ti, tt in enumerate(_TASK_ORDER):
        wins  = int(final_counts[ti, 0].item())
        total = int(final_counts[ti, 1].item())
        metrics[f"success/{tt}"] = wins / total if total else 0.0
    total_wins = int(final_counts[:, 0].sum().item())
    metrics["success/overall"] = total_wins / total_episodes if total_episodes else 0.0

    # 汇总打印（DDP 只在 rank0 打印，避免重复刷屏）
    if not (is_ddp and world_size > 1) or rank == 0:
        logger.info("\n" + "="*70)
        logger.info("Eval Results: %d episodes", total_episodes)
        logger.info("="*70)
        for ti, tt in enumerate(_TASK_ORDER):
            wins  = int(final_counts[ti, 0].item())
            total = int(final_counts[ti, 1].item())
            rate  = metrics[f"success/{tt}"]
            logger.info("  %-25s  %3d/%3d  %.4f  (%.1f%%)", tt, wins, total, rate, rate * 100)
        logger.info("-" * 70)
        logger.info(
            "  %-25s  %3d/%3d  %.4f  (%.1f%%)",
            "OVERALL", total_wins, total_episodes, metrics["success/overall"],
            metrics["success/overall"] * 100
        )
        logger.info("="*70 + "\n")

    return metrics


class _EvalState:
    """batch 路径中一个并行 episode 流的运行状态（一个 env + 它区间内当前 episode）。"""

    def __init__(self, env) -> None:
        self.env = env
        self.next_ep: int = 0        # 区间内下一个要启动的全局 ep_idx
        self.interval_end: int = 0   # 区间上界（不含）
        self.obs: Optional[str] = None
        self.info: Optional[Dict] = None
        self.history: List = []
        self.steps: int = 0
        self.won: bool = False
        self.done: bool = True       # True = 当前无 episode 在跑（可启动下一个）
        self.ep_idx: Optional[int] = None
        self.exhausted: bool = False # 区间已耗尽


def _start_eval_episode(st: _EvalState, n_episodes_total: int, eval_logger=None) -> None:
    """启动 st 的当前区间内下一个 episode（连续 reset，无放回推进）。"""
    obs, info = st.env.reset()
    st.ep_idx = st.next_ep
    st.next_ep += 1
    st.obs, st.info = obs, info
    st.history = []
    st.steps = 0
    st.won = False
    st.done = False
    logger.info(
        "\n=== Episode %d/%d START ===\ntask_type=%s\ntask_desc=%s\ninitial_obs=%s",
        st.ep_idx + 1, n_episodes_total, info["task_type"], info["task_description"], obs[:200],
    )
    if eval_logger is not None:
        eval_logger.info(
            "\n=== Episode %d/%d START ===\ntask_type=%s\ntask_desc=%s\ninitial_obs=%s",
            st.ep_idx + 1, n_episodes_total, info["task_type"], info["task_description"], obs,
        )


def _run_eval_batched(
    trainer,
    env_factory: Callable,
    n_episodes: int,
    eval_batch_size: int,
    eval_logger=None,
) -> Dict[str, float]:
    """
    多 episode 并行评估（滑动窗口，training.eval_batch_size > 1）。

    设计：B 个 env 各负责 slice 内一段**连续区间**（无放回、互不重叠、恰好
    覆盖 slice）。每轮主循环对当前未完成的 episode 批量生成一步（encoder/
    projector/generate 全部 batch 化，greedy），逐条 env.step；done 的
    记录结果并从批中移除，同一 env 连续 reset 启动区间内下一个 episode。

    与串行路径的等价性：每局用的是同一 greedy 生成、同一 prompt 模板、同一
    停止条件，只把"一次生成 1 局"变成"一次生成多局"——每局决策序列与串行
    完全一致，GPU 利用率更高。无放回由 EVAL_ORDER_SEED 统一洗牌 + 连续
    reset 保证（与串行相同机制，只是把 slice 切成 B 段并行）。
    """
    model      = trainer._base_model
    tokenizer  = trainer.tokenizer
    encoder    = trainer.encoder
    projector  = trainer.projector
    skill_lib  = trainer.skill_lib
    device     = trainer.device
    rcfg       = trainer.cfg.get("rollout", {})

    max_steps      = rcfg.get("max_steps", 50)
    max_new_tokens = rcfg.get("max_new_tokens", 128)
    max_prompt_len = rcfg.get("max_prompt_len", 8192)
    history_len    = rcfg.get("history_len", 3)

    is_ddp     = bool(getattr(trainer, "is_ddp", False))
    rank       = int(getattr(trainer, "rank", 0))
    world_size = int(getattr(trainer, "world_size", 1))

    model.eval()
    encoder.eval()
    projector.eval()

    results: Dict[str, List[bool]] = defaultdict(list)
    eval_start = time.time()
    # exp4: encoder 直接吃原始文本，不再需要预先算 embedding 的缓存
    # （旧 skill_emb_cache 删除，与串行路径一致）。

    # ── 无放回顺序 + DDP 分片（与串行相同的机制）──────────────────────────
    probe = env_factory()
    reuse_env = hasattr(probe, "reseed")
    pool_size = getattr(probe, "num_games", None)
    if pool_size and n_episodes > pool_size:
        logger.warning(
            "n_episodes=%d 超过 game 池大小 %d，已截断为 pool 大小（无放回上限）",
            n_episodes, pool_size,
        )
        n_episodes = pool_size
    probe.close()

    if reuse_env:
        np.random.seed(EVAL_ORDER_SEED)   # 所有 env 共用同一洗牌顺序的前提
    if is_ddp and world_size > 1:
        n_per_rank  = -(-n_episodes // world_size)
        slice_start = rank * n_per_rank
        slice_end   = min(slice_start + n_per_rank, n_episodes)
    else:
        slice_start, slice_end = 0, n_episodes
    logger.info("rank=%d eval slice: episodes [%d, %d)", rank, slice_start, slice_end)

    n_slice = slice_end - slice_start
    if n_slice <= 0:
        return _aggregate_eval_metrics({}, is_ddp, world_size, device, rank)

    B = min(eval_batch_size, n_slice)
    chunk = -(-n_slice // B)   # 每个 env 负责的连续 episode 数（ceil）

    # ── 构造 B 个 env，各推进到区间起点，启动第一个 episode ──────────────
    states: List[_EvalState] = []
    for k in range(B):
        start = slice_start + k * chunk
        end   = min(slice_start + (k + 1) * chunk, slice_end)
        if start >= end:
            break
        e = env_factory()
        if reuse_env:
            np.random.seed(EVAL_ORDER_SEED)
            e.reseed(EVAL_ORDER_SEED)
            for _ in range(start):     # 连续 reset 推进到区间起点（无放回）
                e.reset()
        st = _EvalState(e)
        st.next_ep = start
        st.interval_end = end
        _start_eval_episode(st, n_episodes, eval_logger)
        states.append(st)

    n_done = 0
    pbar = tqdm(total=n_slice, desc="Eval", unit="ep", dynamic_ncols=True)

    with torch.no_grad():
        while not all(st.exhausted for st in states):
            # 启动所有"当前无 episode 且区间未耗尽"的流
            for st in states:
                if not st.exhausted and st.done:
                    if st.next_ep < st.interval_end:
                        _start_eval_episode(st, n_episodes, eval_logger)
                    else:
                        st.exhausted = True
            active = [st for st in states if not st.done]
            if not active:
                break

            # ── 批量构造 prompt / state_text / skill_text ──────────────────
            # exp4: encoder 直接吃原始文本（与 training/rollout.py 的
            # state_text 格式一致），不再预先算 embedding。
            prompts: List[str] = []
            state_texts: List[str] = []
            skill_texts: List[str] = []
            for st in active:
                skill = skill_lib.retrieve_for_encoder(
                    st.info["task_description"], task_type=st.info.get("task_type")
                )
                skill_text = skill.grounding_text
                skill_texts.append(skill_text)
                state_texts.append(
                    f"Task: {st.info['task_description']}\nObservation: {st.obs}"
                )
                prompts.append(_build_eval_prompt(
                    st.obs, st.info, st.history, st.steps + 1, skill_lib, skill_text
                ))

            # ── tokenize（padding）──────────────────────────────────────────
            chat_texts = []
            for p in prompts:
                msg = [{"role": "user", "content": p}]
                chat_texts.append(
                    tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                )
            enc = tokenizer(
                chat_texts, return_tensors="pt", padding=True, truncation=True,
                max_length=max_prompt_len,
            ).to(device)

            # ── 批量 encoder → projector → soft prefix ──────────────────────
            mu, log_var, _pooled_state = encoder(state_texts, skill_texts)
            z_tilde = sample_z(mu, log_var)                            # [n, L]
            soft_prefix = projector(z_tilde)                           # [n, m, H]

            embed_layer = model.get_input_embeddings()
            input_embeds = embed_layer(enc.input_ids)                  # [n, seq, H]
            inputs_embeds = torch.cat(
                [soft_prefix.to(input_embeds.dtype), input_embeds], dim=1
            )
            prefix_mask = torch.ones(
                len(active), soft_prefix.size(1),
                dtype=enc.attention_mask.dtype, device=device,
            )
            attention_mask = torch.cat([prefix_mask, enc.attention_mask], dim=1)

            output_ids = model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,            # greedy decoding for eval
                pad_token_id=tokenizer.eos_token_id,
                stopping_criteria=build_action_stop_criteria(tokenizer),
            )
            # output_ids: [n, gen_len]（inputs_embeds 路径只返回新生成的 token）

            # ── 逐条 parse / step / 记录 ────────────────────────────────────
            for j, st in enumerate(active):
                raw_output = tokenizer.decode(output_ids[j], skip_special_tokens=True)
                action_text, _ = parse_action(raw_output)
                matched_action = match_admissible(
                    action_text, st.info["admissible_commands"]
                )
                st.steps += 1
                ep_label = st.ep_idx + 1

                logger.info(
                    "  ep=%d step=%d | raw_output: %s", ep_label, st.steps, raw_output[:200]
                )
                logger.info(
                    "  ep=%d step=%d | parsed_action: %s | matched: %s",
                    ep_label, st.steps, action_text, matched_action,
                )
                if eval_logger is not None:
                    eval_logger.info(
                        "  ep=%d step=%d | raw_output: %s", ep_label, st.steps, raw_output
                    )
                    eval_logger.info(
                        "  ep=%d step=%d | parsed_action: %s | matched: %s",
                        ep_label, st.steps, action_text, matched_action,
                    )

                obs, reward, done, info = st.env.step(matched_action)
                st.obs, st.info = obs, info
                st.history.append((st.steps, obs, matched_action))
                if len(st.history) > history_len:
                    st.history.pop(0)

                logger.info(
                    "  ep=%d step=%d | reward=%.2f done=%s won=%s | obs: %s",
                    ep_label, st.steps, reward, done, info["won"], obs[:150],
                )
                if eval_logger is not None:
                    eval_logger.info(
                        "  ep=%d step=%d | reward=%.2f done=%s won=%s | obs: %s",
                        ep_label, st.steps, reward, done, info["won"], obs,
                    )

                if done:
                    st.won = info["won"]
                    st.done = True
                    results[st.info["task_type"]].append(st.won)
                    n_done += 1
                    pbar.update(1)
                    logger.info(
                        "=== Episode %d/%d DONE === won=%s steps=%d reward=%.2f\n",
                        ep_label, n_episodes, st.won, st.steps, reward,
                    )
                    if eval_logger is not None:
                        eval_logger.info(
                            "=== Episode %d/%d DONE === won=%s steps=%d reward=%.2f\n",
                            ep_label, n_episodes, st.won, st.steps, reward,
                        )
                    elapsed_total = time.time() - eval_start
                    running_sr = n_done and (sum(sum(v) for v in results.values()) / n_done)
                    eta = elapsed_total / n_done * (n_slice - n_done) if n_done else 0.0
                    pbar.set_postfix(
                        task=st.info["task_type"][:12], won=st.won,
                        steps=st.steps, sr=f"{running_sr:.2f}", eta=f"{eta:.0f}s",
                    )

    pbar.close()
    for st in states:
        st.env.close()

    return _aggregate_eval_metrics(results, is_ddp, world_size, device, rank)
