"""
scripts/eval_baseline.py

exp10：7B 原始模型 baseline 评估（无 InfoSkill 模块）。

与 InfoSkill 正式 eval 的唯一差异：不用 encoder/projector 的 VAE 软前缀注入，
模型直接对 prompt 文本（含 skill 检索文本）用 input_ids 生成动作。skill 库
照常检索并灌进 prompt（保留 prompt + skill，隔离"软前缀注入"这个变量的贡献）。

复用的口径（与 eval/evaluate.py 完全一致）：
  - 无放回 140 局（EVAL_ORDER_SEED 洗牌 + 连续 reset）
  - greedy 生成、build_action_stop_criteria 提前停止
  - 6 子数据集 success 率 + overall（_aggregate_eval_metrics 汇总打印）
  - eval_batch_size 多 episode 并行（滑动窗口，默认 4）

Usage:
    python scripts/eval_baseline.py --config configs/alfworld.yaml [--n_episodes 140] [--eval-batch-size 4]
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import yaml
from tqdm import tqdm

logger = logging.getLogger("eval_baseline")


def setup_logging(log_path: str) -> None:
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(fmt))
    logging.getLogger().addHandler(fh)


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


class _EvalState:
    """batch 路径中一个并行 episode 流的运行状态（一个 env + 它区间内当前 episode）。"""

    def __init__(self, env) -> None:
        self.env = env
        self.next_ep: int = 0
        self.interval_end: int = 0
        self.obs = None
        self.info = None
        self.history = []
        self.steps = 0
        self.won = False
        self.done = True       # True = 当前无 episode 在跑（可启动下一个）
        self.ep_idx = None
        self.exhausted = False
        self.ep_record = None  # per-episode 明细（本 episode 完成后写入全局）


def main():
    parser = argparse.ArgumentParser(description="exp10: 7B 原始模型 baseline 评估（无 InfoSkill 模块）")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--n_episodes", type=int, default=140, help="Number of eval episodes (default: 140, full eval_in_distribution)")
    parser.add_argument("--mode", choices=["serial", "batch"], default="batch",
                        help="serial=逐局串行（日志每局完整不交错，便于跟输入输出）；"
                             "batch=多局并行（更快，日志交错）。")
    parser.add_argument("--eval-batch-size", type=int, default=4, help="batch 模式下的并行 episode 数（滑动窗口）")
    parser.add_argument("--log-file", default=None, help="Path to log file (auto-generated if not set)")
    args = parser.parse_args()

    # ── 日志 ───────────────────────────────────────────────────────────────────
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    if args.log_file:
        log_path = args.log_file
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = str(log_dir / f"eval_baseline_{timestamp}.log")
    setup_logging(log_path)
    logger.info("Log file: %s", log_path)

    cfg = load_config(args.config)
    device = torch.device(cfg.get("device", "cuda"))

    # ── 原始模型（不加载任何 checkpoint / aux 模块权重） ─────────────────────
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = cfg["model"]["backbone"]
    logger.info("Loading raw backbone (no InfoSkill modules): %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # ── Skill library（照常检索；baseline 只用它灌 prompt 文本，不喂 encoder） ─
    from skill_library.library import SkillLibrary

    skill_lib = SkillLibrary(
        json_path=cfg["paths"]["skills_json"],
        model=model,
        tokenizer=tokenizer,
        device=device,
        top_k_task=cfg["skill_library"]["top_k_task"],
        max_skills=cfg["skill_library"]["max_skills"],
    )

    # ── 复用正式 eval 的公共 helper（不改 evaluate.py） ──────────────────────
    from eval.evaluate import EVAL_ORDER_SEED, _aggregate_eval_metrics, _build_eval_prompt
    from envs.alfworld_env import AlfworldTextEnv
    from utils.action_parser import match_admissible, parse_action
    from utils.stopping_criteria import build_action_stop_criteria

    rcfg = cfg.get("rollout", {})
    max_steps      = rcfg.get("max_steps", 50)
    max_new_tokens = rcfg.get("max_new_tokens", 128)
    max_prompt_len = rcfg.get("max_prompt_len", 8192)
    history_len    = rcfg.get("history_len", 3)

    n_episodes   = args.n_episodes
    # 串行/并行开关：serial 强制单局（日志不交错、好跟输入输出），
    # batch 用 --eval-batch-size 控制并行度（更快、日志交错）
    eval_batch_size = 1 if args.mode == "serial" else args.eval_batch_size

    config_path = cfg["paths"]["alfworld_config"]
    env_seed_counter = {"count": 0}

    def env_factory():
        s = env_seed_counter["count"]
        env_seed_counter["count"] += 1
        return AlfworldTextEnv(
            config_path=config_path,
            train_eval="eval_in_distribution",
            seed=s,
            max_steps=max_steps,
        )

    # ── 无放回顺序（与正式 eval 同一机制） ───────────────────────────────────
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

    n_slice = n_episodes
    B = min(eval_batch_size, n_slice)
    chunk = -(-n_slice // B)   # 每个 env 负责的连续 episode 数（ceil）

    # ── 构造 B 个 env，各推进到区间起点，启动第一个 episode ──────────────────
    states: list = []
    for k in range(B):
        start = k * chunk
        end = min((k + 1) * chunk, n_slice)
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
        _start_episode(st, n_episodes, logger)
        states.append(st)

    results: dict = defaultdict(list)
    per_episode: list = []
    n_done = 0
    eval_start = time.time()
    pbar = tqdm(total=n_slice, desc="Eval", unit="ep", dynamic_ncols=True)

    with torch.no_grad():
        while not all(st.exhausted for st in states):
            # 启动所有"当前无 episode 且区间未耗尽"的流
            for st in states:
                if not st.exhausted and st.done:
                    if st.next_ep < st.interval_end:
                        _start_episode(st, n_episodes, logger)
                    else:
                        st.exhausted = True
            active = [st for st in states if not st.done]
            if not active:
                break

            # ── 批量构造 prompt（skill 文本照常灌入） ───────────────────────
            prompts = []
            for st in active:
                skill = skill_lib.retrieve_for_encoder(
                    st.info["task_description"], task_type=st.info.get("task_type")
                )
                skill_text = skill.grounding_text
                st.skill_text = skill_text
                prompts.append(_build_eval_prompt(
                    st.obs, st.info, st.history, st.steps + 1, skill_lib, skill_text
                ))

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

            # ── 直接 input_ids 生成（无 soft prefix） ───────────────────────
            output_ids = model.generate(
                input_ids=enc.input_ids,
                attention_mask=enc.attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,            # greedy decoding（与正式 eval 一致）
                pad_token_id=tokenizer.eos_token_id,
                stopping_criteria=build_action_stop_criteria(tokenizer),
            )
            # input_ids 路径返回完整序列 [n, prompt_len + gen_len]，截掉 prompt
            prompt_len = enc.input_ids.shape[1]

            # ── 逐条 parse / step / 记录 ─────────────────────────────────────
            for j, st in enumerate(active):
                gen_ids = output_ids[j, prompt_len:]
                raw_output = tokenizer.decode(gen_ids, skip_special_tokens=True)
                action_text, _ = parse_action(raw_output)
                matched_action = match_admissible(
                    action_text, st.info["admissible_commands"]
                )
                st.steps += 1
                ep_label = st.ep_idx + 1

                # 大模型输入（prompt）+ 输出（raw_output）完整打印，便于调试
                logger.info(
                    "  ep=%d step=%d | PROMPT:\n%s", ep_label, st.steps, prompts[j]
                )
                logger.info(
                    "  ep=%d step=%d | raw_output: %s", ep_label, st.steps, raw_output
                )
                logger.info(
                    "  ep=%d step=%d | parsed_action: %s | matched: %s",
                    ep_label, st.steps, action_text, matched_action,
                )

                obs, reward, done, info = st.env.step(matched_action)
                st.obs, st.info = obs, info
                st.ep_record["actions"].append(
                    {"step": st.steps, "action": matched_action,
                     "parsed": action_text, "raw": raw_output[:200]}
                )
                st.history.append((st.steps, obs, matched_action))
                if len(st.history) > history_len:
                    st.history.pop(0)

                logger.info(
                    "  ep=%d step=%d | reward=%.2f done=%s won=%s | obs: %s",
                    ep_label, st.steps, reward, done, info["won"], obs[:150],
                )

                if done:
                    st.won = info["won"]
                    st.done = True
                    results[st.info["task_type"]].append(st.won)
                    st.ep_record["won"] = st.won
                    st.ep_record["steps"] = st.steps
                    st.ep_record["reward"] = reward
                    per_episode.append(st.ep_record)
                    n_done += 1
                    pbar.update(1)
                    elapsed_total = time.time() - eval_start
                    running_sr = sum(sum(v) for v in results.values()) / n_done
                    eta = elapsed_total / n_done * (n_slice - n_done)
                    logger.info(
                        "  [progress] done=%d/%d  running_sr=%.2f  elapsed=%.0fs  eta=%.0fs",
                        n_done, n_slice, running_sr, elapsed_total, eta,
                    )
                    logger.info(
                        "=== Episode %d/%d DONE === won=%s steps=%d reward=%.2f\n",
                        ep_label, n_episodes, st.won, st.steps, reward,
                    )

    pbar.close()
    for st in states:
        st.env.close()

    # ── 汇总（打印 6 子数据集 + overall） + 存明细 ──────────────────────────
    metrics = _aggregate_eval_metrics(results, False, 1, device, 0)

    # 汇总打印（复用 _aggregate_eval_metrics 已打印的格式，再打一份到 stdout 保险）
    print("\n" + "=" * 70)
    print("exp10 Baseline Eval Results (raw Qwen2.5-7B, no InfoSkill modules)")
    print("=" * 70)
    for tt in _TASK_ORDER:
        print(f"  {tt:<30s} {metrics.get(f'success/{tt}', 0.0):.4f}  ({metrics.get(f'success/{tt}', 0.0)*100:.1f}%)")
    print("-" * 70)
    print(f"  {'OVERALL':<30s} {metrics.get('success/overall', 0.0):.4f}  ({metrics.get('success/overall', 0.0)*100:.1f}%)")
    print("=" * 70)

    per_ep_path = os.path.join(log_dir, f"eval_baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}_per_episode.json")
    with open(per_ep_path, "w", encoding="utf-8") as f:
        json.dump(per_episode, f, ensure_ascii=False, indent=2)
    logger.info("Per-episode detail saved to %s", per_ep_path)


def _start_episode(st, n_episodes_total: int, logger) -> None:
    """启动 st 的当前区间内下一个 episode（连续 reset，无放回推进）。"""
    obs, info = st.env.reset()
    st.ep_idx = st.next_ep
    st.next_ep += 1
    st.obs, st.info = obs, info
    st.history = []
    st.steps = 0
    st.won = False
    st.done = False
    st.ep_record = {
        "ep_idx": st.ep_idx,
        "task_type": info["task_type"],
        "task_desc": info["task_description"],
        "won": False,
        "steps": 0,
        "reward": 0.0,
        "actions": [],
    }
    logger.info(
        "\n=== Episode %d/%d START ===\ntask_type=%s\ntask_desc=%s\ninitial_obs=%s",
        st.ep_idx + 1, n_episodes_total, info["task_type"], info["task_description"], obs[:200],
    )


# 6 个子数据集顺序（与 evaluate.py 一致）
_TASK_ORDER = [
    "pick_and_place",
    "look_at_obj_in_light",
    "clean",
    "heat",
    "cool",
    "examine",
]


if __name__ == "__main__":
    main()
