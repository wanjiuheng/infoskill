"""
training/rollout.py

GroupRolloutCollector: runs G parallel AlfworldTextEnv instances,
collects a full Trajectory Buffer, and returns everything needed for loss computation.

Design:
  - Input is `task_groups`: a list of task groups, each `group_size` envs
    reset to the SAME task (same seed) — one GRPO Group. `task_groups` may
    hold more than one task (rollout.tasks_per_batch > 1); all groups are
    flattened into one batch for generate(), but each group's rewards are
    normalised into GRPO advantages independently (see
    InfoskillTrainer._fast_update, which slices by `buf.group_size`).
  - At each step: batch-forward through Encoder → Projector → LLM.generate().
  - Active Mask tracks which episodes are still running — this is per-episode
    and task-agnostic, so tasks that finish at different times (e.g. one task
    group hits `done` well before another) are handled automatically: the
    batch naturally shrinks to just the still-running episodes.
  - Completed episodes stop calling env.step() but stay in the batch (zero-masked).
  - Returns a TrajectoryBuffer dict ready for loss computation.
"""

from __future__ import annotations

import logging
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any

from envs.alfworld_env import AlfworldTextEnv
from utils.action_parser import parse_action, match_admissible
from utils.stopping_criteria import build_action_stop_criteria

logger = logging.getLogger("rollout")


# ── Prompt template ───────────────────────────────────────────────────────────

_STEP_PROMPT = """You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}
## Retrieved Relevant Experience
{skill_guidance}
## Current Progress
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_len} observations and the corresponding actions you took: {history}
You are now at step {current_step} and your current observation is: {obs}
Your admissible actions of the current situation are: [{admissible}].
Now it's your turn to take an action. You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""


# ── Data container ────────────────────────────────────────────────────────────

@dataclass
class StepRecord:
    """
    One step from one episode within a Group.

    Everything here is produced under torch.no_grad() during rollout — no
    computation graph is kept alive across steps. `_fast_update()` uses the
    raw ids/embeddings stored here to recompute a fresh, differentiable
    log-prob (and z_tilde/soft_prefix) at update time, one mini-batch at a
    time, instead of holding G × max_steps graphs in memory simultaneously.
    """
    ep_idx:      int           # which episode (0..G-1)
    step_idx:    int           # step number within that episode
    state_text:  str           # "Task: {task_description}\nObservation: {obs}"
                                # — fed to the T5 encoder directly (exp4:
                                # no pre-pooled Qwen embedding any more；
                                # _fast_update recomputes mu/log_var/pooled_state
                                # fresh from this text each update).
    skill_text:  str           # skill.grounding_text, also fed to T5 encoder directly
    mu:          torch.Tensor  # [latent_dim], detached
    log_var:     torch.Tensor  # [latent_dim], detached
    z_tilde:     torch.Tensor  # [latent_dim], detached — the actual sampled latent used for generation
    eps:         torch.Tensor  # [latent_dim], detached — reparameterisation noise (z = mu + std*eps),
                                # stored so _fast_update can reconstruct the *exact same* z_tilde with
                                # gradient attached (mu/log_var recomputed fresh through the encoder).
    prompt_ids:  torch.Tensor  # [prompt_len] long, detached — trimmed (no padding)
    gen_ids:     torch.Tensor  # [gen_len] long, detached — trimmed (no padding)
    action:      str
    reward:      float
    done:        bool
    is_valid:    bool          # action was valid (<action> tags + no Chinese)
    is_padding:  bool = False  # True for shape-alignment placeholders (episode already done)


@dataclass
class TrajectoryBuffer:
    """Holds all StepRecords from one full rollout (one or more task groups)."""
    records:      List[StepRecord] = field(default_factory=list)
    total_rewards: List[float]     = field(default_factory=list)  # length G_total
    total_steps:  List[int]        = field(default_factory=list)  # length G_total: 每个 episode 的实际步数
    group_size:   int              = 0  # 每个 GRPO 组的局数；trainer 据此把
                                          # total_rewards/records 按任务组切片
                                          # 分别算 advantage（见 _fast_update）
    success_trajectories: List[Dict] = field(default_factory=list)  # for Slow Module
    task_info_per_episode: List[Dict] = field(default_factory=list)  # 每个 episode 自己的 task 信息


# ── Collector ─────────────────────────────────────────────────────────────────

class GroupRolloutCollector:
    """
    Manages one or more GRPO task groups' environments and collects one
    full rollout (possibly spanning multiple tasks in a single batch).

    Args:
        task_groups: List of task groups. Each inner list has `group_size`
                     AlfworldTextEnv instances reset to the SAME task (same
                     seed) — one GRPO Group. When len(task_groups) > 1
                     (rollout.tasks_per_batch > 1), multiple independent
                     tasks are batched together into one generate() call for
                     throughput, but their rewards are normalised into GRPO
                     advantages separately by the caller (InfoskillTrainer),
                     using `buf.group_size` to slice `buf.total_rewards`.
                     All inner lists must have the same length.
        model:       Qwen2.5-7B-Instruct model with LoRA (on device).
        tokenizer:   Matching tokenizer.
        encoder:     StateConditionalEncoder.
        projector:   Projector.
        skill_lib:   SkillLibrary (for retrieval + grounding text).
        device:      Compute device.
        cfg:         Rollout config dict with keys:
                       max_steps, max_new_tokens, max_prompt_len, temperature,
                       top_p, history_len.
    """

    def __init__(
        self,
        task_groups: List[List[AlfworldTextEnv]],
        model,
        tokenizer,
        encoder,
        projector,
        skill_lib,
        device:     torch.device,
        cfg:        Dict[str, Any],
        action_logger = None,  # Optional logger for recording actions
        success_reward_threshold: float = 5.0,  # skill_library.success_reward_threshold
    ) -> None:
        assert len(task_groups) >= 1, "task_groups must have at least one task group"
        group_size = len(task_groups[0])
        assert all(len(g) == group_size for g in task_groups), (
            f"每个任务组的局数必须一致，got sizes {[len(g) for g in task_groups]}"
        )
        self.group_size = group_size
        # 拍平成一个扁平列表去跑 batch generate()；顺序是任务1的 group_size
        # 个 env、任务2的 group_size 个 env、……——trainer 按同样的 group_size
        # 切片对应回去算 GRPO advantage（见 TrajectoryBuffer.group_size）。
        self.envs       = [env for group in task_groups for env in group]
        self.model      = model
        self.tokenizer  = tokenizer
        self.encoder    = encoder
        self.projector  = projector
        self.skill_lib  = skill_lib
        self.device     = device
        self.G          = len(self.envs)  # 总 batch 大小 = group_size * len(task_groups)
        self.action_logger = action_logger
        self.success_reward_threshold = success_reward_threshold

        self.max_steps       = cfg.get("max_steps", 50)
        self.max_new_tokens  = cfg.get("max_new_tokens", 128)
        self.max_prompt_len  = cfg.get("max_prompt_len", 8192)
        self.temperature     = cfg.get("temperature", 0.9)
        self.top_p           = cfg.get("top_p", 0.9)
        self.history_len     = cfg.get("history_len", 3)
        # exp4: 不再需要 skill embedding 缓存——encoder 直接吃 skill_text
        # 原始文本，不预先算向量（旧 _skill_emb_cache 删除）。

    # ── Public API ────────────────────────────────────────────────────────────

    def collect(self) -> TrajectoryBuffer:
        """
        Run one complete Group rollout (all G episodes until done or max_steps).

        Returns:
            TrajectoryBuffer with all step records and per-episode total rewards.
        """
        buf = TrajectoryBuffer()

        # ── Reset all G envs ──────────────────────────────────────────────────
        obs_list:      List[str]           = []
        info_list:     List[Dict]          = []
        history_list:  List[List[Tuple]]   = [[] for _ in range(self.G)]
        active_mask:   List[bool]          = [True] * self.G
        ep_rewards:    List[float]         = [0.0]  * self.G
        ep_steps_buf:  List[List[Dict]]    = [[]    for _ in range(self.G)]  # for skill gen
        ep_token_total: List[int]          = [0]    * self.G  # 每个 episode 累计生成的 token 数
        ep_step_count:  List[int]          = [0]    * self.G  # 每个 episode 实际走了多少步

        for env in self.envs:
            obs, info = env.reset()
            obs_list.append(obs)
            info_list.append(info)

        buf.group_size = self.group_size
        buf.task_info_per_episode = [
            {
                "task_description": info["task_description"],
                "task_type": info["task_type"],
            }
            for info in info_list
        ]

        # 诊断（tasks_per_batch>1 时确认多任务正确性）：按任务组打印每组
        # task_description 的唯一性。组内（同 seed）必须 1 个唯一值；组间
        # （不同 seed）必须不同——若组间相同，说明底层 seed 洗牌没生效，
        # 两个任务实际是同一个（tpb=2 的正确性前提被破坏）。
        # 注意：Step 日志里 8 个 env token 数相同是 all 语义 batch 同步的
        # 必然结果，不是任务相同的证据，只有这里打印的 task_description 能
        # 真正区分任务。
        for gi in range(0, len(info_list), self.group_size):
            group = info_list[gi:gi + self.group_size]
            descs = {g["task_description"] for g in group}
            logger.info(
                "Task group %d (episodes %d-%d): %d unique task_desc%s",
                gi // self.group_size, gi, gi + self.group_size - 1, len(descs),
                " (SAME!)" if len(descs) == 1 else " (DIFFERENT!)",
            )
            for j, g in enumerate(group):
                logger.info(
                    "  task group %d ep %d: %s", gi // self.group_size, gi + j,
                    g["task_description"],
                )

        # ── Step loop ─────────────────────────────────────────────────────────
        # Everything in this loop runs under no_grad: rollout never needs a
        # computation graph. That graph is rebuilt, one mini-batch at a time,
        # inside InfoskillTrainer._fast_update() — this is what decouples
        # rollout memory from max_steps * group_size.
        with torch.no_grad():
            for step_idx in range(self.max_steps):
                # Early exit if all episodes are done
                active_indices = [i for i in range(self.G) if active_mask[i]]
                if not active_indices:
                    break

                # 1. Retrieve skill (only for active episodes)
                skill_texts_active = self._batch_retrieve_skills_active(
                    obs_list, info_list, active_indices
                )

                # 2. Build state_text (task + current obs) for active episodes
                #    — exp4: encoder 直接吃原始文本（T5 tokenizer 内部编码），
                #    不再预先用 Qwen mean-pool 出向量（旧 _batch_embed_states_active
                #    已删除，见 models/encoder_t5.py 的方案A格式）。
                state_texts_active = self._batch_build_state_texts_active(
                    obs_list, info_list, active_indices
                )

                # 3. Encode → z_tilde (only for active episodes)
                mu_active, log_var_active, pooled_state_active = self.encoder(
                    state_texts_active, skill_texts_active
                )  # mu/log_var: [n_active, L], pooled_state: [n_active, D_t5]
                std_active = torch.exp(0.5 * log_var_active)
                eps_active = torch.randn_like(std_active)
                z_tilde_active = mu_active + std_active * eps_active  # [n_active, L]

                # 4. Project → soft prefix (only for active episodes)
                soft_prefix_active = self.projector(z_tilde_active)  # [n_active, m, H]

                # 5. Build prompts and run LLM.generate() (only for active episodes)
                actions_raw_active, prompt_ids_active, gen_ids_active, token_counts = self._batch_generate_active(
                    obs_list, info_list, history_list, skill_texts_active,
                    soft_prefix_active, active_indices, ep_step_count,
                )

                # Map active results back to full G indices
                # exp4: state_text/skill_text 是 str，直接用列表存（不再是 Tensor），
                # pooled_state 不需要跨 no_grad 边界保留——它只在本 step 内部用于
                # GroundingDecoder 的 grounding loss（若在 rollout 阶段计算的话），
                # 但目前 grounding loss 只在 _fast_update 里按存储的 state_text
                # 重新过 encoder 计算，rollout 阶段不需要保留 pooled_state。
                skill_texts_full = [""] * self.G
                state_texts_full = [""] * self.G
                mu_full = torch.zeros(self.G, mu_active.size(1), device=mu_active.device)
                log_var_full = torch.zeros(self.G, log_var_active.size(1), device=log_var_active.device)
                z_tilde_full = torch.zeros(self.G, z_tilde_active.size(1), device=z_tilde_active.device)
                eps_full = torch.zeros(self.G, eps_active.size(1), device=eps_active.device)
                actions_raw_full = [""] * self.G
                prompt_ids_full = [torch.tensor([]) for _ in range(self.G)]
                gen_ids_full = [torch.tensor([]) for _ in range(self.G)]

                for idx, i in enumerate(active_indices):
                    skill_texts_full[i] = skill_texts_active[idx]
                    state_texts_full[i] = state_texts_active[idx]
                    mu_full[i] = mu_active[idx]
                    log_var_full[i] = log_var_active[idx]
                    z_tilde_full[i] = z_tilde_active[idx]
                    eps_full[i] = eps_active[idx]
                    actions_raw_full[i] = actions_raw_active[idx]
                    prompt_ids_full[i] = prompt_ids_active[idx]
                    gen_ids_full[i] = gen_ids_active[idx]

                # Log per-env token counts for this step, and accumulate per-episode totals
                if token_counts:
                    detail = ", ".join(
                        f"env_{i}={tok}tok" for i, tok in zip(active_indices, token_counts)
                    )
                    logger.info(f"Step {step_idx}: {detail}")
                    for idx, i in enumerate(active_indices):
                        ep_token_total[i] += token_counts[idx]
                        ep_step_count[i]  += 1

                # 6. Parse actions, step envs, record
                for i in range(self.G):
                    if not active_mask[i]:
                        # Still create a placeholder record to keep tensor shapes aligned
                        buf.records.append(StepRecord(
                            ep_idx=i, step_idx=step_idx,
                            state_text=state_texts_full[i], skill_text=skill_texts_full[i],
                            mu=mu_full[i].detach(), log_var=log_var_full[i].detach(),
                            z_tilde=z_tilde_full[i].detach(), eps=eps_full[i].detach(),
                            prompt_ids=prompt_ids_full[i], gen_ids=gen_ids_full[i],
                            action="", reward=0.0, done=True, is_valid=False,
                            is_padding=True,
                        ))
                        continue

                    action_text, is_valid = parse_action(actions_raw_full[i])
                    matched_action = match_admissible(
                        action_text, info_list[i]["admissible_commands"]
                    )

                    # Log full raw output to action logger (不截断)
                    if self.action_logger is not None:
                        self.action_logger.info(
                            "[Episode %d/%d, Step %d] raw_output: %s",
                            i + 1, self.G, step_idx + 1, actions_raw_full[i]
                        )
                        self.action_logger.info(
                            "[Episode %d/%d, Step %d] parsed_action: %s | matched: %s",
                            i + 1, self.G, step_idx + 1, action_text, matched_action
                        )

                    next_obs, reward, done, next_info = self.envs[i].step(matched_action)

                    ep_rewards[i] += reward
                    history_list[i].append((ep_step_count[i], obs_list[i], matched_action))
                    if len(history_list[i]) > self.history_len:
                        history_list[i].pop(0)

                    ep_steps_buf[i].append({"obs": obs_list[i], "action": matched_action})

                    buf.records.append(StepRecord(
                        ep_idx=i, step_idx=step_idx,
                        state_text=state_texts_full[i], skill_text=skill_texts_full[i],
                        mu=mu_full[i].detach(), log_var=log_var_full[i].detach(),
                        z_tilde=z_tilde_full[i].detach(), eps=eps_full[i].detach(),
                        prompt_ids=prompt_ids_full[i], gen_ids=gen_ids_full[i],
                        action=matched_action, reward=reward, done=done, is_valid=is_valid,
                        is_padding=False,
                    ))

                    obs_list[i]  = next_obs
                    info_list[i] = next_info

                    if done:
                        active_mask[i] = False
                        avg_tok = ep_token_total[i] / max(ep_step_count[i], 1)
                        logger.info(
                            f"Episode {i} finished: {ep_step_count[i]} steps, "
                            f"{ep_token_total[i]} tokens total (avg {avg_tok:.1f} tokens/step), "
                            f"reward={ep_rewards[i]:.2f}, success={ep_rewards[i] >= 1.0}"
                        )

        # Any episode that ran to max_steps without setting done=True never hit
        # the "Episode finished" log above — log it here.
        for i in range(self.G):
            if active_mask[i]:
                avg_tok = ep_token_total[i] / max(ep_step_count[i], 1)
                logger.info(
                    f"Episode {i} finished: {ep_step_count[i]} steps, "
                    f"{ep_token_total[i]} tokens total (avg {avg_tok:.1f} tokens/step), "
                    f"reward={ep_rewards[i]:.2f}, success={ep_rewards[i] >= 1.0} (hit max_steps)"
                )

        buf.total_rewards = ep_rewards
        buf.total_steps = ep_step_count  # 记录每个 episode 的步数
        group_success = sum(1 for r in ep_rewards if r >= 1.0)
        logger.info(
            f"Group finished: {self.G} episodes, "
            f"avg {sum(ep_step_count) / self.G:.1f} steps, "
            f"avg {sum(ep_token_total) / self.G:.1f} tokens, "
            f"success_rate={group_success / self.G:.2f}"
        )

        # Collect successful trajectories for Slow Module
        # 阈值来自 skill_library.success_reward_threshold（configs/alfworld.yaml）
        for i in range(self.G):
            if ep_rewards[i] >= self.success_reward_threshold:
                buf.success_trajectories.append({
                    "task": buf.task_info_per_episode[i]["task_description"],
                    "task_type": buf.task_info_per_episode[i]["task_type"],
                    "steps": ep_steps_buf[i],
                })

        return buf

    # ── Internals ─────────────────────────────────────────────────────────────

    def _batch_build_state_texts_active(
        self,
        obs_list:       List[str],
        info_list:      List[Dict],
        active_indices: List[int],
    ) -> List[str]:
        """
        构造 encoder 侧的 state_text（exp4 方案A：直接拼 task_description + obs
        原始文本一次性交给 T5 tokenizer，不做结构化分段）。不再预先用 Qwen
        mean-pool 出向量——encoder（T5StateConditionalEncoder）内部自己
        tokenize + forward。
        """
        return [
            f"Task: {info_list[i]['task_description']}\nObservation: {obs_list[i]}"
            for i in active_indices
        ]

    def _batch_retrieve_skills_active(
        self,
        obs_list:       List[str],
        info_list:      List[Dict],
        active_indices: List[int],
    ) -> List[str]:
        """Retrieve skills only for active episodes. Returns skill_texts only —
        exp4 的 encoder 直接吃 skill_text 原始文本，不需要预先算 embedding
        （旧的 _skill_emb_cache 相应删除，见 __init__）。"""
        skill_texts = []
        for i in active_indices:
            skill = self.skill_lib.retrieve_for_encoder(
                info_list[i]["task_description"],
                task_type=info_list[i].get("task_type"),
            )
            skill_texts.append(skill.grounding_text)
        return skill_texts

    def _build_prompt(
        self,
        obs:        str,
        info:       Dict,
        history:    List[Tuple],
        skill_text: str,
        step_count: int,
    ) -> str:
        """Render the step prompt string for one episode.

        step_count: steps already taken before this step (may exceed
                    len(history), which is window-truncated).
        """
        # History: "(step N) obs → action" pairs with absolute step numbers
        hist_lines = []
        for h_step, h_obs, h_act in history:
            hist_lines.append(f"Step {h_step}: Obs: {h_obs} → Action: {h_act}")
        history_str = "\n".join(hist_lines) if hist_lines else "(none yet)"

        admissible_str = ", ".join(info["admissible_commands"])

        # Skill guidance block
        gen_skills, task_skills = self.skill_lib.retrieve(
            info["task_description"], task_type=info.get("task_type")
        )
        skill_guidance = self.skill_lib.format_for_prompt(gen_skills, task_skills)
        if not skill_guidance:
            skill_guidance = f"- {skill_text}"

        current_step = step_count + 1        # this step's number

        return _STEP_PROMPT.format(
            task_description=info["task_description"],
            skill_guidance=skill_guidance,
            step_count=step_count,
            history_len=len(history),        # most recent N shown (window-truncated)
            history=history_str,
            current_step=current_step,
            obs=obs,
            admissible=admissible_str,
        )

    def _batch_generate_active(
        self,
        obs_list:       List[str],
        info_list:      List[Dict],
        history_list:   List[List[Tuple]],
        skill_texts:    List[str],
        soft_prefix:    torch.Tensor,   # [n_active, m, hidden]
        active_indices: List[int],
        ep_step_count:  List[int],
    ) -> Tuple[List[str], List[torch.Tensor], List[torch.Tensor], List[int]]:
        """
        Run model.generate() only for active episodes with soft_prefix prepended.

        Returns:
            actions_raw:      List[str] of raw LLM outputs (length n_active).
            prompt_ids_list:  List of 1-D LongTensors (length n_active).
            gen_ids_list:     List of 1-D LongTensors (length n_active).
            token_counts:     List[int] of actual token counts generated (length n_active).
        """
        n_active = len(active_indices)
        prompts = [
            self._build_prompt(
                obs_list[i], info_list[i], history_list[i], skill_texts[idx],
                ep_step_count[i],
            )
            for idx, i in enumerate(active_indices)
        ]

        # Apply chat template and tokenise
        chat_texts = []
        for p in prompts:
            msg = [{"role": "user", "content": p}]
            chat_texts.append(
                self.tokenizer.apply_chat_template(
                    msg, tokenize=False, add_generation_prompt=True
                )
            )

        enc = self.tokenizer(
            chat_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_prompt_len,
        ).to(self.device)

        # Get token embeddings for the prompt portion
        embed_layer = self.model.get_input_embeddings()
        input_embeds = embed_layer(enc.input_ids)         # [n_active, seq_len, H]

        # Prepend soft prefix: [soft_prefix | input_embeds]
        soft_prefix = soft_prefix.to(input_embeds.dtype)
        inputs_embeds = torch.cat([soft_prefix, input_embeds], dim=1)  # [n_active, m+seq, H]

        # Extend attention mask to cover the prefix tokens (all 1s)
        prefix_mask = torch.ones(
            n_active, soft_prefix.size(1),
            dtype=enc.attention_mask.dtype, device=self.device,
        )
        attention_mask = torch.cat([prefix_mask, enc.attention_mask], dim=1)

        # Run generate() in eval mode: rollout is entirely under torch.no_grad()
        # (see collect()) and never needs a gradient path, so there is no reason
        # to run it through the train-mode + gradient-checkpointing branch of
        # Qwen2DecoderLayer.forward(). Diagnosis showed that switching back to
        # train mode after the first eval() call corrupts this exact
        # train-mode + gradient-checkpointing + sampling generate() path (output
        # degenerates into repeated tokens ignoring the prompt), while eval-mode
        # greedy generate() from the same weights stays correct throughout. Using
        # eval mode here reuses that known-good inference path. Save/restore the
        # previous mode so we don't clobber it if the model was already in eval
        # mode for some other reason (e.g. scripts/eval_checkpoint.py).
        was_training = self.model.training
        self.model.eval()
        try:
            output_ids = self.model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                # 出现 </action>（动作已给出）立即停止，省掉后续多余的
                # think/收尾 token（rollout 提速）。整批共享：任一序列
                # 触发即整批停；同 GRPO 组内输出长度高度相似，风险很小。
                stopping_criteria=build_action_stop_criteria(self.tokenizer),
            )
        finally:
            if was_training:
                self.model.train()

        # generate() only returns newly generated tokens when called with inputs_embeds
        gen_ids_full = output_ids  # [n_active, gen_len]

        actions_raw = []
        prompt_ids_list: List[torch.Tensor] = []
        gen_ids_list:    List[torch.Tensor] = []
        token_counts:    List[int] = []

        for i in range(n_active):
            text = self.tokenizer.decode(gen_ids_full[i], skip_special_tokens=True)
            actions_raw.append(text)

            # Strip left/right padding from the prompt row
            valid = enc.attention_mask[i].bool()
            prompt_ids_list.append(enc.input_ids[i][valid].detach().cpu())
            gen_ids_list.append(gen_ids_full[i].detach().cpu())

            # Count actual generated tokens (non-padding)
            token_counts.append(len(gen_ids_full[i]))

        return actions_raw, prompt_ids_list, gen_ids_list, token_counts
