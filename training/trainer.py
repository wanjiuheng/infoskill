"""
training/trainer.py

InfoskillTrainer: the main training loop integrating Fast Module and Slow Module.

Fast loop  (every Group):
  1. Collect G-episode rollout via GroupRolloutCollector.
  2. Compute GRPO advantages.
  3. Assemble per-step tensors (z_tilde, log_probs, state_embs, …).
  4. Compute total loss = policy + fidelity + rate + grounding.
  5. Backprop, clip gradients, step optimiser.

Slow loop  (every T global steps):
  6. Evaluate MIG for each skill using recent usage history.
  7. Prune skills with MIG ≤ 0.
  8. Generate new skill candidates from buffered success trajectories.
  9. Add candidates that pass MIG threshold.
  10. Save updated skill library to disk.
"""

from __future__ import annotations

import csv
import os
import json
import logging
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from training.rollout import GroupRolloutCollector, TrajectoryBuffer, StepRecord
from training.grpo import compute_grpo_advantages
from training.losses import compute_total_loss
from envs.alfworld_env import AlfworldTextEnv
from models.encoder import StateConditionalEncoder, PriorNetwork
from models.projector import Projector
from models.reward_predictor import RewardPredictor
from models.grounding_decoder import GroundingDecoder
from skill_library.library import SkillLibrary
from skill_library.skill_updater import SkillUpdater
from utils.embedding import get_text_embedding

logger = logging.getLogger(__name__)


class InfoskillTrainer:
    """
    Orchestrates Fast + Slow modules and the GRPO training loop.

    Args:
        model:        Qwen2.5-7B-Instruct with LoRA applied (main policy / BRAIN).
        tokenizer:    Matching tokenizer.
        encoder:      StateConditionalEncoder.
        prior_net:    PriorNetwork.
        projector:    Projector.
        reward_predictor: RewardPredictor.
        grounding_decoder: GroundingDecoder.
        skill_lib:    SkillLibrary.
        skill_updater: SkillUpdater.
        optimizer:    Single AdamW covering all trainable params.
        device:       Compute device.
        cfg:          Full config dict (loaded from alfworld.yaml).
    """

    def __init__(
        self,
        model,
        tokenizer,
        encoder:            StateConditionalEncoder,
        prior_net:          PriorNetwork,
        projector:          Projector,
        reward_predictor:   RewardPredictor,
        grounding_decoder:  GroundingDecoder,
        skill_lib:          SkillLibrary,
        skill_updater:      SkillUpdater,
        optimizer:          torch.optim.Optimizer,
        device:             torch.device,
        cfg:                Dict[str, Any],
        run_log_dir:        str = None,  # Optional run-specific log dir
        is_ddp:             bool = False,  # DDP mode flag
        rank:               int = 0,       # DDP rank
        world_size:         int = 1,       # DDP world size
    ) -> None:
        self.model             = model
        # Unwrapped model reference for accessing HF methods (get_input_embeddings,
        # generate, save_pretrained) that DDP doesn't forward. forward/backward
        # still go through self.model (the DDP wrapper) for gradient sync.
        self._base_model        = model.module if is_ddp else model
        self.tokenizer         = tokenizer
        self.encoder           = encoder
        self.prior_net         = prior_net
        self.projector         = projector
        self.reward_predictor  = reward_predictor
        self.grounding_decoder = grounding_decoder
        self.skill_lib         = skill_lib
        self.skill_updater     = skill_updater
        self.optimizer         = optimizer
        self.device            = device
        self.cfg               = cfg
        self.is_ddp            = is_ddp
        self.rank              = rank
        self.world_size        = world_size

        # Hyperparameters
        tcfg = cfg.get("training", {})
        lcfg = cfg.get("loss", {})
        rcfg = cfg.get("rollout", {})
        slcfg = cfg.get("skill_library", {})

        self.alpha1  = lcfg.get("alpha1", 0.1)
        self.alpha2  = lcfg.get("alpha2", 0.01)
        self.beta    = lcfg.get("beta", 0.001)
        self.mig_beta = tcfg.get("mig_beta", 0.001)
        self.grad_clip = tcfg.get("grad_clip", 1.0)
        self.slow_interval   = tcfg.get("slow_update_interval", 200)
        self.success_thresh  = tcfg.get("success_threshold", 0.8)
        self.skill_gen_batch = slcfg.get("skill_gen_batch", 5)
        self.checkpoint_dir  = cfg.get("paths", {}).get("checkpoint_dir", "checkpoints")
        self.save_freq       = tcfg.get("save_freq", 100)
        self.eval_freq       = tcfg.get("eval_freq", 50)
        self.G               = rcfg.get("group_size", 8)
        self.num_episodes    = tcfg.get("num_episodes", 5000)

        # 阈值式触发（取代 %）：episode_count 每轮 += G*world_size 后不再是
        # save_freq/eval_freq 的整数倍，用 % 会永不触发。阈值保证多卡/单卡
        # 语义一致——每累计 save_freq/eval_freq 个全局 episode 触发一次。
        self._next_save_ep   = self.save_freq
        self._next_eval_ep   = self.eval_freq
        self._next_slow_step = self.slow_interval

        # Slow Module bookkeeping
        self._global_step: int = 0
        self._episode_count: int = 0
        # usage_history: skill_id → deque of (state_emb [D], advantage [scalar])
        self._usage_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        # pending success trajectories for skill generation
        self._pending_traj: List[Dict] = []

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Metrics tracking for loss curve plots
        self._log_dir = cfg.get("paths", {}).get("log_dir", "logs")
        # Use provided run_log_dir or create a new one
        if run_log_dir is not None:
            self._run_log_dir = run_log_dir
        else:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._run_log_dir = os.path.join(self._log_dir, f"run_{timestamp}")
            os.makedirs(self._run_log_dir, exist_ok=True)
        logger.info("Training run log directory: %s", self._run_log_dir)

        # Setup separate loggers for rollout actions and eval episodes
        self._setup_action_loggers()

        self._metrics_history: List[Dict] = []

        # Eval success rate tracking for plot
        self._eval_history: List[Dict] = []  # [{"episode": int, "success_rate": float}, ...]

        # Average steps per episode tracking for plot
        self._steps_history: List[Dict] = []  # [{"episode": int, "avg_steps": float}, ...]

    def _setup_action_loggers(self) -> None:
        """Setup separate loggers for rollout actions and eval episodes."""
        # Rollout action logger
        self.action_logger = logging.getLogger("rollout_actions")
        self.action_logger.setLevel(logging.INFO)
        self.action_logger.propagate = False  # Don't propagate to root logger
        action_handler = logging.FileHandler(
            os.path.join(self._run_log_dir, "rollout_actions.log"),
            encoding="utf-8"
        )
        action_handler.setFormatter(logging.Formatter("%(message)s"))
        self.action_logger.addHandler(action_handler)

        # Eval episode logger
        self.eval_logger = logging.getLogger("eval_episodes")
        self.eval_logger.setLevel(logging.INFO)
        self.eval_logger.propagate = False
        eval_handler = logging.FileHandler(
            os.path.join(self._run_log_dir, "eval_episodes.log"),
            encoding="utf-8"
        )
        eval_handler.setFormatter(logging.Formatter("%(message)s"))
        self.eval_logger.addHandler(eval_handler)

    # ── Main training loop ────────────────────────────────────────────────────

    def train(
        self,
        train_envs_factory,  # Callable[[int], List[AlfworldTextEnv]]
        eval_env_factory=None,
    ) -> None:
        """
        Outer training loop.

        Args:
            train_envs_factory: Callable(seed) → List[G AlfworldTextEnv].
                Used to create a fresh group of G envs for each training group.
            eval_env_factory:   Optional callable → single AlfworldTextEnv for eval.
        """
        logger.info("InfoskillTrainer: starting training for %d episodes.", self.num_episodes)

        group_idx = 0
        start_time = time.time()

        # 创建进度条
        pbar = tqdm(
            total=self.num_episodes,
            desc="Training",
            unit="ep",
            ncols=100,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )

        while self._episode_count < self.num_episodes:
            # ── Create G envs for this group (same task family, different seeds)
            # 不同 rank 取不同的 seed 块（rank 维度分片），同一 rank 组内 G 个 env
            # 仍共享 seed（GRPO 要求组内同任务）。这样 N 卡一轮 = G*N 个不同任务。
            seed = (group_idx * self.world_size + self.rank) * self.G
            envs: List[AlfworldTextEnv] = train_envs_factory(seed)
            assert len(envs) == self.G, f"Expected {self.G} envs, got {len(envs)}"

            # ── Rollout ───────────────────────────────────────────────────────
            collector = GroupRolloutCollector(
                envs=envs,
                model=self._base_model,  # 解包模型（rollout 需要 get_input_embeddings/generate）
                tokenizer=self.tokenizer,
                encoder=self.encoder,
                projector=self.projector,
                skill_lib=self.skill_lib,
                device=self.device,
                cfg=self.cfg.get("rollout", {}),
                action_logger=self.action_logger,  # Pass action logger to rollout
            )
            buf: TrajectoryBuffer = collector.collect()

            # Clean up envs after rollout
            for env in envs:
                env.close()

            # ── Fast Module update ────────────────────────────────────────────
            loss_dict = self._fast_update(buf)

            # episode_count 反映全局进度：N 卡一轮 = G*N 个 episode
            self._episode_count += self.G * self.world_size
            group_idx += 1
            step_count = sum(1 for r in buf.records if not r.is_padding)
            # 不同 rank 任务步数可能不同 → global_step 用 all_reduce 同步，
            # 保证各 rank 的 _global_step 一致（slow_update 触发条件依赖它，否则
            # rank 间判断不一致会导致 barrier 死锁）。
            if self.is_ddp:
                import torch.distributed as dist
                step_tensor = torch.tensor([step_count], device=self.device, dtype=torch.long)
                dist.all_reduce(step_tensor, op=dist.ReduceOp.SUM)
                step_count = step_tensor.item()
            self._global_step += step_count

            # ── Logging ───────────────────────────────────────────────────────
            sr = sum(1 for r in buf.total_rewards if r >= 1.0) / self.G  # 任务完成率（不考虑步数）
            avg_steps = sum(buf.total_steps) / self.G  # 平均步数
            logger.info(
                "Episode %d | group %d | success_rate=%.2f | avg_steps=%.1f | "
                "loss=%.4f p=%.4f f=%.4f r=%.4f g=%.4f | skills=%d",
                self._episode_count, group_idx, sr, avg_steps,
                loss_dict["total"], loss_dict["policy"],
                loss_dict["fidelity"], loss_dict["rate"], loss_dict["grounding"],
                len(self.skill_lib),
            )

            # 记录 metrics 用于画图
            self._metrics_history.append({
                "episode": self._episode_count,
                "group": group_idx,
                "success_rate": sr,
                "avg_steps": avg_steps,
                "total_loss": loss_dict["total"],
                "policy_loss": loss_dict["policy"],
                "fidelity_loss": loss_dict["fidelity"],
                "rate_loss": loss_dict["rate"],
                "grounding_loss": loss_dict["grounding"],
                "num_skills": len(self.skill_lib),
            })

            # 记录平均步数历史
            self._steps_history.append({
                "episode": self._episode_count,
                "avg_steps": avg_steps,
            })

            # 更新进度条（全局进度 = G * world_size）
            pbar.update(self.G * self.world_size)
            elapsed = time.time() - start_time
            avg_time_per_ep = elapsed / self._episode_count if self._episode_count > 0 else 0
            remaining_eps = self.num_episodes - self._episode_count
            eta_seconds = avg_time_per_ep * remaining_eps
            pbar.set_postfix({
                "sr": f"{sr:.2f}",
                "loss": f"{loss_dict['total']:.4f}",
                "skills": len(self.skill_lib),
                "ETA": f"{eta_seconds/3600:.1f}h" if eta_seconds >= 3600 else f"{eta_seconds/60:.0f}m"
            })

            # 每 50 个 episodes 在 log 中记录进度和剩余时间
            if self._episode_count % 50 == 0:
                progress_pct = 100.0 * self._episode_count / self.num_episodes
                elapsed_str = f"{elapsed/3600:.1f}h" if elapsed >= 3600 else f"{elapsed/60:.0f}m"
                eta_str = f"{eta_seconds/3600:.1f}h" if eta_seconds >= 3600 else f"{eta_seconds/60:.0f}m"
                logger.info(
                    "Progress: %d/%d (%.1f%%) | Elapsed: %s | ETA: %s | SR: %.3f | Loss: %.4f | Skills: %d",
                    self._episode_count, self.num_episodes, progress_pct,
                    elapsed_str, eta_str, sr, loss_dict['total'], len(self.skill_lib)
                )

            # Accumulate pending success trajectories
            self._pending_traj.extend(buf.success_trajectories)

            # ── Slow Module ───────────────────────────────────────────────────
            # Only rank 0 performs slow module updates in DDP mode
            if self._global_step >= self._next_slow_step and self._global_step > 0:
                self._next_slow_step += self.slow_interval
                if self.rank == 0:
                    self._slow_update()
                # Barrier: ensure all ranks wait for rank 0 to finish slow update
                # (skill library might be updated, need to sync before next rollout)
                if self.is_ddp:
                    import torch.distributed as dist
                    dist.barrier()

            # ── Save metrics plot (每个 group 后更新) ──────────────────────────
            # Only rank 0 saves plots/CSV in DDP mode
            if self.rank == 0:
                self._save_metrics_plot()
                self._save_steps_plot()
                self._save_metrics_csv()  # 保存 CSV 数据

            # ── Checkpoint ────────────────────────────────────────────────────
            if self._episode_count >= self._next_save_ep and self._episode_count > 0:
                self._next_save_ep += self.save_freq
                if self.rank == 0:
                    self.save_checkpoint()
                # Barrier: ensure all ranks wait for rank 0 to finish saving checkpoint
                if self.is_ddp:
                    import torch.distributed as dist
                    dist.barrier()

            # ── Eval ──────────────────────────────────────────────────────────
            if eval_env_factory is not None and self._episode_count >= self._next_eval_ep and self._episode_count > 0:
                self._next_eval_ep += self.eval_freq
                if self.rank == 0:
                    from eval.evaluate import run_eval
                    metrics = run_eval(
                        self, eval_env_factory,
                        n_episodes=self.cfg["training"]["eval_episodes"],
                        eval_logger=self.eval_logger  # Pass eval logger
                    )

                    # Record eval result and plot
                    overall_sr = metrics.get("success/overall", 0.0)
                    self._eval_history.append({
                        "episode": self._episode_count,
                        "success_rate": overall_sr,
                    })
                    self._save_eval_plot()
                    self._save_eval_csv()  # 保存 eval CSV 数据
                # Barrier: ensure all ranks wait for rank 0 to finish evaluation
                if self.is_ddp:
                    import torch.distributed as dist
                    dist.barrier()

        logger.info("Training complete after %d episodes.", self._episode_count)
        pbar.close()
        if self.rank == 0:
            self.save_checkpoint(final=True)

    # ── Fast Module update ────────────────────────────────────────────────────

    def _fast_update(self, buf: TrajectoryBuffer) -> Dict[str, float]:
        """
        Fast Module 更新：mini-batch recomputation 版本。

        对比旧版（单次 backward）：
          旧版：rollout 时计算所有 log_prob（带梯度），内存 ∝ G × max_steps
          新版：rollout 时无梯度，update 阶段分批重算 log_prob，内存 ∝ mini_batch_size

        Args:
            buf: TrajectoryBuffer from GroupRolloutCollector.collect()

        Returns:
            loss_dict: {"total", "policy", "fidelity", "rate", "grounding"}
        """

        # 1. 计算 GRPO advantages（每个 episode 一个标量）
        advantages_per_ep = compute_grpo_advantages(buf.total_rewards)  # [G]

        # 2. 过滤出有效的 records（排除 padding placeholders）
        active_records: List[StepRecord] = [
            r for r in buf.records if not r.is_padding
        ]
        if not active_records:
            return {
                "total": 0.0, "policy": 0.0, "fidelity": 0.0,
                "rate": 0.0, "grounding": 0.0
            }

        N = len(active_records)

        # 3. Prepare advantage tensor broadcast to each step
        adv_list = [advantages_per_ep[r.ep_idx] for r in active_records]
        advantages = torch.tensor(adv_list, device=self.device, dtype=torch.float32)  # [N]

        # 4. Mini-batch recomputation loop
        #    这里的 batch_size 是每次 forward + backward 的 step 数量。
        #    可以根据显存调整；默认 16 是保守值，足够小以避免 OOM，
        #    又不至于让 backward 调用次数过多（过多会稍微增加通信开销）。
        batch_size = 4
        self.optimizer.zero_grad()  # 清空梯度，准备累加

        # 累积各项 loss 用于日志（从每个 mini-batch 收集）
        total_loss_accum = 0.0
        p_loss_accum = 0.0
        f_loss_accum = 0.0
        r_loss_accum = 0.0
        g_loss_accum = 0.0
        num_batches = 0

        for i in range(0, N, batch_size):
            batch_records = active_records[i : i + batch_size]
            batch_adv = advantages[i : i + batch_size]  # [B]
            B = len(batch_records)

            # ── 4a. 重新计算 encoder forward（带梯度）──────────────────────────
            state_embs_b = torch.stack(
                [r.state_emb for r in batch_records], dim=0
            ).to(self.device)  # [B, D]
            skill_embs_b = torch.stack(
                [r.skill_emb for r in batch_records], dim=0
            ).to(self.device)  # [B, D]

            mu_new, log_var_new = self.encoder(state_embs_b, skill_embs_b)  # [B, L]

            # ── 4b. 用存储的 eps 重构 z_tilde（带梯度）────────────────────────
            eps_b = torch.stack(
                [r.eps for r in batch_records], dim=0
            ).to(self.device)  # [B, L]
            std_new = torch.exp(0.5 * log_var_new)
            z_tilde_new = mu_new + std_new * eps_b  # [B, L], 梯度连接到 encoder

            # ── 4c. projector → soft_prefix（带梯度）──────────────────────────
            soft_prefix_b = self.projector(z_tilde_new)  # [B, m, H]

            # ── 4d. 重新计算 log_prob（带梯度）────────────────────────────────
            #    对每个 record，用存储的 prompt_ids 和 gen_ids 重建完整序列，
            #    前向 LLM，计算生成 token 的 log-prob。
            log_probs_b = self._recompute_log_probs_batch(
                batch_records, soft_prefix_b
            )  # [B]

            # ── 4e. 计算 prior、reward predictor、grounding loss ──────────────
            prior_mu_b, prior_logvar_b = self.prior_net(state_embs_b)  # [B, L]
            pred_advantage_b = self.reward_predictor(z_tilde_new, state_embs_b)  # [B]

            # Grounding loss：只在第一个 mini-batch 计算（避免重复计算，
            # 因为 grounding 是辅助loss，不需要每个 batch 都算）
            if i == 0:
                grounding_loss_scalar = self._compute_grounding_loss(batch_records)
            else:
                grounding_loss_scalar = torch.tensor(0.0, device=self.device)

            # ── 4f. 计算 total loss（这个 mini-batch）───────────────────────
            total_b, p_b, f_b, r_b, g_b = compute_total_loss(
                log_probs=log_probs_b,
                advantages=batch_adv,
                pred_advantage=pred_advantage_b,
                mu=mu_new,
                log_var=log_var_new,
                prior_mu=prior_mu_b,
                prior_log_var=prior_logvar_b,
                grounding_loss=grounding_loss_scalar,
                alpha1=self.alpha1,
                alpha2=self.alpha2,
                beta=self.beta,
                mask=None,  # 已经过滤掉 padding，不需要 mask
            )

            # ── 4g. Backward（梯度累加）────────────────────────────────────────
            # 注意：不调用 optimizer.zero_grad()，这样梯度会累加到之前的 mini-batch
            total_b.backward()

            # 累积 loss 用于日志（按 batch size 加权平均）
            total_loss_accum += total_b.item() * B
            p_loss_accum += p_b.item() * B
            f_loss_accum += f_b.item() * B
            r_loss_accum += r_b.item() * B
            g_loss_accum += (g_b.item() if isinstance(g_b, torch.Tensor) else g_b) * B
            num_batches += B

        # 5. Gradient clipping + optimizer step（所有 mini-batch 的梯度已累加）
        nn.utils.clip_grad_norm_(self._all_params(), self.grad_clip)
        self.optimizer.step()

        # 6. In DDP mode, synchronize gradients across all processes
        # Note: DDP already handles gradient synchronization automatically during backward(),
        # so no explicit sync is needed here. The gradients are already averaged across ranks.

        # 6. 更新 usage history（用于 Slow Module MIG 计算）
        with torch.no_grad():
            for rec, adv in zip(active_records, adv_list):
                self._usage_history[rec.skill_text[:50]].append(
                    (rec.state_emb.cpu(), float(adv))
                )

        # 7. 返回平均 loss（用于日志）
        return {
            "total": total_loss_accum / num_batches,
            "policy": p_loss_accum / num_batches,
            "fidelity": f_loss_accum / num_batches,
            "rate": r_loss_accum / num_batches,
            "grounding": g_loss_accum / num_batches,
        }

    def _compute_grounding_loss(self, records: List[StepRecord]) -> torch.Tensor:
        """
        Compute GroundingDecoder loss for a capped subset of active records.
        Uses the skill's 'title: principle' text as reconstruction target.
        """
        # Cap at 16 samples to keep memory usage predictable
        sample = records[:16]

        z_batch  = torch.stack([r.z_tilde   for r in sample], dim=0).to(self.device)
        s_batch  = torch.stack([r.state_emb for r in sample], dim=0).to(self.device)

        # Tokenise grounding targets (skill texts)
        skill_texts = [r.skill_text for r in sample]
        enc = self.tokenizer(
            skill_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.cfg.get("fast", {}).get("grounding_max_len", 64),
        ).to(self.device)

        return self.grounding_decoder(z_batch, s_batch, target_ids=enc.input_ids)

    def _recompute_log_probs_batch(
        self,
        batch_records: List[StepRecord],
        soft_prefix: torch.Tensor,  # [B, m, H], already with gradient
    ) -> torch.Tensor:
        """
        对一个 mini-batch 的 records，重新计算每个 step 的 log_prob（带梯度）。

        流程：
          1. 用存储的 prompt_ids embed 成 prompt_embeds
          2. 拼接 [soft_prefix | prompt_embeds]
          3. 用存储的 gen_ids embed 成 gen_embeds
          4. 拼接 [soft_prefix | prompt_embeds | gen_embeds]，得到完整序列
          5. 前向 LLM，得到 logits
          6. 提取生成部分的 token log-probs，计算 mean

        Args:
            batch_records: List of StepRecord (length B)
            soft_prefix:   [B, m, H] tensor with gradient attached

        Returns:
            log_probs: [B] tensor, mean log-prob per step
        """
        B = len(batch_records)
        device = self.device

        # 1. Tokenize prompt_ids → embed
        #    每个 record 的 prompt_ids 长度可能不同（已去除 padding），
        #    需要重新 pad 成统一长度以便 batch forward。
        prompt_ids_list = [r.prompt_ids.to(device) for r in batch_records]
        max_prompt_len = max(t.size(0) for t in prompt_ids_list)

        # Pad prompt_ids to max_prompt_len (left-padding with pad_token_id)
        pad_id = self.tokenizer.pad_token_id
        prompt_ids_padded = []
        prompt_mask = []
        for t in prompt_ids_list:
            pad_len = max_prompt_len - t.size(0)
            if pad_len > 0:
                t_padded = torch.cat([
                    torch.full((pad_len,), pad_id, dtype=torch.long, device=device),
                    t
                ], dim=0)
                mask = torch.cat([
                    torch.zeros(pad_len, dtype=torch.long, device=device),
                    torch.ones(t.size(0), dtype=torch.long, device=device)
                ], dim=0)
            else:
                t_padded = t
                mask = torch.ones(t.size(0), dtype=torch.long, device=device)
            prompt_ids_padded.append(t_padded)
            prompt_mask.append(mask)

        prompt_ids_batch = torch.stack(prompt_ids_padded, dim=0)  # [B, max_prompt_len]
        prompt_mask_batch = torch.stack(prompt_mask, dim=0)       # [B, max_prompt_len]

        embed_layer = self._base_model.get_input_embeddings()
        prompt_embeds = embed_layer(prompt_ids_batch)  # [B, max_prompt_len, H]

        # 2. 拼接 soft_prefix
        soft_prefix = soft_prefix.to(prompt_embeds.dtype)  # dtype match
        inputs_embeds = torch.cat([soft_prefix, prompt_embeds], dim=1)  # [B, m+max_prompt_len, H]

        # Attention mask for prefix + prompt
        m = soft_prefix.size(1)
        prefix_mask = torch.ones(B, m, dtype=torch.long, device=device)
        attn_mask = torch.cat([prefix_mask, prompt_mask_batch], dim=1)  # [B, m+max_prompt_len]

        # 3. Embed gen_ids
        gen_ids_list = [r.gen_ids.to(device) for r in batch_records]
        # gen_ids 在同一个 rollout step 内长度是一致的（generate() 产出统一长度）
        # 但保险起见还是检查并 pad
        max_gen_len = max(t.size(0) for t in gen_ids_list)
        gen_ids_padded = []
        gen_mask = []
        for t in gen_ids_list:
            pad_len = max_gen_len - t.size(0)
            if pad_len > 0:
                t_padded = torch.cat([
                    t,
                    torch.full((pad_len,), pad_id, dtype=torch.long, device=device)
                ], dim=0)
                mask = torch.cat([
                    torch.ones(t.size(0), dtype=torch.long, device=device),
                    torch.zeros(pad_len, dtype=torch.long, device=device)
                ], dim=0)
            else:
                t_padded = t
                mask = torch.ones(t.size(0), dtype=torch.long, device=device)
            gen_ids_padded.append(t_padded)
            gen_mask.append(mask)

        gen_ids_batch = torch.stack(gen_ids_padded, dim=0)  # [B, max_gen_len]
        gen_mask_batch = torch.stack(gen_mask, dim=0)       # [B, max_gen_len]

        gen_embeds = embed_layer(gen_ids_batch)  # [B, max_gen_len, H]

        # 4. 拼接完整序列
        full_embeds = torch.cat([inputs_embeds, gen_embeds], dim=1)  # [B, m+max_prompt_len+max_gen_len, H]
        full_mask = torch.cat([attn_mask, gen_mask_batch], dim=1)    # [B, m+max_prompt_len+max_gen_len]

        # 5. Forward LLM
        logits = self.model(
            inputs_embeds=full_embeds,
            attention_mask=full_mask,
        ).logits  # [B, m+max_prompt_len+max_gen_len, V]

        # 6. 计算生成部分的 log-prob
        #    生成部分在序列的最后 max_gen_len 个位置
        #    logit[t] 预测 token[t+1]，所以生成 token 的 logit 位置是
        #    [m+max_prompt_len-1 : m+max_prompt_len+max_gen_len-1]
        start = m + max_prompt_len - 1
        gen_logits = logits[:, start : start + max_gen_len, :]  # [B, max_gen_len, V]

        log_prob_all = F.log_softmax(gen_logits, dim=-1)  # [B, max_gen_len, V]
        tok_log_probs = log_prob_all.gather(
            dim=-1, index=gen_ids_batch.unsqueeze(-1)
        ).squeeze(-1)  # [B, max_gen_len]

        # Mean over valid (non-padding) generated tokens
        gen_valid_mask = gen_mask_batch.float()  # [B, max_gen_len]
        n_valid = gen_valid_mask.sum(dim=-1).clamp(min=1.0)  # [B]
        mean_log_probs = (tok_log_probs * gen_valid_mask).sum(dim=-1) / n_valid  # [B]

        return mean_log_probs

    # ── Slow Module update ────────────────────────────────────────────────────

    def _slow_update(self) -> None:
        """
        1. Prune skills using MIG.
        2. Generate new skills from buffered success trajectories.
        3. Persist updated library.
        """
        logger.info("Slow Module: library maintenance at global_step=%d", self._global_step)

        # Put auxiliary models into eval mode temporarily
        self.encoder.eval()
        self.prior_net.eval()
        self.reward_predictor.eval()

        # ── 1. Prune ──────────────────────────────────────────────────────────
        # Build usage history tensors keyed by skill_id
        # (We key usage by skill grounding_text prefix in _fast_update;
        #  here we map back to skills by closest match.)
        usage_by_id = self._resolve_usage_history()
        pruned = self.skill_lib.prune(
            encoder=self.encoder,
            prior_net=self.prior_net,
            reward_predictor=self.reward_predictor,
            usage_history=usage_by_id,
            beta=self.mig_beta,
        )
        if pruned:
            logger.info("Slow Module: pruned %d skills: %s", len(pruned), pruned)

        # ── 2. Generate new skills ─────────────────────────────────────────────
        if len(self._pending_traj) >= self.skill_gen_batch:
            batch = self._pending_traj[:self.skill_gen_batch]
            self._pending_traj = self._pending_traj[self.skill_gen_batch:]
            task_type = batch[0].get("task_type", "pick_and_place")

            new_skill = self.skill_updater.generate_skill(batch, task_type=task_type)
            if new_skill:
                added = self.skill_lib.add_skill(new_skill, category=task_type)
                if added:
                    logger.info("Slow Module: added new skill %r", new_skill.get("title"))

        # ── 3. Persist ────────────────────────────────────────────────────────
        save_path = os.path.join(
            self.checkpoint_dir,
            f"skills_step{self._global_step}.json"
        )
        self.skill_lib.save(save_path)

        self.encoder.train()
        self.prior_net.train()
        self.reward_predictor.train()

    def _resolve_usage_history(
        self,
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Convert the deque-based usage_history (keyed by skill text prefix)
        into a dict keyed by skill_id, with stacked tensors.
        """
        result = {}
        for skill in self.skill_lib:
            key = skill.grounding_text[:50]
            if key not in self._usage_history:
                continue
            entries = list(self._usage_history[key])
            if not entries:
                continue
            state_embs = torch.stack([e[0] for e in entries]).to(self.device)
            advantages = torch.tensor([e[1] for e in entries], device=self.device)
            result[skill.skill_id] = (state_embs, advantages)
        return result

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _all_params(self):
        """Generator over all trainable parameters (LoRA + auxiliary modules)."""
        modules = [
            self.model, self.encoder, self.prior_net, self.projector,
            self.reward_predictor, self.grounding_decoder,
        ]
        for m in modules:
            for p in m.parameters():
                if p.requires_grad:
                    yield p

    def save_checkpoint(self, final: bool = False) -> None:
        """Save model weights and auxiliary module states. (DDP: only rank 0 calls this)"""
        tag = "final" if final else f"ep{self._episode_count}"
        path = os.path.join(self.checkpoint_dir, tag)
        os.makedirs(path, exist_ok=True)

        # DDP 模式下用解包模型保存（DDP 对象没有 save_pretrained）
        model_to_save = self._base_model

        # Save LoRA adapter weights
        model_to_save.save_pretrained(os.path.join(path, "lora"))

        # Save auxiliary modules
        torch.save({
            "encoder":           self.encoder.state_dict(),
            "prior_net":         self.prior_net.state_dict(),
            "projector":         self.projector.state_dict(),
            "reward_predictor":  self.reward_predictor.state_dict(),
            "grounding_decoder": self.grounding_decoder.state_dict(),
            "optimizer":         self.optimizer.state_dict(),
            "episode_count":     self._episode_count,
            "global_step":       self._global_step,
        }, os.path.join(path, "aux_modules.pt"))

        # Save current skill library
        self.skill_lib.save(os.path.join(path, "skills.json"))
        logger.info("Checkpoint saved: %s", path)

    def load_checkpoint(self, path: str) -> None:
        """Load from a previously saved checkpoint directory."""
        from peft import PeftModel
        # 从解包模型加载 LoRA，DDP 下再重新包装
        new_model = PeftModel.from_pretrained(
            self._base_model, os.path.join(path, "lora")
        )
        if self.is_ddp:
            from torch.nn.parallel import DistributedDataParallel as DDP
            new_model = DDP(new_model, device_ids=[self.device.index], output_device=self.device.index)
        self.model = new_model
        self._base_model = new_model.module if self.is_ddp else new_model
        state = torch.load(os.path.join(path, "aux_modules.pt"), map_location=self.device)
        self.encoder.load_state_dict(state["encoder"])
        self.prior_net.load_state_dict(state["prior_net"])
        self.projector.load_state_dict(state["projector"])
        self.reward_predictor.load_state_dict(state["reward_predictor"])
        self.grounding_decoder.load_state_dict(state["grounding_decoder"])
        self.optimizer.load_state_dict(state["optimizer"])
        self._episode_count = state["episode_count"]
        self._global_step   = state["global_step"]
        # 恢复阈值（下一个触发点 = 当前进度之后最近的一个 freq 整数倍）
        self._next_save_ep   = (self._episode_count // self.save_freq + 1) * self.save_freq
        self._next_eval_ep   = (self._episode_count // self.eval_freq + 1) * self.eval_freq
        self._next_slow_step = (self._global_step // self.slow_interval + 1) * self.slow_interval
        logger.info("Loaded checkpoint from %s (episode %d)", path, self._episode_count)

    def _save_metrics_plot(self) -> None:
        """
        保存训练指标曲线图和 CSV 文件。
        每个 group 后调用，覆盖写 loss_curve.png 和追加写 metrics.csv。
        """
        if not self._metrics_history:
            return

        os.makedirs(self._log_dir, exist_ok=True)

        # ── Save CSV (追加模式，第一次写表头) ────────────────────────────────
        csv_path = os.path.join(self._log_dir, "metrics.csv")
        file_exists = os.path.exists(csv_path)

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            fieldnames = [
                "episode", "group", "success_rate", "total_loss",
                "policy_loss", "fidelity_loss", "rate_loss", "grounding_loss", "num_skills"
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            # 只写最后一条（当前 group 的 metrics）
            writer.writerow(self._metrics_history[-1])

        # ── Plot curves (覆盖写 PNG) ──────────────────────────────────────────
        try:
            import matplotlib
            matplotlib.use("Agg")  # 无 GUI 后端
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib 未安装，跳过绘图（pip install matplotlib）")
            return

        episodes = [m["episode"] for m in self._metrics_history]
        sr = [m["success_rate"] for m in self._metrics_history]
        total = [m["total_loss"] for m in self._metrics_history]
        p = [m["policy_loss"] for m in self._metrics_history]
        f = [m["fidelity_loss"] for m in self._metrics_history]
        r = [m["rate_loss"] for m in self._metrics_history]
        g = [m["grounding_loss"] for m in self._metrics_history]
        skills = [m["num_skills"] for m in self._metrics_history]

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f"InfoSkill Training Metrics (ep {self._episode_count}/{self.num_episodes})", fontsize=14)

        # Success rate
        axes[0, 0].plot(episodes, sr, label="Success Rate", color="tab:green", linewidth=1.5)
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Success Rate")
        axes[0, 0].set_title("Success Rate")
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()

        # Total loss
        axes[0, 1].plot(episodes, total, label="Total Loss", color="tab:red", linewidth=1.5)
        axes[0, 1].set_xlabel("Episode")
        axes[0, 1].set_ylabel("Loss")
        axes[0, 1].set_title("Total Loss")
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()

        # Loss components
        axes[1, 0].plot(episodes, p, label="Policy", alpha=0.8, linewidth=1)
        axes[1, 0].plot(episodes, f, label="Fidelity", alpha=0.8, linewidth=1)
        axes[1, 0].plot(episodes, r, label="Rate", alpha=0.8, linewidth=1)
        axes[1, 0].plot(episodes, g, label="Grounding", alpha=0.8, linewidth=1)
        axes[1, 0].set_xlabel("Episode")
        axes[1, 0].set_ylabel("Loss")
        axes[1, 0].set_title("Loss Components")
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()

        # Skill library size
        axes[1, 1].plot(episodes, skills, label="# Skills", color="tab:purple", linewidth=1.5)
        axes[1, 1].set_xlabel("Episode")
        axes[1, 1].set_ylabel("Count")
        axes[1, 1].set_title("Skill Library Size")
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()

        plt.tight_layout()
        plot_path = os.path.join(self._run_log_dir, "loss_curve.png")
        plt.savefig(plot_path, dpi=100)
        plt.close(fig)

    def _save_eval_plot(self) -> None:
        """
        Save eval success rate curve to logs/eval_curve.png.
        Called after each eval run, overwrites previous plot.
        """
        if not self._eval_history:
            return

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        episodes = [rec["episode"] for rec in self._eval_history]
        success_rates = [rec["success_rate"] for rec in self._eval_history]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(episodes, success_rates, marker="o", linewidth=2, markersize=6, color="tab:blue")
        ax.set_xlabel("Episode (Checkpoint)", fontsize=12)
        ax.set_ylabel("Success Rate (140 eval samples)", fontsize=12)
        ax.set_title("Eval Success Rate Over Training", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.0])

        # Add value labels on each point
        for ep, sr in zip(episodes, success_rates):
            ax.text(ep, sr + 0.02, f"{sr:.3f}", ha="center", va="bottom", fontsize=9)

        plt.tight_layout()
        plot_path = os.path.join(self._run_log_dir, "eval_curve.png")
        plt.savefig(plot_path, dpi=100)
        plt.close(fig)
        logger.info("Eval plot saved: %s", plot_path)

    def _save_steps_plot(self) -> None:
        """
        Save average steps per episode curve to logs/steps_curve.png.
        Called after each group rollout, overwrites previous plot.
        """
        if not self._steps_history:
            return

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        episodes = [rec["episode"] for rec in self._steps_history]
        avg_steps = [rec["avg_steps"] for rec in self._steps_history]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(episodes, avg_steps, linewidth=1.5, alpha=0.7, color="tab:orange")
        ax.set_xlabel("Episode", fontsize=12)
        ax.set_ylabel("Average Steps per Episode", fontsize=12)
        ax.set_title("Average Interaction Steps Over Training", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(self._run_log_dir, "steps_curve.png")
        plt.savefig(plot_path, dpi=100)
        plt.close(fig)

    def _save_metrics_csv(self) -> None:
        """
        Save training metrics (loss, success rate, steps, skills) to CSV.
        """
        if not self._metrics_history:
            return

        import csv
        csv_path = os.path.join(self._run_log_dir, "training_metrics.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "episode", "group", "success_rate", "avg_steps",
                "total_loss", "policy_loss", "fidelity_loss", "rate_loss", "grounding_loss",
                "num_skills"
            ])
            writer.writeheader()
            writer.writerows(self._metrics_history)

    def _save_eval_csv(self) -> None:
        """
        Save eval success rate history to CSV.
        """
        if not self._eval_history:
            return

        import csv
        csv_path = os.path.join(self._run_log_dir, "eval_metrics.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["episode", "success_rate"])
            writer.writeheader()
            writer.writerows(self._eval_history)

