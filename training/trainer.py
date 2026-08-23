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

import os
import json
import logging
import random
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
        self.skill_gen_batch = slcfg.get("skill_gen_batch", 5)
        self.success_reward_threshold = slcfg.get("success_reward_threshold", 5.0)
        self.checkpoint_dir  = cfg.get("paths", {}).get("checkpoint_dir", "checkpoints")
        self.save_freq       = tcfg.get("save_freq", 100)
        self.eval_freq       = tcfg.get("eval_freq", 50)
        self.G               = rcfg.get("group_size", 8)
        self.num_episodes    = tcfg.get("num_episodes", 5000)
        # _fast_update() 里每次 forward+backward 重算 log_prob 的 mini-batch 大小
        # （不是 rollout.group_size，那是每组并行跑几局任务，语义不同）
        self.mini_batch_size = tcfg.get("mini_batch_size", 10)

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

        # resume 时 self._episode_count 是从 checkpoint 读回来的非零值，
        # 反推出应该从第几组继续（episode_count 与 group_idx 存在确定性的
        # 线性关系：episode_count = group_idx * G * world_size），这样 resume
        # 后不会把 group_idx=0 对应的任务重新喂给模型一遍。
        group_idx = self._episode_count // (self.G * self.world_size)
        envs: Optional[List[AlfworldTextEnv]] = None
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
            # ── Get G envs for this group (same task family, different seeds)
            # 不同 rank 取不同的 seed 块（rank 维度分片），同一 rank 组内 G 个 env
            # 仍共享 seed（GRPO 要求组内同任务）。这样 N 卡一轮 = G*N 个不同任务。
            seed = (group_idx * self.world_size + self.rank) * self.G
            if envs is None:
                # 只在整个训练/resume 生命周期内构造一次 AlfredTWEnv（每次构造都
                # 要重扫 train split 下全部目录，是当前最大的单项耗时开销）。
                # 之后每组只 reseed，复用已建好的 env 对象。
                envs = train_envs_factory(seed)
                assert len(envs) == self.G, f"Expected {self.G} envs, got {len(envs)}"
            else:
                for env in envs:
                    env.reseed(seed)

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
                success_reward_threshold=self.success_reward_threshold,
            )
            buf: TrajectoryBuffer = collector.collect()

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
        if envs is not None:
            for env in envs:
                env.close()
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

        # 2. 过滤出有效的 records（排除 padding placeholders + 无效动作）
        #    is_valid=False 表示动作无法解析（无 <action> 标签、action 内容为空
        #    或含中文）——生成 token 是乱码/无效输出，不能参与 policy loss，否则
        #    成功 episode 里的乱码步会被 advantage>0 强化，放大训练退化阶段的
        #    乱码行为。注意：<think> 缺失或推理里带中文不判无效（那是真实执行过
        #    的动作，见 utils/action_parser.py）。
        active_records: List[StepRecord] = [
            r for r in buf.records if not r.is_padding and r.is_valid
        ]
        if not active_records:
            return {
                "total": 0.0, "policy": 0.0, "fidelity": 0.0,
                "rate": 0.0, "grounding": 0.0
            }

        N = len(active_records)

        # 零方差组（同组全部 reward 相同 → advantage 全 0）：fidelity 的回归
        # target 全 0，若照常训练会把 RewardPredictor 教成"对任何输入都预测 0"，
        # 退化成常数预测器 → MIG 剪枝的 fidelity 项失去区分度。这些组跳过
        # fidelity 监督（policy 项因 advantage=0 天然无梯度，不受影响；rate 是
        # 压缩正则、与方差无关，照常计算）。
        rewards_t = torch.as_tensor(buf.total_rewards, dtype=torch.float32)
        zero_variance_group = bool(rewards_t.std(unbiased=False) < 1e-6)

        # 3. Prepare advantage tensor broadcast to each step
        adv_list = [advantages_per_ep[r.ep_idx] for r in active_records]
        advantages = torch.tensor(adv_list, device=self.device, dtype=torch.float32)  # [N]

        # 4. Mini-batch recomputation loop
        #    self.mini_batch_size（training.mini_batch_size）是每次 forward +
        #    backward 的 step 数量，可根据显存调整；越小越省显存。
        #
        #    DDP note: N (active_records count) is rank-local — it depends on
        #    each rank's own rollout sampling (is_valid filtering), so the two
        #    ranks' number of mini-batches, ceil(N/batch_size), can differ.
        #    self.model is DDP-wrapped, and DDP's default behavior triggers a
        #    gradient all-reduce on every backward() — if one rank calls
        #    backward() more times than the other in this loop, their
        #    collective-op sequences diverge and NCCL deadlocks (each rank
        #    waits for a collective the other rank never issues). Fix: wrap
        #    every backward() except the last mini-batch in self.model.no_sync()
        #    (accumulates gradients locally, skips the all-reduce); only the
        #    final backward() syncs, and it captures the full accumulated
        #    gradient (still the same .grad tensors). This makes each rank
        #    contribute exactly one all-reduce per _fast_update call, always,
        #    regardless of N or how many mini-batches it took.
        batch_size = self.mini_batch_size
        n_mini_batches = -(-N // batch_size)  # ceil(N / batch_size), rank-local
        self.optimizer.zero_grad()  # 清空梯度，准备累加

        # 累积各项 loss 用于日志（从每个 mini-batch 收集）
        total_loss_accum = 0.0
        p_loss_accum = 0.0
        f_loss_accum = 0.0
        r_loss_accum = 0.0
        g_loss_accum = 0.0
        num_batches = 0

        for bi, i in enumerate(range(0, N, batch_size)):
            batch_records = active_records[i : i + batch_size]
            batch_adv = advantages[i : i + batch_size]  # [B]
            B = len(batch_records)
            is_last_mini_batch = (bi == n_mini_batches - 1)

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

            # ── 4e. 计算 prior、reward predictor ─────────────────────────────
            prior_mu_b, prior_logvar_b = self.prior_net(state_embs_b)  # [B, L]
            pred_advantage_b = self.reward_predictor(z_tilde_new, state_embs_b)  # [B]

            # ── 4f. Grounding loss（这个 mini-batch）─────────────────────────
            # 用本 batch 的 encoder 重算 z（带梯度），grounding 梯度能回传
            # encoder，真正把 z 锚定到 skill 文本。此前用 rollout 存的 detached
            # z_tilde，梯度只进 GroundingDecoder，grounding 完全不约束 encoder。
            # 放循环内逐 batch 计算：每个 batch 有独立的计算图，避免共享图在多次
            # backward 时被释放报错。
            grounding_loss_b = self._compute_grounding_loss(batch_records)

            # ── 4g. 计算 total loss（这个 mini-batch）───────────────────────
            # fidelity_mask：零方差组的所有步置 0，跳过 fidelity 监督
            fidelity_mask_b = (
                torch.zeros(B, device=self.device, dtype=torch.float32)
                if zero_variance_group else None
            )
            total_b, p_b, f_b, r_b, g_b = compute_total_loss(
                log_probs=log_probs_b,
                advantages=batch_adv,
                pred_advantage=pred_advantage_b,
                mu=mu_new,
                log_var=log_var_new,
                prior_mu=prior_mu_b,
                prior_log_var=prior_logvar_b,
                grounding_loss=grounding_loss_b,
                alpha1=self.alpha1,
                alpha2=self.alpha2,
                beta=self.beta,
                mask=None,  # 已过滤 padding + 无效动作，不需要 mask
                fidelity_mask=fidelity_mask_b,
            )

            # ── 4h. Backward（梯度累加）────────────────────────────────────────
            # 注意：不调用 optimizer.zero_grad()，这样梯度会累加到之前的 mini-batch
            # DDP: only the last mini-batch's backward() triggers the gradient
            # all-reduce (see comment above the loop); earlier ones accumulate
            # locally via no_sync() so every rank does exactly one all-reduce
            # per call regardless of how many mini-batches it ran.
            if self.is_ddp and not is_last_mini_batch:
                with self.model.no_sync():
                    total_b.backward()
            else:
                total_b.backward()

            # 累积 loss 用于日志（按 batch size 加权平均）
            total_loss_accum += total_b.item() * B
            p_loss_accum += p_b.item() * B
            f_loss_accum += f_b.item() * B
            r_loss_accum += r_b.item() * B
            g_loss_accum += (g_b.item() if isinstance(g_b, torch.Tensor) else g_b) * B
            num_batches += B

        # 5. In DDP mode, sync auxiliary-module gradients across ranks.
        # encoder/prior_net/projector/reward_predictor/grounding_decoder are
        # plain nn.Modules, never wrapped in DDP (only self.model, the LLM, is)
        # — so unlike the LLM their gradients were never cross-rank synced;
        # each rank was silently training its own diverging copy. Doing this
        # here (once, after the mini-batch loop, before optimizer.step()) is
        # safe regardless of how many mini-batches each rank ran: every rank
        # reaches this line exactly once per _fast_update() call.
        if self.is_ddp:
            import torch.distributed as dist
            aux_modules = [
                self.encoder, self.prior_net, self.projector,
                self.reward_predictor, self.grounding_decoder,
            ]
            for m in aux_modules:
                for p in m.parameters():
                    if p.grad is not None:
                        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                        p.grad /= self.world_size

        # 6. Gradient clipping + optimizer step（所有 mini-batch 的梯度已累加）
        nn.utils.clip_grad_norm_(self._all_params(), self.grad_clip)
        self.optimizer.step()

        # 7. 更新 usage history（用于 Slow Module MIG 计算）
        with torch.no_grad():
            for rec, adv in zip(active_records, adv_list):
                self._usage_history[rec.skill_text[:50]].append(
                    (rec.state_emb.cpu(), float(adv))
                )

        # 8. 返回平均 loss（用于日志）
        return {
            "total": total_loss_accum / num_batches,
            "policy": p_loss_accum / num_batches,
            "fidelity": f_loss_accum / num_batches,
            "rate": r_loss_accum / num_batches,
            "grounding": g_loss_accum / num_batches,
        }

    def _compute_grounding_loss(self, records: List[StepRecord]) -> torch.Tensor:
        """
        Compute GroundingDecoder loss for a capped subset of records.
        Recomputes z from the stored eps with the CURRENT encoder, so the
        gradient flows back into the encoder (grounding anchors z to the skill
        text). The rollout-stored z_tilde is detached and would only train the
        GroundingDecoder.
        """
        # 随机抽 ≤16 条，避免"总取前几条"的偏置；cap 控显存
        sample = random.sample(records, min(16, len(records)))

        s_batch = torch.stack([r.state_emb for r in sample], dim=0).to(self.device)
        k_batch = torch.stack([r.skill_emb for r in sample], dim=0).to(self.device)
        eps_batch = torch.stack([r.eps for r in sample], dim=0).to(self.device)

        mu_s, log_var_s = self.encoder(s_batch, k_batch)
        std_s = torch.exp(0.5 * log_var_s)
        z_tilde_s = mu_s + std_s * eps_batch   # [B, latent_dim], 梯度 → encoder

        # Tokenise grounding targets (skill texts)
        skill_texts = [r.skill_text for r in sample]
        enc = self.tokenizer(
            skill_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.cfg.get("fast", {}).get("grounding_max_len", 64),
        ).to(self.device)

        return self.grounding_decoder(z_tilde_s, s_batch, target_ids=enc.input_ids)

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

        # 技能库快照不再单独存 skills_step<N>.json —— 由 save_checkpoint 每
        # save_freq 个 episode 统一落盘（ep<N>/skills.json），load_checkpoint
        # 时恢复，避免冗余存档。
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
            # 运行时状态：不保存则 resume 后 MIG 评估短期失真（usage_history）、
            # 已收集成功轨迹丢失（pending_traj）、曲线/CSV 从 resume 后重画
            # （metrics/eval/steps history）。dict(...) 把 defaultdict 转普通 dict，
            # load 时再重建 defaultdict(deque(maxlen=100))。
            "usage_history":     dict(self._usage_history),
            "pending_traj":      self._pending_traj,
            "metrics_history":   self._metrics_history,
            "eval_history":      self._eval_history,
            "steps_history":     self._steps_history,
        }, os.path.join(path, "aux_modules.pt"))

        # Save current skill library
        self.skill_lib.save(os.path.join(path, "skills.json"))
        logger.info("Checkpoint saved: %s", path)

    def load_checkpoint(self, path: str) -> None:
        """Load from a previously saved checkpoint directory."""
        # 恢复 LoRA：把 adapter 权重加载到现有模型对象（set_peft_model_state_dict
        # 保持参数对象不变，optimizer 持有的参数引用才继续有效）。不要用
        # PeftModel.from_pretrained 重建模型——那会创建新参数对象，resume 后
        # optimizer 更新的旧对象不再被 forward 使用，LoRA 权重将不再更新。
        self._load_lora(path)

        # weights_only=False: this file is our own checkpoint (trusted) and
        # contains non-tensor state (usage_history is a defaultdict(deque)),
        # which torch 2.6's default weights_only=True rejects.
        state = torch.load(os.path.join(path, "aux_modules.pt"), map_location=self.device,
                            weights_only=False)
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

        # 恢复技能库：续训要接上之前的技能积累，否则从初始 seed 重新开始
        skills_path = os.path.join(path, "skills.json")
        if os.path.exists(skills_path):
            self.skill_lib.load(skills_path)
            logger.info("Loaded skill library from %s (%d skills)",
                        skills_path, len(self.skill_lib))
        else:
            logger.warning("No skills.json in %s — keeping current library", path)

        # 恢复运行时状态（旧 checkpoint 无这些字段时用空默认值，保持兼容）：
        # usage_history 丢了 → MIG 评估短期失真（无数据技能全视为 +inf 保留）；
        # pending_traj 丢了 → 已收集的成功轨迹消失，技能生成中断；
        # 绘图历史丢了 → 曲线/CSV 只从 resume 后开始画。
        self._usage_history = defaultdict(
            lambda: deque(maxlen=100),
            {k: deque(v, maxlen=100) for k, v in state.get("usage_history", {}).items()},
        )
        self._pending_traj    = list(state.get("pending_traj", []))
        self._metrics_history = list(state.get("metrics_history", []))
        self._eval_history    = list(state.get("eval_history", []))
        self._steps_history   = list(state.get("steps_history", []))

        logger.info("Loaded checkpoint from %s (episode %d)", path, self._episode_count)

    def _load_lora(self, path: str) -> None:
        """把 checkpoint 的 LoRA adapter 权重加载进现有模型，不重建模型对象。

        保持 self._base_model 及其参数对象不变，这样 train.py 里基于该模型构建的
        optimizer 参数引用在 resume 后仍然有效（否则梯度更新落在无人使用的旧
        参数对象上，LoRA 权重静默冻结）。
        """
        from peft import set_peft_model_state_dict
        lora_dir = os.path.join(path, "lora")
        safetensors_path = os.path.join(lora_dir, "adapter_model.safetensors")
        bin_path = os.path.join(lora_dir, "adapter_model.bin")
        if os.path.exists(safetensors_path):
            from safetensors.torch import load_file
            adapter_state = load_file(safetensors_path)
        elif os.path.exists(bin_path):
            adapter_state = torch.load(bin_path, map_location=self.device)
            if "state_dict" in adapter_state:  # 兼容老格式（带 state_dict 包装）
                adapter_state = adapter_state["state_dict"]
        else:
            raise FileNotFoundError(f"No adapter weights found in {lora_dir}")
        set_peft_model_state_dict(self._base_model, adapter_state)
        logger.info("LoRA adapter loaded from %s", lora_dir)

    def _save_metrics_plot(self) -> None:
        """
        保存训练指标曲线图（loss_curve.png）。
        每个 group 后调用，覆盖写 PNG；CSV 由 _save_metrics_csv 负责。
        """
        if not self._metrics_history:
            return

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

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib 未安装，跳过绘图（pip install matplotlib）")
            return

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

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib 未安装，跳过绘图（pip install matplotlib）")
            return

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

