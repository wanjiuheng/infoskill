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
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

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
    ) -> None:
        self.model             = model
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

        # Slow Module bookkeeping
        self._global_step: int = 0
        self._episode_count: int = 0
        # usage_history: skill_id → deque of (state_emb [D], advantage [scalar])
        self._usage_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        # pending success trajectories for skill generation
        self._pending_traj: List[Dict] = []

        os.makedirs(self.checkpoint_dir, exist_ok=True)

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
        while self._episode_count < self.num_episodes:
            # ── Create G envs for this group (same task family, different seeds)
            seed = group_idx * self.G
            envs: List[AlfworldTextEnv] = train_envs_factory(seed)
            assert len(envs) == self.G, f"Expected {self.G} envs, got {len(envs)}"

            # ── Rollout ───────────────────────────────────────────────────────
            collector = GroupRolloutCollector(
                envs=envs,
                model=self.model,
                tokenizer=self.tokenizer,
                encoder=self.encoder,
                projector=self.projector,
                skill_lib=self.skill_lib,
                device=self.device,
                cfg=self.cfg.get("rollout", {}),
            )
            buf: TrajectoryBuffer = collector.collect()

            # Clean up envs after rollout
            for env in envs:
                env.close()

            # ── Fast Module update ────────────────────────────────────────────
            loss_dict = self._fast_update(buf)

            self._episode_count += self.G
            group_idx += 1
            step_count = sum(1 for r in buf.records if not (r.done and r.ep_idx >= 0 and r.log_prob.item() == 0.0))
            self._global_step += step_count

            # ── Logging ───────────────────────────────────────────────────────
            sr = sum(1 for r in buf.total_rewards if r >= 10.0) / self.G
            logger.info(
                "Episode %d | group %d | success_rate=%.2f | "
                "loss=%.4f p=%.4f f=%.4f r=%.4f g=%.4f | skills=%d",
                self._episode_count, group_idx, sr,
                loss_dict["total"], loss_dict["policy"],
                loss_dict["fidelity"], loss_dict["rate"], loss_dict["grounding"],
                len(self.skill_lib),
            )

            # Accumulate pending success trajectories
            self._pending_traj.extend(buf.success_trajectories)

            # ── Slow Module ───────────────────────────────────────────────────
            if self._global_step % self.slow_interval == 0 and self._global_step > 0:
                self._slow_update()

            # ── Checkpoint ────────────────────────────────────────────────────
            if self._episode_count % self.save_freq == 0:
                self.save_checkpoint()

            # ── Eval ──────────────────────────────────────────────────────────
            if eval_env_factory is not None and self._episode_count % self.eval_freq == 0:
                from eval.evaluate import run_eval
                run_eval(self, eval_env_factory, n_episodes=self.cfg["training"]["eval_episodes"])

        logger.info("Training complete after %d episodes.", self._episode_count)
        self.save_checkpoint(final=True)

    # ── Fast Module update ────────────────────────────────────────────────────

    def _fast_update(self, buf: TrajectoryBuffer) -> Dict[str, float]:
        """Compute GRPO advantages, build tensors, compute + backprop total loss."""

        # 1. GRPO advantages: one scalar per episode → broadcast to its steps
        advantages_per_ep = compute_grpo_advantages(buf.total_rewards)  # [G]

        # 2. Filter to active (non-zero-masked) records
        active_records: List[StepRecord] = [
            r for r in buf.records
            if not (r.done and r.log_prob.item() == 0.0 and r.action == "")
        ]
        if not active_records:
            return {"total": 0.0, "policy": 0.0, "fidelity": 0.0, "rate": 0.0, "grounding": 0.0}

        # 3. Stack tensors
        z_list      = [r.z_tilde   for r in active_records]
        mu_list     = [r.mu        for r in active_records]
        lv_list     = [r.log_var   for r in active_records]
        s_list      = [r.state_emb for r in active_records]
        lp_list     = [r.log_prob  for r in active_records]
        adv_list    = [advantages_per_ep[r.ep_idx] for r in active_records]

        z_tilde     = torch.stack(z_list,   dim=0).to(self.device)  # [N, L]
        mu          = torch.stack(mu_list,  dim=0).to(self.device)
        log_var     = torch.stack(lv_list,  dim=0).to(self.device)
        state_embs  = torch.stack(s_list,   dim=0).to(self.device)  # [N, D]
        log_probs   = torch.stack(lp_list,  dim=0).to(self.device)  # [N]
        advantages  = torch.tensor(adv_list, device=self.device, dtype=torch.float32)

        # 4. Prior
        prior_mu, prior_logvar = self.prior_net(state_embs)

        # 5. RewardPredictor
        pred_advantage = self.reward_predictor(z_tilde, state_embs)

        # 6. Grounding loss — use one batch of active records (capped for speed)
        grounding_loss = self._compute_grounding_loss(active_records)

        # 7. Total loss
        total, p_loss, f_loss, r_loss, g_loss = compute_total_loss(
            log_probs=log_probs,
            advantages=advantages,
            pred_advantage=pred_advantage,
            mu=mu,
            log_var=log_var,
            prior_mu=prior_mu,
            prior_log_var=prior_logvar,
            grounding_loss=grounding_loss,
            alpha1=self.alpha1,
            alpha2=self.alpha2,
            beta=self.beta,
        )

        # 8. Backprop
        self.optimizer.zero_grad()
        total.backward()
        nn.utils.clip_grad_norm_(self._all_params(), self.grad_clip)
        self.optimizer.step()

        # 9. Update usage history for Slow Module
        with torch.no_grad():
            for rec, adv in zip(active_records, adv_list):
                self._usage_history[rec.skill_text[:50]].append(
                    (rec.state_emb.cpu(), float(adv))
                )

        return {
            "total":    total.item(),
            "policy":   p_loss.item(),
            "fidelity": f_loss.item(),
            "rate":     r_loss.item(),
            "grounding": g_loss.item() if isinstance(g_loss, torch.Tensor) else g_loss,
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
        """Save model weights and auxiliary module states."""
        tag = "final" if final else f"ep{self._episode_count}"
        path = os.path.join(self.checkpoint_dir, tag)
        os.makedirs(path, exist_ok=True)

        # Save LoRA adapter weights
        self.model.save_pretrained(os.path.join(path, "lora"))

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
        self.model = PeftModel.from_pretrained(
            self.model, os.path.join(path, "lora")
        )
        state = torch.load(os.path.join(path, "aux_modules.pt"), map_location=self.device)
        self.encoder.load_state_dict(state["encoder"])
        self.prior_net.load_state_dict(state["prior_net"])
        self.projector.load_state_dict(state["projector"])
        self.reward_predictor.load_state_dict(state["reward_predictor"])
        self.grounding_decoder.load_state_dict(state["grounding_decoder"])
        self.optimizer.load_state_dict(state["optimizer"])
        self._episode_count = state["episode_count"]
        self._global_step   = state["global_step"]
        logger.info("Loaded checkpoint from %s (episode %d)", path, self._episode_count)
