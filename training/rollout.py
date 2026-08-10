"""
training/rollout.py

GroupRolloutCollector: runs G parallel AlfworldTextEnv instances,
collects a full Trajectory Buffer, and returns everything needed for loss computation.

Design:
  - G envs are reset to the same task (same seed group) to form one GRPO Group.
  - At each step: batch-forward through Encoder → Projector → LLM.generate().
  - Active Mask tracks which episodes are still running.
  - Completed episodes stop calling env.step() but stay in the batch (zero-masked).
  - Returns a TrajectoryBuffer dict ready for loss computation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any

from envs.alfworld_env import AlfworldTextEnv
from utils.action_parser import parse_action, match_admissible
from utils.embedding import get_text_embedding


# ── Prompt template ───────────────────────────────────────────────────────────

_STEP_PROMPT = """You are an expert agent operating in the ALFWorld household environment.
Your task is to: {task_description}

## Retrieved Skill Guidance
{skill_guidance}

## Action History (last {history_len} steps)
{history}

## Current Observation
{obs}

## Admissible Actions
{admissible}

Reason step-by-step inside <think></think> tags, then output exactly one action inside <action></action> tags.
"""


# ── Data container ────────────────────────────────────────────────────────────

@dataclass
class StepRecord:
    """One step from one episode within a Group."""
    ep_idx:     int           # which episode (0..G-1)
    step_idx:   int           # step number within that episode
    state_text: str
    skill_text: str
    state_emb:  torch.Tensor  # [state_dim], detached
    z_tilde:    torch.Tensor  # [latent_dim], attached for gradient
    mu:         torch.Tensor  # [latent_dim]
    log_var:    torch.Tensor  # [latent_dim]
    log_prob:   torch.Tensor  # scalar — mean log-prob of generated tokens
    action:     str
    reward:     float
    done:       bool
    is_valid:   bool          # action was valid (<action> tags + no Chinese)


@dataclass
class TrajectoryBuffer:
    """Holds all StepRecords from one full Group rollout."""
    records:      List[StepRecord] = field(default_factory=list)
    total_rewards: List[float]     = field(default_factory=list)  # length G
    task_type:    str              = "pick_and_place"
    task_description: str         = ""
    success_trajectories: List[Dict] = field(default_factory=list)  # for Slow Module


# ── Collector ─────────────────────────────────────────────────────────────────

class GroupRolloutCollector:
    """
    Manages G parallel environments and collects one full Group rollout.

    Args:
        envs:        List of G AlfworldTextEnv instances (same task, diff seeds).
        model:       Qwen2.5-7B-Instruct model with LoRA (on device).
        tokenizer:   Matching tokenizer.
        encoder:     StateConditionalEncoder.
        projector:   Projector.
        skill_lib:   SkillLibrary (for retrieval + grounding text).
        device:      Compute device.
        cfg:         Rollout config dict with keys:
                       max_steps, max_new_tokens, temperature, top_p, history_len.
    """

    def __init__(
        self,
        envs:       List[AlfworldTextEnv],
        model,
        tokenizer,
        encoder,
        projector,
        skill_lib,
        device:     torch.device,
        cfg:        Dict[str, Any],
    ) -> None:
        self.envs       = envs
        self.model      = model
        self.tokenizer  = tokenizer
        self.encoder    = encoder
        self.projector  = projector
        self.skill_lib  = skill_lib
        self.device     = device
        self.G          = len(envs)

        self.max_steps       = cfg.get("max_steps", 50)
        self.max_new_tokens  = cfg.get("max_new_tokens", 128)
        self.temperature     = cfg.get("temperature", 0.9)
        self.top_p           = cfg.get("top_p", 0.9)
        self.history_len     = cfg.get("history_len", 3)

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

        for env in self.envs:
            obs, info = env.reset()
            obs_list.append(obs)
            info_list.append(info)

        buf.task_description = info_list[0]["task_description"]
        buf.task_type        = info_list[0]["task_type"]

        # ── Step loop ─────────────────────────────────────────────────────────
        for step_idx in range(self.max_steps):
            if not any(active_mask):
                break

            # 1. Retrieve skill for each active episode
            skill_texts, skill_embs = self._batch_retrieve_skills(
                obs_list, info_list, active_mask
            )

            # 2. Embed states
            state_embs = self._batch_embed_states(obs_list, active_mask)  # [G, D]

            # 3. Encode → z_tilde
            skill_emb_t = torch.stack(skill_embs, dim=0)                  # [G, D]
            mu, log_var = self.encoder(state_embs, skill_emb_t)           # [G, L]
            from models.encoder import sample_z
            z_tilde = sample_z(mu, log_var)                               # [G, L]

            # 4. Project → soft prefix
            soft_prefix = self.projector(z_tilde)                         # [G, m, H]

            # 5. Build prompts and run LLM.generate() with soft prefix
            actions_raw, log_probs = self._batch_generate(
                obs_list, info_list, history_list, skill_texts,
                soft_prefix, active_mask,
            )

            # 6. Parse actions, step envs, record
            for i in range(self.G):
                if not active_mask[i]:
                    # Still create a zero-weight record to keep tensor shapes aligned
                    buf.records.append(StepRecord(
                        ep_idx=i, step_idx=step_idx,
                        state_text=obs_list[i], skill_text=skill_texts[i],
                        state_emb=state_embs[i].detach(),
                        z_tilde=z_tilde[i], mu=mu[i], log_var=log_var[i],
                        log_prob=torch.tensor(0.0, device=self.device),
                        action="", reward=0.0, done=True, is_valid=False,
                    ))
                    continue

                action_text, is_valid = parse_action(actions_raw[i])
                action_text = match_admissible(
                    action_text, info_list[i]["admissible_commands"]
                )

                next_obs, reward, done, next_info = self.envs[i].step(action_text)

                ep_rewards[i] += reward
                history_list[i].append((obs_list[i], action_text))
                if len(history_list[i]) > self.history_len:
                    history_list[i].pop(0)

                ep_steps_buf[i].append({"obs": obs_list[i], "action": action_text})

                buf.records.append(StepRecord(
                    ep_idx=i, step_idx=step_idx,
                    state_text=obs_list[i], skill_text=skill_texts[i],
                    state_emb=state_embs[i].detach(),
                    z_tilde=z_tilde[i], mu=mu[i], log_var=log_var[i],
                    log_prob=log_probs[i],
                    action=action_text, reward=reward, done=done, is_valid=is_valid,
                ))

                obs_list[i]  = next_obs
                info_list[i] = next_info

                if done:
                    active_mask[i] = False

        buf.total_rewards = ep_rewards

        # Collect successful trajectories for Slow Module
        for i in range(self.G):
            if ep_rewards[i] >= 1.0:   # any reward ≥ 1 means at least partial success
                buf.success_trajectories.append({
                    "task": buf.task_description,
                    "task_type": buf.task_type,
                    "steps": ep_steps_buf[i],
                })

        return buf

    # ── Internals ─────────────────────────────────────────────────────────────

    def _batch_embed_states(
        self,
        obs_list:    List[str],
        active_mask: List[bool],
    ) -> torch.Tensor:
        """Embed all G observations; inactive ones reuse last obs (no env call)."""
        # Embed all G obs (active and inactive alike) for tensor-shape consistency
        return get_text_embedding(
            obs_list, self.model, self.tokenizer, self.device
        )  # [G, hidden]

    def _batch_retrieve_skills(
        self,
        obs_list:    List[str],
        info_list:   List[Dict],
        active_mask: List[bool],
    ) -> Tuple[List[str], List[torch.Tensor]]:
        """Retrieve one skill per episode using the task description."""
        skill_texts = []
        skill_embs  = []
        for i in range(self.G):
            skill = self.skill_lib.retrieve_for_encoder(
                info_list[i]["task_description"]
            )
            skill_texts.append(skill.grounding_text)
            emb = get_text_embedding(
                skill.grounding_text, self.model, self.tokenizer, self.device
            )
            if emb.dim() == 2:
                emb = emb.squeeze(0)
            skill_embs.append(emb)
        return skill_texts, skill_embs

    def _build_prompt(
        self,
        obs:        str,
        info:       Dict,
        history:    List[Tuple],
        skill_text: str,
    ) -> str:
        """Render the step prompt string for one episode."""
        # History: "(step N) obs → action" pairs
        hist_lines = []
        for j, (h_obs, h_act) in enumerate(history, 1):
            hist_lines.append(f"Step {j}: Obs: {h_obs[:150]} → Action: {h_act}")
        history_str = "\n".join(hist_lines) if hist_lines else "(none yet)"

        admissible_str = ", ".join(info["admissible_commands"][:20])  # cap for context

        # Skill guidance block
        gen_skills, task_skills = self.skill_lib.retrieve(
            info["task_description"], task_type=info.get("task_type")
        )
        skill_guidance = self.skill_lib.format_for_prompt(gen_skills, task_skills)
        if not skill_guidance:
            skill_guidance = f"- {skill_text}"

        return _STEP_PROMPT.format(
            task_description=info["task_description"],
            skill_guidance=skill_guidance,
            history_len=len(history),
            history=history_str,
            obs=obs,
            admissible=admissible_str,
        )

    def _batch_generate(
        self,
        obs_list:    List[str],
        info_list:   List[Dict],
        history_list: List[List[Tuple]],
        skill_texts: List[str],
        soft_prefix: torch.Tensor,   # [G, m, hidden]
        active_mask: List[bool],
    ) -> Tuple[List[str], List[torch.Tensor]]:
        """
        Run model.generate() with soft_prefix prepended to inputs_embeds.

        For inactive episodes we still include them in the batch (fixed-size)
        but don't use their output.

        Returns:
            actions_raw: List[str] of raw LLM outputs (length G).
            log_probs:   List of scalar tensors (mean log-prob per generated token).
        """
        prompts = [
            self._build_prompt(obs_list[i], info_list[i], history_list[i], skill_texts[i])
            for i in range(self.G)
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
            max_length=2048,
        ).to(self.device)

        # Get token embeddings for the prompt portion
        embed_layer = self.model.get_input_embeddings()
        input_embeds = embed_layer(enc.input_ids)         # [G, seq_len, H]

        # Prepend soft prefix: [soft_prefix | input_embeds]
        # soft_prefix comes from the (float32) Projector while input_embeds is
        # whatever dtype the backbone uses (bfloat16) — cast to match, or
        # torch.cat silently upcasts the whole tensor to float32 and every
        # downstream Linear layer (bfloat16 weights) errors out.
        soft_prefix = soft_prefix.to(input_embeds.dtype)
        inputs_embeds = torch.cat([soft_prefix, input_embeds], dim=1)  # [G, m+seq, H]

        # Extend attention mask to cover the prefix tokens (all 1s)
        prefix_mask = torch.ones(
            self.G, soft_prefix.size(1),
            dtype=enc.attention_mask.dtype, device=self.device,
        )
        attention_mask = torch.cat([prefix_mask, enc.attention_mask], dim=1)

        with torch.no_grad():
            output_ids = self.model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Compute log-probs via a separate forward pass (with gradients for policy loss)
        log_probs = self._compute_log_probs(
            inputs_embeds, attention_mask, output_ids, enc.input_ids.shape[1]
        )

        # Decode generated tokens (new tokens only)
        prompt_len = enc.input_ids.shape[1] + soft_prefix.size(1)
        actions_raw = []
        for i in range(self.G):
            new_tok = output_ids[i, prompt_len:]
            text = self.tokenizer.decode(new_tok, skip_special_tokens=True)
            actions_raw.append(text)

        return actions_raw, log_probs

    def _compute_log_probs(
        self,
        inputs_embeds:   torch.Tensor,   # [G, prefix+prompt_len, H]
        attention_mask:  torch.Tensor,   # [G, prefix+prompt_len]
        output_ids:      torch.Tensor,   # [G, total_len]
        prompt_tok_len:  int,            # number of prompt tokens (before prefix)
    ) -> List[torch.Tensor]:
        """
        Re-run a forward pass on (prefix + prompt + generated tokens) to get
        token-level log-probabilities with gradient.  Returns mean log-prob per
        episode as a scalar tensor.
        """
        prefix_len = inputs_embeds.shape[1] - prompt_tok_len  # = num_prefix

        # Embed the full sequence (prefix + prompt + generated)
        # Generated tokens: output_ids[:, prefix_len + prompt_tok_len:]
        gen_ids = output_ids[:, prefix_len + prompt_tok_len:]    # [G, gen_len]

        embed_layer = self.model.get_input_embeddings()
        gen_embeds  = embed_layer(gen_ids)                       # [G, gen_len, H]
        full_embeds = torch.cat([inputs_embeds, gen_embeds], dim=1)  # [G, total, H]

        # Attention mask for generated portion (all 1s)
        gen_mask = torch.ones(
            self.G, gen_ids.shape[1],
            dtype=attention_mask.dtype, device=self.device,
        )
        full_mask = torch.cat([attention_mask, gen_mask], dim=1)

        # Forward pass with gradient
        logits = self.model(
            inputs_embeds=full_embeds,
            attention_mask=full_mask,
        ).logits                                                 # [G, total, V]

        # Token log-probs at positions corresponding to generated tokens
        # The logit at position t predicts token t+1
        # so for gen token at position i in the gen sequence,
        # the logit is at (prefix_len + prompt_tok_len + i - 1) in the full sequence
        start = prefix_len + prompt_tok_len - 1
        gen_logits = logits[:, start : start + gen_ids.shape[1], :]  # [G, gen_len, V]

        log_prob_all = F.log_softmax(gen_logits, dim=-1)             # [G, gen_len, V]
        tok_log_probs = log_prob_all.gather(
            dim=-1, index=gen_ids.unsqueeze(-1)
        ).squeeze(-1)                                                 # [G, gen_len]

        # Mean over non-padding generated tokens
        gen_valid_mask = (gen_ids != self.tokenizer.pad_token_id).float()
        ep_log_probs = []
        for i in range(self.G):
            n_valid = gen_valid_mask[i].sum().clamp(min=1.0)
            mean_lp = (tok_log_probs[i] * gen_valid_mask[i]).sum() / n_valid
            ep_log_probs.append(mean_lp)

        return ep_log_probs
