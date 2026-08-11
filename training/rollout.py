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
    state_text:  str
    skill_text:  str
    state_emb:   torch.Tensor  # [state_dim], detached
    skill_emb:   torch.Tensor  # [skill_dim], detached
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
        # Everything in this loop runs under no_grad: rollout never needs a
        # computation graph. That graph is rebuilt, one mini-batch at a time,
        # inside InfoskillTrainer._fast_update() — this is what decouples
        # rollout memory from max_steps * group_size.
        with torch.no_grad():
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
                # Inlined reparameterisation (rather than calling sample_z())
                # so we can keep `eps` and persist it in the StepRecord —
                # _fast_update() needs it to reconstruct this exact same
                # z_tilde later, with gradient attached, from a fresh
                # (grad-enabled) encoder forward on the same (state_emb,
                # skill_emb) inputs.
                skill_emb_t = torch.stack(skill_embs, dim=0)                  # [G, D]
                mu, log_var = self.encoder(state_embs, skill_emb_t)           # [G, L]
                std = torch.exp(0.5 * log_var)
                eps = torch.randn_like(std)
                z_tilde = mu + std * eps                                      # [G, L]

                # 4. Project → soft prefix
                soft_prefix = self.projector(z_tilde)                         # [G, m, H]

                # 5. Build prompts and run LLM.generate() with soft prefix
                actions_raw, prompt_ids_list, gen_ids_list = self._batch_generate(
                    obs_list, info_list, history_list, skill_texts,
                    soft_prefix, active_mask,
                )

                # 6. Parse actions, step envs, record
                for i in range(self.G):
                    if not active_mask[i]:
                        # Still create a placeholder record to keep tensor shapes aligned
                        buf.records.append(StepRecord(
                            ep_idx=i, step_idx=step_idx,
                            state_text=obs_list[i], skill_text=skill_texts[i],
                            state_emb=state_embs[i].detach(),
                            skill_emb=skill_emb_t[i].detach(),
                            mu=mu[i].detach(), log_var=log_var[i].detach(),
                            z_tilde=z_tilde[i].detach(), eps=eps[i].detach(),
                            prompt_ids=prompt_ids_list[i], gen_ids=gen_ids_list[i],
                            action="", reward=0.0, done=True, is_valid=False,
                            is_padding=True,
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
                        skill_emb=skill_emb_t[i].detach(),
                        mu=mu[i].detach(), log_var=log_var[i].detach(),
                        z_tilde=z_tilde[i].detach(), eps=eps[i].detach(),
                        prompt_ids=prompt_ids_list[i], gen_ids=gen_ids_list[i],
                        action=action_text, reward=reward, done=done, is_valid=is_valid,
                        is_padding=False,
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
    ) -> Tuple[List[str], List[torch.Tensor], List[torch.Tensor]]:
        """
        Run model.generate() with soft_prefix prepended to inputs_embeds.

        For inactive episodes we still include them in the batch (fixed-size)
        but don't use their output.

        No log-prob is computed here anymore — that used to require a second
        forward pass *with gradient*, and the resulting graph had to stay
        alive until the group's backward() call. Instead we hand back the raw
        token ids needed to recompute that forward pass later, on demand, in
        InfoskillTrainer._fast_update() via recompute_step_batch().

        Returns:
            actions_raw:      List[str] of raw LLM outputs (length G).
            prompt_ids_list:  List of 1-D LongTensors, one per episode — the
                               real (non-padding) prompt token ids for that
                               row, recovered via attention_mask.
            gen_ids_list:     List of 1-D LongTensors, one per episode — the
                               generated token ids (uniform length across the
                               batch for this step, as produced by generate()).
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

        output_ids = self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        # generate() only returns newly generated tokens when called with
        # inputs_embeds (there's no input_ids for it to echo back), so the
        # entire output is the generated portion.
        gen_ids_full = output_ids  # [G, gen_len]

        actions_raw = []
        prompt_ids_list: List[torch.Tensor] = []
        gen_ids_list:    List[torch.Tensor] = []
        for i in range(self.G):
            text = self.tokenizer.decode(gen_ids_full[i], skip_special_tokens=True)
            actions_raw.append(text)

            # Strip left/right padding from the prompt row using
            # attention_mask, so the recompute step in _fast_update() doesn't
            # need to know this batch's padding scheme.
            valid = enc.attention_mask[i].bool()
            prompt_ids_list.append(enc.input_ids[i][valid].detach().cpu())
            gen_ids_list.append(gen_ids_full[i].detach().cpu())

        return actions_raw, prompt_ids_list, gen_ids_list
