"""
eval/evaluate.py

Evaluation loop: runs N episodes with the current policy (no gradient),
reports success rate per task type and overall.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Callable, Dict, List, Optional

import torch

from envs.alfworld_env import AlfworldTextEnv
from utils.action_parser import parse_action, match_admissible
from utils.embedding import get_text_embedding
from models.encoder import sample_z

logger = logging.getLogger(__name__)


def run_eval(
    trainer,               # InfoskillTrainer instance
    env_factory: Callable, # () → AlfworldTextEnv
    n_episodes: int = 64,
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
    model      = trainer.model
    tokenizer  = trainer.tokenizer
    encoder    = trainer.encoder
    projector  = trainer.projector
    skill_lib  = trainer.skill_lib
    device     = trainer.device
    rcfg       = trainer.cfg.get("rollout", {})

    max_steps      = rcfg.get("max_steps", 50)
    max_new_tokens = rcfg.get("max_new_tokens", 128)
    history_len    = rcfg.get("history_len", 3)

    # Put modules in eval mode
    model.eval()
    encoder.eval()
    projector.eval()

    results: Dict[str, List[bool]] = defaultdict(list)

    with torch.no_grad():
        for ep_idx in range(n_episodes):
            env = env_factory()
            obs, info = env.reset()
            task_type = info["task_type"]
            history: List = []
            won = False

            for step in range(max_steps):
                # Retrieve skill
                skill = skill_lib.retrieve_for_encoder(info["task_description"])

                # Embed
                state_emb = get_text_embedding(obs, model, tokenizer, device)  # [D]
                if state_emb.dim() == 1:
                    state_emb = state_emb.unsqueeze(0)                          # [1, D]
                skill_emb = get_text_embedding(
                    skill.grounding_text, model, tokenizer, device
                )
                if skill_emb.dim() == 1:
                    skill_emb = skill_emb.unsqueeze(0)

                # Encode → soft prefix
                mu, log_var = encoder(state_emb, skill_emb)
                z_tilde     = sample_z(mu, log_var)             # [1, L]
                soft_prefix = projector(z_tilde)                # [1, m, H]

                # Build prompt
                from training.rollout import _STEP_PROMPT
                hist_lines = [
                    f"Step {j+1}: Obs: {h_obs[:150]} → Action: {h_act}"
                    for j, (h_obs, h_act) in enumerate(history)
                ]
                history_str = "\n".join(hist_lines) if hist_lines else "(none yet)"
                gen_skills, task_skills = skill_lib.retrieve(
                    info["task_description"], task_type=task_type
                )
                skill_guidance = skill_lib.format_for_prompt(gen_skills, task_skills)
                prompt = _STEP_PROMPT.format(
                    task_description=info["task_description"],
                    skill_guidance=skill_guidance or f"- {skill.grounding_text}",
                    history_len=len(history),
                    history=history_str,
                    obs=obs,
                    admissible=", ".join(info["admissible_commands"][:20]),
                )
                msg = [{"role": "user", "content": prompt}]
                text = tokenizer.apply_chat_template(
                    msg, tokenize=False, add_generation_prompt=True
                )
                enc = tokenizer(
                    text, return_tensors="pt", truncation=True, max_length=2048
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
                )

                prompt_len = soft_prefix.size(1) + enc.input_ids.shape[1]
                raw_output = tokenizer.decode(
                    output_ids[0, prompt_len:], skip_special_tokens=True
                )
                action_text, _ = parse_action(raw_output)
                action_text = match_admissible(
                    action_text, info["admissible_commands"]
                )

                obs, reward, done, info = env.step(action_text)
                history.append((obs, action_text))
                if len(history) > history_len:
                    history.pop(0)

                if done:
                    won = info["won"]
                    break

            results[task_type].append(won)
            env.close()

    # Compute metrics
    metrics: Dict[str, float] = {}
    all_results: List[bool] = []
    for task_type, wins in results.items():
        rate = sum(wins) / len(wins) if wins else 0.0
        metrics[f"success/{task_type}"] = rate
        all_results.extend(wins)
    metrics["success/overall"] = sum(all_results) / len(all_results) if all_results else 0.0

    logger.info("Eval (%d eps): %s", n_episodes, metrics)

    # Restore training mode
    model.train()
    encoder.train()
    projector.train()

    return metrics
