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

import torch
from tqdm import tqdm

from envs.alfworld_env import AlfworldTextEnv
from utils.action_parser import parse_action, match_admissible
from utils.embedding import get_text_embedding
from models.encoder import sample_z

logger = logging.getLogger(__name__)


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
    model      = trainer.model
    tokenizer  = trainer.tokenizer
    encoder    = trainer.encoder
    projector  = trainer.projector
    skill_lib  = trainer.skill_lib
    device     = trainer.device
    rcfg       = trainer.cfg.get("rollout", {})

    max_steps      = rcfg.get("max_steps", 50)
    max_new_tokens = rcfg.get("max_new_tokens", 128)  # 从配置读取，默认 128
    history_len    = rcfg.get("history_len", 3)

    # Put modules in eval mode
    model.eval()
    encoder.eval()
    projector.eval()

    results: Dict[str, List[bool]] = defaultdict(list)
    eval_start = time.time()

    with torch.no_grad():
        pbar = tqdm(range(n_episodes), desc="Eval", unit="ep", dynamic_ncols=True)
        for ep_idx in pbar:
            ep_start = time.time()
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

                history.append((obs, matched_action))
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
            env.close()

            ep_time = time.time() - ep_start
            step_time = ep_time / max(steps, 1)
            so_far = sum(sum(v) for v in results.values())
            total_so_far = sum(len(v) for v in results.values())
            running_sr = so_far / total_so_far if total_so_far else 0.0
            elapsed_total = time.time() - eval_start
            eta = elapsed_total / total_so_far * (n_episodes - total_so_far) if total_so_far else 0.0

            pbar.set_postfix(task=task_type[:12], won=won, steps=steps, sr=f"{running_sr:.2f}", s_ep=f"{ep_time:.1f}s")
            logger.info(
                "ep=%d/%d  task_type=%-25s  won=%-5s  steps=%2d  "
                "ep_time=%.1fs  step_time=%.1fs  running_sr=%.2f  elapsed=%.0fs  eta=%.0fs",
                ep_idx + 1, n_episodes, task_type, won, steps,
                ep_time, step_time, running_sr, elapsed_total, eta,
            )

    # Compute metrics
    metrics: Dict[str, float] = {}
    all_results: List[bool] = []

    # 按照 6 个任务类型的顺序输出（与 _TASK_TYPE_KEYWORDS 一致）
    TASK_ORDER = [
        "pick_and_place",
        "look_at_obj_in_light",
        "clean",
        "heat",
        "cool",
        "examine",
    ]

    logger.info("\n" + "="*70)
    logger.info("Eval Results: %d episodes", n_episodes)
    logger.info("="*70)

    for task_type in TASK_ORDER:
        if task_type in results:
            wins = results[task_type]
            rate = sum(wins) / len(wins) if wins else 0.0
            metrics[f"success/{task_type}"] = rate
            all_results.extend(wins)
            logger.info(
                "  %-25s  %3d/%3d  %.4f  (%.1f%%)",
                task_type, sum(wins), len(wins), rate, rate * 100
            )
        else:
            logger.info("  %-25s  %3d/%3d  %.4f  (%.1f%%)", task_type, 0, 0, 0.0, 0.0)

    overall_rate = sum(all_results) / len(all_results) if all_results else 0.0
    metrics["success/overall"] = overall_rate

    logger.info("-" * 70)
    logger.info(
        "  %-25s  %3d/%3d  %.4f  (%.1f%%)",
        "OVERALL", sum(all_results), len(all_results), overall_rate, overall_rate * 100
    )
    logger.info("="*70 + "\n")

    # Restore training mode
    model.train()
    encoder.train()
    projector.train()

    return metrics
