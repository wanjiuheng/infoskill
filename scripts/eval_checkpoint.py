"""
scripts/eval_checkpoint.py

Standalone evaluator for a saved InfoSkill checkpoint.

Usage:
    python scripts/eval_checkpoint.py \
        --config configs/alfworld.yaml \
        --checkpoint checkpoints/ep1000 \
        --n_episodes 64
"""

import argparse
import logging
import os
import sys
import types
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import yaml


def setup_logging(log_path: str) -> None:
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(fmt))
    logging.getLogger().addHandler(fh)


logger = logging.getLogger("eval_checkpoint")


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved InfoSkill checkpoint on ALFWorld")
    parser.add_argument("--config",      required=True,  help="Path to YAML config (e.g. configs/alfworld.yaml)")
    parser.add_argument("--checkpoint",  required=True,  help="Checkpoint directory (e.g. checkpoints/ep1000)")
    parser.add_argument("--n_episodes",  type=int, default=64, help="Number of eval episodes (default: 64)")
    parser.add_argument("--log_file",    default=None, help="Path to log file (auto-generated if not set)")
    args = parser.parse_args()

    # ── Logging setup ─────────────────────────────────────────────────────────
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    if args.log_file:
        log_path = args.log_file
    else:
        ckpt_name = Path(args.checkpoint).name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = str(log_dir / f"eval_{ckpt_name}_{timestamp}.log")
    setup_logging(log_path)
    logger.info("Log file: %s", log_path)

    cfg    = load_config(args.config)
    device = torch.device(cfg.get("device", "cuda"))

    # ── Backbone + LoRA ───────────────────────────────────────────────────────
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    model_name = cfg["model"]["backbone"]
    logger.info("Loading backbone: %s", model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    for p in base_model.parameters():
        p.requires_grad = False

    lora_path = os.path.join(args.checkpoint, "lora")
    logger.info("Loading LoRA adapter: %s", lora_path)
    model = PeftModel.from_pretrained(base_model, lora_path)
    model.eval()

    # ── Fast modules (only Encoder + Projector needed for eval) ──────────────
    from models.encoder import StateConditionalEncoder
    from models.projector import Projector

    state_dim  = cfg["model"]["hidden_size"]
    latent_dim = cfg["fast"]["latent_dim"]
    num_prefix = cfg["fast"]["num_prefix"]

    encoder   = StateConditionalEncoder(
        state_dim=state_dim, skill_dim=state_dim, latent_dim=latent_dim
    ).to(device)
    projector = Projector(
        latent_dim=latent_dim, num_prefix=num_prefix, llm_hidden_size=state_dim
    ).to(device)

    aux_path = os.path.join(args.checkpoint, "aux_modules.pt")
    logger.info("Loading aux modules: %s", aux_path)
    state = torch.load(aux_path, map_location=device)
    encoder.load_state_dict(state["encoder"])
    projector.load_state_dict(state["projector"])
    encoder.eval()
    projector.eval()

    episode_count = state.get("episode_count", "?")
    logger.info("Checkpoint episode count: %s", episode_count)

    # ── Skill library ─────────────────────────────────────────────────────────
    from skill_library.library import SkillLibrary

    skills_path = os.path.join(args.checkpoint, "skills.json")
    logger.info("Loading skill library: %s", skills_path)
    skill_lib = SkillLibrary(
        json_path=skills_path,
        model=model,
        tokenizer=tokenizer,
        device=device,
        top_k_general=cfg["skill_library"]["top_k_general"],
        top_k_task=cfg["skill_library"]["top_k_task"],
        max_skills=cfg["skill_library"]["max_skills"],
    )

    # ── Trainer proxy (only the fields run_eval accesses) ─────────────────────
    trainer = types.SimpleNamespace(
        model=model,
        tokenizer=tokenizer,
        encoder=encoder,
        projector=projector,
        skill_lib=skill_lib,
        device=device,
        cfg=cfg,
    )

    # ── Eval env factory ──────────────────────────────────────────────────────
    from envs.alfworld_env import AlfworldTextEnv

    config_path = cfg["paths"]["alfworld_config"]
    max_steps   = cfg["rollout"]["max_steps"]

    # Use a counter to generate different seeds for each episode
    eval_env_counter = {"count": 0}

    def eval_env_factory():
        env_seed = eval_env_counter["count"]
        eval_env_counter["count"] += 1
        return AlfworldTextEnv(
            config_path=config_path,
            train_eval="eval_in_distribution",
            seed=env_seed,
            max_steps=max_steps,
        )

    # ── Run eval ──────────────────────────────────────────────────────────────
    from eval.evaluate import run_eval

    logger.info("Starting eval: %d episodes, checkpoint=%s", args.n_episodes, args.checkpoint)
    metrics = run_eval(trainer, eval_env_factory, n_episodes=args.n_episodes)

    lines = [
        "\n=== Eval Results ===",
        f"Checkpoint : {args.checkpoint}  (episode {episode_count})",
    ]
    for k, v in sorted(metrics.items()):
        label = k.replace("success/", "")
        lines.append(f"  {label:<30s} {v:.4f}  ({v*100:.1f}%)")
    lines.append("===================\n")
    result_str = "\n".join(lines)
    print(result_str)
    logger.info("Final results:\n%s", result_str)


if __name__ == "__main__":
    main()
