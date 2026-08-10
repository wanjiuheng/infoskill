"""
train.py

Entry point for InfoSkill training.

Usage:
    python train.py --config configs/alfworld.yaml

The script:
  1. Loads config from YAML.
  2. Loads Qwen2.5-7B-Instruct with LoRA applied.
  3. Instantiates all Fast Module components + SkillLibrary + SkillUpdater.
  4. Creates env factories for train and eval.
  5. Builds a single AdamW optimiser over all trainable params.
  6. Runs InfoskillTrainer.train().
"""

import argparse
import logging
import os
import random

import numpy as np
import torch
import yaml
from peft import LoraConfig, get_peft_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train")


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train InfoSkill on ALFWorld")
    parser.add_argument(
        "--config", type=str, default="configs/alfworld.yaml",
        help="Path to YAML config file."
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint directory to resume from."
    )
    return parser.parse_args()


# ── Config loading ────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Model setup ───────────────────────────────────────────────────────────────

def build_model_and_tokenizer(cfg: dict, device: torch.device):
    """Load Qwen2.5-7B-Instruct (text-only), freeze backbone, apply LoRA."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = cfg["model"]["backbone"]
    logger.info("Loading backbone: %s", model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Freeze backbone
    for param in model.parameters():
        param.requires_grad = False

    # Apply LoRA
    lora_cfg = cfg["model"]["lora"]
    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        target_modules=lora_cfg["target_modules"],
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        bias=lora_cfg.get("bias", "none"),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model = model.to(device)

    return model, tokenizer


# ── Auxiliary modules ─────────────────────────────────────────────────────────

def build_fast_modules(cfg: dict, device: torch.device):
    """Instantiate Encoder, PriorNet, Projector, RewardPredictor, GroundingDecoder."""
    from models.encoder import StateConditionalEncoder, PriorNetwork
    from models.projector import Projector
    from models.reward_predictor import RewardPredictor
    from models.grounding_decoder import GroundingDecoder
    from transformers import AutoTokenizer

    state_dim  = cfg["model"]["hidden_size"]   # 3584
    skill_dim  = state_dim
    latent_dim = cfg["fast"]["latent_dim"]     # 64
    num_prefix = cfg["fast"]["num_prefix"]     # 8
    hidden_sz  = cfg["model"]["hidden_size"]   # 3584

    # Vocab size from tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model"]["backbone"], trust_remote_code=True
    )
    vocab_size = len(tokenizer)

    encoder = StateConditionalEncoder(
        state_dim=state_dim, skill_dim=skill_dim, latent_dim=latent_dim
    ).to(device)
    prior_net = PriorNetwork(
        state_dim=state_dim, latent_dim=latent_dim
    ).to(device)
    projector = Projector(
        latent_dim=latent_dim, num_prefix=num_prefix, llm_hidden_size=hidden_sz
    ).to(device)
    reward_predictor = RewardPredictor(
        latent_dim=latent_dim, state_dim=state_dim
    ).to(device)
    grounding_decoder = GroundingDecoder(
        latent_dim=latent_dim,
        state_dim=state_dim,
        vocab_size=vocab_size,
        hidden_dim=cfg["fast"]["grounding_hidden_dim"],
        max_len=cfg["fast"]["grounding_max_len"],
        pad_token_id=tokenizer.pad_token_id or 0,
    ).to(device)

    return encoder, prior_net, projector, reward_predictor, grounding_decoder


# ── Env factories ─────────────────────────────────────────────────────────────

def make_train_env_factory(cfg: dict):
    """
    Returns a factory(seed) → List[G AlfworldTextEnv].
    All G envs share the same config_path but use consecutive seeds.
    """
    from envs.alfworld_env import AlfworldTextEnv

    config_path = cfg["paths"]["alfworld_config"]
    G           = cfg["rollout"]["group_size"]
    max_steps   = cfg["rollout"]["max_steps"]

    def factory(seed: int):
        return [
            AlfworldTextEnv(
                config_path=config_path,
                train_eval="train",
                seed=seed + i,
                max_steps=max_steps,
            )
            for i in range(G)
        ]
    return factory


def make_eval_env_factory(cfg: dict):
    """Returns a factory() → single AlfworldTextEnv for eval."""
    from envs.alfworld_env import AlfworldTextEnv

    config_path = cfg["paths"]["alfworld_config"]
    max_steps   = cfg["rollout"]["max_steps"]

    def factory():
        return AlfworldTextEnv(
            config_path=config_path,
            train_eval="eval_in_distribution",
            seed=0,
            max_steps=max_steps,
        )
    return factory


# ── Optimiser ─────────────────────────────────────────────────────────────────

def build_optimizer(model, modules: list, cfg: dict) -> torch.optim.Optimizer:
    """Single AdamW over LoRA params + all auxiliary modules."""
    params = []
    # LoRA params
    for p in model.parameters():
        if p.requires_grad:
            params.append(p)
    # Auxiliary modules
    for m in modules:
        params.extend(m.parameters())

    lr = float(cfg["training"].get("learning_rate", 1e-4))
    return torch.optim.AdamW(params, lr=lr)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    cfg    = load_config(args.config)
    wanted_device = cfg.get("device", "cuda")
    if wanted_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "Config requests device=%r but torch.cuda.is_available() is False "
            "(no CUDA GPU visible to torch). This usually means the installed "
            "torch wheel's CUDA build is newer than what the NVIDIA driver "
            "supports — check `nvidia-smi` (driver's max CUDA version) against "
            "`python -c \"import torch; print(torch.version.cuda)\"` (torch's "
            "CUDA build) and reinstall a matching torch wheel. To force CPU "
            "anyway, set device: cpu in the config." % wanted_device
        )
    device = torch.device(wanted_device)
    set_seed(cfg.get("seed", 42))

    logger.info("Device: %s", device)

    # Build model + tokenizer
    model, tokenizer = build_model_and_tokenizer(cfg, device)

    # Build fast modules
    encoder, prior_net, projector, reward_predictor, grounding_decoder = build_fast_modules(cfg, device)

    # Build skill library
    from skill_library.library import SkillLibrary
    skill_lib = SkillLibrary(
        json_path=cfg["paths"]["skills_json"],
        model=model,
        tokenizer=tokenizer,
        device=device,
        top_k_general=cfg["skill_library"]["top_k_general"],
        top_k_task=cfg["skill_library"]["top_k_task"],
        max_skills=cfg["skill_library"]["max_skills"],
    )

    # Build skill updater
    from skill_library.skill_updater import SkillUpdater
    skill_updater = SkillUpdater(model=model, tokenizer=tokenizer, device=device)

    # Build optimiser
    aux_modules = [encoder, prior_net, projector, reward_predictor, grounding_decoder]
    optimizer   = build_optimizer(model, aux_modules, cfg)

    # Build trainer
    from training.trainer import InfoskillTrainer
    trainer = InfoskillTrainer(
        model=model,
        tokenizer=tokenizer,
        encoder=encoder,
        prior_net=prior_net,
        projector=projector,
        reward_predictor=reward_predictor,
        grounding_decoder=grounding_decoder,
        skill_lib=skill_lib,
        skill_updater=skill_updater,
        optimizer=optimizer,
        device=device,
        cfg=cfg,
    )

    # Optionally resume
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Build env factories
    train_factory = make_train_env_factory(cfg)
    eval_factory  = make_eval_env_factory(cfg)

    # Run
    trainer.train(
        train_envs_factory=train_factory,
        eval_env_factory=eval_factory,
    )


if __name__ == "__main__":
    main()
