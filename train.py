"""
train.py

Entry point for InfoSkill training.

Usage:
    # Single GPU
    python train.py --config configs/alfworld.yaml

    # Multi-GPU DDP (specify GPUs with CUDA_VISIBLE_DEVICES)
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train.py --config configs/alfworld.yaml
    CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 train.py --config configs/alfworld.yaml
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train.py --config configs/alfworld.yaml

The script:
  1. Loads config from YAML.
  2. Loads Qwen2.5-7B-Instruct with LoRA applied.
  3. Instantiates all Fast Module components + SkillLibrary + SkillUpdater.
  4. Creates env factories for train and eval.
  5. Builds a single AdamW optimiser over all trainable params.
  6. Runs InfoskillTrainer.train().
  7. Supports DDP (DistributedDataParallel) for multi-GPU training.
"""

import argparse
import datetime
import logging
import os
import random

import numpy as np
import torch
import torch.distributed as dist
import yaml
from peft import LoraConfig, get_peft_model

logger = logging.getLogger("train")


def setup_logging(log_dir: str, run_log_dir: str) -> None:
    """
    Log to both stdout and a timestamped file under log_dir.

    Args:
        log_dir: Base log directory (e.g., "logs")
        run_log_dir: Run-specific directory (e.g., "logs/run_20260817_123456")
    """
    os.makedirs(run_log_dir, exist_ok=True)
    log_path = os.path.join(run_log_dir, "train_progress.log")
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(fmt)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(file_handler)
    root.addHandler(stream_handler)

    logger.info("Logging to %s", log_path)


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

def set_seed(seed: int, rank: int = 0) -> None:
    """Set random seed for reproducibility. Add rank to seed for DDP."""
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + rank)


# ── DDP utilities ─────────────────────────────────────────────────────────────

def setup_ddp():
    """Initialize DDP if launched with torchrun, otherwise return single-GPU info."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # Launched with torchrun
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

        return True, rank, local_rank, world_size
    else:
        # Single GPU mode
        return False, 0, 0, 1


def cleanup_ddp(is_ddp: bool):
    """Clean up DDP process group."""
    if is_ddp:
        dist.destroy_process_group()


# ── Model setup ───────────────────────────────────────────────────────────────

def build_model_and_tokenizer(cfg: dict, device: torch.device, is_ddp: bool = False):
    """Load Qwen2.5-7B-Instruct (text-only), freeze backbone, apply LoRA."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = cfg["model"]["backbone"]
    logger.info("Loading backbone: %s", model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # In DDP mode, don't use device_map="auto" (conflicts with DDP)
    # Load model to specified device directly
    if is_ddp:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        model = model.to(device)
    else:
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

    # In DDP mode, wrap with DistributedDataParallel
    if is_ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[device.index], output_device=device.index)
        logger.info("Wrapped model with DDP")
    else:
        model = model.to(device)

    # Trade compute for memory: without this, the full-sequence forward pass in
    # rollout._compute_log_probs() (needed to get a differentiable log-prob for
    # the policy loss) keeps every layer's activations alive for backward and
    # OOMs on a 7B model even at batch size 2. enable_input_require_grads() is
    # required alongside it — with a frozen base model + inputs_embeds (not
    # input_ids), gradient checkpointing otherwise can't recompute activations
    # on the backward pass because the checkpointed input doesn't require grad.
    base_model = model.module if is_ddp else model
    base_model.gradient_checkpointing_enable()
    base_model.enable_input_require_grads()

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
                seed=seed,  # 组内共享同一个 seed，确保抽到同一个任务（GRPO 理论要求）
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

    # Use a counter to generate different seeds for each episode
    eval_env_counter = {"count": 0}

    def factory():
        env_seed = eval_env_counter["count"]
        eval_env_counter["count"] += 1
        return AlfworldTextEnv(
            config_path=config_path,
            train_eval="eval_in_distribution",
            seed=env_seed,
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
    # Setup DDP first
    is_ddp, rank, local_rank, world_size = setup_ddp()

    args   = parse_args()
    cfg    = load_config(args.config)

    # Only rank 0 creates log directory
    if rank == 0:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_log_dir = os.path.join(cfg["paths"]["log_dir"], f"run_{timestamp}")
        os.makedirs(run_log_dir, exist_ok=True)
    else:
        run_log_dir = None

    # Broadcast run_log_dir from rank 0 to all ranks
    if is_ddp:
        if rank == 0:
            run_log_dir_list = [run_log_dir]
        else:
            run_log_dir_list = [None]
        dist.broadcast_object_list(run_log_dir_list, src=0)
        run_log_dir = run_log_dir_list[0]

    setup_logging(cfg["paths"]["log_dir"], run_log_dir)

    if is_ddp:
        logger.info("DDP initialized: rank=%d, local_rank=%d, world_size=%d", rank, local_rank, world_size)

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

    # In DDP mode, use local_rank as device
    if is_ddp:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(wanted_device)

    set_seed(cfg.get("seed", 42), rank)

    logger.info("Device: %s", device)

    # Build model + tokenizer
    model, tokenizer = build_model_and_tokenizer(cfg, device, is_ddp)

    # Build fast modules
    encoder, prior_net, projector, reward_predictor, grounding_decoder = build_fast_modules(cfg, device)

    # Build skill library - need to pass the base model (unwrapped)
    from skill_library.library import SkillLibrary
    base_model = model.module if is_ddp else model
    skill_lib = SkillLibrary(
        json_path=cfg["paths"]["skills_json"],
        model=base_model,
        tokenizer=tokenizer,
        device=device,
        top_k_general=cfg["skill_library"]["top_k_general"],
        top_k_task=cfg["skill_library"]["top_k_task"],
        max_skills=cfg["skill_library"]["max_skills"],
    )

    # Build skill updater
    from skill_library.skill_updater import SkillUpdater
    skill_updater = SkillUpdater(model=base_model, tokenizer=tokenizer, device=device)

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
        run_log_dir=run_log_dir,  # Pass the run-specific log dir
        is_ddp=is_ddp,
        rank=rank,
        world_size=world_size,
    )

    # Optionally resume
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Build env factories
    train_factory = make_train_env_factory(cfg)
    eval_factory  = make_eval_env_factory(cfg)

    # Run
    try:
        trainer.train(
            train_envs_factory=train_factory,
            eval_env_factory=eval_factory,
        )
    finally:
        cleanup_ddp(is_ddp)


if __name__ == "__main__":
    main()
