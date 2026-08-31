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
  2. Loads Qwen2.5-7B full-parameter (no LoRA).
  3. Instantiates all Fast Module components + SkillLibrary + SkillUpdater.
  4. Creates env factories for train and eval.
  5. Builds a single 8-bit AdamW optimiser over all trainable params.
  6. Runs InfoskillTrainer.train().
  7. Supports DDP (DistributedDataParallel) for multi-GPU training.
"""

import argparse
import logging
import os
import random

import numpy as np
import torch
import torch.distributed as dist
import yaml

logger = logging.getLogger("train")


def setup_logging(log_dir: str, run_log_dir: str, rank: int = 0, is_ddp: bool = False) -> None:
    """
    Log to both stdout and a timestamped file under log_dir.

    Args:
        log_dir: Base log directory (e.g., "logs")
        run_log_dir: Run-specific directory (e.g., "logs/run_20260817_123456")
        rank: Process rank (0 for single-GPU).
        is_ddp: True when running under torchrun multi-GPU. When True the
            progress file gets a _rank{N} suffix so concurrent ranks never
            interleave lines into one shared file — interleaved logs make it
            impossible to tell which GPU hit an error, or how far each rank got
            before a DDP deadlock hang.
    """
    os.makedirs(run_log_dir, exist_ok=True)
    suffix = f"_rank{rank}" if is_ddp else ""
    log_path = os.path.join(run_log_dir, f"train_progress{suffix}.log")
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
    parser.add_argument(
        "--run-name", type=str, default=None,
        help="Experiment name, used as a subdir under paths.checkpoint_dir and "
             "paths.log_dir so parallel experiments (different hparams) never "
             "overwrite each other's checkpoints/logs. Defaults to an "
             "auto-generated name built from the config's key hparams "
             "(lr/mini_batch_size/max_new_tokens/max_steps) plus a timestamp, "
             "e.g. lr1e-06_bs10_mnt512_ms30_run_20260818_153000, so the "
             "directory name alone tells you which hparams produced it."
    )
    return parser.parse_args()


def default_run_name(cfg: dict) -> str:
    """
    从 config 的关键超参数自动拼出一个可读的 run_name，这样不用手动在
    --run-name 里重复输入 lr/bs/max_new_tokens/max_steps，也不会写错或漏写
    ——目录名和实际生效的超参数始终保持一致。
    """
    from datetime import datetime
    tcfg = cfg.get("training", {})
    rcfg = cfg.get("rollout", {})
    lr             = float(tcfg.get("learning_rate", 1e-4))
    mini_batch_size = tcfg.get("mini_batch_size", 10)
    max_new_tokens  = rcfg.get("max_new_tokens", 512)
    max_steps       = rcfg.get("max_steps", 50)
    tasks_per_batch = rcfg.get("tasks_per_batch", 1)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        f"lr{lr:.0e}_bs{mini_batch_size}_mnt{max_new_tokens}_ms{max_steps}"
        f"_tpb{tasks_per_batch}_run_{timestamp}"
    )


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
    """Load Qwen2.5-7B (text-only), full-parameter training (no LoRA)."""
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
            attn_implementation="flash_attention_2",   # FA2：加速 rollout + update 前向/反向
            trust_remote_code=True,
        )
        model = model.to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",   # FA2：加速 rollout + update 前向/反向
            device_map="auto",
            trust_remote_code=True,
        )

    # exp8：全参训练。不 freeze、不套 LoRA，所有参数保持 from_pretrained 默认的
    # requires_grad=True（build_optimizer 会遍历所有可训练参数）。

    # DDP 模式下不包 DistributedDataParallel。观察证实（run 20260824, torch 2.6）
    # no_sync() 无法抑制每个 mini-batch backward 触发的梯度 allreduce，而各 rank
    # 的 active records 数 N 不同 → backward 次数不同，两 rank 的 collective 序
    # 永久错位，NCCL 必然死锁。改为完全手动同步：初始权重由 train.py main()
    # 从 rank0 广播；_fast_update() 循环后对可训练参数梯度手动 all_reduce 一次
    # （trainer.py 第 5 步），collective 数与 N / mini-batch 数彻底无关。
    if not is_ddp:
        model = model.to(device)

    # exp9：重新开 gradient checkpointing。7B 全参（8bit Adam 后静态 ~47GB）的
    # 激活仍是显存大头——exp6 观察证实 7B 不开梯度检查点 batch=2 都 OOM。全参下
    # embedding 本身可训练，inputs_embeds 自带梯度，因此**不需要**
    # enable_input_require_grads()（那是 LoRA 冻结 embedding 时才需要的）。
    model.gradient_checkpointing_enable()

    return model, tokenizer


# ── Auxiliary modules ─────────────────────────────────────────────────────────

def build_fast_modules(cfg: dict, device: torch.device):
    """Instantiate Encoder, PriorNet, Projector, RewardPredictor, GroundingDecoder.

    cfg["decoder_type"]（默认 "lstm"）选择 GroundingDecoder 的具体实现：
    "lstm"（现有实现）或 "transformer"（消融实验用，2层 decoder-only
    transformer，接口与 lstm 版完全一致，调用方无需区分）。
    """
    from models.encoder import StateConditionalEncoder, PriorNetwork
    from models.projector import Projector
    from models.reward_predictor import RewardPredictor
    from transformers import AutoTokenizer

    decoder_type = cfg.get("decoder_type", "lstm")
    if decoder_type == "lstm":
        from models.grounding_decoder import GroundingDecoder
    elif decoder_type == "transformer":
        from models.grounding_decoder_transformer import (
            GroundingDecoder as GroundingDecoder,
        )
    else:
        raise ValueError(
            f"Unknown decoder_type={decoder_type!r}, expected 'lstm' or 'transformer'."
        )

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
    """Single 8-bit AdamW (bitsandbytes) over all trainable params + aux modules."""
    import bitsandbytes as bnb

    params = []
    # Backbone params（exp9 全参：所有可训练参数）
    for p in model.parameters():
        if p.requires_grad:
            params.append(p)
    # Auxiliary modules
    for m in modules:
        params.extend(m.parameters())

    lr = float(cfg["training"].get("learning_rate", 1e-4))
    # 8-bit Adam：优化器状态 fp32 → 8bit，静态显存从 ~91GB 降到 ~47GB，是 7B
    # 全参能在 80GB 单卡（配合梯度检查点）跑起来的关键。要求 bnb>=0.41 以支持
    # bf16 参数（state 内部反量化到 fp32 更新再量化回，参数本身仍保持 bf16）。
    return bnb.optim.AdamW8bit(params, lr=lr)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Setup DDP first
    is_ddp, rank, local_rank, world_size = setup_ddp()

    args   = parse_args()
    cfg    = load_config(args.config)

    # Run name: identifies this experiment's checkpoints/logs so parallel runs
    # with different hparams never overwrite each other. Falls back to a
    # timestamp so a bare `python train.py` still gets its own subdir.
    if rank == 0:
        run_name = args.run_name if args.run_name else default_run_name(cfg)
    else:
        run_name = None

    # Broadcast run_name from rank 0 to all ranks (all ranks must agree on the
    # same dirs, and only rank 0 can see the wall-clock timestamp branch above)
    if is_ddp:
        run_name_list = [run_name] if rank == 0 else [None]
        dist.broadcast_object_list(run_name_list, src=0)
        run_name = run_name_list[0]

    # Namespace checkpoint_dir under the run name (e.g. checkpoints/lr1e-4_bs16/)
    cfg["paths"]["checkpoint_dir"] = os.path.join(cfg["paths"]["checkpoint_dir"], run_name)

    run_log_dir = os.path.join(cfg["paths"]["log_dir"], run_name)
    setup_logging(cfg["paths"]["log_dir"], run_log_dir, rank=rank, is_ddp=is_ddp)
    if rank == 0:
        logger.info("Run name: %s (checkpoint_dir=%s, run_log_dir=%s)",
                     run_name, cfg["paths"]["checkpoint_dir"], run_log_dir)
        # 把本次实际生效的完整 config 存一份快照，避免日后只看 run_name/目录名
        # 猜不出当时到底用了什么超参数（run_name 只是人起的标签，不保证准确）
        config_snapshot_path = os.path.join(run_log_dir, "config.yaml")
        with open(config_snapshot_path, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, allow_unicode=True, sort_keys=False)
        logger.info("Config snapshot saved to %s", config_snapshot_path)

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

    # DDP: 初始权重从 rank0 广播（不包 DDP 后没有构造期广播）。set_seed(seed+rank)
    # 按 rank 播种 → backbone 与 aux 模块每 rank 随机初始化不同；不同步则各 rank
    # 从不同起点训练，即使梯度 allreduce 后也永久保持初始偏移分叉。之前只有 LoRA
    # 被 DDP 构造器广播，aux 模块从未同步（潜在分叉 bug），这里一并修复。
    if is_ddp:
        for p in model.parameters():
            if p.requires_grad:
                dist.broadcast(p.data, src=0)
        for m in (encoder, prior_net, projector, reward_predictor, grounding_decoder):
            for p in m.parameters():
                if p.requires_grad:
                    dist.broadcast(p.data, src=0)
        logger.info("Broadcast initial trainable weights from rank 0 (backbone + aux)")

    # Build skill library - need to pass the base model (unwrapped)
    from skill_library.library import SkillLibrary
    base_model = model
    skill_lib = SkillLibrary(
        json_path=cfg["paths"]["skills_json"],
        model=base_model,
        tokenizer=tokenizer,
        device=device,
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
