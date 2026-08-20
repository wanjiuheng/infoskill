"""
scripts/diagnose_grad_norms.py

一次性诊断：跑一组 GRPO rollout，对 policy / fidelity / rate / grounding 四项
分别单独 backward，记录每项给各个模块组（encoder+projector / reward_predictor /
prior_net / grounding_decoder / lora）贡献的梯度范数，以及训练实际使用的
"加权有效梯度"里 policy 与辅助项的占比。

回答 loss 平衡的核心问题：policy 路径是否在梯度层面碾压 fidelity/rate/grounding，
还是辅助项其实也在起作用。loss 值只是粗略代理，梯度范数才是决定性的。

用法（在训练机上跑，如 H20）：
    python scripts/diagnose_grad_norms.py \
        --config configs/alfworld.yaml \
        --checkpoint checkpoints/<run_name>/ep<N> \
        [--n-batches 1]

    # 不带 --checkpoint 时用随机初始化的 fast module + LoRA（仅用于对照）
"""

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import yaml

from training.grpo import compute_grpo_advantages
from training.losses import (
    compute_policy_loss, compute_fidelity_loss, compute_rate_loss,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("diagnose_grad_norms")


def grad_norm(params) -> float:
    """L2 norm over all given params' gradients (missing grad counts as 0)."""
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += p.grad.detach().float().norm().item() ** 2
    return total ** 0.5


def acc_norm(acc) -> float:
    """L2 norm of an accumulation dict {param_id: tensor}."""
    return sum(torch.norm(acc[k]) ** 2 for k in acc).item() ** 0.5


def main():
    parser = argparse.ArgumentParser(description="Per-loss-term grad-norm diagnostic")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", default=None,
                        help="Checkpoint ep<N> dir (optional; fresh init if omitted)")
    parser.add_argument("--n-batches", type=int, default=1,
                        help="Number of mini-batches to measure (default 1)")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    device = torch.device(cfg.get("device", "cuda"))

    import train as train_main

    # ── 模型 + fast modules + skill library + trainer（复用 train.py 的构建）──
    model, tokenizer = train_main.build_model_and_tokenizer(cfg, device, is_ddp=False)
    encoder, prior_net, projector, reward_predictor, grounding_decoder = \
        train_main.build_fast_modules(cfg, device)

    from skill_library.library import SkillLibrary
    from skill_library.skill_updater import SkillUpdater
    from training.trainer import InfoskillTrainer

    skill_lib = SkillLibrary(
        json_path=cfg["paths"]["skills_json"],
        model=model, tokenizer=tokenizer, device=device,
        top_k_general=cfg["skill_library"]["top_k_general"],
        top_k_task=cfg["skill_library"]["top_k_task"],
        max_skills=cfg["skill_library"]["max_skills"],
    )
    skill_updater = SkillUpdater(model=model, tokenizer=tokenizer, device=device)
    aux_modules = [encoder, prior_net, projector, reward_predictor, grounding_decoder]
    optimizer = train_main.build_optimizer(model, aux_modules, cfg)

    trainer = InfoskillTrainer(
        model=model, tokenizer=tokenizer,
        encoder=encoder, prior_net=prior_net, projector=projector,
        reward_predictor=reward_predictor, grounding_decoder=grounding_decoder,
        skill_lib=skill_lib, skill_updater=skill_updater, optimizer=optimizer,
        device=device, cfg=cfg, is_ddp=False, rank=0, world_size=1,
    )

    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
        logger.info("Loaded checkpoint %s (episode %d)", args.checkpoint,
                    trainer._episode_count)
    else:
        logger.info("No checkpoint — measuring fresh random init")

    # ── 跑一组 rollout（一次真实采集）────────────────────────────────────────
    from training.rollout import GroupRolloutCollector
    from envs.alfworld_env import AlfworldTextEnv

    G = cfg["rollout"]["group_size"]
    max_steps = cfg["rollout"]["max_steps"]
    config_path = cfg["paths"]["alfworld_config"]
    seed = 0
    envs = [
        AlfworldTextEnv(config_path=config_path, train_eval="train",
                        seed=seed, max_steps=max_steps)
        for _ in range(G)
    ]
    collector = GroupRolloutCollector(
        envs=envs, model=model, tokenizer=tokenizer,
        encoder=encoder, projector=projector, skill_lib=skill_lib,
        device=device, cfg=cfg.get("rollout", {}), action_logger=None,
        success_reward_threshold=trainer.success_reward_threshold,
    )
    logger.info("Collecting one group (G=%d, max_steps=%d)...", G, max_steps)
    buf = collector.collect()
    for env in envs:
        env.close()

    adv_per_ep = compute_grpo_advantages(buf.total_rewards)
    active = [r for r in buf.records if not r.is_padding and r.is_valid]
    logger.info("Group rewards: %s  | success=%d/%d | active records=%d",
                [round(float(r), 2) for r in buf.total_rewards],
                sum(1 for r in buf.total_rewards if r >= 1.0), G, len(active))
    if not active:
        logger.error("No active (valid) records — abort.")
        return

    # ── 模块组 ──────────────────────────────────────────────────────────────
    ep_params = list(encoder.parameters()) + list(projector.parameters())
    module_groups = {
        "encoder+projector": ep_params,
        "reward_predictor":  list(reward_predictor.parameters()),
        "prior_net":         list(prior_net.parameters()),
        "grounding_decoder": list(grounding_decoder.parameters()),
        "lora":              [p for p in model.parameters() if p.requires_grad],
    }
    alpha1 = trainer.alpha1
    alpha2 = trainer.alpha2
    beta   = trainer.beta
    term_weights = {
        "policy":   1.0,
        "fidelity": alpha1,
        "rate":     alpha1 * beta,
        "grounding": alpha2,
    }

    # ── 逐 mini-batch 测量 ─────────────────────────────────────────────────
    N = len(active)
    batch_size = trainer.mini_batch_size
    n_batches = min(args.n_batches, -(-N // batch_size))  # cap at available batches

    # 聚合结果（跨 batch 用 L2 平均，避免 batch 数影响占比）
    agg_loss   = {"policy": 0.0, "fidelity": 0.0, "rate": 0.0, "grounding": 0.0}
    agg_norm   = {t: {g: 0.0 for g in module_groups} for t in term_weights}
    agg_pol_ep = 0.0   # encoder+projector 上 policy 有效梯度范数（累计平方）
    agg_aux_ep = 0.0   # encoder+projector 上辅助项合计有效梯度范数（累计平方）

    n_records = 0
    for bi in range(n_batches):
        batch_records = active[bi * batch_size : (bi + 1) * batch_size]
        B = len(batch_records)
        n_records += B

        # 与 _fast_update 一致：encoder → z → projector → LLM log_prob 全带梯度
        state_embs_b = torch.stack([r.state_emb for r in batch_records], dim=0).to(device)
        skill_embs_b = torch.stack([r.skill_emb for r in batch_records], dim=0).to(device)
        mu_new, log_var_new = trainer.encoder(state_embs_b, skill_embs_b)
        eps_b = torch.stack([r.eps for r in batch_records], dim=0).to(device)
        std_new = torch.exp(0.5 * log_var_new)
        z_tilde_new = mu_new + std_new * eps_b
        soft_prefix_b = trainer.projector(z_tilde_new)
        log_probs_b = trainer._recompute_log_probs_batch(batch_records, soft_prefix_b)
        prior_mu_b, prior_logvar_b = trainer.prior_net(state_embs_b)
        pred_adv_b = trainer.reward_predictor(z_tilde_new, state_embs_b)
        adv_b = torch.tensor(
            [adv_per_ep[r.ep_idx] for r in batch_records],
            device=device, dtype=torch.float32,
        )

        terms = {
            "policy":   compute_policy_loss(log_probs_b, adv_b),
            "fidelity": compute_fidelity_loss(pred_adv_b, adv_b),
            "rate":     compute_rate_loss(mu_new, log_var_new, prior_mu_b, prior_logvar_b),
            # 注意：grounding 用的是 rollout 存的 detached z_tilde → 梯度只进
            # grounding_decoder，不回传 encoder（诊断会如实显示这一点）
            "grounding": trainer._compute_grounding_loss(batch_records),
        }
        for t, v in terms.items():
            agg_loss[t] += float(v.item()) * B

        # 逐项单独 backward，记录梯度范数 + 累积有效梯度
        eff_policy = {id(p): torch.zeros_like(p.data) for p in ep_params}
        eff_aux    = {id(p): torch.zeros_like(p.data) for p in ep_params}
        term_names = list(term_weights.keys())
        for ti, t in enumerate(term_names):
            optimizer.zero_grad()
            terms[t].backward(retain_graph=(ti < len(term_names) - 1))
            for g, ps in module_groups.items():
                agg_norm[t][g] += grad_norm(ps) ** 2
            w = term_weights[t]
            for p in ep_params:
                if p.grad is not None:
                    g = w * p.grad.detach()
                    (eff_policy if t == "policy" else eff_aux)[id(p)] += g

        agg_pol_ep += acc_norm(eff_policy) ** 2
        agg_aux_ep += acc_norm(eff_aux) ** 2
        optimizer.zero_grad()  # 清理，避免干扰后续

    denom = max(n_records, 1)
    # ── 输出 ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print(f"Loss-balance diagnostic  checkpoint={args.checkpoint or 'fresh'}")
    print(f"  alpha1={alpha1}  alpha2={alpha2}  beta={beta}  "
          f"mini_batch_size={batch_size}  groups measured={n_batches}")
    print("=" * 72)

    print("\n[1] Raw loss values per term (mean per record, matches training plot):")
    for t in term_names:
        print(f"    {t:<10s} {agg_loss[t]/denom:8.3f}")

    print("\n[2] Per-term grad norm by module group (L2 over batches):")
    header = f"    {'term':<10s} " + "".join(f"{g:>20s}" for g in module_groups)
    print(header)
    for t in term_names:
        row = f"    {t:<10s} "
        for g in module_groups:
            row += f"{agg_norm[t][g]**0.5:20.4f}"
        print(row)

    pol_ep = agg_pol_ep ** 0.5
    aux_ep = agg_aux_ep ** 0.5
    total_ep = (agg_pol_ep + agg_aux_ep) ** 0.5
    print("\n[3] Effective (weighted) grad norm on encoder+projector — "
          "what training actually applies:")
    print(f"    policy path : {pol_ep:8.4f}  ({100*pol_ep/max(total_ep,1e-12):5.1f}%)")
    print(f"    aux  path   : {aux_ep:8.4f}  ({100*aux_ep/max(total_ep,1e-12):5.1f}%)")
    print(f"    ratio policy/aux = {pol_ep/max(aux_ep,1e-12):.2f}")
    print("    (aux = alpha1*fidelity + alpha1*beta*rate + alpha2*grounding)")
    print("=" * 72)


if __name__ == "__main__":
    main()
