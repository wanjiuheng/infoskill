"""
training/losses.py

All loss terms for InfoSkill (paper Eq. 8):

  total_loss = −policy_loss
             + α1 × (fidelity_loss + β × rate_loss)
             + α2 × grounding_loss

Where:
  policy_loss    = E[A × log π(a|s, prefix)]       — REINFORCE / GRPO
  fidelity_loss  = MSE(pred_advantage, actual_adv)  — InfoBottleneck fidelity
  rate_loss      = KL(q(z|s,skill) ‖ p(z|s))       — InfoBottleneck rate
  grounding_loss = CE(decoder_logits, skill_tokens)  — Auxiliary reconstruction
"""

import torch
import torch.nn.functional as F
from typing import Optional


# ── Individual loss terms ─────────────────────────────────────────────────────

def compute_policy_loss(
    log_probs:  torch.Tensor,   # [N]  — log π(a|s) for each recorded step
    advantages: torch.Tensor,   # [N]  — GRPO advantages broadcast to each step
    mask:       Optional[torch.Tensor] = None,  # [N] float, 0 for inactive slots
) -> torch.Tensor:
    """
    REINFORCE-style policy gradient loss (we *minimise* -J, so this is -(A·logπ)).

    Args:
        log_probs:  Per-step log probabilities of the generated action tokens.
                    Computed as mean log-prob over the generated token sequence.
        advantages: GRPO advantages (per episode, broadcast to all steps).
        mask:       Active-episode mask (1 = active, 0 = done / padding).

    Returns:
        Scalar policy loss (to be *minimised*, i.e. already negated).
    """
    loss_per_step = -(log_probs * advantages)
    if mask is not None:
        loss_per_step = loss_per_step * mask
        denom = mask.sum().clamp(min=1.0)
        return loss_per_step.sum() / denom
    return loss_per_step.mean()


def compute_fidelity_loss(
    pred_advantage:   torch.Tensor,   # [N]  — RewardPredictor output
    actual_advantage: torch.Tensor,   # [N]  — GRPO advantages (detached)
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """MSE between predicted and actual GRPO advantages."""
    diff_sq = (pred_advantage - actual_advantage.detach()) ** 2
    if mask is not None:
        diff_sq = diff_sq * mask
        return diff_sq.sum() / mask.sum().clamp(min=1.0)
    return diff_sq.mean()


def compute_rate_loss(
    mu:            torch.Tensor,   # [N, latent_dim] — posterior mean
    log_var:       torch.Tensor,   # [N, latent_dim] — posterior log-var
    prior_mu:      torch.Tensor,   # [N, latent_dim] — prior mean
    prior_log_var: torch.Tensor,   # [N, latent_dim] — prior log-var
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    KL(q(z|s,skill) ‖ p(z|s)) closed-form for two Gaussians:
      KL = 0.5 × Σ[ prior_logvar − logvar
                    + (var + (mu−prior_mu)²) / prior_var
                    − 1 ]
    summed over latent dims, averaged over batch (active samples only).
    """
    var       = torch.exp(log_var)
    prior_var = torch.exp(prior_log_var).clamp(min=1e-8)

    kl_per_dim = 0.5 * (
        prior_log_var - log_var
        + (var + (mu - prior_mu) ** 2) / prior_var
        - 1.0
    )                                        # [N, latent_dim]
    kl_per_sample = kl_per_dim.sum(dim=-1)  # [N]

    if mask is not None:
        return (kl_per_sample * mask).sum() / mask.sum().clamp(min=1.0)
    return kl_per_sample.mean()


# ── Combined loss ─────────────────────────────────────────────────────────────

def compute_total_loss(
    log_probs:        torch.Tensor,
    advantages:       torch.Tensor,
    pred_advantage:   torch.Tensor,
    mu:               torch.Tensor,
    log_var:          torch.Tensor,
    prior_mu:         torch.Tensor,
    prior_log_var:    torch.Tensor,
    grounding_loss:   torch.Tensor,
    alpha1:           float = 0.1,
    alpha2:           float = 0.01,
    beta:             float = 0.001,
    mask:             Optional[torch.Tensor] = None,
    fidelity_mask:    Optional[torch.Tensor] = None,
) -> tuple:
    """
    Compute the total InfoSkill loss (paper Eq. 8).

    Args:
        fidelity_mask: 独立于 mask 的 fidelity-only mask。零方差组（advantage 全 0）
                       用全 0 掩码跳过 fidelity 监督，避免 RewardPredictor 被教成
                       常数预测器；policy 项因 advantage=0 天然无梯度，rate 是
                       压缩正则、与方差无关，两者都不受该 mask 影响。

    Returns:
        (total_loss, policy_loss, fidelity_loss, rate_loss, grounding_loss)
        — all scalars.  total_loss is ready for .backward().
    """
    p_loss = compute_policy_loss(log_probs, advantages, mask)
    f_loss = compute_fidelity_loss(
        pred_advantage, advantages,
        fidelity_mask if fidelity_mask is not None else mask,
    )
    r_loss = compute_rate_loss(mu, log_var, prior_mu, prior_log_var, mask)
    g_loss = grounding_loss   # already a scalar from GroundingDecoder

    total = p_loss + alpha1 * (f_loss + beta * r_loss) + alpha2 * g_loss
    return total, p_loss, f_loss, r_loss, g_loss
