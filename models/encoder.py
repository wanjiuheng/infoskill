"""
models/encoder.py

Fast Module — VAE-style encoder and prior network.

StateConditionalEncoder:
    Inputs:  state_emb [B, state_dim] + skill_emb [B, skill_dim]
    Outputs: mu [B, latent_dim], log_var [B, latent_dim]
    Samples z_tilde via reparameterisation.

PriorNetwork:
    Inputs:  state_emb [B, state_dim]
    Outputs: prior_mu [B, latent_dim], prior_log_var [B, latent_dim]
    Provides the "prior" distribution p(z|s) for the Rate (KL) loss.
"""

import torch
import torch.nn as nn
from typing import Tuple


class StateConditionalEncoder(nn.Module):
    """
    Compress (state, skill) → Gaussian latent distribution.

    Architecture: Concat → LayerNorm → 2-layer MLP → (mu, log_var)
    Kept intentionally shallow — the LLM backbone does the heavy lifting;
    this module only needs to learn *what part* of the skill matters now.
    """

    def __init__(
        self,
        state_dim: int,     # e.g. 3584 for Qwen2.5-7B
        skill_dim: int,     # same as state_dim (same embedding source)
        latent_dim: int,    # z_tilde dimension, e.g. 64
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        in_dim = state_dim + skill_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
        )
        self.fc_mu     = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)

        # Initialise log_var head bias to 0 for numerical stability at start
        nn.init.zeros_(self.fc_logvar.bias)

    def forward(
        self,
        state_emb: torch.Tensor,   # [B, state_dim]
        skill_emb: torch.Tensor,   # [B, skill_dim]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (mu, log_var) each of shape [B, latent_dim]."""
        x = torch.cat([state_emb, skill_emb], dim=-1)
        h = self.net(x)
        mu      = self.fc_mu(h)
        log_var = self.fc_logvar(h).clamp(-10.0, 2.0)   # clamp for stability
        return mu, log_var


class PriorNetwork(nn.Module):
    """
    Learnable prior p(z | s): maps state embedding → Gaussian parameters.

    Using a learned (rather than standard-normal) prior allows the model to
    capture state-dependent structure in the latent space, reducing unnecessary
    KL penalty for state-relevant compression.
    """

    def __init__(
        self,
        state_dim: int,
        latent_dim: int,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.fc_mu     = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        nn.init.zeros_(self.fc_logvar.bias)

    def forward(
        self,
        state_emb: torch.Tensor,   # [B, state_dim]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (prior_mu, prior_log_var) each of shape [B, latent_dim]."""
        h = self.net(state_emb)
        prior_mu     = self.fc_mu(h)
        prior_logvar = self.fc_logvar(h).clamp(-10.0, 2.0)
        return prior_mu, prior_logvar


# ── Reparameterisation ────────────────────────────────────────────────────────

def sample_z(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """
    Reparameterised sample: z = mu + eps * exp(0.5 * log_var).
    Gradient flows through mu and log_var (not through eps).

    Args:
        mu:      [B, latent_dim]
        log_var: [B, latent_dim]

    Returns:
        z_tilde: [B, latent_dim]
    """
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return mu + std * eps
