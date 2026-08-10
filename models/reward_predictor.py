"""
models/reward_predictor.py

Fast Module — RewardPredictor: estimate GRPO Advantage from z_tilde + state_emb.

Fidelity loss = MSE(predicted_advantage, actual_grpo_advantage.detach())
Minimising this loss forces z_tilde to retain information predictive of the
episode outcome — the core "information bottleneck keeps useful bits" claim.
"""

import torch
import torch.nn as nn


class RewardPredictor(nn.Module):
    """
    MLP that predicts the GRPO Advantage scalar from (z_tilde, state_emb).

    Architecture: Concat → 3-layer MLP → scalar
    Output is a single float per sample (the predicted advantage value).
    """

    def __init__(
        self,
        latent_dim: int,    # z_tilde dimension
        state_dim: int,     # state_emb dimension (LLM hidden size)
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        in_dim = latent_dim + state_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        z_tilde:   torch.Tensor,   # [B, latent_dim]
        state_emb: torch.Tensor,   # [B, state_dim]
    ) -> torch.Tensor:
        """
        Returns:
            pred_advantage: [B]  (squeeze the last dim)
        """
        x = torch.cat([z_tilde, state_emb], dim=-1)
        return self.net(x).squeeze(-1)
