"""
models/projector.py

Fast Module — Projector: z_tilde → Soft Prefix.

Maps the latent code z_tilde [B, latent_dim] to m continuous vectors in the
LLM's embedding space [B, m, hidden_size], which are prepended to the LLM's
input_embeds before generation.
"""

import torch
import torch.nn as nn


class Projector(nn.Module):
    """
    Linear projection from latent space to soft-prefix embedding space.

    Output shape: [B, num_prefix, llm_hidden_size]
    These vectors are directly concatenated onto the front of input_embeds,
    so their dimension must match the LLM's hidden size exactly.

    A two-layer MLP (with intermediate normalisation) is used rather than a
    bare Linear to give the model more capacity to reshape the distribution of
    z_tilde into the LLM's embedding manifold.
    """

    def __init__(
        self,
        latent_dim: int,         # z_tilde dimension, e.g. 64
        num_prefix: int,         # m, number of soft prefix tokens, e.g. 8
        llm_hidden_size: int,    # Qwen2.5-7B: 3584
    ) -> None:
        super().__init__()
        self.num_prefix     = num_prefix
        self.llm_hidden_size = llm_hidden_size
        out_dim = num_prefix * llm_hidden_size

        self.proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 4),
            nn.GELU(),
            nn.Linear(latent_dim * 4, out_dim),
        )

    def forward(self, z_tilde: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_tilde: [B, latent_dim]

        Returns:
            soft_prefix: [B, num_prefix, llm_hidden_size]
        """
        B = z_tilde.size(0)
        flat = self.proj(z_tilde)                                    # [B, m * hidden]
        return flat.view(B, self.num_prefix, self.llm_hidden_size)   # [B, m, hidden]
