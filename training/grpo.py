"""
training/grpo.py

GRPO (Group Relative Policy Optimisation) advantage computation.

Given G episode total rewards from one Group (same task, G parallel episodes),
compute the group-normalised advantage for each:
  A_i = (R_i − mean(R)) / (std(R) + ε)

This scalar A_i serves as:
  1. The weight on log_prob in the policy loss.
  2. The regression target for the RewardPredictor (fidelity loss).
"""

import torch
from typing import List, Union


def compute_grpo_advantages(
    rewards: Union[List[float], torch.Tensor],
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute GRPO group-normalised advantages.

    Args:
        rewards: Length-G list or 1-D tensor of episode total rewards.
        eps:     Denominator stability constant.

    Returns:
        advantages: torch.FloatTensor of shape [G].
    """
    if not isinstance(rewards, torch.Tensor):
        rewards = torch.tensor(rewards, dtype=torch.float32)
    rewards = rewards.float()

    mean_r = rewards.mean()
    std_r  = rewards.std(unbiased=False)   # population std (unbiased=True can give nan for G=1)
    advantages = (rewards - mean_r) / (std_r + eps)
    return advantages
