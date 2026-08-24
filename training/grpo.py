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


def compute_grpo_advantages_grouped(
    rewards: Union[List[float], torch.Tensor],
    group_size: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    GRPO advantages with rewards sliced into independent groups.

    Used when a single rollout batch spans multiple tasks
    (rollout.tasks_per_batch > 1): each task's `group_size` rewards are
    normalised independently, so task-difficulty differences never bleed
    into another task's advantage. `rewards` is ordered as task1's
    group_size rewards, then task2's, ... (GroupRolloutCollector flattens
    task_groups in exactly this order). With group_size == len(rewards)
    this degenerates to plain `compute_grpo_advantages` (single task group).

    Returns:
        advantages: torch.FloatTensor of shape [len(rewards)].
    """
    if not isinstance(rewards, torch.Tensor):
        rewards = torch.tensor(rewards, dtype=torch.float32)
    rewards = rewards.float()
    assert group_size > 0, f"group_size must be positive, got {group_size}"
    assert rewards.numel() % group_size == 0, (
        f"rewards length {rewards.numel()} must be divisible by group_size {group_size}"
    )
    chunks = [
        compute_grpo_advantages(rewards[i:i + group_size], eps)
        for i in range(0, rewards.numel(), group_size)
    ]
    return torch.cat(chunks)
