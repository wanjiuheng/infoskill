"""
envs/base.py

Abstract base class for environment wrappers.
Concrete implementations: AlfworldTextEnv (ALFWorld), future WebShopEnv, etc.
"""
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional


class BaseEnvWrapper(ABC):
    """
    Minimal interface that the rollout collector and trainer depend on.
    Every environment must speak this protocol.
    """

    @abstractmethod
    def reset(self) -> Tuple[str, Dict[str, Any]]:
        """
        Reset to a new episode.

        Returns:
            obs_text: Human-readable observation string.
            info:     Dict with at least:
                        'task_description' (str)  — goal sentence
                        'admissible_commands' (List[str]) — valid actions
                        'won' (bool)
                        'done' (bool)
        """

    @abstractmethod
    def step(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        """
        Execute one action.

        Args:
            action: Raw action string (already extracted from <action> tags).

        Returns:
            obs_text: Next observation string.
            reward:   Scalar reward (0 or 10 for won).
            done:     Whether the episode has ended.
            info:     Same keys as reset().
        """

    @abstractmethod
    def close(self) -> None:
        """Release environment resources."""

    @property
    @abstractmethod
    def task_description(self) -> str:
        """Current episode's task goal string."""
