"""
envs/alfworld_env.py

Single-instance ALFWorld text-mode wrapper (AlfredTWEnv).
Implements BaseEnvWrapper so the rollout collector can treat all envs uniformly.

Design notes:
- One AlfworldTextEnv instance = one env sub-process calling TextWorld under the hood.
- For the G-parallel rollout, GroupRolloutCollector instantiates G of these.
- Action parsing (extract <action>...</action>) lives in utils/action_parser.py;
  this wrapper expects the *already-extracted* action string.
- Reward: 10.0 if 'won', else 0.0  (matches SkillRL's AlfworldWorker.compute_reward).
"""

import os
import re
import yaml
from typing import Tuple, Dict, Any, List, Optional

from envs.base import BaseEnvWrapper


# ── Task-type keyword map (mirrors SkillRL's _detect_task_type) ───────────────
_TASK_TYPE_KEYWORDS: Dict[str, List[str]] = {
    "pick_and_place":      ["pick", "put", "place"],
    "look_at_obj_in_light":["examine", "look", "light"],
    "clean":               ["clean", "wash", "rinse"],
    "heat":                ["heat", "microwave", "warm"],
    "cool":                ["cool", "fridge", "refrigerator"],
    "examine":             ["examine", "study", "inspect"],
}


def detect_task_type(task_description: str) -> str:
    """Keyword-based task category detection (used for skill retrieval routing)."""
    desc = task_description.lower()
    for task_type, keywords in _TASK_TYPE_KEYWORDS.items():
        if any(kw in desc for kw in keywords):
            return task_type
    return "pick_and_place"   # safe default


class AlfworldTextEnv(BaseEnvWrapper):
    """
    Wraps a single AlfredTWEnv instance.

    Args:
        config_path:  Path to alfworld_base_config.yaml.
        train_eval:   'train', 'eval_in_distribution', or 'eval_out_of_distribution'.
        seed:         Random seed for this sub-environment.
        max_steps:    Episode step budget.
    """

    def __init__(
        self,
        config_path: str,
        train_eval: str = "train",
        seed: int = 42,
        max_steps: int = 50,
    ) -> None:
        self._config_path = config_path
        self._train_eval = train_eval
        self._seed = seed
        self._max_steps = max_steps

        self._env = None          # lazy-initialised TextWorld gym env
        self._task_desc: str = ""
        self._step_count: int = 0

        self._init_env()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def _init_env(self) -> None:
        """Load config and initialise the underlying AlfredTWEnv."""
        assert os.path.exists(self._config_path), (
            f"ALFWorld config not found: {self._config_path}"
        )
        with open(self._config_path) as f:
            config = yaml.safe_load(f)

        # Import here so the rest of the codebase doesn't hard-depend on alfworld
        from alfworld.agents.environment import get_environment
        base_env = get_environment("AlfredTWEnv")(config, train_eval=self._train_eval)
        self._env = base_env.init_env(batch_size=1)
        self._env.seed(self._seed)

        # Monkey patch: 禁用 shuffle，改为顺序采样（临时方案，用于遍历全部 3553 个训练样本）
        # 原始的 TextworldBatchGymEnv.seed() 会调用 np.random.shuffle(self.game_files)
        # 我们替换成空操作，让 reset() 按 game_files 的原始顺序（文件系统字母序）逐个取游戏
        def _no_shuffle_seed(seed_value):
            """空操作版本的 seed()，跳过 shuffle 步骤。"""
            pass
        self._env.seed = _no_shuffle_seed

    def close(self) -> None:
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
            self._env = None

    # ── BaseEnvWrapper interface ───────────────────────────────────────────────

    def reset(self) -> Tuple[str, Dict[str, Any]]:
        """
        Reset to a fresh episode.

        Returns:
            obs_text: Initial observation string.
            info:     {task_description, admissible_commands, task_type, won, done}
        """
        obs_list, infos = self._env.reset()
        self._step_count = 0

        obs_text: str = obs_list[0] if isinstance(obs_list, (list, tuple)) else obs_list

        # Extract task description from the first observation or infos
        # ALFWorld prepends "Your task is to: ..." in the first obs
        self._task_desc = self._extract_task(obs_text, infos)

        admissible: List[str] = self._get_admissible(infos)
        info = {
            "task_description":    self._task_desc,
            "task_type":           detect_task_type(self._task_desc),
            "admissible_commands": admissible,
            "won":                 False,
            "done":                False,
            "step":                0,
        }
        return obs_text, info

    def step(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        """
        Execute one action step.

        Args:
            action: Extracted action string (no <action> tags).

        Returns:
            obs_text, reward, done, info
        """
        self._step_count += 1

        obs_list, scores, dones_list, infos = self._env.step([action])
        obs_text: str = obs_list[0] if isinstance(obs_list, (list, tuple)) else obs_list
        won: bool = bool(infos.get("won", [False])[0])
        done: bool = bool(dones_list[0]) or (self._step_count >= self._max_steps)

        # Reward shaping: 基础成功奖励 10.0，减去步数惩罚（每步 -0.1）
        # 鼓励更短、更高效的轨迹；失败则 reward=0.0
        if won:
            reward = 10.0 - 0.1 * self._step_count
        else:
            reward = 0.0

        admissible: List[str] = self._get_admissible(infos)
        info = {
            "task_description":    self._task_desc,
            "task_type":           detect_task_type(self._task_desc),
            "admissible_commands": admissible,
            "won":                 won,
            "done":                done,
            "step":                self._step_count,
        }
        return obs_text, reward, done, info

    @property
    def task_description(self) -> str:
        return self._task_desc

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_task(obs_text: str, infos: Dict) -> str:
        """Pull the task goal sentence from the initial observation."""
        # Pattern: "Your task is to: <goal>."
        match = re.search(r"[Yy]our task is to[:\s]+(.+?)[\.\n]", obs_text)
        if match:
            return match.group(1).strip()
        # Fallback: use the gamefile path tail if available
        gamefile = infos.get("extra.gamefile", [""])[0] if isinstance(
            infos.get("extra.gamefile", ""), list) else infos.get("extra.gamefile", "")
        if gamefile:
            return os.path.basename(os.path.dirname(str(gamefile)))
        return obs_text[:120]   # last resort: truncate raw obs

    @staticmethod
    def _get_admissible(infos: Dict) -> List[str]:
        """Extract admissible commands list from infos dict."""
        cmds = infos.get("admissible_commands", [[]])[0]
        if isinstance(cmds, (list, tuple)):
            return list(cmds)
        return []
