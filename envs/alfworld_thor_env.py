"""
envs/alfworld_thor_env.py

Single-instance ALFWorld vision-mode wrapper (AlfredThorEnv + mrcnn_astar controller).

Option 2 (perception-robustness experiment, see docs/adr/0002-*.md):
BUTLER::VISION = pretrained Mask-RCNN detector (via the `mrcnn_astar` controller),
BUTLER::BODY = ThorEnv (ai2thor), both untouched from alfworld-master. The controller
still emits a plain-text feedback string identical in style to AlfredTWEnv's output
(see BaseAgent.print_intro / MaskRCNNAgent.step in alfworld-master), so the rest of
the interface -- task extraction, admissible-command lookup, reward, done -- is
unchanged from the text-mode wrapper. Only the environment backend differs, so this
subclasses AlfworldTextEnv and swaps in "AlfredThorEnv".

Requires a headless display on the Linux training machine (e.g. `xvfb-run`), since
ai2thor renders real RGB frames even though this wrapper only ever surfaces the
resulting text to the caller (BUTLER::BRAIN never sees pixels directly).
"""

import os
import yaml

from envs.alfworld_env import AlfworldTextEnv


class AlfredThorTextEnv(AlfworldTextEnv):
    """
    Wraps a single AlfredThorEnv instance.

    Same reset()/step()/task_description interface as AlfworldTextEnv, so the
    rollout collector and trainer can use either wrapper interchangeably via
    BaseEnvWrapper -- only envs/alfworld_thor_config.yaml (env.type: AlfredThorEnv,
    controller.type: mrcnn_astar) needs to be passed in as config_path.
    """

    def _init_env(self) -> None:
        assert os.path.exists(self._config_path), (
            f"ALFWorld config not found: {self._config_path}"
        )
        with open(self._config_path) as f:
            config = yaml.safe_load(f)

        from alfworld.agents.environment import get_environment
        base_env = get_environment("AlfredThorEnv")(config, train_eval=self._train_eval)
        self._env = base_env.init_env(batch_size=1)
        self._env.seed(self._seed)
