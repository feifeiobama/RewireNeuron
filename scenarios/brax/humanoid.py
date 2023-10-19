#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import brax
from brax.envs.humanoid import Humanoid, _SYSTEM_CONFIG as humanoid_config
from brax.envs import env as _env
from brax import jumpy as jp
from google.protobuf import text_format
import numpy as np

OBS_SHAPE = 376
ACTION_SHAPE = 17

class Humanoid(Humanoid):
    def __init__(self, env_task: str, **kwargs) -> None:
        self._forward_reward_weight = 1.25
        self._ctrl_cost_weight = 0.1
        self._healthy_reward = 5.0
        self._terminate_when_unhealthy = True
        self._healthy_z_range = (0.8,2.1)
        self._reset_noise_scale = 1e-2
        self._exclude_current_positions_from_observation = (True)

        config = text_format.Parse(humanoid_config, brax.Config())
        env_specs = env_tasks[env_task]
        self.obs_mask = jp.concatenate(np.ones((1,OBS_SHAPE)))
        self.action_mask = jp.concatenate(np.ones((1,ACTION_SHAPE)))
        for spec,coeff in env_specs.items():
            if spec == "gravity":
                config.gravity.z *= coeff
            elif spec == "friction":
                config.friction *= coeff
            elif spec == "obs_mask":
                zeros = int(coeff * OBS_SHAPE)
                ones = OBS_SHAPE - zeros
                np.random.seed(0)
                self.obs_mask = jp.concatenate(np.random.permutation(([0]*zeros)+([1]*ones)).reshape(1,-1))
            elif spec == "action_mask":
                self.action_mask[coeff] = 0.
            elif spec == "action_swap":
                self.action_mask[coeff] = -1.
            else:
                for body in config.bodies:
                    if spec in body.name:
                        pass
                        body.mass *= coeff
                        body.colliders[-1].capsule.radius *= coeff
        self.sys = brax.System(config)

env_tasks = {
    # environmental changes
    "normal": {},
    "moon": {"gravity":0.15},
    "carrystuff":{"torso":4.,"lwaist":4.},
    "tinyfeet":{"shin":0.5},
}
