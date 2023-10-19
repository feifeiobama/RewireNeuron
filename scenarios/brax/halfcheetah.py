#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import brax
import numpy as np
from brax import jumpy as jp
from brax.envs import env as _env
from brax.envs.half_cheetah import _SYSTEM_CONFIG as halfcheetah_config
from brax.envs.half_cheetah import Halfcheetah
from google.protobuf import text_format

OBS_DIM = 18
ACT_DIM = 6

class Halfcheetah(Halfcheetah):
    def __init__(self, env_task: str, **kwargs) -> None:
        self._forward_reward_weight = 1.0
        self._ctrl_cost_weight = 0.1
        self._reset_noise_scale = 0.1
        self._exclude_current_positions_from_observation = (True)
        config = text_format.Parse(halfcheetah_config, brax.Config())
        env_specs = env_tasks[env_task]
        self.obs_mask = jp.concatenate(np.ones((1,OBS_DIM)))
        self.action_mask = jp.concatenate(np.ones((1,ACT_DIM)))
        for spec,coeff in env_specs.items():
            if spec == "gravity":
                config.gravity.z *= coeff
            elif spec == "friction":
                config.friction *= coeff
            elif spec == "obs_mask":
                zeros = int(coeff * OBS_DIM)
                ones = OBS_DIM - zeros
                np.random.seed(0)
                self.obs_mask = jp.concatenate(np.random.permutation(([0]*zeros)+([1]*ones)).reshape(1,-1))
            elif spec == "action_mask":
                self.action_mask[coeff] = 0.
            elif spec == "action_swap":
                self.action_mask[coeff] = -1.
            else:
                for body in config.bodies:
                    if spec in body.name:
                        body.mass *= coeff
                        body.colliders[0].capsule.radius *= coeff
        self.sys = brax.System(config)

    def _get_obs(self, qp: brax.QP, info: brax.Info) -> jp.ndarray:
        """Observe halfcheetah body position and velocities."""
        joint_angle, joint_vel = self.sys.joints[0].angle_vel(qp)
        # qpos: position and orientation of the torso and the joint angles
        # TODO: convert rot to just y-ang component
        if self._exclude_current_positions_from_observation:
            qpos = [qp.pos[0, 2:], qp.rot[0, (0, 2)], joint_angle]
        else:
            qpos = [qp.pos[0, (0, 2)], qp.rot[0, (0, 2)], joint_angle]
        # qvel: velocity of the torso and the joint angle velocities
        qvel = [qp.vel[0, (0, 2)], qp.ang[0, 1:2], joint_vel]
        return jp.concatenate(qpos + qvel) * self.obs_mask

    def reset(self, rng: jp.ndarray) -> _env.State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2 = jp.random_split(rng, 3)

        qpos = self.sys.default_angle() + self._noise(rng1)
        qvel = self._noise(rng2)

        qp = self.sys.default_qp(joint_angle=qpos, joint_velocity=qvel)
        self._qps = [qp]
        obs = self._get_obs(qp, self.sys.info(qp))
        reward, done, zero = jp.zeros(3)
        metrics = {
            'x_position': zero,
            'x_velocity': zero,
            'reward_ctrl': zero,
            'reward_run': zero,
        }
        return _env.State(qp, obs, reward, done, metrics)


    def step(self, state: _env.State, action: jp.ndarray) -> _env.State:
        """Run one timestep of the environment's dynamics."""
        action = action * self.action_mask
        qp, info = self.sys.step(state.qp, action)
        self._qps.append(qp)

        velocity = (qp.pos[0] - state.qp.pos[0]) / self.sys.config.dt
        forward_reward = self._forward_reward_weight * velocity[0]
        ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))

        obs = self._get_obs(qp, info)
        reward = forward_reward - ctrl_cost
        state.metrics.update(
            x_position=qp.pos[0, 0],
            x_velocity=velocity[0],
            reward_run=forward_reward,
            reward_ctrl=-ctrl_cost)
        return state.replace(qp=qp, obs=obs, reward=reward)

    def get_qps(self):
        return self._qps

env_tasks = {
    "normal":{},
    "carry_stuff":{"torso": 4.,"thigh": 1.,"shin": 1.,"foot": 1.},
    'carry_stuff_hugegravity': {'torso': 4.0,'thigh': 1.0,'shin': 1.0,'foot': 1.0,'gravity': 1.5},
    "defective_module":{"obs_mask":0.5},
    "hugefoot":{"foot":1.5},
    "hugefoot_rainfall": {'foot': 1.5, 'friction': 0.4},
    "inverted_actions":{"action_swap":[0,1,2,3,4,5]},
    "moon":{"gravity":0.15},
    "tinyfoot":{"foot":0.5},
    "tinyfoot_moon": {'foot': 0.5, 'gravity': 0.15},
    "rainfall":{"friction":0.4},
}