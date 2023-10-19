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
from brax.envs.ant import _SYSTEM_CONFIG as ant_config
from brax.envs.ant import Ant
from google.protobuf import text_format

OBS_DIM = 27
ACT_DIM = 8

class Ant(Ant):
    def __init__(self, env_task: str, **kwargs) -> None:
        self._ctrl_cost_weight = 0.5
        self._use_contact_forces = False
        self._contact_cost_weight = 5e-4
        self._healthy_reward = 1.0
        self._terminate_when_unhealthy = True
        self._healthy_z_range = (0.2,1.0)
        self._reset_noise_scale = 0.1
        self._exclude_current_positions_from_observation = True

        config = text_format.Parse(ant_config, brax.Config())
        env_specs = env_tasks[env_task]
        self.obs_mask = jp.concatenate(np.ones((1,OBS_DIM)))
        self.action_mask = jp.concatenate(np.ones((1,ACT_DIM)))
        for spec,coeff in env_specs.items():
            if spec == "gravity":
                config.gravity.z *= coeff
            elif spec == "friction":
                config.friction *= coeff
            elif spec == "mask":
                zeros = int(coeff*OBS_DIM)
                ones = OBS_DIM-zeros
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
        """Observe ant body position and velocities."""
        joint_angle, joint_vel = self.sys.joints[0].angle_vel(qp)

        # qpos: position and orientation of the torso and the joint angles.
        if self._exclude_current_positions_from_observation:
            qpos = [qp.pos[0, 2:], qp.rot[0], joint_angle]
        else:
            qpos = [qp.pos[0], qp.rot[0], joint_angle]

        # qvel: velocity of the torso and the joint angle velocities.
        qvel = [qp.vel[0], qp.ang[0], joint_vel]

        # external contact forces:
        # delta velocity (3,), delta ang (3,) * 10 bodies in the system
        if self._use_contact_forces:
            cfrc = [
                jp.clip(info.contact.vel, -1, 1),
                jp.clip(info.contact.ang, -1, 1)
            ]
        # flatten bottom dimension
            cfrc = [jp.reshape(x, x.shape[:-2] + (-1,)) for x in cfrc]
        else:
            cfrc = []
        return jp.concatenate(qpos + qvel + cfrc) * self.obs_mask

    def step(self, state: _env.State, action: jp.ndarray) -> _env.State:
        """Run one timestep of the environment's dynamics."""
        action = action * self.action_mask
        qp, info = self.sys.step(state.qp, action)

        velocity = (qp.pos[0] - state.qp.pos[0]) / self.sys.config.dt
        forward_reward = velocity[0]

        min_z, max_z = self._healthy_z_range
        is_healthy = jp.where(qp.pos[0, 2] < min_z, x=0.0, y=1.0)
        is_healthy = jp.where(qp.pos[0, 2] > max_z, x=0.0, y=is_healthy)
        if self._terminate_when_unhealthy:
            healthy_reward = self._healthy_reward
        else:
            healthy_reward = self._healthy_reward * is_healthy
        ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))
        contact_cost = (self._contact_cost_weight *
                        jp.sum(jp.square(jp.clip(info.contact.vel, -1, 1))))
        obs = self._get_obs(qp, info)
        reward = forward_reward + healthy_reward - ctrl_cost - contact_cost
        done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0
        state.metrics.update(
            reward_forward=forward_reward,
            reward_survive=healthy_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            x_position=qp.pos[0, 0],
            y_position=qp.pos[0, 1],
            distance_from_origin=jp.norm(qp.pos[0]),
            x_velocity=velocity[0],
            y_velocity=velocity[1],
            forward_reward=forward_reward,
        )

        return state.replace(qp=qp, obs=obs, reward=reward, done=done)

env_tasks = {
    "normal":{},
    "hugefoot":{'Body':1.5},
    "moon":{'gravity':0.7},
    "rainfall":{'friction':0.375},
    "disabled_hard1":{"action_mask":[2,3,4,5,6,7]},
    "disabled_hard2":{"action_mask":[0,1,4,5,6,7]},
    "disabled_forefeet":{"action_mask":[0,1,2,3]},
    "disabled_backfeet":{"action_mask":[4,5,6,7]},
    "disabled_first_diagonal":{"action_mask":[0,1,4,5]},
    "disabled_second_diagonal":{"action_mask":[2,3,6,7]},
    "inverted_actions":{"action_swap":[0,1,2,3,4,5,6,7]},
}