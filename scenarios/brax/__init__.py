#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import typing as tp

import jax
from brax.envs import Env as BraxEnv, wrappers
from core import Scenario, Task

from .ant import Ant
from .halfcheetah import Halfcheetah
from .humanoid import Humanoid

brax_domains = {
    "halfcheetah": Halfcheetah,
    "ant": Ant,
    "humanoid": Humanoid
}


class VectorGymWrapper(wrappers.VectorGymWrapper):
    def __init__(self,
               env: BraxEnv,
               seed: int = 0,
               backend: tp.Optional[str] = None):
        super().__init__(env,seed,backend)
        def step(state, action):
            state = self._env.step(state, action)
            info = {**state.metrics, **state.info, "qp":state.qp}
            return state, state.obs, state.reward, state.done, info
        self._step = jax.jit(step, backend=self.backend)


def make_brax_env(seed = 0,
            batch_size = None,
            max_episode_steps = 1000,
            action_repeat = 1,
            backend = None,
            auto_reset = True,
            domain = "halfcheetah",
            env_task = "normal",
            **kwargs):

    env: BraxEnv = brax_domains[domain](env_task, **kwargs)
    sys = env.sys
    if max_episode_steps is not None:
        env = wrappers.EpisodeWrapper(env, max_episode_steps, action_repeat)
    if batch_size:
        env = wrappers.VectorWrapper(env, batch_size)
    if auto_reset:
        env = wrappers.AutoResetWrapper(env)
    if batch_size is None:
        env = wrappers.GymWrapper(env, seed=seed, backend=backend)
    else:
        env = VectorGymWrapper(env, seed=seed, backend=backend)
    env.sys = sys
    return env


class BraxScenario(Scenario):
    def __init__(self,n_train_envs,n_evaluation_envs,n_steps,domain,tasks, repeat_scenario, **kwargs):
        super().__init__()
        tasks = list(tasks) * repeat_scenario
        print("Domain:",domain)
        print("Scenario:",tasks)
        for k,task in enumerate(tasks):
            agent_cfg={
                "classname":"agents.AutoResetBraxAgent",
                "make_env_fn":make_brax_env,
                "make_env_args":{
                                "domain":domain,
                                "max_episode_steps":1000,
                                "env_task":task},
                "n_envs":n_train_envs
            }
            self._train_tasks.append(Task(agent_cfg,k,n_steps))
            test_cfg={
                "classname":"agents.NoAutoResetBraxAgent",
                "make_env_fn":make_brax_env,
                "make_env_args":{
                                "domain":domain,
                                "max_episode_steps":1000,
                                "env_task":task},
                "n_envs":n_evaluation_envs
            }
            self._test_tasks.append(Task(test_cfg,k))