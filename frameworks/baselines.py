#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import re

import torch
from salina import instantiate_class

from core import Framework


def get_checkpoint(path, keyword="policy_"):
    files = [x for x in os.listdir(path) if keyword in x]
    file = max(files, key = lambda x: int(x[re.search("_",x).end():re.search("\.",x).start()]))
    stage = int(file[re.search("_",file).end():re.search("\.",file).start()]) + 1
    return path+"/"+file, stage

class OneStep(Framework):
    """
    OneStep framework. 1 algorithm.
    """
    def __init__(self,seed,params):
        super().__init__(seed,params)
        self.algorithm = instantiate_class(self.cfg.algorithm)
        self.policy_agent = None
        self.critic_agent = None
        if "path" in self.cfg:
            print("Found a checkpoint. Loading last policy checkpointed...")
            policy_path, stage = get_checkpoint( self.cfg.path, keyword = "policy_")
            self._stage = stage
            print("self._stage="+str(self._stage))
            self.policy_agent = torch.load(policy_path)
            print("Policy loaded successfully ! Resuming on stage "+str(self._stage))

    def _create_policy_agent(self,task,logger):
        logger.message("Creating Policy Agent")
        policy_agent_cfg = self.cfg.policy_agent
        policy_agent_cfg.input_dimension = task.input_dimension()
        policy_agent_cfg.output_dimension = task.output_dimension()
        self.policy_agent = instantiate_class(policy_agent_cfg)

    def _create_critic_agent(self,task,logger):
        logger.message("Creating Critic Agent")
        critic_agent_cfg = self.cfg.critic_agent
        critic_agent_cfg.obs_dimension = task.input_dimension()
        critic_agent_cfg.action_dimension = task.output_dimension()
        self.critic_agent = instantiate_class(critic_agent_cfg)

    def _train(self,task,logger):
        if self.policy_agent is None:
            self._create_policy_agent(task,logger)
        else:
            self.policy_agent.set_task()
        if self.critic_agent is None:
            self._create_critic_agent(task,logger)
        env_agent = task.make()
        info = {"task_id":task.task_id()}
        r,self.policy_agent,self.critic_agent, info = self.algorithm.run(self.policy_agent, self.critic_agent, env_agent, logger, self.seed, n_max_interactions = task.n_interactions(), info = info)

        if self.cfg.checkpoint:
            torch.save(self.critic_agent,os.getcwd()+"/critic_"+str(task._task_id)+".dat")
            torch.save(self.policy_agent,os.getcwd()+"/policy_"+str(task._task_id)+".dat")

        return r

    def memory_size(self):
        pytorch_total_params = sum(p.numel() for p in self.policy_agent.state_dict().values())
        return {"n_parameters":pytorch_total_params}

    def get_evaluation_agent(self,task_id):
        self.policy_agent.set_task(task_id)
        return self.policy_agent

class TwoSteps(OneStep):
    """
    A model that is using 2 algorithms (algorithm 2 is for example a regularization method).
    """
    def __init__(self,seed,params):
        super().__init__(seed,params)
        self.algorithm2 = instantiate_class(self.cfg.algorithm2)

    def _train(self,task,logger):
        if self.policy_agent is None:
            self._create_policy_agent(task,logger)
        else:
            self.policy_agent.set_task()
        if self.critic_agent is None:
            self._create_critic_agent(task,logger)
        env_agent = task.make()
        info = {"task_id":task.task_id()}
        r1,self.policy_agent,self.critic_agent, info = self.algorithm.run(self.policy_agent, self.critic_agent, env_agent,logger, self.seed, n_max_interactions = task.n_interactions(), info = info)
        r2,self.policy_agent,self.critic_agent, info = self.algorithm2.run(self.policy_agent, self.critic_agent, logger, info)
        if self.cfg.checkpoint:
            torch.save(self.critic_agent,os.getcwd()+"/critic_"+str(task._task_id)+".dat")
            torch.save(self.policy_agent,os.getcwd()+"/policy_"+str(task._task_id)+".dat")
        return r1