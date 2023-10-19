#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import copy
import os
import re

import torch
from salina import Workspace, instantiate_class
from salina.agents import Agents, TemporalAgent
from salina.agents.brax import EpisodesDone
from torch.distributions.dirichlet import Dirichlet

from core import Framework


def get_checkpoint(path, keyword="policy_"):
    files = [x for x in os.listdir(path) if keyword in x]
    file = max(files, key = lambda x: int(x[re.search("_",x).end():re.search("\.",x).start()]))
    stage = int(file[re.search("_",file).end():re.search("\.",file).start()]) + 1
    return path+"/"+file, stage

class Subspace(Framework):
    """
    Model for the subspace method.
    """
    def __init__(self,seed,params):
        super().__init__(seed,params)
        self.train_algorithm = instantiate_class(self.cfg.algorithm)
        self.alpha_search = instantiate_class(self.cfg.alpha_search)
        self.policy_agent = None
        self.critic_agent = None
        self.lr_policy = self.cfg.algorithm.params.optimizer_policy.lr
        if "path" in self.cfg:
            print("Found a checkpoint. Loading last policy checkpointed...")
            policy_path, stage = get_checkpoint( self.cfg.path, keyword = "policy_")
            self._stage = stage
            print("self._stage="+str(self._stage))
            self.policy_agent = torch.load(policy_path)
            print("Policy loaded successfully ! Resuming on stage "+str(self._stage))

    def _create_policy_agent(self,task,logger):
        logger.message("Creating policy Agent")
        assert self.policy_agent is None
        input_dimension = task.input_dimension()
        output_dimension = task.output_dimension()
        policy_agent_cfg = self.cfg.policy_agent
        policy_agent_cfg.input_dimension = input_dimension
        policy_agent_cfg.output_dimension = output_dimension
        self.policy_agent = instantiate_class(policy_agent_cfg)

    def _create_critic_agent(self,task,logger):
        logger.message("Creating Critic Agent")
        obs_dimension = task.input_dimension()
        action_dimension = task.output_dimension()
        critic_agent_cfg = self.cfg.critic_agent
        critic_agent_cfg.obs_dimension = obs_dimension
        critic_agent_cfg.action_dimension = action_dimension
        critic_agent_cfg.n_anchors = self.policy_agent[0].n_anchors
        self.critic_agent = instantiate_class(critic_agent_cfg)

    def _train(self,task,logger):
        task_id = task.task_id()
        if self.policy_agent is None:
            self._create_policy_agent(task,logger)
        if self.critic_agent is None:
            self._create_critic_agent(task,logger)
        env_agent = task.make()
        self.train_algorithm.cfg.optimizer_policy.lr = self.lr_policy * (1 + task_id * self.cfg.lr_scaling)
        logger.message("Setting policy_lr to "+str(self.train_algorithm.cfg.optimizer_policy.lr))
        info = {"task_id":task_id}
        if task_id > 0:
            self.policy_agent.add_anchor(logger = logger)
            self.critic_agent.add_anchor(n_anchors = self.policy_agent[0].n_anchors,logger = logger)
        burnin_interactions = self.cfg.alpha_search.params.n_rollouts * self.cfg.alpha_search.params.n_validation_steps
        r1, self.policy_agent, self.critic_agent, info = self.train_algorithm.run(self.policy_agent, self.critic_agent, env_agent,logger, self.seed, n_max_interactions = task.n_interactions() - burnin_interactions, info = info)
        r2, self.policy_agent, self.critic_agent, info = self.alpha_search.run(self.policy_agent, self.critic_agent, task, logger, self.seed, info = info)

        if self.cfg.checkpoint:
            torch.save(self.critic_agent,os.getcwd()+"/critic_"+str(task_id)+".dat")
            torch.save(self.policy_agent,os.getcwd()+"/policy_"+str(task_id)+".dat")
            del info["replay_buffer"]
            torch.save(info,os.getcwd()+"/info_"+str(task_id)+".dat")
            del info
        return r1

    def memory_size(self):
        pytorch_total_params = sum(p.numel() for p in self.policy_agent.state_dict().values())
        return {"n_parameters":pytorch_total_params}

    def get_evaluation_agent(self,task_id):
        self.policy_agent.set_task(task_id)
        return copy.deepcopy(self.policy_agent),self.critic_agent

    def evaluate(self,test_tasks,logger):
        """ Evaluate a model over a set of test tasks
        Args:
            test_tasks: The set of tasks to evaluate on
            logger
        Returns:
            evaluation: Some statistics about the evaluation (i.e metrics)
        """
        logger.message("Starting evaluation...")
        with torch.no_grad():
            evaluation={}
            for k,task in enumerate(test_tasks):
                metrics=self._evaluate_single_task(task,logger)
                evaluation[task.task_id()]=metrics
                logger.message("Evaluation over task "+str(k)+":"+str(metrics))

        logger.message("-- End evaluation...")
        return evaluation

    def _evaluate_single_task(self,task,logger):
        device = self.cfg.evaluation.device
        env_agent = task.make()
        policy_agent, _ =self.get_evaluation_agent(task.task_id())
        policy_agent.eval()
        acquisition_agent = TemporalAgent(Agents(env_agent, EpisodesDone(), policy_agent)).to(device)
        acquisition_agent.seed(self.seed*13+self._stage*100)

        #Evaluating best alpha
        rewards = []
        w = Workspace()
        for i in range(self.cfg.evaluation.n_rollouts):
            with torch.no_grad():
                acquisition_agent(w, t = 0, stop_variable = "env/done")
            ep_lengths= w["env/done"].max(0)[1]+1
            B = ep_lengths.size()[0]
            arange = torch.arange(B).to(device)
            cr = w["env/cumulated_reward"][ep_lengths-1,arange]
            rewards.append(cr)
        rewards = torch.stack(rewards, dim = 0).mean()
        metrics={ "avg_reward" : rewards.item()}
        del w

        #Evaluating oracle
        if self.cfg.evaluation.oracle_rollouts>0:
            rewards = []
            w = Workspace()
            n_anchors = policy_agent[0].n_anchors
            alphas = Dirichlet(torch.ones(n_anchors)).sample(torch.Size([B])).to(device)
            for i in range(self.cfg.evaluation.oracle_rollouts):
                with torch.no_grad():
                    acquisition_agent(w, t = 0, alphas = alphas, stop_variable = "env/done")
                ep_lengths= w["env/done"].max(0)[1]+1
                B = ep_lengths.size()[0]
                arange = torch.arange(B).to(device)
                cr = w["env/cumulated_reward"][ep_lengths-1,arange]
                rewards.append(cr)
            rewards = torch.stack(rewards, dim = 0).mean(0).max()
            metrics["oracle_reward"] = rewards.item()
            del w
        return metrics