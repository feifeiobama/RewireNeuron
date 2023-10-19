import copy
import os
import re

import numpy as np
import torch
from salina import Workspace, instantiate_class
from salina.agents import Agents, TemporalAgent
from salina.agents.brax import EpisodesDone

from core import Framework


def get_checkpoint(path, keyword="policy_"):
    files = [x for x in os.listdir(path) if keyword in x]
    file = max(files, key = lambda x: int(x[re.search("_",x).end():re.search("\.",x).start()]))
    stage = int(file[re.search("_",file).end():re.search("\.",file).start()]) + 1
    return path+"/"+file, stage


class Rewire(Framework):
    """
    Model for the rewire method.
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
        self.algorithm2 = instantiate_class(self.cfg.algorithm2)

    def _create_policy_agent(self,task,logger):
        logger.message("Creating policy Agent")
        assert self.policy_agent is None
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
        task_id = task.task_id()
        if self.policy_agent is None:
            self._create_policy_agent(task,logger)
        self.policy_agent.set_task()
        if task_id > 0:
            r0 = self._evaluate_single_task(task, self.policy_agent)['avg_reward']
        self.policy_agent.train()
        if self.critic_agent is None:
            self._create_critic_agent(task,logger)
        env_agent = task.make()
        logger.message("Setting policy_lr to " + str(self.algorithm.cfg.optimizer_policy.lr))
        info = {"task_id":task_id}
        r1, self.policy_agent, self.critic_agent, info = self.algorithm.run(self.policy_agent, self.critic_agent, env_agent, logger, self.seed, n_max_interactions = task.n_interactions(), info = info)
        if task_id > 0:  # follow CSP
            rs = []
            for k_ in range(self.policy_agent.agents[-1].k):
                self.policy_agent.agents[-1].k_ = k_
                rs.append(self._evaluate_single_task(task, self.policy_agent)['avg_reward'])
            print('Compare:', r0, rs)
            if max(rs) < r0 * (1 + self.cfg.improvement_threshold):
                self.policy_agent.agents[-1].roll_back()
            self.policy_agent.agents[-1].k_ = np.array(rs).argmax()
        r2, self.policy_agent, self.critic_agent, info = self.algorithm2.run(self.policy_agent, self.critic_agent, logger, info)
        self.policy_agent.set_task(-1)

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
        return copy.deepcopy(self.policy_agent), self.critic_agent

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
                metrics=self._evaluate_single_task(task)
                evaluation[task.task_id()]=metrics
                logger.message("Evaluation over task "+str(k)+":"+str(metrics))

        logger.message("-- End evaluation...")
        return evaluation

    def _evaluate_single_task(self, task, policy_agent=None):
        device = self.cfg.evaluation.device
        env_agent = task.make()
        if policy_agent is None:
            policy_agent, _ = self.get_evaluation_agent(task.task_id())
        policy_agent.eval()
        acquisition_agent = TemporalAgent(Agents(env_agent, EpisodesDone(), policy_agent)).to(device)
        acquisition_agent.seed(self.seed*13+self._stage*100)

        #Evaluating best alpha
        rewards = []
        successes = []
        w = Workspace()
        for i in range(self.cfg.evaluation.n_rollouts):
            with torch.no_grad():
                acquisition_agent(w, t=0, stop_variable = "env/done")
            ep_lengths= w["env/done"].max(0)[1]+1
            B = ep_lengths.size()[0]
            arange = torch.arange(B).to(device)
            cr = w["env/cumulated_reward"][ep_lengths-1,arange]
            rewards.append(cr)
            if self.cfg.evaluation.evaluate_success:
                cr = w["env/success"][ep_lengths-1,arange]
                successes.append(cr)
        rewards = torch.stack(rewards, dim = 0).mean()
        metrics={ "avg_reward" : rewards.item()}
        if self.cfg.evaluation.evaluate_success:
            successes = torch.stack(successes, dim = 0).mean()
            metrics["success_rate"] = successes.item()
        del w
        return metrics