#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import typing as tp

import salina
import torch
import torch.utils.data
from salina import Agent, Workspace, instantiate_class
from salina.agents import Agents, TemporalAgent
from salina.agents.brax import EpisodesDone


class Task:
    """A Reinforcement Learning task defined as a SaLinA agent. Use make() method
    to instantiate the salina agent corresponding to the task. 

    Parameters
    ----------
    env_agent_cfg   : The OmegaConf (or dict) that allows to configure the SaLinA agent
    task_id         : An identifier of the task
    n_interactions  : Defaults to None. Number of env interactions allowed for training
    input_dimension : The input dimension of the observations
    output_dimension: The output dimension of the actions (i.e size of the output tensor, or number of actions if discrete actions)
    """
    def __init__(self,env_agent_cfg: dict,
                      task_id: int,
                      n_interactions: tp.Union[None,int] = None,
                      input_dimension: tp.Union[None,int] = None,
                      output_dimension: tp.Union[None,int] = None,
                      )  -> None:
        self._task_id = task_id
        self._n_interactions = n_interactions
        self._env_agent_cfg = env_agent_cfg

        if input_dimension is None or output_dimension is None:
            env = env_agent_cfg["make_env_fn"](**env_agent_cfg["make_env_args"])
            self._input_dimension = env.observation_space.shape[0]
            self._output_dimension = env.action_space.shape[0]
        else:
            self._input_dimension = input_dimension
            self._output_dimension = output_dimension


    def input_dimension(self) -> int:
        return self._input_dimension

    def output_dimension(self) -> int:
        return self._output_dimension

    def task_id(self) -> int:
        return self._task_id

    def env_cfg(self) -> dict:
        return self._env_agent_cfg

    def make(self) -> salina.Agent:
        agent = instantiate_class(self._env_agent_cfg)
        agent.set_name("env")
        return agent

    def n_interactions(self) -> int:
        return self._n_interactions

class Scenario:
    """ 
    A scenario is a sequence of train tasks and a sequence of test tasks.
    """

    def __init__(self) -> None:
        self._train_tasks = []
        self._test_tasks = []

    def train_tasks(self) -> tp.List[Task]:
        return self._train_tasks

    def test_tasks(self) -> tp.List[Task]:
        return self._test_tasks



class Framework:
    """A (CRL) Model can be updated over one new task, and evaluated over any task
    
    Parameters
    ----------
    seed 
    params : The OmegaConf (or dict) that allows to configure the model
    """
    def __init__(self,seed: int,params: dict) -> None:
        self.seed=seed
        self.cfg=params
        self._stage=0

    def memory_size(self) -> dict:    
        raise NotImplementedError

    def get_stage(self) -> int:
        return self._stage

    def train(self,task: Task,logger: tp.Any, **extra_args) -> None:
        """ Update a model over a particular task.

        Parameters
        ----------
        task: The task to train on
        logger: a salina logger to log metrics and messages
        """
        logger.message("-- Train stage "+str(self._stage))
        output=self._train(task,logger.get_logger("stage_"+str(self._stage)+"/"))
        [logger.add_scalar("monitor_per_stage/"+k,output[k],self._stage) for k in output]
        self._stage+=1

    def evaluate(self,test_tasks: tp.List[Task], logger: tp.Any) -> dict:
        """ Evaluate a model over a set of test tasks
        
        Parameters
        ----------
        test_tasks: The set of tasks to evaluate on
        logger: a salina logger

        Returns
        ----------
        evaluation: A dict containing some evaluation metrics
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

    def _train(self,task: Task,logger: tp.Any) -> None:
        raise NotImplementedError

    def get_evaluation_agent(self,task_id: int) -> salina.Agent:
        raise NotImplementedError

    def _evaluate_single_task(self,task: Task) -> dict:
        metrics={}
        env_agent=task.make()
        policy_agent=self.get_evaluation_agent(task.task_id())

        if not policy_agent is None:
            policy_agent.eval()
            no_autoreset = EpisodesDone()
            acquisition_agent = TemporalAgent(Agents(env_agent,no_autoreset,policy_agent))
            acquisition_agent.seed(self.seed*13+self._stage*100)
            acquisition_agent.to(self.cfg.evaluation.device)

            avg_reward=0.0
            n=0
            avg_success=0.0
            for r in range(self.cfg.evaluation.n_rollouts):
                workspace=Workspace()
                acquisition_agent(workspace,t=0,stop_variable="env/done")
                ep_lengths=workspace["env/done"].max(0)[1]+1
                B=ep_lengths.size()[0]
                arange=torch.arange(B).to(ep_lengths.device)
                cr=workspace["env/cumulated_reward"][ep_lengths-1,arange]
                avg_reward+=cr.sum().item()
                if self.cfg.evaluation.evaluate_success:
                    cr=workspace["env/success"][ep_lengths-1,arange]
                    avg_success+=cr.sum().item()
                n+=B
            avg_reward /= n
            metrics["avg_reward"] = avg_reward

            if self.cfg.evaluation.evaluate_success:
                avg_success/=n
                metrics["success_rate"]=avg_success
        return metrics


class CRLAgent(Agent):
    """A salina Agent that is able to apply set_task() and add_regularizer() methods
    """
    def set_task(self,task_id: tp.Union[None,int] = None) -> None:
        pass

    def add_regularizer(self, *args) -> torch.Tensor:
        return torch.Tensor([0.]).to(list(self.parameters())[0].device)

class CRLAgents(Agents):
    """A batch of CRL Agents called sequentially.
    """
    def set_task(self,task_id: tp.Union[None,int] = None) -> None:
        for agent in self:
            agent.set_task(task_id)

    def add_regularizer(self, *args) -> torch.Tensor:
        return torch.cat([agent.add_regularizer(*args) for agent in self]).sum()
