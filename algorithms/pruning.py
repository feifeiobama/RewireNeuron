#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import copy

import numpy as np
import torch
from salina import get_arguments, get_class

from .tools import soft_update_params


class PruneAndFinetune:
    def __init__(self,params):
        self.cfg = params
    

    def run(self,action_agent, q_agent, logger, info):
        replay_buffer = info['replay_buffer']
        logger.message("adding filter")
        action_agent[0].add_filter()
        action_agent = self.run_sac(action_agent,q_agent,logger,replay_buffer)
        return {}, action_agent.to('cpu'), q_agent.to('cpu'), info

    def run_sac(self,action_agent, q_agent, logger, replay_buffer):
        
        logger.message("Starting finetuning procedure")
        
        action_agent.train()
        action_agent.to(self.cfg.device)
        q_agent.train()
        q_agent.to(self.cfg.device)
        q_target_agent = copy.deepcopy(q_agent)
        q_target_agent.to(self.cfg.device)

        log_entropy = torch.tensor(np.log(self.cfg.init_temperature), requires_grad=True, device=self.cfg.device)

        # == configuring SAC entropy
        optimizer_args = get_arguments(self.cfg.optimizer_entropy)
        action_card = np.prod(np.array(replay_buffer.get(self.cfg.batch_size).to(self.cfg.device)["action"].size()[2:]))
        target_entropy = - self.cfg.target_multiplier * action_card
        log_entropy = torch.tensor(np.log(self.cfg.init_temperature), requires_grad=True, device=self.cfg.device)
        optimizer_entropy = get_class(self.cfg.optimizer_entropy)([log_entropy], **optimizer_args)

        optimizer_args = get_arguments(self.cfg.optimizer_q)
        optimizer_q = get_class(self.cfg.optimizer_q)(q_agent.parameters(), **optimizer_args)
        optimizer_args = get_arguments(self.cfg.optimizer_policy)
        optimizer_action = get_class(self.cfg.optimizer_policy)(action_agent[0].parameters(), **optimizer_args)
        optimizer_entropy = get_class(self.cfg.optimizer_entropy)([log_entropy], **optimizer_args)

        for iteration in range(self.cfg.iterations):
            entropy = log_entropy.exp()
            replay_workspace = replay_buffer.get(self.cfg.batch_size).to(self.cfg.device)
            done, reward = replay_workspace["env/done", "env/reward"]
            not_done = 1.0 - done.float()
            reward = reward * self.cfg.reward_scaling

            # == q1 and q2 losses
            q_agent(replay_workspace)
            q_1 = replay_workspace["q1"]
            q_2 = replay_workspace["q2"]
            with torch.no_grad():
                action_agent(replay_workspace, q_update = True, finetune = True)
                q_target_agent(replay_workspace, q_update = True)
                q_target_1 = replay_workspace["q1"]
                q_target_2 = replay_workspace["q2"]
                _logp = replay_workspace["action_logprobs"]
                q_target = torch.min(q_target_1, q_target_2)
                target = (reward[1:]+ self.cfg.discount_factor * not_done[1:] * (q_target[1:] - (entropy * _logp[1:]).detach()))
            td_1 = ((q_1[:-1] - target) ** 2).mean()
            td_2 = ((q_2[:-1] - target) ** 2).mean()
            optimizer_q.zero_grad()
            loss = td_1 + td_2
            logger.add_scalar("finetune/td_loss_1",td_1.item(),iteration)
            logger.add_scalar("finetune/td_loss_2",td_2.item(),iteration)
            loss.backward()
            if self.cfg.clip_grad > 0:
                n = torch.nn.utils.clip_grad_norm_(q_agent.parameters(), self.cfg.clip_grad)
                logger.add_scalar("finetune/grad_norm_q", n.item(), iteration)
            optimizer_q.step()
            
            # == Actor and entropy losses
            if iteration % self.cfg.policy_update_delay == 0:
                action_agent(replay_workspace, policy_update = True, finetune = True)
                q_agent(replay_workspace, policy_update = True)
                logp = replay_workspace["action_logprobs"]
                q1 = replay_workspace["q1"]
                q2 = replay_workspace["q2"]
                qloss = torch.min(q1,q2).mean()
                entropy_loss = (entropy.detach() * logp).mean()
                optimizer_action.zero_grad()
                loss = - qloss + entropy_loss
                loss.backward()
                if self.cfg.clip_grad > 0:
                    n = torch.nn.utils.clip_grad_norm_(action_agent.parameters(), self.cfg.clip_grad)
                    logger.add_scalar("finetune/grad_norm_action", n.item(), iteration)
                logger.add_scalar("finetune/q_loss", qloss.item(), iteration)
                optimizer_action.step()

                optimizer_entropy.zero_grad()
                entropy_loss = - (log_entropy.exp() * (logp + target_entropy).detach()).mean()
                entropy_loss.backward()
                if self.cfg.clip_grad > 0:
                    n = torch.nn.utils.clip_grad_norm_(log_entropy, self.cfg.clip_grad)
                    logger.add_scalar("finetune/grad_norm_entropy", n.item(), iteration)
                optimizer_entropy.step()
                logger.add_scalar("finetune/entropy_loss", entropy_loss.item(), iteration)
                logger.add_scalar("finetune/entropy_value", entropy.item(), iteration)

                # == Target network update
                if iteration % self.cfg.target_update_delay == 0:
                    tau = self.cfg.update_target_tau
                    soft_update_params(q_agent[0], q_target_agent[0], tau)
                    soft_update_params(q_agent[1], q_target_agent[1], tau)
    
        #to lastly prune the model
        action_agent(replay_workspace, policy_update = True, finetune = True)
        return action_agent.to('cpu')