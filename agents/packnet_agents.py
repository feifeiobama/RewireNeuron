#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import copy

import numpy as np
import torch
import torch.nn.functional as F
from core import CRLAgent
from torch import nn


class PacknetAction(CRLAgent):
    def __init__(self, input_dimension,output_dimension, hidden_size, start_steps = 0, input_name = "env/env_obs", activation = nn.LeakyReLU(negative_slope=0.2), layer_norm = False, prune_coeff = 0.5):
        super().__init__()
        self.start_steps = start_steps
        self.counter = 0
        self.iname = input_name
        self.task_id = 0
        self.output_dimension = output_dimension
        self.hs = hidden_size
        self.input_size = input_dimension
        self.activation = activation
        self.layer_norm = layer_norm
        self.prune_coeff = prune_coeff
        self.model = nn.ModuleList([self.make_model()])
        self.task_mask = self.make_model()
        for param in self.task_mask.parameters():
            param.requires_grad = False
            param.data.fill_(-1.)

    def make_model(self):
        if self.layer_norm:
            return nn.Sequential(
            nn.Linear(self.input_size,self.hs),
            nn.LayerNorm(self.hs),
            nn.Tanh(),
            nn.Linear(self.hs,self.hs),
            self.activation,
            nn.Linear(self.hs,self.hs),
            self.activation,
            nn.Linear(self.hs,self.output_dimension * 2),
        )       
        else:
            return nn.Sequential(
            nn.Linear(self.input_size,self.hs),
            self.activation,
            nn.Linear(self.hs,self.hs),
            self.activation,
            nn.Linear(self.hs,self.hs),
            self.activation,
            nn.Linear(self.hs,self.output_dimension * 2),
        )

    def forward(self, t = None, finetune = False, **kwargs):
        model_id = min(self.task_id,len(self.model) - 1)
        if not self.training:
            input = self.get((self.iname, t))
            mu, _ = self.model[model_id](input).chunk(2, dim=-1)
            action = torch.tanh(mu)
            self.set(("action", t), action)
        elif not (t is None):
            self.prune(finetune)
            input = self.get((self.iname, t)).detach()
            if self.counter <= self.start_steps:
                action = torch.rand(input.shape[0],self.output_dimension).to(input.device) * 2 - 1
            else:
                mu, log_std = self.model[model_id](input).chunk(2, dim=-1)
                log_std = torch.clip(log_std, min=-20., max=2.)
                std = log_std.exp()
                action = mu + torch.randn(*mu.shape).to(mu.device) * std
                action = torch.tanh(action)
            self.set(("action", t), action)
            self.counter += 1
        else:
            self.prune(finetune)
            input = self.get(self.iname).detach()
            mu, log_std = self.model[model_id](input).chunk(2, dim=-1)
            log_std = torch.clip(log_std, min=-20., max=2.)
            std = log_std.exp()
            action = mu + torch.randn(*mu.shape).to(mu.device) * std
            log_prob = (-0.5 * (((action - mu) / (std + 1e-8)) ** 2 + 2 * log_std + np.log(2 * np.pi))).sum(-1, keepdim=True)
            log_prob -= (2 * np.log(2) - action - F.softplus( - 2 * action)).sum(-1, keepdim=True)
            action = torch.tanh(action)
            self.set("action", action)
            self.set("action_logprobs", log_prob)

    def prune(self,finetune = False):
        assert self.task_id == len(self.model) - 1, "wrong task id"
        if self.task_id == 0:
            if finetune:
                for layer,task_mask in zip(self.model[-1],self.task_mask):
                    if isinstance(layer,nn.Linear):
                        layer.weight.data = layer.weight.data * (task_mask.weight.data == self.task_id)
                        layer.bias.data = layer.bias.data * (task_mask.bias.data == self.task_id)
        else:
            for layer,last_layer, task_mask in zip(self.model[-1],self.model[-2],self.task_mask):
                if isinstance(layer,nn.Linear):
                    if finetune:
                        layer.weight.data = last_layer.weight.data * (task_mask.weight.data > -1) * (task_mask.weight.data < self.task_id) + layer.weight.data * (task_mask.weight.data == self.task_id)
                        layer.bias.data = last_layer.bias.data * (task_mask.bias.data > -1) * (task_mask.bias.data < self.task_id) + layer.bias.data * (task_mask.bias.data == self.task_id)
                    else:
                        layer.weight.data = last_layer.weight.data * (task_mask.weight.data > -1) + layer.weight.data * (task_mask.weight.data == -1)
                        layer.bias.data = last_layer.bias.data * (task_mask.bias.data > -1) + layer.bias.data * (task_mask.bias.data == -1)


    def set_task(self,task_id = None):
        if task_id is None:
            self.model.append(copy.deepcopy(self.model[-1]))
            self.task_id = len(self.model) - 1
        else:
            self.task_id = task_id

    def add_filter(self):
        for layer,task_mask in zip(self.model[-1],self.task_mask):
            if isinstance(layer,nn.Linear):

                # Weight mask
                nb_weights = np.prod(layer.weight.shape)
                cutoff_rank = int(nb_weights * self.prune_coeff)
                cutoff_value = (layer.weight.data * (task_mask.weight.data == -1)).abs().view(-1).kthvalue(cutoff_rank).values
                mask = (layer.weight.data * (task_mask.weight.data == -1)).abs() > cutoff_value
                mask = mask.float()
                task_mask.weight.data = task_mask.weight.data * (1 - mask) +  self.task_id * mask

                # Bias mask
                nb_biases = np.prod(layer.bias.shape)
                cutoff_rank = int(nb_biases * self.prune_coeff)
                cutoff_value = (layer.bias.data * (task_mask.bias.data == -1)).abs().view(-1).kthvalue(cutoff_rank).values
                mask = (layer.bias.data * (task_mask.bias.data == -1)).abs() > cutoff_value
                mask = mask.float()
                task_mask.bias.data = task_mask.bias.data * (1 - mask) +  self.task_id * mask