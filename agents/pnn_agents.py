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


class PNNColumn(nn.Module):
    def __init__(self, input_dimension,output_dimension, hidden_size, activation = nn.LeakyReLU(negative_slope=0.2), column_id = 0):
        super().__init__()
        self.column_id = column_id
        self.activation = activation  
        self.output_dimension = output_dimension
        self.hs = hidden_size
        self.input_dimension = input_dimension

        # Column
        self.w = nn.ModuleList([
            nn.Linear(self.input_dimension,self.hs),
            nn.Linear(self.hs,self.hs),
            nn.Linear(self.hs,self.hs),
            nn.Linear(self.hs,self.output_dimension * 2)])

        # Laterals
        if self.column_id >0:
            self.u = nn.ModuleList([
                nn.Identity(),
                nn.Linear(self.hs,self.hs),
                nn.Linear(self.hs,self.hs),
                nn.Linear(self.hs,self.output_dimension * 2)])
            self.v = nn.ModuleList([
                nn.Identity(),
                nn.Linear(self.hs * self.column_id,self.hs, bias=False),
                nn.Linear(self.hs * self.column_id,self.hs, bias=False),
                nn.Linear(self.hs * self.column_id, self.hs, bias=False)])
            self.alpha = nn.ParameterList([nn.Parameter(torch.ones(1))])
            self.alpha.extend([nn.Parameter(torch.rand(1)/self.column_id) for _ in range(3)])

    def forward(self, x, h, **kwargs):

        # First input
        x = self.activation(self.w[0](x))
        new_h = [x if h is None else torch.cat([h[0],x], dim = -1)]
        #print("\n\n********* COLUMN",self.column_id," *********")
        for i in range(1,len(self.w)):
            # First column
            if self.column_id == 0:
                x = self.activation(self.w[i](x))
                new_h.append(x)
            # Columns with laterals
            else:
                #if i > 0:
                #    print("--- self.v[i].weight.shape:",self.v[i].weight.shape)
                _h = self.activation( self.v[i]( self.alpha[i] * h[i-1] ) )
                ##print("--- _h_before:",_h.shape)
                _h = self.u[i]( _h )
                #print("--- _h:",_h.shape)
                #print("--- x:",x.shape)
                x = self.w[i](x) + _h
                if i < len(self.w) - 1:
                    x = self.activation( x )
                #print("--- self.activation( self.w[i](x) + _h ):",x.shape)
                #print("--- torch.cat([h[i],x]:",torch.cat([h[i],x], dim = -1).shape)
                new_h.append(torch.cat([h[i],x], dim = -1))
        return new_h,x

class PNNAction(CRLAgent):
    def __init__(self, input_dimension,output_dimension, hidden_size, start_steps = 0, input_name = "env/env_obs", activation = nn.LeakyReLU(negative_slope=0.2)):
        super().__init__()
        self.iname = input_name
        self.task_id = 0
        self.activation = activation
        self.start_steps = start_steps
        self.counter = 0

        self.output_dimension = output_dimension
        self.hs = hidden_size
        self.input_dimension = input_dimension
        self.columns = nn.ModuleList([PNNColumn(self.input_dimension,self.output_dimension,self.hs, activation = self.activation, column_id = 0)])

    def _forward(self,x,column_id):
        h = None
        for i in range(column_id + 1):
            h, output = self.columns[i](x,h)
        return output

    def forward(self, t = None, **kwargs):
        column_id = min(self.task_id,len(self.columns) - 1)
        if not self.training:
            input = self.get((self.iname, t))
            mu, _ = self._forward(input,column_id).chunk(2, dim=-1)
            action = torch.tanh(mu)
            self.set(("action", t), action)
        elif not (t is None):
            input = self.get((self.iname, t)).detach()
            if self.counter <= self.start_steps:
                action = torch.rand(input.shape[0],self.output_dimension).to(input.device) * 2 - 1
            else:
                mu, log_std = self._forward(input,column_id).chunk(2, dim=-1)
                log_std = torch.clip(log_std, min=-20., max=2.)
                std = log_std.exp()
                action = mu + torch.randn(*mu.shape).to(mu.device) * std
                action = torch.tanh(action)
            self.set(("action", t), action)
            self.counter += 1
        else:
            input = self.get(self.iname).detach()
            mu, log_std = self._forward(input,column_id).chunk(2, dim=-1)
            log_std = torch.clip(log_std, min=-20., max=2.)
            std = log_std.exp()
            action = mu + torch.randn(*mu.shape).to(mu.device) * std
            log_prob = (-0.5 * (((action - mu) / (std + 1e-8)) ** 2 + 2 * log_std + np.log(2 * np.pi))).sum(-1, keepdim=True)
            log_prob -= (2 * np.log(2) - action - F.softplus( - 2 * action)).sum(-1, keepdim=True)
            action = torch.tanh(action)
            self.set("action", action)
            self.set("action_logprobs", log_prob)

    def set_task(self,task_id = None):
        if task_id is None:
            for param in self.columns[-1].parameters():
                param.requires_grad = False
            self.columns.append(PNNColumn(self.input_dimension,self.output_dimension,self.hs, activation = self.activation, column_id = len(self.columns)))
            self.task_id = len(self.columns) - 1 
        else:
            self.task_id = task_id