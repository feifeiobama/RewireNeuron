#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import copy

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.dirichlet import Dirichlet


def create_dist(dist_type,n_anchors):
    n_anchors = max(1,n_anchors)
    if dist_type == "flat":
        dist = Dirichlet(torch.ones(n_anchors))
    if dist_type == "peaked":
        dist = Dirichlet(torch.Tensor([1.] * (n_anchors-1) + [n_anchors ** 2]))
    elif dist_type == "categorical":
        dist = Categorical(torch.ones(n_anchors))
    elif dist_type == "last_anchor":
        dist = Categorical(torch.Tensor([0] * (n_anchors-1) + [1]))
    return dist

class LinearSubspace(nn.Module):
    def __init__(self, n_anchors, in_channels, out_channels, bias = True, same_init = False, freeze_anchors = True):
        super().__init__()
        self.n_anchors = n_anchors
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.is_bias = bias
        self.freeze_anchors = freeze_anchors

        if same_init:
            anchor = nn.Linear(in_channels,out_channels,bias = self.is_bias)
            anchors = [copy.deepcopy(anchor) for _ in range(n_anchors)]
        else:
            anchors = [nn.Linear(in_channels,out_channels,bias = self.is_bias) for _ in range(n_anchors)]
        self.anchors = nn.ModuleList(anchors)

    def forward(self, x, alpha):
        #print("---anchor:",max(x.abs().max() for x in self.anchors.parameters()))
        #check = (not torch.is_grad_enabled()) and (alpha[0].max() == 1.)
        xs = [anchor(x) for anchor in self.anchors]
        #if check:
        #    copy_xs = xs
        #    argmax = alpha[0].argmax()
        xs = torch.stack(xs,dim=-1)

        alpha = torch.stack([alpha] * self.out_channels, dim=-2)
        xs = (xs * alpha).sum(-1)
        #if check:
        #    print("sanity check:",(copy_xs[argmax] - xs).sum().item())
        return xs

    def add_anchor(self,alpha = None):
        if self.freeze_anchors:
            for param in self.parameters():
                param.requires_grad = False

        # Midpoint by default
        if alpha is None:
            alpha = torch.ones((self.n_anchors,)) / self.n_anchors

        new_anchor = nn.Linear(self.in_channels,self.out_channels,bias=self.is_bias)
        new_weight = torch.stack([a * anchor.weight.data for a,anchor in zip(alpha,self.anchors)], dim = 0).sum(0)
        new_anchor.weight.data.copy_(new_weight)
        if self.is_bias:
            new_bias = torch.stack([a * anchor.bias.data for a,anchor in zip(alpha,self.anchors)], dim = 0).sum(0)
            new_anchor.bias.data.copy_(new_bias)
        self.anchors.append(new_anchor)
        self.n_anchors +=1

    def L2_norms(self):
        L2_norms = {}
        with torch.no_grad():
            for i in range(self.n_anchors):
                for j in range(i+1,self.n_anchors):
                    w1 = self.anchors[i].weight
                    w2 = self.anchors[j].weight
                    L2_norms["θ"+str(i+1)+"θ"+str(i+2)] = torch.norm(w1 - w2, p=2).item()
        return L2_norms

    def cosine_similarities(self):
        cosine_similarities = {}
        with torch.no_grad():
            for i in range(self.n_anchors):
                for j in range(i+1,self.n_anchors):
                    w1 = self.anchors[i].weight
                    w2 = self.anchors[j].weight
                    p = ((w1 * w2).sum() / max(((w1 ** 2).sum().sqrt() * (w2 ** 2).sum().sqrt()),1e-8)) ** 2
                    cosine_similarities["θ"+str(i+1)+"θ"+str(i+2)] = p.item()
        return cosine_similarities

class Sequential(nn.Sequential):
    def forward(self, input, alpha):
        for module in self:
            input = module(input,alpha) if isinstance(module,LinearSubspace) else module(input)
        return input