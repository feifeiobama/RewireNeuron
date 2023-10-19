import copy

import torch
import torch.nn as nn
import numpy as np


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)


def softsort(
    scores,
    tau=1.0,
    beta=0.0
):
    scores = scores + sample_gumbel(scores.shape).to(scores.device) * beta
    scores = scores.unsqueeze(-1)
    sorted = scores.sort(dim=-2)[0]
    pairwise_diff = (scores.transpose(-2, -1) - sorted).abs().neg()
    soft = torch.softmax(pairwise_diff / tau, dim=-1)
    hard = torch.zeros_like(soft).scatter_(-1, soft.argmax(dim=-1, keepdim=True), 1)
    return hard + soft - soft.detach()


def init_sort(v=None, n=None, k=None, scale=1.):
    if v is None:
        if k is None:
            k = 1
        v = torch.stack([torch.arange(n) for _ in range(k)]).float()
    else:
        v = v.argsort(dim=1).argsort(dim=1).float()
    return (v - v.min(dim=1)[0][:, np.newaxis]) / (v.max(dim=1)[0] - v.min(dim=1)[0])[:, np.newaxis] * scale


def mul_grad(x, k):
    return x * k - x.detach() * (k - 1)


class LinearRewire(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, tau=1.0, beta=1.0, k=3, tau2=1.0, beta2=1.0, cycle=-1, note=3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.is_bias = bias
        self.linear = nn.Linear(in_channels, out_channels, bias=self.is_bias)

        self.tau = tau
        self.beta = beta
        self.k = k
        self.tau2 = tau2
        self.beta2 = beta2
        self.cycle = cycle
        self.note = note  # 1 post, 2 pre, 3 both
        if self.note != 2:
            self.v = nn.Parameter(init_sort(n=out_channels, k=k))
        if self.beta == 0:
            self.dropout = nn.Dropout(0.1)  # alternative way to introduce stochasticity

        self.cnt = 0
        self.t = None

    def forward(self, x, is_train=False, k=0):
        out = self.linear(x)
        if self.beta == 0:
            out = self.dropout(out)
        if self.note == 2:
            return out
        elif self.t is None:
            if is_train:
                p = softsort(mul_grad(self.v[k], self.k), tau=self.tau, beta=self.beta)
                return out @ p.T
            else:
                return out[..., self.v[k].argsort()]
        else:
            if self.cycle == -1:
                t = min(self.t, self.cnt - 1)
            else:
                t = min(self.t % self.cycle, self.cnt - 1)
            return out[..., self.get_buffer(f"v{t}v")]

    def set_task(self, task_id=None, k=0):
        if self.note == 2:
            if task_id == -1:
                self.cnt += 1
        elif task_id == -1:
            cnt = self.cnt
            if self.cycle != -1:
                cnt = cnt % self.cycle
            self.register_buffer(f"v{cnt}v", self.v[k].argsort())
            self.v.data = torch.stack([self.v[k].data.clone() for _ in range(self.k)])
            self.cnt += 1
        else:
            self.t = task_id
            if task_id is None:
                if (self.cycle != -1) and (self.cnt >= self.cycle):
                    last_v = self.get_buffer(f"v{self.cnt % self.cycle}v").argsort()
                    self.v.data = torch.stack([last_v.clone() for _ in range(self.k)])
                self.v.data = init_sort(v=self.v.data)

    def pre_register_and_consolidate(self):
        av, zv = None, None
        if self.cnt > 0:
            if self.note // 2 == 1:
                av = self.av.argsort().argsort()
            if self.note % 2 == 1:
                zv = self.zv.argsort()
        return av, zv

    def register_and_consolidate(self, zv, next_av):
        # update self.vnv
        if (self.cnt > 0) and (self.note != 2):
            cnt = self.cnt
            if self.cycle != -1:
                cnt = min(cnt, self.cycle)
            for t in range(cnt):
                if next_av is not None:
                    self.register_buffer(f"v{t}v", zv[self.get_buffer(f"v{t}v")][next_av])
                else:
                    self.register_buffer(f"v{t}v", zv[self.get_buffer(f"v{t}v")])
        # register mean
        for name, param in self.named_parameters():
            if name.endswith('v'): continue
            name = name.replace('.', '_')
            self.register_buffer(f"{name}_mean", param.data.clone())
        # initialize self.vv
        if self.note // 2 == 1:
            self.register_parameter('av', nn.Parameter(init_sort(n=self.in_channels)[0]))
        if self.note % 2 == 1:
            self.register_parameter('zv', nn.Parameter(init_sort(n=self.out_channels)[0]))

    def add_regularizer(self):
        losses = []
        if self.note // 2 == 1:
            ap = softsort(self.av, tau=self.tau2, beta=self.beta2)
        if self.note % 2 == 1:
            zp = softsort(self.zv, tau=self.tau2, beta=self.beta2)
        for name, param in self.named_parameters():
            if name.endswith('v'): continue
            name = name.replace('.', '_')
            mean = self.get_buffer(f"{name}_mean")
            if (self.note == 1) or ((self.note == 3) and name.endswith('bias')):
                losses.append(- 2 * ((zp.T @ mean) * param).sum() + (param ** 2).sum())
            elif (self.note == 2) and name.endswith('weight'):
                losses.append(- 2 * ((mean @ ap) * param).sum() + (param ** 2).sum())
            elif (self.note == 3) and name.endswith('weight'):
                losses.append(- 2 * ((zp.T @ mean @ ap) * param).sum() + (param ** 2).sum())
            else:  # self.note == 2 and name.endswith('bias')
                losses.append(((mean - param) ** 2).sum())
        return losses

    def roll_back(self):
        for name, param in self.named_parameters():
            if name.endswith('v'): continue
            name = name.replace('.', '_')
            param.data = self.get_buffer(f"{name}_mean")
        if (self.cycle != -1) and (self.cnt >= self.cycle):
            last_cnt = self.cnt % self.cycle
        else:
            last_cnt = self.cnt - 1
        last_v = self.get_buffer(f"v{last_cnt}v").argsort()
        self.v.data = torch.stack([last_v.clone() for _ in range(self.k)])
        self.v.data = init_sort(v=self.v.data)
        if self.note // 2 == 1:
            self.register_parameter('av', nn.Parameter(init_sort(n=self.in_channels)[0]))
        if self.note % 2 == 1:
            self.register_parameter('zv', nn.Parameter(init_sort(n=self.out_channels)[0]))


class LinearExpand(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, cycle=-1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.is_bias = bias
        self.cycle = cycle
        self.linears = nn.ModuleList([])

        self.cnt = 0
        self.t = None

    def forward(self, x):
        if self.t is None:
            t = self.cnt
            if self.cycle != -1:
                t = t % self.cycle
        else:
            if self.cycle == -1:
                t = min(self.t, self.cnt - 1)
            else:
                t = min(self.t % self.cycle, self.cnt - 1)
        return self.linears[t](x)

    def set_task(self, task_id=None):
        if task_id == -1:
            self.cnt += 1
        else:
            self.t = task_id
            if task_id is None:
                if self.cnt == 0:
                    self.linears.append(nn.Linear(self.in_channels, self.out_channels, bias=self.is_bias).cuda())
                else:
                    if (self.cycle == -1) or (self.cnt < self.cycle):
                        self.linears.append(copy.deepcopy(self.linears[-1]))
                        cnt = self.cnt
                        last_cnt = self.cnt - 1
                    else:
                        cnt = self.cnt % self.cycle
                        last_cnt = (self.cnt - 1) % self.cycle
                    for name, param in self.linears[cnt].named_parameters():
                        param.requires_grad = True
                        name = name.replace('.', '_')
                        self.register_buffer(f"{name}_mean", param.data.clone())
                    for param in self.linears[last_cnt].parameters():
                        param.requires_grad = False

    def roll_back(self):
        if (self.cycle == -1) or (self.cnt < self.cycle):
            cnt = self.cnt
        else:
            cnt = self.cnt % self.cycle
        for name, param in self.linears[cnt].named_parameters():
            name = name.replace('.', '_')
            param.data = self.get_buffer(f"{name}_mean")


class SequentialRewire(nn.Sequential):
    def forward(self, input, is_train=False, k=0):
        for module in self:
            input = module(input, is_train, k) if isinstance(module, LinearRewire) else module(input)
        return input

    def set_task(self, task_id=None, k=0):
        for module in self:
            if isinstance(module, LinearRewire):
                module.set_task(task_id, k)
            elif isinstance(module, LinearExpand):
                module.set_task(task_id)
