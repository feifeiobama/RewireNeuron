import torch.nn.functional as F

from core import CRLAgent
from .rewire_tools import *
import random


class RewireAction(CRLAgent):
    def __init__(self, input_dimension, output_dimension, hs, start_steps, tau, beta, k, tau2, beta2, rewire_last, cycle, divergence, coeff, input_name="env/env_obs"):
        super().__init__()
        self.start_steps = start_steps
        self.counter = 0
        self.iname = input_name
        self.task_id = 0  # useless
        self.input_size = input_dimension
        self.output_dimension = output_dimension
        self.hs = hs
        self.k = k
        self.k_ = 0
        self.divergence = divergence
        self.coeff = coeff
        self.regularize = False

        if rewire_last == 'expand':
            head = LinearExpand(self.hs, self.output_dimension * 2, cycle=cycle)
        elif rewire_last == True:
            head = LinearRewire(self.hs, self.output_dimension * 2, tau=tau, beta=beta, k=k, tau2=tau2, beta2=beta2, cycle=cycle, note=2)
        else:
            head = nn.Linear(self.hs, self.output_dimension * 2)

        self.model = SequentialRewire(
            LinearRewire(self.input_size, self.hs, tau=tau, beta=beta, k=k, tau2=tau2, beta2=beta2, cycle=cycle, note=1),
            nn.LeakyReLU(negative_slope=0.2),
            LinearRewire(self.hs, self.hs, tau=tau, beta=beta, k=k, tau2=tau2, beta2=beta2, cycle=cycle),
            nn.LeakyReLU(negative_slope=0.2),
            LinearRewire(self.hs, self.hs, tau=tau, beta=beta, k=k, tau2=tau2, beta2=beta2, cycle=cycle),
            nn.LeakyReLU(negative_slope=0.2),
            head,
        )

    def forward(self, t = None, **kwargs):
        if not self.training:
            x = self.get((self.iname, t))
            mu, _ = self.model(x, k=self.k_).chunk(2, dim=-1)
            action = torch.tanh(mu)
            self.set(("action", t), action)
        elif not (t is None):  # fake train, does not bp
            x = self.get((self.iname, t))
            if self.counter <= self.start_steps:
                action = torch.rand(x.shape[0], self.output_dimension).to(x.device) * 2 - 1
            else:
                mu, log_std = self.model(x, k=random.randint(0, self.k-1)).chunk(2, dim=-1)
                log_std = torch.clip(log_std, min=-20., max=2.)
                std = log_std.exp()
                action = mu + torch.randn(*mu.shape).to(mu.device) * std
                action = torch.tanh(action)
            self.set(("action", t), action)
            self.counter += 1
        else:
            input = self.get(self.iname)
            k = random.randint(0, self.k-1)
            if self.k > 1:
                with torch.no_grad():
                    mu0, log_std0 = self.model(input, k=(k + random.randint(1, self.k-1)) % self.k).chunk(2, dim=-1)
                    log_std0 = torch.clip(log_std0, min=-20., max=2.)
                    mu0 = torch.clip(mu0, min=-20, max=20)
            mu, log_std = self.model(input, is_train=True, k=k).chunk(2, dim=-1)
            log_std = torch.clip(log_std, min=-20., max=2.)
            std = log_std.exp()
            action = mu + torch.randn(*mu.shape).to(mu.device) * std
            log_prob = (-0.5 * (((action - mu) / (std + 1e-8)) ** 2 + 2 * log_std + np.log(2 * np.pi))).sum(-1, keepdim=True)
            log_prob -= (2 * np.log(2) - action - F.softplus( - 2 * action)).sum(-1, keepdim=True)
            action = torch.tanh(action)
            self.set("action", action)
            self.set("action_logprobs", log_prob)
            if self.k > 1:
                # self.divergence_loss = (log_std - log_std0 + (std0 ** 2 + (mu0 - mu) ** 2) / (2. * (std ** 2 + 1e-8)) - 0.5).sum()
                self.divergence_loss = ((mu - mu0) ** 2 + (log_std - log_std0) ** 2).sum()  # more stable in optimization
            else:
                self.divergence_loss = torch.Tensor([0.])

    def set_task(self, task_id=None):
        if task_id == -1:
            if hasattr(self, 'divergence_loss'):
                del self.divergence_loss
        self.model.set_task(task_id, self.k_)

    def register_and_consolidate(self):
        # linear
        for name, param in self.model.named_parameters():
            if ('linear' in name) or name.endswith('v'): continue
            name = name.replace('.', '_')
            self.model.register_buffer(f"{name}_mean", param.data.clone())
        # rewire
        perms = []
        for layer in self.model.children():
            if not isinstance(layer, LinearRewire): continue
            perms.append(layer.pre_register_and_consolidate())
        perms.append((None, None))
        i = 0
        for layer in self.model.children():
            if not isinstance(layer, LinearRewire): continue
            layer.register_and_consolidate(perms[i][1], perms[i+1][0])
            i += 1
        self.regularize = True

    def add_regularizer(self, *args):
        divergence_loss = (self.divergence * self.divergence_loss).view(1).to(list(self.parameters())[0].device)
        if self.regularize:
            losses = []
            # linear
            for name, param in self.model.named_parameters():
                if ('linear' in name) or name.endswith('v'): continue
                name = name.replace('.', '_')
                mean = getattr(self.model, f"{name}_mean")
                losses.append(((param - mean.detach())**2).sum())
            # rewire
            for layer in self.model.children():
                if not isinstance(layer, LinearRewire): continue
                losses += layer.add_regularizer()
            return divergence_loss + self.coeff*sum(losses).view(1).to(divergence_loss.device)
        else:
            return divergence_loss

    def roll_back(self):
        for name, param in self.model.named_parameters():
            if ('linear' in name) or name.endswith('v'): continue
            name = name.replace('.', '_')
            param.data = self.model.get_buffer(f"{name}_mean")
        for layer in self.model.children():
            if isinstance(layer, LinearRewire) or isinstance(layer, LinearExpand):
                layer.roll_back()
