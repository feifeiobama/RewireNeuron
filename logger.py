#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import time
from typing import *

import wandb
from omegaconf import DictConfig


class WandbLogger:
    """A wandb logger"""

    def __init__(
        self,
        project: Optional[str] = None,
        group: Optional[str] = None,
        job_type: Optional[str] = None,
        every_n_seconds: float = 0.,
        verbose: bool =False,
        log_loss: bool = False,
        **kwargs) -> None:
        wandb.init(project='salina_cl', group = project, job_type = job_type)
        self.logs = {}
        self.every_n_seconds = every_n_seconds
        self.save_time = - float("inf")
        self.verbose = verbose
        self.log_loss = log_loss

    def _to_dict(self, h: Union[dict,DictConfig]) -> dict:
        if isinstance(h, dict) or isinstance(h, DictConfig):
            return {k: self._to_dict(v) for k, v in h.items()}
        else:
            return h

    def save_hps(self, hps, verbose = True):
        print(hps)
        wandb.config.update(self._to_dict(hps))

    def get_logger(self, prefix):
        return self

    def message(self, msg, from_name=""):
        print("[", from_name, "]: ", msg)

    def add_images(self, name, value, iteration):
        pass

    def add_figure(self, name, value, iteration):
        pass

    def add_scalar(self, name, value, iteration):
        if ("loss" in name) and (not self.log_loss):
            pass
        else:
            if self.verbose:
                print("['" + name + "' at " + str(iteration) + "] = " + str(value))
            if "/" in name:
                iteration_name = "/".join(name.split("/")[:-1]+["iteration"])
            else:
                iteration_name = "iteration"
            self.logs[name] = value
            self.logs[iteration_name] = iteration
            t = time.time()
            if self.every_n_seconds == 0.:
                wandb.log(self.logs, commit = True)
                self.save_time = t
                self.logs = {}
            elif ((t - self.save_time) > self.every_n_seconds) or ("evaluation/iteration" in self.logs):
                wandb.log(self.logs, commit = True)
                self.save_time = t
                self.logs = {}

    def add_video(self, name, value, iteration, fps=10):
        pass

    def add_html(self,name,value):
        wandb.log({name: wandb.Html(value)})

    def close(self):
        pass