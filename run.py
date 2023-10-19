#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import time

import hydra
import torch

from salina import instantiate_class


@hydra.main(config_path="configs/", config_name="rewire.yaml")
def main(cfg):
    _start = time.time()
    logger = instantiate_class(cfg.logger)
    logger.save_hps(cfg, verbose =False)
    framework = instantiate_class(cfg.framework)
    scenario = instantiate_class(cfg.scenario)
    #logger_evaluation = logger.get_logger("evaluation/")
    #logger_evaluation.logger.modulo = 1
    stage = framework.get_stage()
    for train_task in scenario.train_tasks()[stage:]:
        framework.train(train_task,logger)
        evaluation = framework.evaluate(scenario.test_tasks(),logger)
        metrics = {}
        for tid in evaluation:
            for k,v in evaluation[tid].items():
                logger.add_scalar("evaluation/"+str(tid)+"_"+k,v,stage)
                metrics[k] = v + metrics.get(k,0)
        for k,v in metrics.items():
            logger.add_scalar("evaluation/aggregate_"+k,v / len(evaluation),stage)
        m_size = framework.memory_size()
        for k,v in m_size.items():
            logger.add_scalar("memory/"+k,v,stage)
        stage+=1
    logger.close()
    logger.message("time elapsed: "+str(round((time.time()-_start),0))+" sec")

if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn")
    CUDA_AVAILABLE = torch.cuda.is_available()
    if CUDA_AVAILABLE:
        v = torch.ones(1, device="cuda:0")
    main()