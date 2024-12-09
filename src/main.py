
"""
Detectron2 training script with a plain training loop.

This script reads a given config file and runs the training or evaluation.
It is an entry point that is able to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as a library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.

Compared to "train_net.py", this script supports fewer default features.
It also includes fewer abstraction, therefore is easier to add custom logic.
"""

import logging
import os
import time
import datetime
import torch
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import pandas as pd

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.engine import default_argument_parser, default_setup, default_writers, launch, hooks
from detectron2.evaluation import inference_on_dataset, print_csv_format
from detectron2.utils.events import EventStorage
from detectron2.config import LazyConfig, instantiate
from detectron2.engine.defaults import create_ddp_model
from detectron2.engine.hooks import HookBase
from detectron2.solver import LRMultiplier
from fvcore.common.timer import Timer
import albumentations as albu
from sklearn.metrics import roc_auc_score
import numpy as np
import detectron2.data.transforms as T
from detectron2_train.ema_trainer import EMAAMPTrainer, EMASimpleTrainer, MeanTeacherDetectModelTrainer, MeanTeacherDetectModelAMPTrainer
from util import setup_logger

# test the wsidataset
# modified by nowandfuture
from dataset import WSIDataset, build_wsi_test_dataloader, build_wsi_train_dataloader


logger = logging.getLogger("detectron2")

class IterationTimer(HookBase):
    """
    Track the time spent for each iteration (each run_step call in the trainer).
    Print a summary in the end of training.
    This hook uses the time between the call to its :meth:`before_step`
    and :meth:`after_step` methods.
    Under the convention that :meth:`before_step` of all hooks should only
    take negligible amount of time, the :class:`IterationTimer` hook should be
    placed at the beginning of the list of hooks to obtain accurate timing.
    """

    def __init__(self, storage, start_iter, warmup_iter=3):
        """
        Args:
            warmup_iter (int): the number of iterations at the beginning to exclude
                from timing.
        """
        self._warmup_iter = warmup_iter
        self._step_timer = Timer()
        self._start_time = time.perf_counter()
        self._total_timer = Timer()

        self.storage = storage
        self.start_iter = start_iter

    def before_train(self):
        self._start_time = time.perf_counter()
        self._total_timer.reset()
        self._total_timer.pause()

    def after_train(self):
        total_time = time.perf_counter() - self._start_time
        total_time_minus_hooks = self._total_timer.seconds()
        hook_time = total_time - total_time_minus_hooks

        num_iter = self.storage.iter + 1 - self.start_iter - self._warmup_iter

        if num_iter > 0 and total_time_minus_hooks > 0:
            # Speed is meaningful only after warmup
            # NOTE this format is parsed by grep in some scripts
            logger.info(
                "Overall training speed: {} iterations in {} ({:.4f} s / it)".format(
                    num_iter,
                    str(datetime.timedelta(seconds=int(total_time_minus_hooks))),
                    total_time_minus_hooks / num_iter,
                )
            )

        logger.info(
            "Total training time: {} ({} on hooks)".format(
                str(datetime.timedelta(seconds=int(total_time))),
                str(datetime.timedelta(seconds=int(hook_time))),
            )
        )

    def before_step(self):
        self._step_timer.reset()
        self._total_timer.resume()

    def after_step(self):
        # +1 because we're in after_step, the current step is done
        # but not yet counted
        iter_done = self.storage.iter - self.start_iter + 1
        if iter_done >= self._warmup_iter:
            sec = self._step_timer.seconds()
            self.storage.put_scalars(time=sec)
        else:
            self._start_time = time.perf_counter()
            self._total_timer.reset()

        self._total_timer.pause()

def do_test(cfg, model):
    if "evaluator" in cfg.dataloader:
        cfg.dataloader.evaluator.output_dir = os.path.join(cfg.train.output_dir, "inference")
        ret = inference_on_dataset(
            model, instantiate(cfg.dataloader.test), instantiate(cfg.dataloader.evaluator)
        )
        print_csv_format(ret)
        return ret
    
def do_train(args, cfg):
    """
    Args:
        cfg: an object with the following attributes:
            model: instantiate to a module
            dataloader.{train,test}: instantiate to dataloaders
            dataloader.evaluator: instantiate to evaluator for test set
            optimizer: instantaite to an optimizer
            lr_multiplier: instantiate to a fvcore scheduler
            train: other misc config defined in `common_train.py`, including:
                output_dir (str)
                init_checkpoint (str)
                amp.enabled (bool)
                max_iter (int)
                eval_period, log_period (int)
                device (str)
                checkpointer (dict)
                ddp (dict)
    """
    model = instantiate(cfg.model)
    logger = logging.getLogger("detectron2")
    logger.info("Model:\n{}".format(model))
    model.to(cfg.train.device)

    cfg.optimizer.params.model = model
    optim = instantiate(cfg.optimizer)

    train_loader = instantiate(cfg.dataloader.train)
    unlabeled_data_loader = instantiate(cfg.dataloader.train_unlabeled)

    model = create_ddp_model(model, **cfg.train.ddp)
    trainer = (MeanTeacherDetectModelAMPTrainer if cfg.train.amp.enabled else MeanTeacherDetectModelTrainer)(model, train_loader, unlabeled_data_loader, optim)
    
    checkpointer_swa = DetectionCheckpointer(
        model,
        cfg.train.output_dir,
        trainer=trainer,
    )
    
    checkpointer = DetectionCheckpointer(
        model,
        cfg.train.output_dir,
        trainer=trainer,
    )
    
    checkpointer_swa.resume_or_load(cfg.train.init_checkpoint, resume=args.resume)
    trainer.init_swa_model()
    
    # update the model with swa model
    swa_model = trainer.swa_model.module 
    checkpointer_swa.model = swa_model

    if args.resume and checkpointer_swa.has_checkpoint():
        # The checkpoint stores the training iteration that just finished, thus we start
        # at the next iteration
        start_iter = trainer.iter + 1
    else:
        start_iter = 0
        
    # checkpointer = DetectionCheckpointer(
    #     model,
    #     cfg.train.output_dir,
    #     trainer=trainer,
    # )
    trainer.register_hooks(
        [
            hooks.IterationTimer(),
            hooks.LRScheduler(scheduler=instantiate(cfg.lr_multiplier)),
            hooks.PeriodicCheckpointer(checkpointer, **cfg.train.checkpointer)
            if comm.is_main_process()
            else None,
            # hooks.PeriodicCheckpointer(checkpointer_swa, **cfg.train.checkpointer, file_prefix="swa_model")
            # if comm.is_main_process()
            # else None,
            hooks.EvalHook(cfg.train.eval_period, lambda: do_test(cfg, model)),
            hooks.PeriodicWriter(
                default_writers(cfg.train.output_dir, cfg.train.max_iter),
                period=cfg.train.log_period,
            )
            if comm.is_main_process()
            else None,
        ]
    )

    trainer.train(start_iter, cfg.train.max_iter)

def main(args):
    print(args.config_file)
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    default_setup(cfg, args)
    setup_logger(cfg, args)

    if args.eval_only:
        model = instantiate(cfg.model)
        model.to(cfg.train.device)
        model = create_ddp_model(model)
        DetectionCheckpointer(model, cfg.train.output_dir).resume_or_load(cfg.train.init_checkpoint,
            resume=args.resume)
        logger.info(do_test(cfg, args, model))
    else:
        do_train(args, cfg)


if __name__ == "__main__":
    args_parser = default_argument_parser()
    args_parser.add_argument("--train-file", default="dataset/train.txt", type=str)
    args_parser.add_argument("--test-file", default="dataset/val.txt", type=str)
    # args_parser.add_argument("--config-file", default="/nasdata/private/mbzuai/Retrie_Retina/src/config.py", type=str)
    args = args_parser.parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
