# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Set up custom environment before nearly anything else is imported
from skimage.transform import resize
# NOTE: this should be the first import (no not reorder)
from utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import torch
import torch.nn as nn
import os

from config import cfg
from data import make_train_loader
from solver import make_lr_scheduler
from solver import make_optimizer
#from engine.inference import inference
from engine.trainer import do_train, do_co_train
from modeling.detector import build_detection_model
from utils.checkpoint import DetectronCheckpointer
from utils.collect_env import collect_env_info
from utils.comm import synchronize, get_rank
from utils.imports import import_file
from utils.logger import setup_logger
from utils.miscellaneous import mkdir, save_config

import pdb

# See if we can use apex.DistributedDataParallel instead of the torch default,
# and enable mixed-precision via apex.amp
# try:
#     from apex import amp
# except ImportError:
#     raise ImportError('Use APEX for multi-precision via apex.amp')


def train(cfg, logger, local_rank, distributed):
    
    model = build_detection_model(cfg)
    if len(cfg.MODEL.DEVICE) > 1:
        model = nn.DataParallel(model, device_ids=cfg.MODEL.DEVICE)
    device = torch.device("cuda", cfg.MODEL.DEVICE[0])
    model.to(device)

    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)

    # Initialize mixed-precision training
    # use_mixed_precision = cfg.DTYPE == "float16"
    # amp_opt_level = 'O1' if use_mixed_precision else 'O0'
    # model, optimizer = amp.initialize(model, optimizer, opt_level=amp_opt_level)

    # if distributed:
    #     model = torch.nn.parallel.DistributedDataParallel(
    #         model, device_ids=[local_rank], output_device=local_rank,
    #         # this should be removed if we update BatchNorm stats
    #         broadcast_buffers=False,
    #     )

    arguments = {}

    output_dir = cfg.OUTPUT_DIR

    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk
    )
    extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT)
    arguments.update(extra_checkpoint_data)

    arguments["iteration"] = 0
    train_loader, valid_loader = make_train_loader(
            cfg,
            is_distributed=distributed,
            start_iter=arguments["iteration"])
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    test_period = cfg.SOLVER.ITER_PER_EPOCH
    
    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)


    do_train(
            cfg,
            model,
            train_loader,
            valid_loader,
            optimizer,
            scheduler,
            checkpointer,
            device,
            checkpoint_period,
            test_period,
            arguments,
            logger
        )
    return model


def main(args):
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("DeepTomo_Segmentation", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    output_config_path = os.path.join(cfg.OUTPUT_DIR, 'config.yml')
    logger.info("Saving config into: {}".format(output_config_path))
    
    # save overloaded model config in the output directory
    save_config(cfg, output_config_path)

    model = train(cfg, logger, args.local_rank, args.distributed)

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepTomo Detection Module Training")
    
    parser.add_argument("--config-file", default="./config/config_finetune.yaml", metavar="FILE", help="path to config file", type=str)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--skip-test", dest="skip_test", help="Do not test the final model", action="store_true")
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()
    main(args)
