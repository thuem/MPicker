# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Set up custom environment before nearly anything else is imported
from skimage.transform import resize
# NOTE: this should be the first import (no not reorder)
from utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import torch
import torch.nn as nn
import os
import pdb

from config import cfg
from data import make_test_loader
from engine.inference import inference
from modeling.detector import build_detection_model
from utils.checkpoint import DetectronCheckpointer
from utils.collect_env import collect_env_info
from utils.comm import synchronize, get_rank
from utils.logger import setup_logger
from utils.miscellaneous import mkdir

# Check if we can enable mixed-precision via apex.amp
# try:
#     from apex import amp
# except ImportError:
#     raise ImportError('Use APEX for mixed precision via apex.amp')


def main(args):
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    logger = setup_logger("DeepTomo_Segmentation", None, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)
    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    model = build_detection_model(cfg)
    if len(cfg.MODEL.DEVICE) > 1:
        model = nn.DataParallel(model, device_ids=cfg.MODEL.DEVICE)
    device = torch.device("cuda", cfg.MODEL.DEVICE[0])
    model.to(device)

    
    checkpointer = DetectronCheckpointer(cfg, model)
    ckpt = cfg.MODEL.WEIGHT if args.ckpt is None else args.ckpt
    _ = checkpointer.load(ckpt, use_latest=args.ckpt is None)

    output_folder = os.path.abspath(cfg.OUTPUT_TEST_DIR)
    os.makedirs(output_folder, exist_ok=True)
    data_loaders_test = make_test_loader(cfg, is_distributed=distributed)
    
    for data_loader_test in data_loaders_test:
        # img_name = data_loader_test.dataset.img_path.split("/")[-1].split(".mrc")[0] + "_seg.mrc"
        img_name = os.path.basename(data_loader_test.dataset.img_path).split("_img.npy")[0].split(".mrc")[0] + "_seg.mrc"
        output_folder = os.path.join(cfg.OUTPUT_TEST_DIR, img_name)
        inference(
            cfg,
            model,
            data_loader_test,
            device=device,
            output_folder=output_folder,
            logger=logger
        )
        synchronize()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepTomo Segmentation Module Inferencing")
    parser.add_argument("--config-file", default="./config/config_predict.yaml", metavar="FILE", help="path to config file", type=str)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--ckpt",
        help="The path to the checkpoint for test, default is the latest checkpoint.",
        default=None,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    main(args)
