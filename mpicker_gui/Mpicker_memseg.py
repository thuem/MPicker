#!/usr/bin/env python3

# Copyright (C) 2024  Xiaofeng Yan, Shudong Li
# Xueming Li Lab, Tsinghua University

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import argparse
from pathlib import Path
import sys, os
from shutil import rmtree
from skimage.transform import resize


memseg_path = "../memseg_v3"
memseg_path = Path(__file__).parent/memseg_path
sys.path.append(str(memseg_path.resolve()))
delete_tmp = True
small_batch = False # just for test on laptop


def absolute_path(path):
    # path = Path(path) # str or Path
    return os.path.abspath(str(path))


def make_tmpdir(path):
    path = Path(path) # parent path
    if not path.is_dir():
        path.mkdir(parents=True, exist_ok=True)
        print("makedirs:", str(path))
    i = 0
    while True:
        name = "tmp_memseg_" + str(i)
        tmp = path/name
        if not tmp.exists():
            break
        i += 1
    tmp.mkdir()
    return absolute_path(tmp)


def remove_tmpdir(path):
    path = Path(path)
    if path.is_dir():
        path = absolute_path(path)
        try:
            rmtree(path)
        except Exception as e:
            print(e)
            print("failed to remove", path)


def main_seg(input, output, model, gpuid="0", batch=2, thread=4):
    from yacs.config import CfgNode as CN
    from test_net import main as seg

    if output is None or output=='None':
        output = input+".segraw.mrc"
        print("set output_name:", output)

    # output is a file
    if not Path(output).parent.is_dir():
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        print("makedirs:", str(Path(output).parent))

    try:
        tmp_path = make_tmpdir(Path(output).parent)
    except:
        tmp_path = make_tmpdir(Path(input).parent)

    if model is None or model=='None':
        model = memseg_path/"pretrained_model/model.pth"

    gpuid = gpuid.strip().split(",")
    gpuid = [int(i) for i in gpuid]

    _C = CN()
    _C.DATASETS = CN()
    _C.DATASETS.TEST = absolute_path(input)
    _C.DATASETS.TEST_SAVE = absolute_path(Path(tmp_path)/"test")
    _C.MODEL = CN()
    _C.MODEL.WEIGHT = absolute_path(model)
    _C.MODEL.DEVICE = gpuid
    _C.OUTPUT_TEST_DIR = absolute_path(tmp_path)
    _C.TEST = CN()
    _C.DATALOADER = CN()
    _C.TEST.IMS_PER_BATCH = int(batch)
    _C.DATALOADER.NUM_WORKERS = int(thread)
    config_path = Path(tmp_path)/"config_predict.yaml"
    config_path = absolute_path(config_path)
    with open(config_path, 'w') as f:
        f.write(_C.dump())

    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", default=config_path, metavar="FILE", type=str)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--ckpt", default=None)
    args = parser.parse_args([])

    seg(args)
    result_name = os.path.basename(absolute_path(input)).split(".mrc")[0] + "_seg.mrc" # from test_net.py
    result_name = os.path.join(absolute_path(tmp_path), result_name)
    os.rename(result_name, output)

    if delete_tmp:
        remove_tmpdir(tmp_path)
    return


def main_post(input, output, thres, gauss, voxel_cut):
    from utils.process import seg_post_process as postprocess

    # output is a file
    if not Path(output).parent.is_dir():
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        print("makedirs:",str(Path(output).parent))

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--threshold', type=float, default=thres)
    parser.add_argument('-l', '--lowpass', type=float, default=gauss)
    parser.add_argument('-v', '--voxel_cutoff', type=float, default=voxel_cut)
    parser.add_argument('-i', '--input', type=str, default=absolute_path(input))
    parser.add_argument('-o', '--output', type=str,  default=absolute_path(output))
    args = parser.parse_args([])

    postprocess(args)
    return


def main_finetune_pre(data, mask, output, z_range="-1,-1", crop_size="300,300", stride="200,200", overwrite=True):
    from utils.finetune import make_finetune_dataset

    if output is None or output=='None':
        output = make_tmpdir(Path(mask).parent)
        print("output in:", output)
    output = Path(output)

    # output is a dir
    if not output.is_dir():
        output.mkdir(parents=True, exist_ok=True)
        print("makedirs:", str(output))

    dataset_dir = absolute_path(output/"finetune_dataset")
    if overwrite:
        print(f"clear {dataset_dir} at first")
        remove_tmpdir(dataset_dir)
    
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--img_dir", type=str, default=absolute_path(data))
    parser.add_argument("--seg_dir", type=str, default=absolute_path(mask))
    parser.add_argument("--output_dir", type=str, default=dataset_dir)
    parser.add_argument("--suffix", type=str, default="recon")
    parser.add_argument("--crop_size", type=str, default=crop_size)
    parser.add_argument("--stride", type=str, default=stride)
    parser.add_argument("--z_range", type=str, default=z_range)
    parser.add_argument("--coord_dir", type=str, default=None)
    parser.add_argument("--box_size", type=int)
    parser.add_argument("--seg_only", type=bool, default=True)
    args = parser.parse_args([])

    make_finetune_dataset(args)
    return dataset_dir


def main_finetune_train(data, dataset, model=None, iters=40, gpuid="0", thread=4):
    from yacs.config import CfgNode as CN
    from train_net import main as finetune

    output = Path(dataset).parent
    dataset_dir = absolute_path(dataset) # output/"finetune_dataset"
    model_dir = absolute_path(output/"finetune_model")
    config_path = absolute_path(output/"config_finetune.yaml")
    tmp_dir1 = make_tmpdir(output)
    # tmp_dir2 = make_tmpdir(output)

    if model is None or model=='None':
        model = memseg_path/"pretrained_model/model.pth"

    gpuid = gpuid.strip().split(",")
    gpuid = [int(i) for i in gpuid]

    _C = CN()
    _C.DATASETS = CN()
    _C.DATASETS.TRAIN = dataset_dir
    _C.DATASETS.TRAIN_SAVE = tmp_dir1
    _C.DATASETS.TRAIN_SUFFIX = "recon."
    _C.SOLVER = CN()
    _C.SOLVER.ITER_PER_EPOCH = iters
    _C.SOLVER.MAX_EPOCH = 4
    _C.SOLVER.IMS_PER_BATCH = 6
    _C.MODEL = CN()
    _C.MODEL.WEIGHT = absolute_path(model)
    _C.MODEL.DEVICE = gpuid
    _C.OUTPUT_DIR = model_dir
    _C.DATASETS.IS_VALID = False
    # _C.DATASETS.VALID = absolute_path(data)
    # _C.DATASETS.VALID_SAVE = tmp_dir2
    _C.DATALOADER = CN()
    _C.DATALOADER.NUM_WORKERS = int(thread)
    if small_batch:
        _C.SOLVER.MAX_EPOCH = 2
        _C.SOLVER.IMS_PER_BATCH = 1

    with open(config_path, 'w') as f:
        f.write(_C.dump())

    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", default=config_path, metavar="FILE", type=str)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--skip-test", dest="skip_test", action="store_true")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args([])

    finetune(args)
    if delete_tmp:
        remove_tmpdir(tmp_dir1)
        # remove_tmpdir(tmp_dir2)
    return


def main_finetune(data, mask, output, z_range="-1,-1", model=None, iters=40, gpuid=0, thread=4, crop_size="300,300", stride="200,200"):
    dataset_dir = main_finetune_pre(data, mask, output, z_range, crop_size, stride)
    main_finetune_train(data, dataset_dir, model, iters, gpuid, thread)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="use memseg_v3",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mode', type=str, default='seg',
                        choices=['seg', 'post', 'finetune', 'finetune_pre', 'finetune_train'],
                        help='task you want to do')
    parser.add_argument('--input_mrc', type=str, required=True,
                        help='the name of mrc')
    parser.add_argument('--output_mrc', type=str, 
                        help='output name')
    parser.add_argument('--thres', type=float, default=0.5,
                        help='threshold to use when postprocess')
    parser.add_argument('--gauss', type=float, default=0.5,
                        help='gaussian filter when postprocess')
    parser.add_argument('--voxel_cut', type=int, default=0,
                        help='pixel cutoff number of 2d connect component when postprocess, cutoff will *100 in 3d')
    parser.add_argument('--input_mask', type=str, 
                        help='postprocessed 01 mask used in finetune')
    parser.add_argument('--output_finetune', type=str,
                        help='path to save finetune result')
    parser.add_argument("--z_range", type=str, default="-1,-1", 
                        help="Z range for cropping generated tomogram,iIn format 'int, int', default all, thicker than 50")
    parser.add_argument('--not_overwrite', action='store_true',
                        help='not delete dataset dir if existed when finetune_pre')
    parser.add_argument('--dataset', type=str,
                        help='input datset dir when finetune_train. results will be saved in dataset/../finetune_model')
    parser.add_argument('--input_model', type=str,
                        help='model file (.pth) used to finetune from, or to seg after finetune')
    parser.add_argument('--iters', type=int, default=40,
                        help='train iters per epoch when finetune. 4 epoch in total.')
    parser.add_argument('--gpuid', type=str, default="0",
                        help='gpuids, such as "0,1,2"')
    parser.add_argument('--batch', type=int, default=2,
                        help='batch size when predict, gpu_number*2 is a good test')
    parser.add_argument('--thread', type=int, default=2,
                        help='NUM_WORKERS for DATALOADER, same as batch size is enough')
    args = parser.parse_args()

    if args.mode == 'seg':
        main_seg(args.input_mrc, args.output_mrc, args.input_model, args.gpuid, args.batch, args.thread)
    elif args.mode == 'post':
        main_post(args.input_mrc, args.output_mrc, args.thres, args.gauss, args.voxel_cut)
    elif args.mode == 'finetune':
        # overwrite dataset by default
        main_finetune(args.input_mrc, args.input_mask, args.output_finetune, args.z_range, 
                      args.input_model, args.iters, args.gpuid, args.thread)
    elif args.mode == 'finetune_pre':
        overwrite = not args.not_overwrite
        dataset_dir = main_finetune_pre(args.input_mrc, args.input_mask, args.output_finetune, args.z_range, overwrite=overwrite)
        print("datset dir:", dataset_dir)
    elif args.mode == 'finetune_train':
        main_finetune_train(args.input_mrc, args.dataset, args.input_model, args.iters, args.gpuid, args.thread)