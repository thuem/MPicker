# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm

from data import make_train_loader
from utils.comm import get_world_size, synchronize
from utils.metric_logger import MetricLogger
# from torch.utils.tensorboard import SummaryWriter

# from utils.plotting import plot_valid_box
import numpy as np
import mrcfile


#from engine.inference import inference
#from apex import amp

import pdb

def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def do_train(
    cfg,
    model,
    data_loader,
    data_loader_val,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    test_period,
    arguments,
    logger
):

    # writer = SummaryWriter('./output/tensorboard')
    #logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    model.train()
    start_training_time = time.time()
    end = time.time()

    output_val_dir = cfg.OUTPUT_DIR + '/visualization/'
    if not os.path.exists(output_val_dir):
        os.makedirs(output_val_dir)

    if data_loader_val is not None:
        valid_iterator = iter(data_loader_val)

    torch.autograd.set_detect_anomaly(True)
    time_dict = {}

    start_time = time.time()
    for iteration, (img, seg, box_coord, box_class) in enumerate(data_loader, start_iter):
        iteration = iteration + 1
        arguments["iteration"] = iteration
                
        img = torch.Tensor(img)
        img = img.unsqueeze(dim=1)
        img = img.to(device)

        seg = torch.Tensor(seg)
        seg = seg.to(device)

        data_time = time.time()
        # print("Time for reading batch data: {}s".format(data_time - start_time))
        loss = model(img, seg, box_coord, box_class)
        loss = loss.mean()
        
        time_dict["Load_data"] = data_time - start_time
        # print("Time for forwarding data: {}s".format(forward_time - data_time))
        start_time = time.time()
        # reduce losses over all GPUs for logging purposes
        meters.update(loss=loss)
        optimizer.zero_grad()
        loss.backward()
        # Note: If mixed precision is not used, this ends up doing nothing
        # Otherwise apply loss scaling for mixed-precision recipe
        # with amp.scale_loss(losses, optimizer) as scaled_losses:
        #     scaled_losses.backward()
        optimizer.step()
        scheduler.step()

        time_dict["Backward_loss"] = time.time() - start_time
        #print("Time for backwarding data: {}s".format(backward_time - forward_time))

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        logger.info("Iter {}, Loss: {:.3f}, Batch_time: {:.3f}".format(
                        iteration, loss.item(),  end - start_time))
        start_time = time.time()

        if iteration % 20 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
        
        if iteration % checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
        
        if data_loader_val is not None and test_period > 0 and iteration % test_period == 0:
            meters_val = MetricLogger(delimiter="  ")
            synchronize()
            model.eval()
            with torch.no_grad():
                # Should be one image for each GPU:
                img_val, seg_val, box_coord_val, box_class_val = valid_iterator.next()
                
                img_val = torch.Tensor(img_val)
                img_val = img_val.unsqueeze(dim=1)
                img_val = img_val.to(device)

                seg_val = torch.Tensor(seg_val)
                seg_val = seg_val.to(device)

                seg_predict = model(img_val, seg_val)
            predict = np.transpose(np.array(seg_predict[0, 1].cpu()), [2, 1, 0])
            with mrcfile.new('{}/seg_vis_iter{}.mrc'.format(output_val_dir, iteration), overwrite=True) as tomo:
                tomo.set_data(np.float32(predict))
                
                        
            synchronize()
            logger.info(
                meters_val.delimiter.join(
                    [
                        "[Validation]: ",
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters_val),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
            model.train()
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )


def do_co_train(
    cfg,
    model,
    data_loader,
    data_loader_pse,
    data_loader_val,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    test_period,
    arguments,
    logger
):

    # writer = SummaryWriter('./output/tensorboard')
    #logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    model.train()
    start_training_time = time.time()
    end = time.time()

    output_val_dir = cfg.OUTPUT_DIR + '/visualization/'
    if not os.path.exists(output_val_dir):
        os.makedirs(output_val_dir)
    valid_iterator = iter(data_loader_val)
    finetune_iterator = iter(data_loader_pse)

    start_time = time.time()
    for iteration, (img_s, seg_s, _, _) in enumerate(data_loader, start_iter):
        iteration = iteration + 1
        arguments["iteration"] = iteration

        img_s = torch.Tensor(img_s)
        img_s = img_s.unsqueeze(dim=1)
        img_s = img_s.to(device)

        seg_s = torch.Tensor(seg_s)
        seg_s = seg_s.to(device)

        img_t, seg_t, _, _ = next(finetune_iterator)

        if not seg_s.shape[0] == seg_t.shape[0]:
            keep = min(seg_s.shape[0], seg_t.shape[0])
            img_s = img_s[:keep]
            seg_s = seg_s[:keep]
            img_t = img_t[:keep]
            seg_t = seg_t[:keep]

        img_t = torch.Tensor(img_t)
        img_t = img_t.unsqueeze(dim=1)
        img_t = img_t.to(device)

        seg_t = torch.Tensor(seg_t)
        seg_t = seg_t.to(device)

        img_dacs = img_t * (1 - seg_s.unsqueeze(dim=1)) + img_s * seg_s.unsqueeze(dim=1)
        seg_dacs = seg_t * (1 - seg_s) + seg_s

        data_time = time.time()
        #print("Time for reading batch data: {}s".format(data_time - start_time))
        loss_dict_s, _ = model(img_s, seg_s, None, None)
        loss_dict_dacs, _ = model(img_dacs, seg_dacs, None, None)
        forward_time = time.time()
        
        #print("Time for forwarding data: {}s".format(forward_time - data_time))

        losses_s = sum(loss for loss in loss_dict_s.values())
        losses_dacs = sum(loss for loss in loss_dict_dacs.values())

        losses = losses_s + losses_dacs

        loss_dict = {}
        loss_dict['loss_seg_source'] = losses_s
        loss_dict['loss_seg_dacs'] = losses_dacs

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        optimizer.zero_grad()
        losses.backward()
        # Note: If mixed precision is not used, this ends up doing nothing
        # Otherwise apply loss scaling for mixed-precision recipe
        # with amp.scale_loss(losses, optimizer) as scaled_losses:
        #     scaled_losses.backward()
        optimizer.step()
        scheduler.step()

        backward_time = time.time()
        #print("Time for backwarding data: {}s".format(backward_time - forward_time))

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if not cfg.MODEL.SEG_ONLY:
            logger.info("Iter {}, Loss: {:.3f} | RPN_class: {:.3f}, RPN_box: {:.3f}, RCNN_class: {:.3f}, RCNN_box: {:.3f}, "
                        "Segmentation: {:.3f}, Batch_time: {:.3f}".format(
                        iteration, losses.item(), loss_dict['loss_objectness'].item(), loss_dict['loss_rpn_box_reg'].item(),
                        loss_dict['loss_classifier'].item(), loss_dict['loss_box_reg'].item(), loss_dict['loss_seg'].item(), end - start_time))
        else:
            logger.info("Iter {}, Loss: {:.3f}| Seg_Source: {:.3f}, Seg_DACS {:.3f}, Batch_time: {:.3f}".format(
                        iteration, losses.item(), loss_dict['loss_seg_source'], loss_dict['loss_seg_dacs'],  end - start_time))
        start_time = time.time()

        if iteration % 20 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
        
        if iteration % checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
        
        if data_loader_val is not None and test_period > 0 and iteration % test_period == 0:
            meters_val = MetricLogger(delimiter="  ")
            synchronize()
            model.eval()
            with torch.no_grad():
                # Should be one image for each GPU:
                img_val, seg_val, box_coord_val, box_class_val = valid_iterator.next()
                
                img_val = torch.Tensor(img_val)
                img_val = img_val.unsqueeze(dim=1)
                img_val = img_val.to(device)

                seg_val = torch.Tensor(seg_val)
                seg_val = seg_val.to(device)

                _, seg_predict = model(img_val, seg_val)

            predict = np.transpose(np.array(seg_predict[0, 1].cpu()), [2, 1, 0])
            with mrcfile.new('{}/predict_iter{}.mrc'.format(output_val_dir, iteration), overwrite=True) as tomo:
                tomo.set_data(np.float32(predict))
                        
            synchronize()
            logger.info(
                meters_val.delimiter.join(
                    [
                        "[Validation]: ",
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters_val),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
            model.train()
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )