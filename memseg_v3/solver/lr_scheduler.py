# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from bisect import bisect_right
import pdb
import torch


# FIXME ideally this would be achieved with a CombinedLRScheduler,
# separating MultiStepLR with WarmupLR
# but the current LRScheduler design doesn't allow it
class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        lr_dict,
        iter_per_epoch,
        max_epoch,
        last_epoch=-1,
    ):
        self.lr_dict = lr_dict
        self.max_epoch = max_epoch
        self.iter_per_epoch = iter_per_epoch
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)


    def get_lr(self):
        ind = (4 * (self.last_epoch // self.iter_per_epoch)) // self.max_epoch
        if ind == len(self.lr_dict):
            ind = -1
        for idx in range(len(self.base_lrs)):
            self.base_lrs[idx] = self.lr_dict[ind]
        return self.base_lrs
