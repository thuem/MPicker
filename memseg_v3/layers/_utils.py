# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch.autograd import Variable

def log2(x):
    """Implementatin of Log2. Pytorch doesn't have a native implemenation."""
    ln2 = Variable(torch.log(torch.FloatTensor([2.0])), requires_grad=False)
    if x.is_cuda:
        ln2 = ln2.cuda()
    return torch.log(x) / ln2
