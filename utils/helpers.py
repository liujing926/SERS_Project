#!/usr/bin/env python
# coding: utf-8

import random
import numpy as np
import torch


def set_seed(seed=42):
    """Set all random seeds to ensure reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False     


def count_parameters(model):
    """Count the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)