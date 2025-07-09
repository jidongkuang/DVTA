# utils.py
import torch
import numpy as np
import random
from math import pi, cos

def setup_seed(seed):
    """Sets the seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def create_ground_truth_matrix(labels):
    """
    Generates a ground truth matrix for contrastive loss.
    An element (i, j) is 1 if sample i and sample j have the same label, and 0 otherwise.
    """
    # Using NumPy's broadcasting for efficiency
    labels_np = labels.cpu().numpy()
    gt_matrix = (labels_np[:, np.newaxis] == labels_np).astype(np.float32)
    return torch.from_numpy(gt_matrix)

def adjust_learning_rate(optimizer, current_epoch, max_epoch, lr_min, lr_max, warmup_epoch=15):
    """Cosine annealing learning rate scheduler."""
    if current_epoch < warmup_epoch:
            lr = lr_max * current_epoch / warmup_epoch
    else:
        lr = lr_min + (lr_max-lr_min)*(1 + cos(pi * (current_epoch - warmup_epoch) / (max_epoch - warmup_epoch))) / 2
    # lr = lr_min + (lr_max - lr_min) * (1 + cos(pi * current_epoch / max_epoch)) / 2
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr