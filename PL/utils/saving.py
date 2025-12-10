## Standard libraries
import os
import numpy as np
import random
import math
import time
import copy
import argparse
import torch
import gc
import h5py

## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import os
import numpy as np
import h5py
import torch

METRIC_NAMES = [
    "epoch",
    "norm_J",
    "train_loss",
    "learning_rate",
    "vali_loss",
    "vali_loss_f",
    "vali_loss_gen",
    "vali_loss_max",
    "vali_loss_f_max",
    "vali_loss_gen_max",
    "x_norm",
]


def init_training_h5(h5_path, model):
    """
    Create the .h5 file at the beginning of training and save the *untrained* model as save 0.
    """
    os.makedirs(os.path.dirname(h5_path), exist_ok=True)

    with h5py.File(h5_path, "w") as f:
        # untrained model = model_0
        g = f.create_group("model_0")
        g.attrs["epoch"] = 0

        state_dict = model.state_dict()
        for name, tensor in state_dict.items():
            f.create_dataset(name, data=tensor.detach().cpu().numpy())

        f.attrs["num_saves"] = 1  # currently only the untrained model


def save_training(h5_path, model, epoch, history, save_idx):
    """
    Save current state:
    - overwrite metric datasets (epoch, norm_J, train_loss, ...) with the *full* history lists
    - save model in a new group model_{save_idx} with its epoch.
    Returns the next save index (save_idx + 1).
    """
    with h5py.File(h5_path, "a") as f:
        # metrics
        for key, values in history.items():
            arr = np.array(values, dtype=np.float64)
            if key in f:
                del f[key]
            f.create_dataset(key, data=arr)

        # model snapshot
        group_name = f"model_{save_idx}"
        if group_name in f:
            del f[group_name]
        g = f.create_group(group_name)
        g.attrs["epoch"] = int(epoch)

        state_dict = model.state_dict()
        for name, tensor in state_dict.items():
            g.create_dataset(name, data=tensor.detach().cpu().numpy())

        f.attrs["num_saves"] = int(save_idx + 1)

    return save_idx + 1


def load_training(h5_path, save_idx, model, device=None):
    """
    Load model (into the given model instance) and metrics from a given save index.

    save_idx = 0 -> untrained model, 1 -> first saved checkpoint, etc.

    Returns:
        model        : the same instance with loaded weights
        metrics      : dict of numpy arrays for each METRIC_NAMES key
        epoch_saved  : epoch stored for that checkpoint
    """
    if device is None:
        device = next(model.parameters()).device

    with h5py.File(h5_path, "r") as f:
        group_name = f"model_{save_idx}"
        if group_name not in f:
            raise ValueError(f"Save index {save_idx} not found in {h5_path}")

        g = f[group_name]
        epoch_saved = int(g.attrs["epoch"])

        # rebuild state_dict
        state_dict = {}
        for name in g.keys():
            arr = np.array(g[name])
            tensor = torch.from_numpy(arr).to(device)
            state_dict[name] = tensor

        model.load_state_dict(state_dict)

        metrics = {}
        for key in METRIC_NAMES:
            if key in f:
                metrics[key] = np.array(f[key])
            else:
                metrics[key] = None

    return model, metrics, epoch_saved
