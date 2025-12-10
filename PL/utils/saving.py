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

import io
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


def init_training_h5(h5_path, model, optimizer=None):
    """
    Initialize the HDF5 file for training:

    - Overwrites any existing file at h5_path.
    - Creates an 'untrained' checkpoint with index 0:
        * model_0        : state_dict of the untrained model
        * optimizer_0    : (optional) state_dict of the optimizer at epoch 0
        * metric datasets: empty arrays for each METRIC_NAMES key
    """
    # Ensure we start from a clean file
    if os.path.exists(h5_path):
        os.remove(h5_path)

    # Empty history for all metrics
    history = {name: [] for name in METRIC_NAMES}

    # This will create the file and write model_0 / optimizer_0 / empty metrics
    save_training(
        h5_path=h5_path,
        model=model,
        optimizer=optimizer,
        epoch=0,
        history=history,
        save_idx=0,
    )




def save_training(h5_path, model, optimizer, epoch, history, save_idx):
    """
    Save current state:
    - overwrite metric datasets (epoch, norm_J, train_loss, ...) with the *full* history lists
    - save model in a new group model_{save_idx} with its epoch.
    - save optimizer state in optimizer_{save_idx} as serialized bytes.
    Returns the next save index (save_idx + 1).
    """
    with h5py.File(h5_path, "a") as f:
        # metrics
        for key, values in history.items():
            arr = np.array(values, dtype=np.float64)
            if key in f:
                del f[key]
            f.create_dataset(key, data=arr)

        # -------------------
        # model snapshot
        # -------------------
        group_name = f"model_{save_idx}"
        if group_name in f:
            del f[group_name]
        g = f.create_group(group_name)
        g.attrs["epoch"] = int(epoch)

        state_dict = model.state_dict()
        for name, tensor in state_dict.items():
            g.create_dataset(name, data=tensor.detach().cpu().numpy())

        # -------------------
        # optimizer snapshot
        # -------------------
        opt_group_name = f"optimizer_{save_idx}"
        if opt_group_name in f:
            del f[opt_group_name]
        g_opt = f.create_group(opt_group_name)
        g_opt.attrs["epoch"] = int(epoch)

        # serialize optimizer.state_dict() to bytes
        opt_state_dict = optimizer.state_dict()
        buffer = io.BytesIO()
        torch.save(opt_state_dict, buffer)
        byte_data = np.frombuffer(buffer.getvalue(), dtype=np.uint8)

        g_opt.create_dataset("state", data=byte_data)

        f.attrs["num_saves"] = int(save_idx + 1)

    return save_idx + 1

def load_training(h5_path, save_idx, model, optimizer=None, device=None):
    """
    Load model (into the given model instance), optimizer (optional)
    and metrics from a given save index.

    save_idx = 0 -> untrained model, 1 -> first saved checkpoint, etc.

    Args:
        h5_path : path to .h5 file
        save_idx: which checkpoint index to load
        model   : model instance whose state_dict will be overwritten
        optimizer: (optional) optimizer instance; if given and a state is
                   found, its state_dict will be loaded.
        device  : device to map tensors to (defaults to model device)

    Returns:
        model        : the same instance with loaded weights
        optimizer    : the same optimizer instance (or None), possibly with loaded state
        metrics      : dict of numpy arrays for each METRIC_NAMES key
        epoch_saved  : epoch stored for that checkpoint
    """
    if device is None:
        device = next(model.parameters()).device

    with h5py.File(h5_path, "r") as f:
        if save_idx == -1:
            if "num_saves" in f.attrs:
                num_saves = int(f.attrs["num_saves"])
                if num_saves <= 0:
                    raise ValueError(f"No saves found in {h5_path}")
                save_idx = num_saves - 1
            else:
                # Fallback: infer from groups model_*
                indices = []
                for key in f.keys():
                    if key.startswith("model_"):
                        try:
                            idx = int(key.split("_")[1])
                            indices.append(idx)
                        except ValueError:
                            pass
                if not indices:
                    raise ValueError(f"No model_* groups found in {h5_path}")
                save_idx = max(indices)

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

        # -------------------
        # optimizer state (optional)
        # -------------------
        opt_group_name = f"optimizer_{save_idx}"
        if optimizer is not None and opt_group_name in f:
            g_opt = f[opt_group_name]
            byte_data = np.array(g_opt["state"], dtype=np.uint8)
            buffer = io.BytesIO(byte_data.tobytes())
            opt_state_dict = torch.load(buffer, map_location=device)
            optimizer.load_state_dict(opt_state_dict)

        # -------------------
        # metrics history
        # -------------------
        metrics = {}
        for key in METRIC_NAMES:
            if key in f:
                metrics[key] = np.array(f[key])
            else:
                metrics[key] = None

    return model, optimizer, metrics, epoch_saved