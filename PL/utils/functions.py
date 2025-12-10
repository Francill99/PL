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

## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

def compute_asymmetry(J: np.ndarray) -> float:
    # Ensure that J is a NumPy array
    J = np.array(J)

    # Step 1: Compute the difference between J and its transpose
    asymmetry_matrix = J - J.T

    # Step 2: Square each element in the asymmetry matrix
    squared_diff_matrix = np.square(asymmetry_matrix)

    # Step 3: Compute the mean of the squared differences
    asymmetry = np.mean(squared_diff_matrix)

    return asymmetry

def start_overlap(
    xi: torch.Tensor,
    init_overlap: float,
    spin_type: str = "vector",
    gamma: float = None,
) -> torch.Tensor:
    """
    Build initial configurations at a given overlap with the reference patterns xi.

    Args
    ----
    xi : torch.Tensor
        Reference patterns, shape [X, N, d].
    init_overlap : float
        Desired (global) overlap with xi.
    spin_type : str, optional
        "vector" or "continuous".
        - "vector", d=1: binary ±1 spins (current behaviour).
        - "vector", d>1: vectors on the sphere; we replace some vectors by new
          random unit vectors.
        - "continuous": we replace some spins by Gaussian with variance 1/gamma.
    gamma : float, optional
        Regularization parameter for the continuous case; the new spins are
        drawn from N(0, 1/gamma).

    Returns
    -------
    init_vectors : torch.Tensor
        Initial configurations with the requested overlap, same shape as xi.
    """
    # Copy xi to avoid modifying the input directly
    init_vectors = xi.clone()

    # Get the dimensions of the input tensor
    X, N, d = init_vectors.shape
    device = init_vectors.device

    for x in range(X):
        if spin_type == "vector":
            if d == 1:
                # Number of rows to flip in order to obtain the desired overlap:
                # m_new = 1 - 2 K / N  =>  K/N = (1 - m_new) / 2
                num_rows_to_flip = int(N * ((1.0 - init_overlap) / 2.0))
                if num_rows_to_flip <= 0:
                    continue

                indices_to_flip = torch.randperm(N, device=device)[:num_rows_to_flip]
                init_vectors[x, indices_to_flip, :] *= -1

            else:
                # Here random new vectors are independent of the old ones, so their expected
                # local overlap is 0. To get global overlap m, we keep a fraction m unchanged
                # and replace a fraction (1 - m) by new random unit vectors.
                num_rows_to_change = int(N * (1.0 - init_overlap))
                if num_rows_to_change <= 0:
                    continue

                indices_to_change = torch.randperm(N, device=device)[:num_rows_to_change]
                # Draw new vectors from N(0,1) and normalize on the sphere
                new_vecs = torch.randn(num_rows_to_change, d, device=device)
                norms = new_vecs.norm(dim=-1, keepdim=True) + 1e-9
                new_vecs = new_vecs / norms
                init_vectors[x, indices_to_change, :] = new_vecs

        elif spin_type == "continuous":
            # ------------------------ continuous spins (any d) ------------------------ #
            if gamma is None:
                raise ValueError("gamma must be provided for spin_type='continuous'.")

            num_rows_to_change = int(N * (1.0 - init_overlap))
            if num_rows_to_change <= 0:
                continue

            indices_to_change = torch.randperm(N, device=device)[:num_rows_to_change]
            std = (1.0 / gamma) ** 0.5

            # For d=1 this is scalar; for d>1 it is a vector with i.i.d. components
            new_vals = torch.randn(num_rows_to_change, d, device=device) * std
            init_vectors[x, indices_to_change, :] = new_vals

        else:
            raise ValueError("spin_type must be 'vector' or 'continuous'.")

    return init_vectors

import numpy as np
import torch
from torch.utils.data import DataLoader


def basins_of_attraction_inp_vectors(
    init_overlaps_array,
    model,
    dataset,
    num_of_run,
    n,
    device,
    batch_size=None,
):
    """
    For each init_overlap in init_overlaps_array and for each independent run
    (num_of_run), do:

      - build initial configurations via start_overlap at that overlap
        relative to the dataset patterns,
      - run the dynamics for n steps,
      - compute per-sample max overlap over time and final overlap.

    Parallelization over the dataset is always done via a DataLoader.

    Returns
    -------
    max_overlap_xi_array  : np.ndarray
        Shape [len(init_overlaps_array), num_of_run, num_samples]
    final_overlap_xi_array: np.ndarray
        Shape [len(init_overlaps_array), num_of_run, num_samples]
    """
    model.eval()

    # Infer spin_type / gamma from model if available
    spin_type = getattr(model, "spin_type", "vector")
    gamma = getattr(model, "gamma", None)

    num_samples = len(dataset)

    # If batch_size is None, use all samples in one batch
    if batch_size is None:
        batch_size = num_samples

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    max_overlap_xi_list = []   # will hold [num_of_init, num_of_run, num_samples]
    final_overlap_xi_list = []

    with torch.no_grad():
        # Loop over different initial overlaps
        for init_overlap in init_overlaps_array:
            runs_max = []    # list of tensors, each [num_samples]
            runs_final = []  # list of tensors, each [num_samples]

            # For each independent run, regenerate random initial data
            for run_idx in range(num_of_run):
                per_run_max_batches = []
                per_run_final_batches = []
                processed = 0

                # Loop over dataset in batches (DataLoader -> uses __getitem__)
                for batch in dataloader:
                    # If dataset returns (data, label, ...), keep only data
                    if isinstance(batch, (tuple, list)):
                        inp_data = batch[0]
                    else:
                        inp_data = batch

                    inp_data = inp_data.to(device)  # [B, N, d] or [B, N]
                    B = inp_data.shape[0]

                    targets = inp_data  # reference patterns

                    # 1) initial configurations at given overlap (new randomness every run)
                    x = start_overlap(
                        targets,
                        init_overlap,
                        spin_type=spin_type,
                        gamma=gamma,
                    )

                    max_ov = torch.zeros(B, device=device)
                    final_ov = None

                    # 2) iterate dynamics for n steps
                    for _ in range(n):
                        # follow your other code: dyn_step(x, n)
                        x = model.dyn_step(x, n)
                        ov = overlap(x, targets, spin_type=spin_type)  # [B]

                        # 3) track max and final overlaps
                        max_ov = torch.maximum(max_ov, ov)
                        final_ov = ov

                    per_run_max_batches.append(max_ov)    # [B]
                    per_run_final_batches.append(final_ov)

                    processed += B

                # Concatenate all batches for this run -> [num_samples]
                run_max_all = torch.cat(per_run_max_batches, dim=0)
                run_final_all = torch.cat(per_run_final_batches, dim=0)

                # Just in case, trim to exact num_samples
                run_max_all = run_max_all[:num_samples]
                run_final_all = run_final_all[:num_samples]

                runs_max.append(run_max_all)
                runs_final.append(run_final_all)

            # Stack runs for this init_overlap -> [num_of_run, num_samples]
            runs_max = torch.stack(runs_max, dim=0)
            runs_final = torch.stack(runs_final, dim=0)

            max_overlap_xi_list.append(runs_max.cpu().numpy())
            final_overlap_xi_list.append(runs_final.cpu().numpy())

    # Final shape: [len(init_overlaps_array), num_of_run, num_samples]
    max_overlap_xi_array = np.array(max_overlap_xi_list)
    final_overlap_xi_array = np.array(final_overlap_xi_list)

    return max_overlap_xi_array, final_overlap_xi_array

def compute_validation_overlap(model, dataloader, device, init_overlap, n):
    model.eval()  # Set model to evaluation mode
    vali_loss = 0.0
    max_vloss = 0.0
    counter = 0

    with torch.no_grad():  # Disable gradient computation for evaluation
        for batch_element in dataloader:
            counter += 1
            inp_data = batch_element.to(device)

            # Overlap and model dynamics computations
            input_vectors = start_overlap(inp_data, init_overlap)
            x_new = input_vectors.clone()
            overlaps = torch.zeros((inp_data.shape[0], n))

            for i_n in range(n):
                x_new = model.dyn_step(x_new, n)
                overlaps[:,i_n] = overlap(x_new, inp_data)

            final_overlaps = overlaps[:, -1]
            max_input_overlap = torch.max(overlaps, dim=-1)[0]

            # Compute validation loss
            vloss = final_overlaps.mean().cpu().numpy()
            max_vloss += max_input_overlap.mean().cpu().numpy()
            vali_loss += vloss

    if counter != 0:
        vali_loss = vali_loss / counter
        max_vloss = max_vloss / counter

    return vali_loss, max_vloss

import torch

def overlap(x: torch.Tensor,
            y: torch.Tensor,
            spin_type: str = "vector",
            eps: float = 1e-8) -> torch.Tensor:
    """
    Compute the overlap between configurations x and y.

    The result is always in [0, 1], where:
        0 -> completely uncorrelated
        1 -> equal (up to numerical precision)

    Parameters
    ----------
    x : torch.Tensor
        Tensor of shape [..., N, d].
    y : torch.Tensor
        Tensor of shape [..., N, d] or [N, d] (broadcastable to x).
    spin_type : str
        "vector"    -> binary spins (d=1) or unit vectors on the sphere (d>1).
                       Uses global cosine similarity.
        "continuous"-> generic continuous variables.
                       Uses Pearson correlation (mean-subtracted).
    eps : float
        Numerical epsilon to avoid division by zero.

    Returns
    -------
    torch.Tensor
        Overlap values in [0, 1], with shape equal to the broadcasted
        leading dimensions of x and y (i.e. no N,d dimensions).
    """

    # Make sure x and y have the same broadcasted shape
    x, y = torch.broadcast_tensors(x, y)

    # Flatten spatial dims (N, d) into a single dimension
    # so we can work with generic leading dims: [...]
    x_flat = x.reshape(*x.shape[:-2], -1)
    y_flat = y.reshape(*y.shape[:-2], -1)

    if spin_type == "vector":
        # Overlap = global cosine similarity, clipped to [0,1]
        # For binary ±1 spins, this is (1/N) sum_i s_i t_i.
        num = (x_flat * y_flat).sum(dim=-1)
        x_norm = torch.sqrt((x_flat ** 2).sum(dim=-1) + eps)
        y_norm = torch.sqrt((y_flat ** 2).sum(dim=-1) + eps)
        cos = num / (x_norm * y_norm + eps)

        # We only care about positive correlation; clamp to [0,1]
        ov = torch.absolute(cos)
        return ov

    elif spin_type == "continuous":
        # Overlap = Pearson correlation across all coordinates:
        #   rho = Cov(X, Y) / (sigma_X * sigma_Y)
        # then clipped to [0,1].
        x_mean = x_flat.mean(dim=-1, keepdim=True)
        y_mean = y_flat.mean(dim=-1, keepdim=True)

        x_c = x_flat - x_mean
        y_c = y_flat - y_mean

        num = (x_c * y_c).sum(dim=-1)

        x_var = (x_c ** 2).sum(dim=-1)
        y_var = (y_c ** 2).sum(dim=-1)

        denom = torch.sqrt(x_var * y_var + eps)

        rho = num / (denom + eps)

        # Again keep [0,1], treating negative (anti-)correlation as 0
        ov = torch.clamp(rho, min=0.0, max=1.0)
        return ov

    else:
        raise ValueError("spin_type must be 'vector' or 'continuous'.")
