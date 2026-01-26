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

from PL.model.model_classifier import Classifier
from PL.dataset.teacher_student import Dataset_Teacher
from PL.utils.saving import init_training_h5, save_training, load_training
from PL.utils.functions import overlap, compute_asymmetry

METRIC_NAMES = [
    "epoch",
    "norm_J",
    "train_loss",
    "train_accuracy",
    "R",
    "learning_rate",
    "diff_hebb",
]

def initialize(N=1000, P=400, d=1, lr=0.1, spin_type="vector",  device='cuda', gamma=0., init_Hebb=True, downf=1.):
    # Initialize the dataset
    dataset = Dataset_Teacher(P, N, d, seed=444, sigma=0.5, spin_type=spin_type)

    # Initialize the model
    model = Classifier(N, d, gamma=gamma, spin_type=spin_type, downf=downf)
    model.to(device)  # Move the model to the specified device
        # create optimizer (vanilla SGD; full-batch equivalence if dataloader is full batch)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # Apply the Hebb rule
    if init_Hebb:
        model.Hebb(dataset.xi, dataset.y, 'Tensorial')

    # Return the dataset and model
    return dataset, model, optimizer

def train_model(model, fixed_norm, dataset, dataloader, epochs, 
                learning_rate, max_grad, device, data_PATH, l, optimizer, J2, 
                norm_J2, valid_every, epochs_to_save, model_name_base, save, l2, loss_type, verbose=True):

    # New: metric history for saving to h5
    history = {name: [] for name in METRIC_NAMES}

    print("# epoch lambda train_loss learning_rate train_metric R")
    t_in = time.time()

    # ---- HDF5 file + untrained model (save 0) ----
    h5_path = os.path.join(data_PATH, model_name_base + ".h5")
    init_training_h5(h5_path, model, METRIC_NAMES, optimizer)
    next_save_idx = 1  # 0 is untrained

    # Training loop
    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        counter = 0

        # Training batch-wise
        for batch_element in dataloader:
            counter += 1
            xi, y = batch_element
            xi = xi.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            # Compute loss (now via compute_crossentropy)
            loss = model.loss(xi, y, lambd=l, loss_type=loss_type, l2=l2)
            # Check for valid loss values (no NaN or Inf)
            if torch.isfinite(loss):
                optimizer.zero_grad()
                loss.backward()

                # gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad)

                # optimizer step
                optimizer.step()

                train_loss += loss.item()
            else:
                print(f"Detected NaN/Inf {model_name_base} epoch {epoch} lr {learning_rate}")
                with torch.no_grad():
                    model.J.data *= 0.1
                learning_rate *= 0.1
                # update optimizer LR as well
                for pg in optimizer.param_groups:
                    pg["lr"] = learning_rate
            
            if fixed_norm==True:
                model.normalize_J()
            y_pred = model(xi)
            accuracy = torch.mean(overlap(y, y_pred))
            train_acc += accuracy.item()

        # Average training loss
        train_loss = train_loss / max(counter, 1)
        train_acc = train_acc / max(counter, 1)
        model.eval()

        # Validation and model saving
        if epoch % valid_every == 0 and epoch > 0:
            R = (torch.einsum("jab,jab->j", dataset.T.to(device), model.J)/(torch.norm(model.J)*torch.norm(dataset.T))).sum().item()

            # print("xi", torch.norm(xi, dim=-1))
            # print("y", torch.norm(y))
            # print("J", torch.norm(model.J))
            # print("u", torch.norm(torch.einsum("jab,mjb->ma", model.J, xi),dim=-1))

            # Compute model parameters for logging
            J = model.J.squeeze().cpu().detach().numpy()
            norm_J = torch.norm(model.J).item()
            diff_Hebb = np.linalg.norm(J2 * norm_J / norm_J2 - J) / norm_J

            if verbose == True:
                print(epoch, norm_J, train_loss, learning_rate, train_acc, R)

            # Append to history used for h5 saving
            history["epoch"].append(epoch)
            history["norm_J"].append(norm_J)
            history["train_loss"].append(train_loss)
            history["train_accuracy"].append(train_acc)
            history["learning_rate"].append(learning_rate)
            history["R"].append(R)
            history["diff_hebb"].append(diff_Hebb)

            # Save checkpoints with h5py
            if (epoch in epochs_to_save) and save is True:
                next_save_idx = save_training(
                    h5_path=h5_path,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    history=history,
                    save_idx=next_save_idx,
                )

    #############################################

    # Final evaluation after training
    model.eval()
    R = (torch.einsum("jab,jab->j", dataset.T.to(device), model.J)/(torch.norm(model.J)*torch.norm(dataset.T))).mean().item()
    # Compute model parameters for logging
    J = model.J.squeeze().cpu().detach().numpy()
    norm_J = torch.norm(model.J, dim=1).mean().item()
    diff_Hebb = np.linalg.norm(J2 * norm_J / norm_J2 - J) / norm_J

    # Append to history used for h5 saving
    history["epoch"].append(epoch)
    history["norm_J"].append(norm_J)
    history["train_loss"].append(train_loss)
    history["train_accuracy"].append(train_acc)
    history["learning_rate"].append(learning_rate)
    history["R"].append(R)
    history["diff_hebb"].append(diff_Hebb)

    if save is True:
        # final SAVE HERE with h5py
        next_save_idx = save_training(
            h5_path=h5_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            history=history,
            save_idx=next_save_idx,
        )

def main(N, alpha_P, l, d, spin_type, batch_size, device, data_PATH, epochs, learning_rate, valid_every, max_grad, loss_type):
    P = int(alpha_P * N)
    print("P={}, lambda={}".format(P, l))
    model_name_base = "{}_capacity_N_{}_P_{}_D{}_l_{}_epochs{}_lr{}_spin{}".format(spin_type, N, P,  l, epochs, learning_rate, spin_type)

    torch.cuda.empty_cache()
    gc.collect()

    dataset, model, optimizer = initialize(N, P, d, learning_rate, spin_type, l, device)
    model2 = Classifier(N, d, spin_type=spin_type)
    model2.to(device)
    model2.Hebb(dataset.xi, 'Tensorial')  # Applying the Hebb rule
    J2 = model2.J.squeeze().cpu().detach().numpy()
    norm_J2 = np.linalg.norm(J2)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=2)
 
    epochs_to_save = [10000]
    save = False

    print("epochs:{} lr:{} max_norm:{} l:{}".format(epochs, learning_rate, max_grad, l))

    # Train the model
    train_model(
        model, dataset, dataloader, epochs, 
        learning_rate, max_grad, device, data_PATH, l, optimizer, J2, norm_J2, valid_every, 
        epochs_to_save, model_name_base, save, l2=None, alpha=None, loss_type=loss_type
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training GD")

    # Define all the parameters
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--alpha_P", type=float, required=True)
    parser.add_argument("--l", type=float, required=True)
    parser.add_argument("--d", type=int, default=1)
    parser.add_argument("--on_sphere", type=bool, default=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--data_PATH", type=str, default="savings")
    parser.add_argument("--epochs", type=int, default=401)
    parser.add_argument("--learning_rate", type=float, default=10.)
    parser.add_argument("--max_grad", type=float, default=20.)
    parser.add_argument("--valid_every", type=int, default=10)
    parser.add_argument("--loss_type", type=str, default="CE")

    args = parser.parse_args()

    # Run the main function with the parsed arguments
    main(args.N, args.alpha_P, args.l, args.d, args.on_sphere, args.device, args.data_PATH, args.epochs, args.learning_rate, args.max_grad, args.valid_every, args.loss_type)
