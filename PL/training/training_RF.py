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

from PL.model.model import TwoBodiesModel
from PL.dataset.random_features import RandomFeaturesDataset, GeneralDataset
from PL.utils.saving import init_training_h5, save_training, load_training
from PL.utils.functions import start_overlap, compute_asymmetry, compute_validation_overlap

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
    "asymmetry",
    "diff_hebb"
]


def initialize(N=1000, P=400, D=0, d=1, lr=0.1, spin_type="vector", device='cuda', L=3, gamma=0., init_Hebb=True):
    # Initialize the dataset
    dataset = RandomFeaturesDataset(P, N, d, seed=444, sigma=0.5, spin_type=spin_type, coefficients="binary", L=L, D=D)

    # Initialize the model
    model = TwoBodiesModel(N, d, gamma=gamma, spin_type=spin_type)
    model.to(device)  # Move the model to the specified device
        # create optimizer (vanilla SGD; full-batch equivalence if dataloader is full batch)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # Apply the Hebb rule
    if init_Hebb:
        model.Hebb(dataset.xi.to(device), 'Tensorial')

    # Return the dataset and model
    return dataset, model, optimizer

def train_model(model, dataloader, dataloader_f, dataloader_gen, epochs, learning_rate, max_grad, device, data_PATH, init_overlap, n, l, optimizer, J2, norm_J2, valid_every, epochs_to_save, model_name_base, save, l2, alpha, loss_type):

    # New: metric history for saving to h5
    history = {name: [] for name in METRIC_NAMES}

    print("# epoch lambda train_loss learning_rate train_metric features_metric generalization_metric // // // norm_x")


    # ---- HDF5 file + untrained model (save 0) ----
    h5_path = os.path.join(data_PATH, model_name_base + ".h5")
    init_training_h5(h5_path, model, METRIC_NAMES, optimizer)
    next_save_idx = 1  # 0 is untrained

    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        counter = 0
        train_loss_t = torch.zeros((), device=device)

        # Training batch-wise
        for batch_element in dataloader:
            counter += 1
            inp_data = batch_element.to(device)
            # Compute loss (now via compute_crossentropy)
            loss = model.loss(inp_data, lambd=l, loss_type=loss_type, l2=l2, alpha=alpha)
            # Check for valid loss values (no NaN or Inf)
            if torch.isfinite(loss):
                optimizer.zero_grad()
                loss.backward()

                # gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad)

                # optimizer step
                optimizer.step()

                with torch.no_grad():
                    train_loss_t += loss.detach()
            else:
                print(f"Detected NaN/Inf {model_name_base} epoch {epoch} lr {learning_rate}")
                with torch.no_grad():
                    model.J.data *= 0.1
                learning_rate *= 0.1
                # update optimizer LR as well
                for pg in optimizer.param_groups:
                    pg["lr"] = learning_rate

        model.eval()

        # Validation and model saving
        if epoch % valid_every == 0 and epoch > 0:
            train_loss = (train_loss_t / counter).item()
            vali_loss, vali_loss_max = compute_validation_overlap(
                model=model, dataloader=dataloader, device=device,
                init_overlap=init_overlap, n=n,
            )
            vali_loss_f, vali_loss_f_max = compute_validation_overlap(
                model=model, dataloader=dataloader_f, device=device,
                init_overlap=init_overlap, n=n,
            )
            vali_loss_gen, vali_loss_gen_max = compute_validation_overlap(
                model=model, dataloader=dataloader_gen, device=device,
                init_overlap=init_overlap, n=n,
            )

            # dynamics to compute x_norm (using last batch's inp_data)
            x_new = inp_data.clone()
            for _ in range(n):
                x_new = model.dyn_step(x_new)
            x_norm = torch.norm(x_new).cpu().item()

            # Compute model parameters for logging
            J = model.J.squeeze().cpu().detach().numpy()
            norm_J = torch.norm(model.J, dim=1).mean().item()
            asymmetry = compute_asymmetry(J)
            diff_Hebb = np.linalg.norm(J2 * norm_J / norm_J2 - J) / norm_J

            print(epoch, norm_J, train_loss, learning_rate,
                  vali_loss, vali_loss_f, vali_loss_gen,
                  vali_loss_max, vali_loss_f_max, vali_loss_gen_max, x_norm)

            # Append to history used for h5 saving
            history["epoch"].append(epoch)
            history["norm_J"].append(norm_J)
            history["train_loss"].append(train_loss)
            history["learning_rate"].append(learning_rate)
            history["vali_loss"].append(vali_loss)
            history["vali_loss_f"].append(vali_loss_f)
            history["vali_loss_gen"].append(vali_loss_gen)
            history["vali_loss_max"].append(vali_loss_max)
            history["vali_loss_f_max"].append(vali_loss_f_max)
            history["vali_loss_gen_max"].append(vali_loss_gen_max)
            history["x_norm"].append(x_norm)
            history["asymmetry"].append(asymmetry)
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
    vali_loss, vali_loss_max = compute_validation_overlap(
        model=model, dataloader=dataloader, device=device,
        init_overlap=init_overlap, n=n,
    )
    vali_loss_f, vali_loss_f_max = compute_validation_overlap(
        model=model, dataloader=dataloader_f, device=device,
        init_overlap=init_overlap, n=n,
    )
    vali_loss_gen, vali_loss_gen_max = compute_validation_overlap(
        model=model, dataloader=dataloader_gen, device=device,
        init_overlap=init_overlap, n=n,
    )

    # compute x_norm again for this final evaluation (first batch)
    with torch.no_grad():
        for batch_element in dataloader:
            inp_data = batch_element.to(device)
            x_new = inp_data.clone()
            for i_n in range(n):
                x_new = model.dyn_step(x_new)
            x_norm = torch.norm(x_new).cpu().item()
            break

    # Compute model parameters for logging
    J = model.J.squeeze().cpu().detach().numpy()
    norm_J = torch.norm(model.J, dim=1).mean().item()
    asymmetry = compute_asymmetry(J)
    diff_Hebb = np.linalg.norm(J2 * norm_J / norm_J2 - J) / (norm_J + 1e-9)

    # Also append to h5 metric history
    history["epoch"].append(epoch)  # epoch is still last loop value
    history["norm_J"].append(norm_J)
    history["train_loss"].append(train_loss)
    history["learning_rate"].append(learning_rate)
    history["vali_loss"].append(vali_loss)
    history["vali_loss_f"].append(vali_loss_f)
    history["vali_loss_gen"].append(vali_loss_gen)
    history["vali_loss_max"].append(vali_loss_max)
    history["vali_loss_f_max"].append(vali_loss_f_max)
    history["vali_loss_gen_max"].append(vali_loss_gen_max)
    history["x_norm"].append(x_norm)
    history["asymmetry"].append(asymmetry)
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

def main(N, alpha_P, alpha_D, l, L, d, spin_type, init_overlap, n, device, data_PATH, epochs, learning_rate, valid_every, max_grad, P_generalization, loss_type):
    P = int(alpha_P * N)
    D = int(alpha_D * N)
    print("P={}, D={}, L={}, lambda={}".format(P, D, L, l))
    model_name_base = "{}_capacity_N_{}_P_{}_D{}_l_{}_epochs{}_lr{}_spin{}".format(spin_type, N, P, D, l, epochs, learning_rate, spin_type)

    torch.cuda.empty_cache()
    gc.collect()

    dataset, model, optimizer = initialize(N, P, D, d, learning_rate, spin_type, device, L)
    if D>0:
        dataset_f = GeneralDataset(D, dataset.f)
        xi_generalization = dataset.get_generalization(P_generalization)
        dataset_generalization = GeneralDataset(P_generalization, xi_generalization)
        batch_size = P
        batch_size_f = D
    else:
        dataset_f = dataset
        dataset_generalization = dataset
        batch_size = P
        batch_size_f = P


    model2 = TwoBodiesModel(N, d, spin_type=spin_type)
    model2.to(device)
    model2.Hebb(dataset.xi, 'Tensorial')  # Applying the Hebb rule
    J2 = model2.J.squeeze().cpu().detach().numpy()
    norm_J2 = np.linalg.norm(J2)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=2)
    dataloader_f = torch.utils.data.DataLoader(dataset_f, batch_size=batch_size_f, shuffle=False, drop_last=False, num_workers=2)
    dataloader_generalization = torch.utils.data.DataLoader(dataset_generalization, batch_size=P_generalization, shuffle=False, drop_last=False, num_workers=2)
    
    epochs_to_save = [1000]
    save = False

    print("epochs:{} lr:{} max_norm:{} init_overlap:{} n:{} l:{}".format(epochs, learning_rate, max_grad, init_overlap, n, l))

    # Train the model
    train_model(
        model, dataloader, dataloader_f,dataloader_generalization, epochs, 
        learning_rate, max_grad, device, data_PATH, init_overlap, 
        n, l, optimizer, J2, norm_J2, valid_every, epochs_to_save, model_name_base, save, loss_type
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training GD")

    # Define all the parameters
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--alpha_P", type=float, required=True)
    parser.add_argument("--alpha_D", type=float, required=True)
    parser.add_argument("--l", type=float, required=True)  # lambda: inverse temperature
    parser.add_argument("--d", type=int, default=1)
    parser.add_argument("--spin_type", type=str, default="vector")
    parser.add_argument("--init_overlap", type=float, default=1.0)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--data_PATH", type=str, default="savings")
    parser.add_argument("--epochs", type=int, default=401)
    parser.add_argument("--learning_rate", type=float, default=10.)
    parser.add_argument("--max_grad", type=float, default=20.)
    parser.add_argument("--valid_every", type=int, default=10)
    parser.add_argument("--P_generalization", type=int, default=1000)
    parser.add_argument("--L", type=int, default=3)
    parser.add_argument("--loss_type", type=str, default="CE")

    args = parser.parse_args()

    # Run the main function with the parsed arguments
    main(args.N, args.alpha_P, args.alpha_D, args.l, args.L, args.d, args.on_sphere, args.init_overlap, args.n, args.device, args.data_PATH, args.epochs, args.learning_rate, args.max_grad, args.valid_every, args.P_generalization, args.loss_type)
