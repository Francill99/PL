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


device = torch.device("cpu")
print("Device:", device)


def initialize(N=1000, P=400, D=0, d=1, on_sphere=True, l=1, device='cuda'):
    # Initialize the dataset
    dataset = CustomDataset(P, N, D, d, seed=444, sigma=0.5, on_sphere=on_sphere, coefficients="binary")
    if D>0:
        dataset.RF(seed=444)

    # Initialize the model
    model = TwoBodiesModel(N, d, on_sphere)
    model.to(device)  # Move the model to the specified device

    # Apply the Hebb rule
    model.Hebb(dataset.xi, 'Tensorial')

    # Return the dataset and model
    return dataset, model

def train_model(model, dataloader, dataloader_f, dataloader_gen, epochs, learning_rate, max_norm, device, data_PATH, model_name, init_overlap, n, l, fake_opt, J2, norm_J2, valid_every, ALPHA, epochs_to_save, model_name_base, save):
    # Initial setup
    norm_0 = torch.tensor(1)
    norm = torch.tensor(1)
    save_model_epoch = np.empty(len(epochs_to_save), dtype=object)

    # Initialize SaveModel class
    save_model = Save_Model(data_PATH + model_name, print=False)
    for i_e, e in enumerate(epochs_to_save):
        save_model_epoch[i_e] = Save_Model(data_PATH+model_name_base+"ep{}.pth".format(e), print=False)
    aa = 0
    # Initialize histories
    hist_loss = []
    hist_vloss = []
    hist_asymm = []
    hist_diff = []
    hist_J_norm = []

    t_in = time.time()

    # Training loop
    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        counter = 0

        # Training batch-wise
        for batch_element in dataloader:
            counter += 1
            inp_data = batch_element.to(device)

            # Compute loss
            loss = model(inp_data, lambd=l)

            # Check for valid loss values (no NaN or Inf)
            if (torch.isnan(loss).any() == False) and (torch.isinf(loss).any() == False):
                model.zero_grad()
                with torch.no_grad():
                    # Backward and gradient descent
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                    for param in model.parameters():
                        param.data -= learning_rate * param.grad
                    train_loss += loss.item()
            else:
                print("Detected nan "+ model_name_base+" epoch{} lr{}".format(epoch, learning_rate))
                model.J.data *= 0.1
                learning_rate *= 0.1

        # Average training loss
        train_loss = train_loss / counter
        hist_loss.append(train_loss)
        model.eval()

        # Validation and model saving
        if epoch % valid_every == 0 and epoch > 0:
            vali_loss = 0.0
            counter = 0

            with torch.no_grad():
                for batch_element in dataloader:
                    a=1
                    counter += 1
                    inp_data = batch_element.to(device)
                    # Overlap and model dynamics computations
                    input_vectors = start_overlap_binary(inp_data, init_overlap)
                    input_vectors = model.normalize_x(input_vectors)
                    if (epoch == epochs-1):
                        n = 100             
                    #x_new = model.dyn_n_step(input_vectors, n)
                    '''
                    # Overlap calculations
                    overlaps = torch.einsum('bnid,bid->bni', x_new, inp_data).mean(dim=-1)  # [b,n]
                    final_overlaps = overlaps[:, -1]
                    max_input_overlap, _ = torch.max(overlaps, dim=-1)

                    vloss = final_overlaps.mean().cpu().numpy()
                    vali_loss += vloss
                    '''

            if counter != 0:
                vali_loss = vali_loss / counter
            hist_vloss.append(vali_loss)

            counter_f = 0
            vali_loss_f = 0

            with torch.no_grad():
                for batch_element in dataloader_f:
                    a=1
                    
                    counter_f +=1
                    inp_data = batch_element
                    inp_data = inp_data.to(device)
                    input_vectors = start_overlap_binary(inp_data, init_overlap)
                    input_vectors = model.normalize_x(input_vectors)
                    if (epoch == epochs-1):
                        n = 100
                    #x_new = model.dyn_n_step(input_vectors, n)
                    '''
                    overlaps = torch.einsum('bnid,bid->bni', x_new, inp_data).mean(dim=-1)         #[b,n]

                    final_overlaps = overlaps[:,-1]

                    vloss_f = final_overlaps.mean().cpu().numpy()
                    ###################################
                    vali_loss_f += vloss_f
                    #data = model.parallel_step(inp_data, g=g, iterations)
                    '''
            if counter_f !=0:
                vali_loss_f = vali_loss_f/(counter_f)

            counter_gen = 0
            vali_loss_gen = 0

            with torch.no_grad():  # Disable gradient computation for evaluation
                for batch_element in dataloader_gen:
                    a=1
                    counter_gen += 1
                    inp_data = batch_element.to(device)
        
                    # Overlap and model dynamics computations
                    input_vectors = start_overlap_binary(inp_data, init_overlap)
                    input_vectors = model.normalize_x(input_vectors)
                    if (epoch == epochs-1):
                        n = 100
                    #x_new = model.dyn_n_step(input_vectors, n)
                    '''
                    # Overlap calculations
                    overlaps = torch.einsum('bnid,bid->bni', x_new, inp_data).mean(dim=-1)
                    final_overlaps = overlaps[:, -1]
                    max_input_overlap, _ = torch.max(overlaps, dim=-1)
        
                    # Compute validation loss
                    vloss_gen = final_overlaps.mean().cpu().numpy()
                    vali_loss_gen += vloss_gen
                    '''

            if counter_gen != 0:
                vali_loss_gen = vali_loss_gen / counter_gen

            elapsed_time = time.time() - t0
            time_from_in = time.time() - t_in

            #Save checkpoints
            if (epoch in epochs_to_save) and save==True:
                save_model_epoch[aa](vali_loss, epoch, model, fake_opt, hist_vloss, time_from_in)
                aa +=1

            # Save last model
            if (epoch == epochs-1):
                if save==True:
                    save_model(vali_loss, epoch, model, fake_opt, hist_vloss, time_from_in)
                else:
                    to_save = np.array([vali_loss,vali_loss_f,vali_loss_gen])
                    #np.save(data_PATH + model_name_base+"overlaps",to_save)
                    #print(to_save)
            # Compute model parameters for logging
            J = model.J.squeeze().cpu().detach().numpy()
            norm_J = np.linalg.norm(J)
            asymmetry = compute_asymmetry(J)
            diff_Hebb = np.linalg.norm(J2 * norm_J / norm_J2 - J) / norm_J

            # Append to history
            hist_asymm.append(asymmetry)
            hist_diff.append(diff_Hebb)
            hist_J_norm.append(norm_J)
    #############################################
            
    model.eval()
    vali_loss = 0.0
    counter = 0
    with torch.no_grad():
        for batch_element in dataloader:
            counter += 1
            inp_data = batch_element.to(device)
            # Overlap and model dynamics computations
            input_vectors = start_overlap_binary(inp_data, init_overlap)
            input_vectors = model.normalize_x(input_vectors)
            if (epoch == epochs-1):
                n = 100
            x_new = model.dyn_n_step(input_vectors, n)
            # Overlap calculations
            overlaps = torch.einsum('bnid,bid->bni', x_new, inp_data).mean(dim=-1)  # [b,n]
            final_overlaps = overlaps[:, -1]
            max_input_overlap, _ = torch.max(overlaps, dim=-1)
            vloss = final_overlaps.mean().cpu().numpy()
            vali_loss += vloss
    if counter != 0:
        vali_loss = vali_loss / counter
    hist_vloss.append(vali_loss)
    counter_f = 0
    vali_loss_f = 0
    with torch.no_grad():
        for batch_element in dataloader_f:
            counter_f +=1
            inp_data = batch_element
            inp_data = inp_data.to(device)
            input_vectors = start_overlap_binary(inp_data, init_overlap)
            input_vectors = model.normalize_x(input_vectors)
            if (epoch == epochs-1):
                n = 100
            x_new = model.dyn_n_step(input_vectors, n)
            overlaps = torch.einsum('bnid,bid->bni', x_new, inp_data).mean(dim=-1)         #[b,n]
            final_overlaps = overlaps[:,-1]
            vloss_f = final_overlaps.mean().cpu().numpy()
            ###################################
            vali_loss_f += vloss_f
            #data = model.parallel_step(inp_data, g=g, iterations)
    if counter_f !=0:
        vali_loss_f = vali_loss_f/(counter_f)
    counter_gen = 0
    vali_loss_gen = 0
    with torch.no_grad():  # Disable gradient computation for evaluation
        for batch_element in dataloader_gen:
            counter_gen += 1
            inp_data = batch_element.to(device)

            # Overlap and model dynamics computations
            input_vectors = start_overlap_binary(inp_data, init_overlap)
            input_vectors = model.normalize_x(input_vectors)
            if (epoch == epochs-1):
                n = 100
            x_new = model.dyn_n_step(input_vectors, n)

            # Overlap calculations
            overlaps = torch.einsum('bnid,bid->bni', x_new, inp_data).mean(dim=-1)
            final_overlaps = overlaps[:, -1]
            max_input_overlap, _ = torch.max(overlaps, dim=-1)

            # Compute validation loss
            vloss_gen = final_overlaps.mean().cpu().numpy()
            vali_loss_gen += vloss_gen
    if counter_gen != 0:
        vali_loss_gen = vali_loss_gen / counter_gen
    elapsed_time = time.time() - t0
    time_from_in = time.time() - t_in
    #Save checkpoints
    if (epoch in epochs_to_save) and save==True:
        save_model_epoch[aa](vali_loss, epoch, model, fake_opt, hist_vloss, time_from_in)
        aa +=1
    # Save last model
    if (epoch == epochs-1):
        if save==True:
            save_model(vali_loss, epoch, model, fake_opt, hist_vloss, time_from_in)
        else:
            to_save = np.array([vali_loss,vali_loss_f,vali_loss_gen])
            #np.save(data_PATH + model_name_base+"overlaps",to_save)
            print(to_save)
            J = model.J.squeeze().cpu().detach().numpy()
            asymmetry = compute_asymmetry(J)
            print(asymmetry)
    # Compute model parameters for logging
    J = model.J.squeeze().cpu().detach().numpy()
    norm_J = np.linalg.norm(J)
    asymmetry = compute_asymmetry(J)
    diff_Hebb = np.linalg.norm(J2 * norm_J / norm_J2 - J) / norm_J
    # Append to history
    hist_asymm.append(asymmetry)
    hist_diff.append(diff_Hebb)
    hist_J_norm.append(norm_J)

    # Return training history for further analysis
    return hist_loss, hist_vloss, hist_asymm, hist_diff, hist_J_norm

def main(N, alpha_P, alpha_D, l, D, d, on_sphere, init_overlap, n, device, data_PATH, epochs, learning_rate, valid_every, max_norm=20,):
    P = int(alpha_P * N)
    D = int(alpha_D * N)
    P_generalization=1000
    print("P={}, D={}, lambda={}".format(P, D, l))
    model_name_base = "GD_capacity_N_{}_P_{}_D{}_l_{}_".format(N, P, D, l)
    
    if ALPHA is None:
        model_name = "GD_capacity_N_{}_P_{}_D{}_l_{}_10kepochs_lr100.pth".format(N, P, D, l, d)
    else:
        model_name = "GD_capacity_N_{}_P_{}_D{}_l_{}_A_{}_10kepochs_lr100.pth".format(N, P, D, l, d, ALPHA)

    torch.cuda.empty_cache()
    gc.collect()
    on_sphere=True

    dataset, model = initialize(N, P, D, d, on_sphere, l, device)
    dataset_f = DatasetF(D, dataset.f)
    xi_generalization = dataset.get_generalization(P_generalization)
    dataset_generalization = DatasetF(P_generalization, xi_generalization)

    model2 = TwoBodiesModel(N, d, on_sphere)
    model2.to(device)
    model2.Hebb(dataset.xi, 'Tensorial')  # Applying the Hebb rule
    J2 = model2.J.squeeze().cpu().detach().numpy()
    norm_J2 = np.linalg.norm(J2)

    batch_size = P
    batch_size_f = D

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=2)
    dataloader_f = torch.utils.data.DataLoader(dataset_f, batch_size=batch_size_f, shuffle=False, drop_last=False, num_workers=2)
    dataloader_generalization = torch.utils.data.DataLoader(dataset_generalization, batch_size=P_generalization, shuffle=False, drop_last=False, num_workers=2)
    
    epochs_to_save = [1000]
    save = False

    fake_opt = torch.optim.SGD(model.parameters(), lr=learning_rate)

    print("epochs{} lr{} max_norm{} init_overlap{} n{} l{}".format(epochs, learning_rate, max_norm, init_overlap, n, l))
    

    # Train the model
    hist_loss, hist_vloss, hist_asymm, hist_diff, hist_J_norm = train_model(
        model, dataloader, dataloader_f,dataloader_generalization, epochs, learning_rate, max_norm, device, data_PATH, model_name, init_overlap, n, l, fake_opt, J2, norm_J2, valid_every, ALPHA, epochs_to_save, model_name_base, save,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training GD")

    # Define all the parameters
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--alpha_P", type=float, required=True)
    parser.add_argument("--alpha_D", type=float, required=True)
    parser.add_argument("--l", type=float, required=True)
    parser.add_argument("--D", type=int, default=0)
    parser.add_argument("--d", type=int, default=1)
    parser.add_argument("--on_sphere", type=bool, default=True)
    parser.add_argument("--init_overlap", type=float, default=1.0)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--data_PATH", type=str, required=True)
    parser.add_argument("--ALPHA", type=float, default=None)

    args = parser.parse_args()

    # Run the main function with the parsed arguments
    main(args.N, args.alpha_P, args.alpha_D, args.l, args.D, args.d, args.on_sphere, args.init_overlap, args.n, args.device, args.data_PATH, args.ALPHA)