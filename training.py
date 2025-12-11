## Standard libraries
import numpy as np
import time
import argparse
import torch
import gc

## PyTorch
import torch

from PL.model.model import TwoBodiesModel
from PL.dataset.dataset import CustomDataset, DatasetF
from PL.utils.saving import Save_Model
from PL.utils.functions import compute_asymmetry, compute_validation_overlap


device = torch.device("cpu")
print("Device:", device)


def initialize(N=1000, P=400, D=0, d=1, spin_type="vector", l=1, device='cuda', L=3, gamma=0., init_Hebb=True):
    # Initialize the dataset
    dataset = CustomDataset(P, N, D, d, seed=444, sigma=0.5, spin_type=spin_type, coefficients="binary", L=L)
    if D>0:
        dataset.RF(seed=444)

    # Initialize the model
    model = TwoBodiesModel(N, d, gamma=gamma, spin_type=spin_type)
    model.to(device)  # Move the model to the specified device

    # Apply the Hebb rule
    if init_Hebb:
        model.Hebb(dataset.xi.to(device), 'Tensorial')

    # Return the dataset and model
    return dataset, model

def train_model(model, dataloader, dataloader_f, dataloader_gen, epochs, learning_rate, max_grad, device, data_PATH,
                model_name, init_overlap, n, l, fake_opt, J2, norm_J2, valid_every, epochs_to_save, model_name_base, save):
    # Initial setup
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

    print("# epoch lambda train_loss learning_rate train_metric features_metric generalization_metric // // // norm_x")

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
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad)
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
            vali_loss, vali_loss_max = compute_validation_overlap(model=model, dataloader=dataloader, device=device,  
                                            init_overlap=init_overlap, n=n,
            )
            vali_loss_f, vali_loss_f_max = compute_validation_overlap(model=model, dataloader=dataloader_f, device=device,  
                                            init_overlap=init_overlap, n=n,
            )
            vali_loss_gen, vali_loss_gen_max = compute_validation_overlap(model=model, dataloader=dataloader_gen, device=device,  
                                            init_overlap=init_overlap, n=n,
            )

            elapsed_time = time.time() - t0
            time_from_in = time.time() - t_in

            x_new = inp_data.clone()
            for i_n in range(n):
                x_new = model.dyn_step(x_new)

            x_norm = torch.norm(x_new).cpu().numpy()

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
            # Compute model parameters for logging
            J = model.J.squeeze().cpu().detach().numpy()
            norm_J = torch.norm(model.J, dim=1).mean().item()
            asymmetry = compute_asymmetry(J)
            diff_Hebb = np.linalg.norm(J2 * norm_J / norm_J2 - J) / norm_J

            print(epoch, norm_J, train_loss, learning_rate, vali_loss, vali_loss_f, vali_loss_gen, vali_loss_max, vali_loss_f_max, vali_loss_gen_max, x_norm)

            # Append to history
            hist_asymm.append(asymmetry)
            hist_diff.append(diff_Hebb)
            hist_J_norm.append(norm_J)
    #############################################
            
    model.eval()
    vali_loss = 0.0
    counter = 0
    vali_loss = compute_validation_overlap(model=model, dataloader=dataloader, device=device,  
                                    init_overlap=init_overlap, n=n,
    )
    vali_loss_f = compute_validation_overlap(model=model, dataloader=dataloader_f, device=device,  
                                    init_overlap=init_overlap, n=n,
    )
    vali_loss_gen = compute_validation_overlap(model=model, dataloader=dataloader_gen, device=device,  
                                    init_overlap=init_overlap, n=n,
    )
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
    diff_Hebb = np.linalg.norm(J2 * norm_J / norm_J2 - J) / (norm_J+1e-9)
    # Append to history
    hist_asymm.append(asymmetry)
    hist_diff.append(diff_Hebb)
    hist_J_norm.append(norm_J)

    # Return training history for further analysis
    return hist_loss, hist_vloss, hist_asymm, hist_diff, hist_J_norm

def main(N, alpha_P, alpha_D, l, L, d, spin_type, init_overlap, n, device, data_PATH, epochs, learning_rate, valid_every, max_grad, P_generalization):
    P = int(alpha_P * N)
    D = int(alpha_D * N)
    print("P={}, D={}, L={}, lambda={}".format(P, D, L, l))
    model_name_base = "GD_capacity_N_{}_P_{}_D{}_l_{}_epochs{}_lr{}".format(N, P, D, l, epochs, learning_rate)
    
    model_name = "GD_capacity_N_{}_P_{}_D{}_l_{}_epochs{}_lr{}.pth".format(N, P, D, l, epochs, learning_rate)

    torch.cuda.empty_cache()
    gc.collect()

    dataset, model = initialize(N, P, D, d, spin_type, l, device, L)
    if D>0:
        dataset_f = DatasetF(D, dataset.f)
        xi_generalization = dataset.get_generalization(P_generalization)
        dataset_generalization = DatasetF(P_generalization, xi_generalization)
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

    fake_opt = torch.optim.SGD(model.parameters(), lr=learning_rate)

    print("epochs:{} lr:{} max_norm:{} init_overlap:{} n:{} l:{}".format(epochs, learning_rate, max_grad, init_overlap, n, l))

    # Train the model
    hist_loss, hist_vloss, hist_asymm, hist_diff, hist_J_norm = train_model(
        model, dataloader, dataloader_f,dataloader_generalization, epochs, 
        learning_rate, max_grad, device, data_PATH, model_name, init_overlap, 
        n, l, fake_opt, J2, norm_J2, valid_every, epochs_to_save, model_name_base, save,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training GD")

    # Define all the parameters
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--alpha_P", type=float, required=True)
    parser.add_argument("--alpha_D", type=float, required=True)
    parser.add_argument("--l", type=float, required=True)   # the inverse temperature lambda
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

    args = parser.parse_args()

    # Run the main function with the parsed arguments
    main(args.N, args.alpha_P, args.alpha_D, args.l, args.L, args.d, args.spin_type, args.init_overlap, args.n, args.device, args.data_PATH, args.epochs, args.learning_rate, args.max_grad, args.valid_every, args.P_generalization)
