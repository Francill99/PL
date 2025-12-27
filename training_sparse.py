## Standard libraries
import os
import numpy as np
import argparse
import torch
import gc

## PyTorch
import torch

from PL.model.model import TwoBodiesModel
from PL.dataset.dataset import BasicDataset
from PL.utils.saving import init_training_h5, save_training
from PL.utils.functions import compute_asymmetry, compute_validation_overlap

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


def initialize(N=1000, P=400, P_generalization=400, d=1, lr=0.1, spin_type="vector", device='cuda', gamma=0., 
               init_Hebb=True, data_train=None, data_test=None, coefficients="binary"):
    # Initialize the dataset
    dataset = BasicDataset(P, N, d, spin_type=spin_type, coefficients=coefficients, xi=data_train)
    dataset_gen = BasicDataset(P_generalization, N, d, spin_type=spin_type, coefficients=coefficients, xi=data_test)

    # Initialize the model
    model = TwoBodiesModel(N, d, gamma=gamma, spin_type=spin_type, device=device)
    model.to(device)  # Move the model to the specified device
        # create optimizer (vanilla SGD; full-batch equivalence if dataloader is full batch)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # Apply the Hebb rule
    if init_Hebb:
        model.Hebb(dataset.xi.to(device), 'Tensorial')

    # Return the dataset and model
    return dataset, dataset_gen, model, optimizer



def train_model(model, dataloader, dataloader_gen, epochs, learning_rate, max_grad, 
                device, data_PATH, init_overlap, n, l, optimizer, J2, norm_J2, valid_every, 
                save_every, model_name_base, save, extra_steps=0, factor_lr_decay=0.999, 
                factor_lr_diminish_when_error=0.9, patience_lr=50, factor_J_diminish_when_error=0.9):

    # New: metric history for saving to h5
    history = {name: [] for name in METRIC_NAMES}

    print("# epoch lambda train_loss learning_rate train_overlap generalization_overlap // // // norm_x")


    # ---- HDF5 file + untrained model (save 0) ----
    h5_path = os.path.join(data_PATH, model_name_base + ".h5")


    # IMPORTANT: change init_training_h5 to also store optimizer (see below)
    init_training_h5(h5_path, model, optimizer)
    next_save_idx = 1  # 0 is untrained

    # Training loop
    epoch = 1
    while epoch < epochs + 1:
        model.train()
        train_loss = 0.0
        counter = 0
        counter_diminish_lr = 0
        error_detected = False
        # Training batch-wise
        for batch_element in dataloader:
            counter += 1
            inp_data = batch_element.to(device)

            # Compute loss (now via compute_crossentropy)
            loss = model.compute_crossentropy(inp_data, lambd=l)

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
                # Reduce J and learning rate
                with torch.no_grad():
                    model.J.data *= factor_J_diminish_when_error
                if counter_diminish_lr >= patience_lr:
                    learning_rate *= factor_lr_diminish_when_error
                    # update optimizer LR as well
                    for pg in optimizer.param_groups:
                        pg["lr"] = learning_rate
                    counter_diminish_lr = 0
                else:
                    counter_diminish_lr += 1
                error_detected = True
        if error_detected:
            epoch -= 1  # redo this epoch

        # Average training loss
        train_loss = train_loss / max(counter, 1)
        model.eval()

        # Validation and model saving
        if epoch % valid_every == 0 and epoch > 0:
            vali_loss, vali_loss_max = compute_validation_overlap(
                model=model, dataloader=dataloader, device=device,
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
                  vali_loss, vali_loss_gen,
                  vali_loss_max, vali_loss_gen_max, x_norm)

            # Append to history used for h5 saving
            history["epoch"].append(epoch)
            history["norm_J"].append(norm_J)
            history["train_loss"].append(train_loss)
            history["learning_rate"].append(learning_rate)
            history["vali_loss"].append(vali_loss)
            history["vali_loss_gen"].append(vali_loss_gen)
            history["vali_loss_max"].append(vali_loss_max)
            history["vali_loss_gen_max"].append(vali_loss_gen_max)
            history["x_norm"].append(x_norm)
            history["asymmetry"].append(asymmetry)
            history["diff_hebb"].append(diff_Hebb)

            # Save checkpoints with h5py
            if epoch % save_every == 0 and save is True:
                next_save_idx = save_training(
                    h5_path=h5_path,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    history=history,
                    save_idx=next_save_idx,
                )
        # Decay learning rate every epoch
        learning_rate *= factor_lr_decay
        for pg in optimizer.param_groups:
            pg["lr"] = learning_rate

        epoch += 1

    #############################################

    epoch -= 1  # last epoch value
    # Final evaluation after training
    model.eval()
    vali_loss, vali_loss_max = compute_validation_overlap(
        model=model, dataloader=dataloader, device=device,
        init_overlap=init_overlap, n=extra_steps,
    )
    vali_loss_gen, vali_loss_gen_max = compute_validation_overlap(
        model=model, dataloader=dataloader_gen, device=device,
        init_overlap=init_overlap, n=extra_steps,
    )

    
    # compute x_norm again for this final evaluation (first batch)
    with torch.no_grad():
        for batch_element in dataloader:
            inp_data = batch_element.to(device)
            x_new = inp_data.clone()
            for _ in range(extra_steps):
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
    history["vali_loss_gen"].append(vali_loss_gen)
    history["vali_loss_max"].append(vali_loss_max)
    history["vali_loss_gen_max"].append(vali_loss_gen_max)
    history["x_norm"].append(x_norm)
    history["asymmetry"].append(asymmetry)
    history["diff_hebb"].append(diff_Hebb)

    
    print(epoch, norm_J, train_loss, learning_rate,
          vali_loss, vali_loss_gen,
          vali_loss_max, vali_loss_gen_max, x_norm)

    
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


def load_data(data_file, P, N, d, skip=3):
    if data_file is not None:
        print("Loading data from file:", data_file)
        data_tensor = torch.zeros(P, N, d)
        with open(data_file, 'r') as f:
            for _ in range(skip):
                next(f)  # Skip header lines
            all_lines = f.readlines()
            if len(all_lines) < P:
                raise ValueError(f"Data file contains only {len(all_lines)} patterns, but P={P} was requested.")
            random_indices = np.random.choice(len(all_lines), P, replace=False)
            for i in range(P):
                line = all_lines[random_indices[i]]
                values = list(map(float, line.strip().split()))
                if len(values) != N * d:
                    raise ValueError(f"Line {i+1} in data file does not have the correct number of values.")
                data_tensor[i] = torch.tensor(values).view(N, d)
        return data_tensor
    else:
        return None


def main(N, P, l, d, spin_type, init_overlap, n, device, data_PATH, epochs, learning_rate, 
         max_grad, valid_every, P_generalization, save_every=10, data_file=None, save=False,
         seed=42, extra_steps=0, factor_lr_decay=0.999, factor_lr_diminish_when_error=0.9, 
         patience_lr=50, factor_J_diminish_when_error=0.9):
    if P_generalization is None:
        P_generalization = P
    data_train = load_data(data_file, P, N, d)
    data_test = load_data(data_file, P_generalization, N, d)
    print("N={}, P={}, lambda={}, d={}".format(N, P, l, d))
    model_name_base = "PseudoLikelihood_N_{}_P_{}_seed_{}_l_{}_d_{}_epochs_{}_lr_{}_spin_{}".format(N, P, seed, l, d, epochs, learning_rate, spin_type)

    torch.cuda.empty_cache()
    gc.collect()

    dataset, dataset_gen, model, optimizer = initialize(N=N, P=P, P_generalization=P_generalization, d=d, lr=learning_rate, spin_type=spin_type, 
                                                        device=device, data_train=data_train, data_test=data_test)
    
    batch_size = P
    batch_size_gen = P_generalization


    model2 = TwoBodiesModel(N, d, spin_type=spin_type, device=device)
    model2.to(device)
    model2.Hebb(dataset.xi, 'Tensorial')  # Applying the Hebb rule
    J2 = model2.J.squeeze().cpu().detach().numpy()
    norm_J2 = np.linalg.norm(J2)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=2)
    dataloader_gen = torch.utils.data.DataLoader(dataset_gen, batch_size=batch_size_gen, shuffle=False, drop_last=False, num_workers=2)
    

    print("epochs:{} lr:{} max_norm:{} init_overlap:{} n:{} l:{}".format(epochs, learning_rate, max_grad, init_overlap, n, l))

    # Train the model
    train_model(
        model, dataloader, dataloader_gen, epochs, 
        learning_rate, max_grad, device, data_PATH, init_overlap, 
        n, l, optimizer, J2, norm_J2, valid_every, save_every, model_name_base, 
        save, extra_steps=extra_steps, factor_lr_decay=factor_lr_decay, 
        factor_lr_diminish_when_error=factor_lr_diminish_when_error, patience_lr=patience_lr, 
        factor_J_diminish_when_error=factor_J_diminish_when_error
    )


def parse_arguments():
    parser = argparse.ArgumentParser(description="Training GD")

    # Define all the parameters
    parser.add_argument("--N", type=int, required=True, help="Number of sites")
    parser.add_argument("--P", type=int, required=True, help="Number of patterns")
    parser.add_argument("--l", type=float, required=True, help="lambda: inverse temperature")
    parser.add_argument("--d", type=int, default=1, help="Dimensionality of each site")
    parser.add_argument("--spin_type", type=str, default="vector", help="Type of spins: 'vector' or 'continuous'")
    parser.add_argument("--init_overlap", type=float, default=1.0, help="Initial overlap for validation")
    parser.add_argument("--n", type=int, default=10, help="Number of dynamics steps for validation")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use: 'cpu' or 'cuda'")
    parser.add_argument("--data_PATH", type=str, default="savings", help="Path to save training data")
    parser.add_argument("--epochs", type=int, default=401, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=10., help="Initial learning rate")
    parser.add_argument("--max_grad", type=float, default=20., help="Maximum gradient norm for clipping")
    parser.add_argument("--valid_every", type=int, default=10, help="Frequency of validation during training")
    parser.add_argument("--P_generalization", type=int, default=None, help="Batch size for generalization dataset")
    parser.add_argument("--save", action='store_true', help="Whether to save the training progress") 
    parser.add_argument("--save_every", type=int, default=10, help="Frequency at which to save the model during training")
    parser.add_argument("--data_file", type=str, default=None, help="Path to data file for predefined patterns")
    parser.add_argument("--seed", type=int, default=444, help="Seed for random number generators")
    parser.add_argument("--extra_steps", type=int, default=0, help="Extra steps to perform after training")
    parser.add_argument("--factor_lr_decay", type=float, default=1.0, help="Factor for learning rate decay each epoch")
    parser.add_argument("--factor_lr_diminish_when_error", type=float, default=0.9, help="Factor to diminish learning rate when error detected")
    parser.add_argument("--patience_lr", type=int, default=50, help="Patience for learning rate adjustment")
    parser.add_argument("--factor_J_diminish_when_error", type=float, default=0.9, help="Factor to diminish J when error detected")

    return parser.parse_args()



if __name__ == "__main__":
    print("GPU available:", torch.cuda.is_available())

    args = parse_arguments()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Run the main function with the parsed arguments
    main(args.N, args.P, args.l, args.d, args.spin_type, args.init_overlap, args.n, 
         args.device, args.data_PATH, args.epochs, args.learning_rate, args.max_grad, 
         args.valid_every, args.P_generalization, save_every=args.save_every, 
         data_file=args.data_file, save=args.save, seed=args.seed, extra_steps=args.extra_steps, 
         factor_lr_decay=args.factor_lr_decay, factor_lr_diminish_when_error=args.factor_lr_diminish_when_error, 
         patience_lr=args.patience_lr, factor_J_diminish_when_error=args.factor_J_diminish_when_error)
