## Standard libraries
import os
import numpy as np
import argparse
import torch
import gc

## PyTorch
import torch
from torch.utils.data import DataLoader, TensorDataset

from PL.model.model_classifier import Classifier
from PL.dataset.teacher_student import Dataset_Teacher
from PL.dataset.dictionary import (
    power_law_covariance_eigenvalues,
    random_dictionary,
    resolve_dictionary_type,
)
from PL.utils.saving import init_training_h5, save_training
from PL.utils.functions import overlap

METRIC_NAMES = [
    "epoch",
    "norm_J",
    "train_loss",
    "train_accuracy",
    "train_mse",
    "test_loss",
    "test_accuracy",
    "test_mse",
    "R",
    "learning_rate",
    "diff_hebb",
]


def load_dictionary_from_path(path: str) -> torch.Tensor:
    """Load an external dictionary tensor from .pt/.pth or .npy."""
    if path is None:
        return None

    if path.endswith(".npy"):
        dictionary = torch.from_numpy(np.load(path))
    else:
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, torch.Tensor):
            dictionary = obj
        elif isinstance(obj, dict):
            for key in ("dictionary", "dict", "dict_vecs", "vectors", "atoms"):
                if key in obj and isinstance(obj[key], torch.Tensor):
                    dictionary = obj[key]
                    break
            else:
                raise ValueError(
                    "Dictionary file is a dict, but no tensor was found under one of: "
                    "'dictionary', 'dict', 'dict_vecs', 'vectors', 'atoms'."
                )
        else:
            raise TypeError(f"Unsupported dictionary object type: {type(obj)!r}")

    return dictionary.float()


def build_dictionary_if_needed(
    spin_type: str,
    label_type: str,
    d: int,
    dictionary_path: str | None,
    dictionary_size: int | None,
    dictionary_on_the_sphere: bool | None = None,
    dictionary_sigma: float = 1.0,
    dictionary_seed: int | None = None,
    dictionary_type: str | None = "spherical",
    dictionary_eta: float = 0.0,
) -> torch.Tensor | None:
    """
    Return a dictionary tensor only when an input or output dictionary is needed.

    Generated dictionaries support two distributions:
      - ``spherical`` (default): unit-norm vectors, preserving previous behavior;
      - ``gaussian``: vectors from N(0, sigma^2 Sigma_eta), without any
        per-vector normalization. Sigma_eta has power-law eigenvalues with
        trace d; eta=0 gives the identity covariance.

    ``dictionary_on_the_sphere`` is kept only for backward compatibility with
    older Python calls. When it is not None, it maps True/False to
    spherical/Gaussian respectively.
    """
    uses_dictionary = (spin_type == "dictionary") or (label_type == "dictionary")
    if not uses_dictionary:
        return None

    resolved_type = resolve_dictionary_type(
        dictionary_type,
        on_the_sphere=dictionary_on_the_sphere,
    )

    if dictionary_eta < 0:
        raise ValueError(
            f"dictionary_eta must be non-negative, got dictionary_eta={dictionary_eta}."
        )
    if dictionary_path is not None and float(dictionary_eta) != 0.0:
        raise ValueError(
            "dictionary_eta applies only to generated dictionaries; it cannot "
            "be combined with dictionary_path."
        )
    if resolved_type == "spherical" and float(dictionary_eta) != 0.0:
        raise ValueError(
            "A non-zero dictionary_eta requires dictionary_type='gaussian'."
        )

    if dictionary_path is not None:
        # External dictionaries are used exactly as stored: no normalization.
        dictionary = load_dictionary_from_path(dictionary_path)
    elif dictionary_size is not None:
        dictionary = random_dictionary(
            dictionary_size,
            d,
            sigma=dictionary_sigma,
            dictionary_type=resolved_type,
            dictionary_eta=dictionary_eta,
            seed=dictionary_seed,
        )
    else:
        raise ValueError(
            "A dictionary is required when spin_type='dictionary' or label_type='dictionary'. "
            "Pass either --dictionary_path or --dictionary_size."
        )

    if dictionary.ndim != 2 or dictionary.shape[1] != d:
        raise ValueError(
            f"dictionary must have shape [D, d] with d={d}; got {tuple(dictionary.shape)}."
        )
    if label_type == "dictionary" and d <= 1:
        raise ValueError("label_type='dictionary' is only supported for d > 1.")

    return dictionary


def initialize(
    N=1000,
    P=400,
    d=1,
    lr=0.1,
    spin_type="vector",
    label_type="vector",
    dictionary=None,
    device="cuda",
    gamma=1.0,
    init_Hebb=True,
    downf=1.0,
    seed=444,
    optimizer_type="SGD",
):
    """
    Build teacher dataset, classifier model, and optimizer.

    The same dictionary can be used independently for input spins, output labels,
    or both, depending on spin_type and label_type.
    """
    dataset = Dataset_Teacher(
        P,
        N,
        d,
        seed=seed,
        sigma=0.5,
        spin_type=spin_type,
        label_type=label_type,
        dictionary=dictionary,
    )

    model = Classifier(
        N,
        d,
        gamma=gamma,
        spin_type=spin_type,
        label_type=label_type,
        dictionary=dictionary,
        downf=downf,
        device=device,
    )
    model.to(device)

    if optimizer_type == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    elif optimizer_type == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0)
    elif optimizer_type == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")

    if init_Hebb:
        model.Hebb(dataset.xi, dataset.y, "Tensorial")

    return dataset, model, optimizer


@torch.no_grad()
def _dictionary_accuracy_from_field(model, xi, y):
    """
    Exact dictionary-label accuracy using raw, unnormalized prediction scores.

    Prediction:
        argmax_k u(x) · v_k, with u(x)=Jx.

    The target ``y`` is already one dictionary atom. Its index must therefore be
    recovered by matching it to the stored atoms, not by ``argmax_k y·v_k``.
    The latter is only safe for equal-norm dictionaries and is wrong in general
    for Gaussian dictionaries, where another longer atom can have a larger dot
    product with ``y`` than ``y`` has with itself.
    """
    if getattr(model, "dictionary", None) is None:
        raise ValueError("model.dictionary is required when label_type='dictionary'.")

    dictionary = model.dictionary.to(device=xi.device, dtype=xi.dtype)
    J = model.J.to(device=xi.device, dtype=xi.dtype)
    u = torch.einsum("jab,mjb->ma", J, xi)

    # y is exactly one row of the dictionary. Squared-distance matching recovers
    # its class independently of dictionary norms.
    y_sq = (y * y).sum(dim=-1, keepdim=True)
    d_sq = (dictionary * dictionary).sum(dim=-1).unsqueeze(0)
    distances_sq = y_sq + d_sq - 2.0 * (y @ dictionary.T)
    target_idx = distances_sq.argmin(dim=-1)

    # The model prediction remains the required raw-overlap argmax.
    pred_idx = (u @ dictionary.T).argmax(dim=-1)
    return (target_idx == pred_idx).float().mean()


@torch.no_grad()
def _batch_accuracy(model, xi, y, y_pred):
    if getattr(model, "label_type", None) == "dictionary":
        return _dictionary_accuracy_from_field(model, xi, y)
    return overlap(y, y_pred).mean()


@torch.no_grad()
def _teacher_alignment(dataset, model, device):
    teacher = getattr(dataset, "Teacher", getattr(dataset, "T", None))
    if teacher is None:
        return float("nan")
    teacher = teacher.to(device=device, dtype=model.J.dtype)
    denom = teacher.norm() * model.J.norm()
    if float(denom.detach().cpu()) == 0.0:
        return float("nan")
    return (torch.einsum("iab,iab->", teacher, model.J) / denom).detach().cpu().item()


@torch.no_grad()
def evaluate_dataset(model, dataloader, device, lambd, loss_type="CE", l2=None):
    """
    Evaluate loss, accuracy, and MSE on a dataloader.

    For dictionary labels:
      - accuracy is 1[argmax_k v_k·Jx == argmax_k v_k·y]
      - MSE is mean((model(x)-y)^2), where model(x) is the predicted dictionary atom.
    """
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_mse = 0.0
    total_n = 0

    for xi, y in dataloader:
        xi = xi.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        loss = model.loss(xi, y, lambd=lambd, loss_type=loss_type, l2=l2)
        y_pred = model(xi)
        acc = _batch_accuracy(model, xi, y, y_pred)
        mse = ((y_pred - y) ** 2).mean()

        b = xi.shape[0]
        total_loss += float(loss.detach().cpu()) * b
        total_acc += float(acc.detach().cpu()) * b
        total_mse += float(mse.detach().cpu()) * b
        total_n += b

    denom = max(total_n, 1)
    return total_loss / denom, total_acc / denom, total_mse / denom


def train_model(
    model,
    fixed_norm,
    dataset,
    dataloader,
    test_dataloader,
    epochs,
    learning_rate,
    max_grad,
    device,
    data_PATH,
    l,
    optimizer,
    J2,
    norm_J2,
    valid_every,
    epochs_to_save,
    model_name_base,
    save,
    l2,
    loss_type,
    verbose=True,
):
    history = {name: [] for name in METRIC_NAMES}

    print("# epoch norm train_loss train_accuracy train_mse test_loss test_accuracy test_mse learning_rate R")
    os.makedirs(data_PATH, exist_ok=True)

    h5_path = os.path.join(data_PATH, model_name_base + ".h5")
    init_training_h5(h5_path, model, METRIC_NAMES, optimizer)
    next_save_idx = 1

    epoch = 0

    # Initial validation before training.
    train_loss, train_acc, train_mse = evaluate_dataset(model, dataloader, device, l, loss_type=loss_type, l2=l2)
    test_loss, test_acc, test_mse = evaluate_dataset(model, test_dataloader, device, l, loss_type=loss_type, l2=l2)
    R = _teacher_alignment(dataset, model, device)
    norm_J = (torch.norm(model.J) / torch.sqrt(torch.tensor(model.J.shape[-1], device=model.J.device))).detach().cpu().item()
    diff_Hebb = np.nan

    history["epoch"].append(0)
    history["norm_J"].append(norm_J)
    history["train_loss"].append(train_loss)
    history["train_accuracy"].append(train_acc)
    history["train_mse"].append(train_mse)
    history["test_loss"].append(test_loss)
    history["test_accuracy"].append(test_acc)
    history["test_mse"].append(test_mse)
    history["learning_rate"].append(learning_rate)
    history["R"].append(R)
    history["diff_hebb"].append(diff_Hebb)

    if verbose:
        print(0, norm_J, train_loss, train_acc, train_mse, test_loss, test_acc, test_mse, learning_rate, R)

    for epoch in range(1, epochs + 1):
        model.train()

        for batch_element in dataloader:
            xi, y = batch_element
            xi = xi.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            loss = model.loss(xi, y, lambd=l, loss_type=loss_type, l2=l2)

            if torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                
                if fixed_norm is True:
                    model.project_J_gradient_()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad)
                optimizer.step()
            else:
                print(f"Detected NaN/Inf {model_name_base} epoch {epoch} lr {learning_rate}")
                with torch.no_grad():
                    model.J.data *= 0.1
                learning_rate *= 0.1
                for pg in optimizer.param_groups:
                    pg["lr"] = learning_rate

        if (epoch % valid_every == 0) or (epoch == epochs):
            train_loss, train_acc, train_mse = evaluate_dataset(model, dataloader, device, l, loss_type=loss_type, l2=l2)
            test_loss, test_acc, test_mse = evaluate_dataset(model, test_dataloader, device, l, loss_type=loss_type, l2=l2)
            R = _teacher_alignment(dataset, model, device)

            J = model.J.detach().cpu().numpy()
            norm_J = (torch.norm(model.J) / torch.sqrt(torch.tensor(model.J.shape[-1], device=model.J.device))).detach().cpu().item()
            diff_Hebb = np.linalg.norm(J2 * norm_J / (norm_J2 + 1e-12) - J) / (norm_J + 1e-12)

            if verbose:
                print(epoch, norm_J, train_loss, train_acc, train_mse, test_loss, test_acc, test_mse, learning_rate, R)

            history["epoch"].append(epoch)
            history["norm_J"].append(norm_J)
            history["train_loss"].append(train_loss)
            history["train_accuracy"].append(train_acc)
            history["train_mse"].append(train_mse)
            history["test_loss"].append(test_loss)
            history["test_accuracy"].append(test_acc)
            history["test_mse"].append(test_mse)
            history["learning_rate"].append(learning_rate)
            history["R"].append(R)
            history["diff_hebb"].append(diff_Hebb)

            if (epoch in epochs_to_save) and save is True:
                next_save_idx = save_training(
                    h5_path=h5_path,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    history=history,
                    save_idx=next_save_idx,
                )

    if save is True:
        next_save_idx = save_training(
            h5_path=h5_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            history=history,
            save_idx=next_save_idx,
        )

    return history


def main(
    N,
    alpha_P,
    l,
    d,
    spin_type,
    label_type,
    batch_size,
    device,
    data_PATH,
    epochs,
    learning_rate,
    valid_every,
    max_grad,
    loss_type,
    dictionary_path=None,
    dictionary_size=None,
    dictionary_on_the_sphere=None,
    dictionary_sigma=1.0,
    dictionary_seed=None,
    seed=444,
    test_seed=None,
    P_test=1000,
    gamma=1.0,
    downf=1.0,
    optimizer_type="SGD",
    fixed_norm=False,
    init_Hebb=True,
    save=False,
    l2=None,
    dictionary_type=None,
    dictionary_eta=0.0,
):
    P = int(alpha_P * N)
    if batch_size is None:
        batch_size = P

    dictionary = build_dictionary_if_needed(
        spin_type=spin_type,
        label_type=label_type,
        d=d,
        dictionary_path=dictionary_path,
        dictionary_size=dictionary_size,
        dictionary_on_the_sphere=dictionary_on_the_sphere,
        dictionary_sigma=dictionary_sigma,
        dictionary_seed=dictionary_seed,
        dictionary_type=dictionary_type,
        dictionary_eta=dictionary_eta,
    )
    D_dict = None if dictionary is None else dictionary.shape[0]
    resolved_dictionary_type = resolve_dictionary_type(
        dictionary_type,
        on_the_sphere=dictionary_on_the_sphere,
    )
    if dictionary is None:
        dictionary_source = None
    else:
        dictionary_source = "external" if dictionary_path is not None else resolved_dictionary_type

    print(
        f"P={P}, P_test={P_test}, lambda={l}, spin_type={spin_type}, "
        f"label_type={label_type}, D_dict={D_dict}, dictionary_type={dictionary_source}, "
        f"dictionary_eta={dictionary_eta if dictionary_source == 'gaussian' else None}"
    )
    if dictionary is not None:
        mean_norm = dictionary.norm(dim=-1).mean().item()
        empirical_coordinate_variance = dictionary.var(dim=0, unbiased=False)
        mean_coordinate_variance = empirical_coordinate_variance.mean().item()
        print(
            f"dictionary mean_norm={mean_norm:.6g}, "
            f"mean_coordinate_variance={mean_coordinate_variance:.6g}"
        )
        if dictionary_path is None and dictionary_source == "gaussian":
            target_eigenvalues = (
                dictionary_sigma ** 2
                * power_law_covariance_eigenvalues(
                    d, eta=dictionary_eta, dtype=dictionary.dtype
                )
            )
            print(
                f"dictionary covariance_trace={target_eigenvalues.sum().item():.6g}, "
                f"lambda_max={target_eigenvalues.max().item():.6g}, "
                f"lambda_min={target_eigenvalues.min().item():.6g}"
            )

    # Keep the historical spherical filename exactly unchanged. Only the new
    # generated Gaussian case receives an explicit suffix.
    model_name_base = (
        f"classifier_N_{N}_P_{P}_Ptest_{P_test}_l_{l}_d_{d}_epochs_{epochs}_lr_{learning_rate}"
        f"_spin_{spin_type}_label_{label_type}_Ddict_{D_dict}_seed_{seed}"
    )
    if dictionary is not None and dictionary_path is None and dictionary_source == "gaussian":
        model_name_base += "_dict_gaussian"
        if float(dictionary_eta) != 0.0:
            model_name_base += f"_eta_{dictionary_eta:g}"

    torch.cuda.empty_cache()
    gc.collect()

    dataset, model, optimizer = initialize(
        N=N,
        P=P,
        d=d,
        lr=learning_rate,
        spin_type=spin_type,
        label_type=label_type,
        dictionary=dictionary,
        device=device,
        gamma=gamma,
        init_Hebb=init_Hebb,
        downf=downf,
        seed=seed,
        optimizer_type=optimizer_type,
    )

    # Fresh test data from the same teacher and the same dictionary.
    if test_seed is None:
        test_seed = int(seed) + 1000003
    torch.manual_seed(test_seed)
    xi_test, y_test = dataset.generate_test_set(P_test)
    test_dataset = TensorDataset(xi_test, y_test)

    model2 = Classifier(
        N,
        d,
        gamma=gamma,
        spin_type=spin_type,
        label_type=label_type,
        dictionary=dictionary,
        device=device,
    )
    model2.to(device)
    model2.Hebb(dataset.xi, dataset.y, "Tensorial")
    J2 = model2.J.detach().cpu().numpy()
    norm_J2 = np.linalg.norm(J2)

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=2)
    eval_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=2)
    test_batch_size = min(batch_size, P_test)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, drop_last=False, num_workers=2)

    epochs_to_save = [10000]
    print(f"epochs:{epochs} lr:{learning_rate} max_norm:{max_grad} l:{l}")

    return train_model(
        model=model,
        fixed_norm=fixed_norm,
        dataset=dataset,
        dataloader=eval_loader,
        test_dataloader=test_loader,
        epochs=epochs,
        learning_rate=learning_rate,
        max_grad=max_grad,
        device=device,
        data_PATH=data_PATH,
        l=l,
        optimizer=optimizer,
        J2=J2,
        norm_J2=norm_J2,
        valid_every=valid_every,
        epochs_to_save=epochs_to_save,
        model_name_base=model_name_base,
        save=save,
        l2=l2,
        loss_type=loss_type,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training GD for supervised teacher-student classifier")

    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--alpha_P", type=float, required=True)
    parser.add_argument("--l", type=float, required=True)
    parser.add_argument("--d", type=int, default=1)
    parser.add_argument("--spin_type", type=str, default="vector", choices=["vector", "continuous", "dictionary"])
    parser.add_argument("--label_type", type=str, default="vector", choices=["vector", "continuous", "dictionary"])
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--data_PATH", type=str, default="savings")
    parser.add_argument("--epochs", type=int, default=401)
    parser.add_argument("--learning_rate", type=float, default=10.0)
    parser.add_argument("--max_grad", type=float, default=20.0)
    parser.add_argument("--valid_every", type=int, default=10)
    parser.add_argument("--loss_type", type=str, default="CE", choices=["CE", "MSE"])

    parser.add_argument("--P_test", type=int, default=1000)
    parser.add_argument("--test_seed", type=int, default=None)

    parser.add_argument("--dictionary_path", type=str, default=None)
    parser.add_argument("--dictionary_size", type=int, default=None)
    parser.add_argument(
        "--dictionary_type",
        type=str,
        default="spherical",
        choices=["spherical", "gaussian"],
        help=(
            "Distribution used to generate the dictionary. 'spherical' is the "
            "backward-compatible default; 'gaussian' draws vectors from "
            "N(0, sigma^2 Sigma_eta) without normalization."
        ),
    )
    parser.add_argument(
        "--dictionary_off_sphere",
        action="store_true",
        help="Deprecated alias for --dictionary_type gaussian.",
    )
    parser.add_argument(
        "--dictionary_sigma",
        type=float,
        default=1.0,
        help=(
            "Global scale of Gaussian atoms. Their covariance is "
            "sigma^2 Sigma_eta; sigma=1 uses trace d."
        ),
    )
    parser.add_argument(
        "--dictionary_eta",
        type=float,
        default=0.0,
        help=(
            "Power-law exponent of the Gaussian covariance eigenvalues: "
            "lambda_m = d m^{-eta} / sum_j j^{-eta}. "
            "The default eta=0 gives Sigma=I_d. Requires "
            "--dictionary_type gaussian when non-zero."
        ),
    )
    parser.add_argument("--dictionary_seed", type=int, default=None)

    parser.add_argument("--seed", type=int, default=444)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--downf", type=float, default=1.0)
    parser.add_argument("--optimizer_type", type=str, default="SGD", choices=["SGD", "Adam", "AdamW"])
    parser.add_argument("--fixed_norm", action="store_true")
    parser.add_argument("--no_init_Hebb", action="store_true")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--l2", type=float, default=None)

    args = parser.parse_args()

    # Preserve the old flag while making the explicit type the primary API.
    if args.dictionary_off_sphere:
        args.dictionary_type = "gaussian"

    main(
        N=args.N,
        alpha_P=args.alpha_P,
        l=args.l,
        d=args.d,
        spin_type=args.spin_type,
        label_type=args.label_type,
        batch_size=args.batch_size,
        device=args.device,
        data_PATH=args.data_PATH,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        valid_every=args.valid_every,
        max_grad=args.max_grad,
        loss_type=args.loss_type,
        P_test=args.P_test,
        test_seed=args.test_seed,
        dictionary_path=args.dictionary_path,
        dictionary_size=args.dictionary_size,
        dictionary_on_the_sphere=None,
        dictionary_sigma=args.dictionary_sigma,
        dictionary_seed=args.dictionary_seed,
        dictionary_type=args.dictionary_type,
        dictionary_eta=args.dictionary_eta,
        seed=args.seed,
        gamma=args.gamma,
        downf=args.downf,
        optimizer_type=args.optimizer_type,
        fixed_norm=args.fixed_norm,
        init_Hebb=not args.no_init_Hebb,
        save=args.save,
        l2=args.l2,
    )
