import math
import torch
from torch.utils.data import Dataset
from PL.model.model_classifier import Classifier

class Dataset_Teacher(Dataset):
    """
    Teacher-driven dataset for generic d, with a local teacher tensor per site:
        T: [N, d, d]
    Labels are produced as a "single spin" object:
        y_mu,i = g( T_i @ xi_mu,i + noise )
    where g enforces the same nature as xi (vector/continuous).

    Generation of xi is aligned with RandomFeaturesDataset:
      - base Gaussian xi ~ N(0, sigma^2)
      - optional Random Features construction xi = sum_k c_{mu,k} f_k
      - for spin_type="vector" normalize spins on S^{d-1} (d=1 -> ±1)
      - coefficients ("binary" or "gaussian") independent of spin_type
      - sparsification via L active features per pattern
    """

    def __init__(
        self,
        P: int,
        N: int,
        d: int,
        seed: int,
        sigma: float,
        spin_type: str = "continuous",        # "vector" or "continuous" 
    ):
        self.P = P
        self.N = N
        self.d = d
        self.sigma = sigma
        self.spin_type = spin_type
        torch.manual_seed(seed)
        # Isotropic: Gaussian then normalized to fixed Frobenius norm.
        self.T = torch.randn(N, d, d)  #[N,d,d]
        teacher_norm2 = float(N * d)
        self.Teacher = self._normalize_frobenius(self.T, target_norm2=teacher_norm2)
        self.xi = torch.randn(P, N, d) * sigma #[P,N,d]

        if self.spin_type == "vector":
            self.xi = self.xi*math.sqrt(self.d)/torch.norm(self.xi, dim=-1, keepdim=True)

        self.y = self._make_labels(self.xi) #[P,d]

    def _make_labels(self, xi: torch.Tensor) -> torch.Tensor:
        """
        xi: [P,N,d]
        returns y: [P,d] with same "nature" as xi
        """
        # local linear map per site: h_{p,i,a} = sum_b T_{i,a,b} xi_{p,i,b}
        h = torch.einsum("iab,pib->pa", self.Teacher, xi)

        # enforce same nature as a single spin of xi
        if self.spin_type == "vector":
            # for d=1: this becomes sign(h) (up to numerical eps)
            y = self.normalize(h)
        elif self.spin_type == "continuous":
            y = h
        else:
            raise ValueError("spin_type must be 'vector' or 'continuous'")

        return y

    @staticmethod
    def normalize(x: torch.Tensor) -> torch.Tensor:
        norms = x.norm(dim=-1,keepdim=True) + 1e-9
        return x / norms

    @staticmethod
    def _normalize_frobenius(T: torch.Tensor, target_norm2: float) -> torch.Tensor:
        """
        Enforce sum_{i,a,b} T[i,a,b]^2 == target_norm2 exactly.
        """
        current = (T * T).sum()
        scale = math.sqrt(float(target_norm2) / float(current + 1e-12))
        return T * scale

    def __len__(self) -> int:
        return self.P

    def __getitem__(self, index: int):
        return self.xi[index], self.y[index]


import torch

def error_generalization_dictionary(
    model: "Classifier",
    dataset: "Dataset_Teacher",
    D: int,
    d: int,
    test_xi: torch.Tensor,   # shape [P_test, N, d]
) -> torch.Tensor:
    """
    Dictionary-based generalization error:
    - build a random dictionary of D vectors in R^d with norm sqrt(d)
    - compute student and teacher labels on test_xi
    - for each test sample, pick the dictionary atom with max cosine similarity
    - return a tensor [P_test] with 0 if argmax matches, 1 otherwise
    """
    if test_xi.ndim != 3:
        raise ValueError(f"test_xi must have shape [P_test, N, d], got {tuple(test_xi.shape)}")
    if test_xi.shape[-1] != d:
        raise ValueError(f"test_xi last dim must be d={d}, got {test_xi.shape[-1]}")
    if D <= 0:
        raise ValueError("D must be > 0")

    device = test_xi.device
    dtype = test_xi.dtype

    # 1) Generate D random vectors in R^d with norm exactly sqrt(d)
    dict_vecs = torch.randn(D, d, device=device, dtype=dtype)
    dict_vecs = dict_vecs / dict_vecs.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    dict_vecs = dict_vecs * (float(d) ** 0.5)  # each row has norm sqrt(d)

    # 2) Student labels y_hat via model.forward
    model_device = next(model.parameters()).device if any(True for _ in model.parameters()) else device
    xi_in = test_xi.to(model_device)
    y_student = model(xi_in)  # expected shape [P_test, d]

    # 3) Teacher labels from dataset.T (same einsum as in forward)
    T = dataset.T.to(model_device)  # expected shape [N, d, d] (i.e., "jab")
    u_teacher = torch.einsum("jab,mjb->ma", T, xi_in)

    if getattr(model, "spin_type", None) == "vector":
        # reuse model's normalization to match its conventions
        y_teacher = model.normalize_x(u_teacher)
    elif getattr(model, "spin_type", None) == "continuous":
        y_teacher = u_teacher.clone()
    else:
        raise ValueError(f"Unknown/unsupported model.spin_type={getattr(model,'spin_type',None)}")

    # Move dictionary to model device too (for the matmuls)
    dict_vecs = dict_vecs.to(model_device)

    # 4) Cosine similarity with all D dictionary vectors
    # cos(y, v) = (y·v) / (||y|| ||v||)
    y_student_n = y_student / y_student.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    y_teacher_n = y_teacher / y_teacher.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    dict_n = dict_vecs / dict_vecs.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    # similarities: [P_test, D]
    sim_student = y_student_n @ dict_n.T
    sim_teacher = y_teacher_n @ dict_n.T

    # 5) Argmax over D for each test point
    arg_student = sim_student.argmax(dim=1)  # [P_test]
    arg_teacher = sim_teacher.argmax(dim=1)  # [P_test]

    # 6) 0 if same argmax, 1 otherwise
    err = (arg_student != arg_teacher).to(torch.long)  # [P_test]

    # return on the same device as test_xi (optional; you can drop this if you want model_device)
    return err.to(device)
