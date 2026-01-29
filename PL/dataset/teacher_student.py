import math
import torch
from torch.utils.data import Dataset

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
