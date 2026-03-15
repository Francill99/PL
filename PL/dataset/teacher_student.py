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
    """

    def __init__(
        self,
        P: int,
        N: int,
        d: int,
        seed: int,
        sigma: float,
        spin_type: str = "continuous",   # "vector" or "continuous"
        label_type: str = "continuous",
    ):
        self.P = P
        self.N = N
        self.d = d
        self.sigma = sigma
        self.spin_type = spin_type
        self.label_type = label_type

        torch.manual_seed(seed)

        # Teacher
        self.T = torch.randn(N, d, d)
        teacher_norm2 = float(N * d)
        self.Teacher = self._normalize_frobenius(self.T, target_norm2=teacher_norm2)

        # Training data
        self.xi = torch.randn(P, N, d) * sigma
        if self.spin_type == "vector":
            self.xi = self.xi * math.sqrt(self.d) / torch.norm(self.xi, dim=-1, keepdim=True)

        self.y = self._make_labels(self.xi)

    def _make_labels(self, xi: torch.Tensor) -> torch.Tensor:
        """
        xi: [P,N,d]
        returns y: [P,d]
        """
        h = torch.einsum("iab,pib->pa", self.Teacher, xi)

        if self.label_type == "vector":
            y = self.normalize(h)
        elif self.label_type == "continuous":
            y = h
        else:
            raise ValueError("label_type must be 'vector' or 'continuous'")

        return y

    def generate_test_set(self, P_test: int):
        """
        Generate fresh test data from the same teacher.

        Parameters
        ----------
        P_test : int
            Number of test samples.
        seed : int or None
            Optional seed for reproducibility.

        Returns
        -------
        xi_test : torch.Tensor
            Shape [P_test, N, d]
        y_test : torch.Tensor
            Shape [P_test, d]
        """

        xi_test = torch.randn(P_test, self.N, self.d) * self.sigma

        if self.spin_type == "vector":
            xi_test = xi_test * math.sqrt(self.d) / torch.norm(xi_test, dim=-1, keepdim=True)

        y_test = self._make_labels(xi_test)
        return xi_test, y_test

    @staticmethod
    def normalize(x: torch.Tensor) -> torch.Tensor:
        norms = x.norm(dim=-1, keepdim=True) + 1e-9
        return x / norms

    @staticmethod
    def _normalize_frobenius(T: torch.Tensor, target_norm2: float) -> torch.Tensor:
        current = (T * T).sum()
        scale = math.sqrt(float(target_norm2) / float(current + 1e-12))
        return T * scale

    def __len__(self) -> int:
        return self.P

    def __getitem__(self, index: int):
        return self.xi[index], self.y[index]