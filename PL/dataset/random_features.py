## Standard libraries
import math
import torch

## PyTorch
import torch
import torch.utils.data as data
from torch.utils.data import Dataset



class BasicDataset(Dataset):
    def __init__(
        self,
        P, N, d,
        seed=None,
        sigma=1.0,
        spin_type="continuous",        # "vector" or "continuous"
        coefficients="binary",
        xi=None,
    ):
        """
        P: Number of patterns
        N: Number of sites
        d: Dimensionality of each site
        sigma: Std of Gaussian noise
        spin_type: "vector" (O(d) spins on unit sphere) or "continuous"
                   (unconstrained real spins). Binary Ising is the
                   special case spin_type="vector", d=1.
        coefficients: "binary" or "gaussian" (for c), independent of spin type.
        xi: Optional pre-defined patterns (P x N x d tensor)
        """
        self.P = P
        self.N = N
        self.d = d
        self.spin_type = spin_type
        self.coefficients = coefficients
        ...

        if seed is not None:
            torch.manual_seed(seed)

        if xi is not None:
            # Take provided patterns
            self.xi = xi
        else:    
            # Fill with class default
            self.xi = self.random_patterns(P, N, d, sigma)

        # Vectorial case: normalize spins on S^{d-1}
        if self.spin_type == "vector":
            self.xi = self.normalize(self.xi)


    def random_patterns(self, P, N, d, sigma):
        # Base patterns xi ~ Gaussian
        return torch.randn(P, N, d) * sigma


    def normalize(self, x):
        # Normalize each d-dimensional vector in x along the last dimension
        norms = x.norm(dim=-1, keepdim=True)+1e-9
        return x / norms

    def __len__(self):
        # Return the number of patterns P
        return self.P

    def __getitem__(self, index):
        # Return the pattern xi at the given index
        return self.xi[index]


class RandomFeaturesDataset(BasicDataset):
    def __init__(
        self,
        P, N, d,
        sigma=1.0,
        seed=None,
        spin_type="continuous",        # "vector" or "continuous"
        coefficients="binary",
        D=0,
        L=None,
    ):
        """
        P: Number of patterns
        N: Number of sites
        D: Number of random features
        d: Dimensionality of each site
        sigma: Std of Gaussian noise
        spin_type: "vector" (O(d) spins on unit sphere) or "continuous"
                   (unconstrained real spins). Binary Ising is the
                   special case spin_type="vector", d=1.
        coefficients: "binary" or "gaussian" (for c), independent of spin type.
        xi: Optional pre-defined patterns (P x N x d tensor)
        """
        super().__init__(P, N, d, sigma=sigma, seed=seed, spin_type=spin_type, coefficients=coefficients)
        self.D = D
        self.L = L if L is not None else D
        self.sigma = sigma
        
        # If D > 0, build xi from random features
        if self.D > 0:
            self.RF(seed=seed)


    def RF(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
    
        # --- Features f: depend on spin_type, not on coefficients ---
        if self.spin_type == "vector":
            # Vectorial spins: Gaussian then normalized → O(d), and for d=1 gives binary ±1
            self.f = torch.randn(self.D, self.N, self.d) * self.sigma
            self.f = self.normalize(self.f)
        elif self.spin_type == "continuous":
            # Continuous spins: unconstrained Gaussian
            self.f = torch.randn(self.D, self.N, self.d) * self.sigma
        else:
            raise ValueError("spin_type must be 'vector' or 'continuous'")
    
        # --- Coefficients c: STILL 'binary' or 'gaussian', independent of spin_type ---
        if self.coefficients == "binary":
            self.c = torch.randint(0, 2, (self.P, self.D)).float() * 2 - 1
        elif self.coefficients == "gaussian":
            self.c = torch.randn(self.P, self.D)
        else:
            raise ValueError("coefficients must be 'binary' or 'gaussian'")
    
        self.c = self.c / math.sqrt(self.D)
    
        # Zero out D-L features per pattern
        indices_to_zero = torch.rand(self.P, self.D)
        indices_to_zero = indices_to_zero.argsort(dim=1)[:, : self.D - self.L]
        self.c = self.c.scatter(1, indices_to_zero, 0)
    
        # Build xi from random features
        self.xi = torch.einsum('pk,kia->pia', self.c, self.f)
    
        # Fill zeros in xi with *Gaussian* noise, then (if needed) renormalize
        mask = self.xi == 0
        filler = torch.randn_like(self.xi) * self.sigma
        self.xi = torch.where(mask, filler, self.xi)
    
        if self.spin_type == "vector":
            self.xi = self.normalize(self.xi)


    def get_generalization(self, P_hat):
        # c stays 'binary' or 'gaussian', independent of spin_type
        if self.coefficients == "binary":
            c = torch.randint(0, 2, (P_hat, self.D)).float() * 2 - 1
        elif self.coefficients == "gaussian":
            c = torch.randn(P_hat, self.D)
        else:
            raise ValueError("coefficients must be 'binary' or 'gaussian'")

        c = c / math.sqrt(self.D)

        # Same sparsification logic as in RF
        indices_to_zero = torch.rand(P_hat, self.D)
        indices_to_zero = indices_to_zero.argsort(dim=1)[:, : self.D - self.L]
        c = c.scatter(1, indices_to_zero, 0)

        # Build new patterns from features
        self.xi_new = torch.einsum('pk,kia->pia', c, self.f)

        # Fill zeros with Gaussian noise
        mask = self.xi_new == 0
        filler = torch.randn_like(self.xi_new) * self.sigma
        # FIX: write into xi_new, not xi
        self.xi_new = torch.where(mask, filler, self.xi_new)

        # Vectorial case: normalize -> for d=1 gives ±1 (binary)
        if self.spin_type == "vector":
            self.xi_new = self.normalize(self.xi_new)

        return self.xi_new
    


    def normalize(self, x):
        # Normalize each d-dimensional vector in x along the last dimension
        norms = x.norm(dim=-1, keepdim=True)+1e-9
        return  x / norms



class GeneralDataset(Dataset):
    def __init__(self, D, f):
        self.D = D
        self.f = data

    def __len__(self):
        return self.D

    def __getitem__(self, index):
        return self.data[index]

