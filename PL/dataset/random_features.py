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
        self.f = f

    def __len__(self):
        return self.D

    def __getitem__(self, index):
        return self.f[index]
    


class RandomDatasetPowerLaw(Dataset):
    """Binary random–feature dataset with optional power‑law coefficients **and**
    synthetic multi‑class labels generated by a teacher–student scheme.

    Parameters
    ----------
    P : int
        Number of patterns (samples)
    N : int
        Number of sites per pattern
    D : int
        Number of random features in the latent field
    d : int
        Dimensionality of each site (for images set d=1 and N = #pixels)
    eta : float
        Exponent controlling biased feature selection (if pick_biased is not None)
    sigma : float, default 1.
        Std‑dev of the Gaussian entries used for both *xi* and teacher weights
    L : int or None, default None
        If not None, only the first **L** of the **D** features are kept (others are zeroed)
    seed : int or None
        Random seed for reproducibility (affects *xi*, *f* and teacher weights)
    on_sphere : bool, default True
        If True, each vector is L2‑normalised
    coefficients : {'binary', 'gaussian'}
        Distribution of coefficients *c* multiplying the random field
    variances : {'power_law', 'exponential', None}
        Optional variance modulation of *c*
    exponent : float or None
        Power‑law exponent if variances == 'power_law'
    shift : int, default 0
        Shift applied to the power‑law index |k + shift|^{‑exponent}
    pick_biased : bool or None
        If not None, randomly zero‑out D‑L coefficients with a bias ∝|k|^{‑eta}
    C : int, default None
        If provided, number of *teacher* classes.  Labels are y = argmax_c w_c·xi.
    teacher_seed : int or None
        Seed used **only** for teacher weights (allows independent sampling)
    """

    def __init__(self, P, N, D, d, eta, sigma=1., L=None, seed=None, *, spin_type="vector",
                 coefficients="binary", variances=None, exponent=None, shift=0,
                 pick_biased=None, C=None, teacher_seed=None):

        super().__init__()
        self.P, self.N, self.D, self.L = P, N, D,  L or D
        if self.L > self.D:
            raise ValueError("L must be ≤ D")
        self.d, self.sigma, self.spin_type = d, sigma, spin_type
        self.coefficients, self.variances = coefficients, variances
        self.exponent, self.shift, self.pick_biased = exponent, shift, pick_biased
        self.eta, self.C = eta, C 

        if seed is not None:
            torch.manual_seed(seed)

        self.xi = torch.randn(P, N, d, dtype=torch.float32) * sigma

        if self.D > 0:
            self._apply_random_field(seed=seed + 111 if seed is not None else None)

        if spin_type=="vector":
            self.xi = self._l2_normalise(self.xi)

        if self.C is not None and self.C > 1:
            self._make_teacher_labels(teacher_seed)
        else:
            self.labels = None  

        self.sample_weights = torch.ones(P, dtype=torch.float32)

    def _apply_random_field(self, *, seed=None):
        """Construct features f_k and coefficients c_{μk}, then replace xi.«
        xi[p] = Σ_k c_{pk} f_k."""
        if seed is not None:
            torch.manual_seed(seed)

        f = torch.randn(self.D, self.N, self.d, dtype=torch.float32) * self.sigma
        if self.spin_type=="vector":
            f = self._l2_normalise(f)
        self.f = f  

        if self.coefficients == "binary":
            c = torch.randint(0, 2, (self.P, self.D), dtype=torch.float32) * 2 - 1
        elif self.coefficients == "gaussian":
            c = torch.randn(self.P, self.D, dtype=torch.float32)
        else:
            raise ValueError("coefficients must be 'binary' or 'gaussian'")

        c = c / math.sqrt(self.D)  

        if self.variances == "power_law":
            expnt = self.exponent if self.exponent is not None else 1.0
            variances = torch.arange(1 + self.shift, self.D + 1 + self.shift, dtype=torch.float32) ** (-expnt)
            c *= variances 
        elif self.variances == "exponential":
            lam = 10.0 / float(self.D)
            variances = torch.exp(-torch.arange(1, self.D + 1, dtype=torch.float32) * lam)
            c *= variances

        if self.pick_biased is not None and self.L < self.D:
            idx = torch.arange(self.D)
            bias = (1+idx).float().pow(-self.eta)
            bias /= bias.sum()
            keep_idx = torch.stack([torch.multinomial(bias, self.L, replacement=False) for _ in range(self.P)])
            mask = torch.zeros_like(c).bool()
            mask.scatter_(1, keep_idx, True)
            c = c.masked_fill(~mask, 0.0)

        self.xi = torch.einsum('pk,kid->pid', c, f)
        self.xi = self._l2_normalise(self.xi) if (self.spin_type=="vector") else self.xi

    def _make_teacher_labels(self, teacher_seed=None):
        """Generate C random teacher weight vectors and assign labels by argmax."""
        if teacher_seed is not None:
            torch.manual_seed(teacher_seed)

        self.teacher_w = torch.randn(self.C, self.N, self.d, dtype=torch.float32) * self.sigma
        if self.spin_type=="vector":
            self.teacher_w = self._l2_normalise(self.teacher_w)

        xi_flat = self.xi.view(self.P, -1)
        w_flat = self.teacher_w.view(self.C, -1).t() 
        logits = xi_flat @ w_flat  
        self.labels = torch.argmax(logits, dim=1)

    def get_generalization(self, P_hat, L=None):
        """Return *new* patterns (without labels) sampled with independent coeffs."""
        L = L or self.L
        if L > self.D:
            raise ValueError("L must be ≤ D")
        if self.coefficients == "binary":
            c = torch.randint(0, 2, (P_hat, self.D), dtype=torch.float32) * 2 - 1
        elif self.coefficients == "gaussian":
            c = torch.randn(P_hat, self.D, dtype=torch.float32)
        c = c / math.sqrt(self.D)
        if self.variances == "power_law":
            expnt = self.exponent if self.exponent is not None else 1.0
            variances = torch.arange(1 + self.shift, self.D + 1 + self.shift, dtype=torch.float32) ** (-expnt)
            c *= variances
        elif self.variances == "exponential":
            lam = 10.0 / float(self.D)
            variances = torch.exp(-torch.arange(1, self.D + 1, dtype=torch.float32) * lam)
            c *= variances
        if self.pick_biased is not None and L < self.D:
            idx = torch.arange(self.D)
            bias = (self.D - idx).float().pow(-self.eta)
            bias /= bias.sum()
            keep_idx = torch.stack([torch.multinomial(bias, L, replacement=False) for _ in range(P_hat)])
            mask = torch.zeros_like(c).bool()
            mask.scatter_(1, keep_idx, True)
            c = c.masked_fill(~mask, 0.0)
        xi_new = torch.einsum('pk,kid->pid', c, self.f)
        xi_new = self._l2_normalise(xi_new) if (self.spin_type=="vector") else xi_new
        return xi_new
    
    def get_generalization_selected_features(self, P_hat, feature_range, L=None):
        # c stays 'binary' or 'gaussian', independent of spin_type
        if self.coefficients == "binary":
            c = torch.randint(0, 2, (P_hat, self.D)).float() * 2 - 1
        elif self.coefficients == "gaussian":
            c = torch.randn(P_hat, self.D)
        else:
            raise ValueError("coefficients must be 'binary' or 'gaussian'")

        c = c / math.sqrt(self.D)

        # Select only the features specified in feature_range
        c = c[:, feature_range]  # Select the columns corresponding to feature_range

        # Same sparsification logic as in RF, applied only to the selected features
        indices_to_zero = torch.rand(P_hat, len(feature_range))
        indices_to_zero = indices_to_zero.argsort(dim=1)[:, : len(feature_range) - self.L]
        c = c.scatter(1, indices_to_zero, 0)

        # Ensure that the tensor `self.f` is indexed correctly and matches dimensions
        # Select only the corresponding features from self.f (shape: self.D, self.N, self.d)
        selected_f = self.f[feature_range, :, :]  # Select only the relevant features from self.f

        # Build new patterns from selected features
        # c has shape (P_hat, len(feature_range))
        # selected_f has shape (len(feature_range), self.N, self.d)
        self.xi_new = torch.einsum('pk,knd->pnd', c, selected_f)

        # Fill zeros with Gaussian noise
        mask = self.xi_new == 0
        filler = torch.randn_like(self.xi_new) * self.sigma
        self.xi_new = torch.where(mask, filler, self.xi_new)

        # Vectorial case: normalize -> for d=1 gives ±1 (binary)
        if self.spin_type == "vector":
            self.xi_new = self._l2_normalise(self.xi_new)

        return self.xi_new



    def _l2_normalise(self, x):
        """Normalise along last dimension to unit L2 norm (handles broadcasting)."""
        if self.d == 1:
            result = torch.sign(x)
        else:
            eps = 1e-8
            norm = torch.norm(x)+eps
            result = x / norm
        return result
        
    def __len__(self):
        return self.P

    def __getitem__(self, idx):
        return self.xi[idx]

    # Convenience helpers
    def get_num_visibles(self):
        """Number of visible units (flattened)."""
        return self.N * self.d

    def get_num_classes(self):
        return self.C if self.C is not None else 0



