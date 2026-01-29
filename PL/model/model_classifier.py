import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from PL.utils.k_d import LogKd

class Classifier(nn.Module):
    def __init__(self, N, d, gamma=0., l=1, device=None, downf=1., spin_type: str = "vector"):
        """
        spin_type:
            - 'vector'     : vector spins (binary if d=1, fixed-norm vector if d>1)
            - 'continuous' : continuous spins (independent of d)
        """
        super(Classifier, self).__init__()
        self.N = N
        self.d = d
        self.sqrt_d = torch.sqrt(torch.tensor(d))
        self.gamma = gamma
        self.l = l
        self.device = device
        self.spin_type = spin_type  # NEW

        self.norm0 = downf*math.sqrt(d)
        J_ = torch.randn(N, d, d)
        norm_ = torch.norm(J_)
        self.J0 = J_*self.norm0/(norm_)
        self.J = nn.Parameter(self.J0)

    def normalize_J(self):
        norm = torch.norm(self.J.data)
        with torch.no_grad():
            self.J.data *= self.norm0/norm

    def normalize_x(self, x):
        if self.spin_type=="vector":
            with torch.no_grad():
                norms = x.norm(dim=-1, keepdim=True) + 1e-9
                x = x / norms
        return x

    def Hebb(self, xi, y, form="Tensorial"):
        """
        Supervised Hebbian initialization of J using (xi_mu, y_mu).
    
        Args:
            xi: [P, N, d] input patterns
            y : [P, N, d] labels (same "single-spin nature" as xi)
            form: "Isotropic" or "Tensorial"
    
        Builds:
            Tensorial:  J[i,j,a,b] = (1/N) * sum_mu y[mu,i,a] * xi[mu,j,b]
            Isotropic:  J[i,j,:,:] = (1/N) * sum_mu <y[mu,i], xi[mu,j]>   (broadcast over a,b)
    
        Diagonal J[i,i,:,:] is set to 0 (as in your classic Hebb).
        """
        if form not in ["Isotropic", "Tensorial"]:
            raise ValueError("Form must be either 'Isotropic' or 'Tensorial'")
    
        if xi.ndim != 3 or y.ndim != 2:
            raise ValueError(f"xi and y must be 3D and 2D tensors. Got xi {xi.shape}, y {y.shape}")
    
        P, N, d = xi.shape
        if N != self.N or d != self.d:
            raise ValueError(f"Shape mismatch: xi is {xi.shape} but model expects N={self.N}, d={self.d}")

        xi = xi.to(self.device)
        y = y.to(self.device)
    
        with torch.no_grad():
            self.J.zero_()
    
            if form == "Tensorial":
                # Vectorized over mu: J_ijab = (1/N) sum_mu y_mia * xi_mjb
                self.J.copy_(torch.einsum("pa,pjb->jab", y, xi))
                self.normalize_J()
    
            else:  # "Isotropic"
                # s_ij = (1/N) sum_mu dot(y_mi, xi_mj)
                s = torch.einsum("pia,pja->j", y, xi) / self.N
                self.J.copy_(s[:, :, None, None])  # broadcast to [N,d,d]
    
    def Z_i_mu_func(self, y_i_mu, lambd, r=1):
        if self.d == 1:
            Z_i_mu = 2 * torch.cosh(lambd * r * y_i_mu)  # [M, N] or [M]
        else:
            print("Z_i_mu_func defined only for d=1")
        return Z_i_mu
    
    def loss(self, xi_batch, y_batch, loss_type="CE", lambd=1., r=1, l2=None):
        if loss_type=="CE":
            return self.compute_crossentropy(xi_batch, y_batch, lambd, r, l2)
        elif loss_type=="MSE":
            return self.compute_MSE(xi_batch, y_batch, lambd, l2)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def compute_MSE(self, xi_batch, y_batch, lambd, l2):
        diagonal = self.J.data.diagonal(dim1=0, dim2=1)
        diagonal.fill_(0)
        J_x = torch.einsum('jab,mjb->ma', self.J, xi_batch)
        energy_mu = ((y_batch-lambd*J_x)**2).mean()
        if l2 is not None:
            l2_term = l2*(self.J ** 2).mean()
            energy_mu = energy_mu + l2_term
        return energy_mu

    def compute_crossentropy(self, xi_batch, y_batch, lambd, r, l2):
        """
        Dispatch to the appropriate forward depending on spin_type and d.
        """
        if self.spin_type == "vector":
            if self.d == 1:
                return self._crossentropy_vector_d1(xi_batch, y_batch, lambd, r, l2)
            else:
                return self._crossentropy_vector_ddim(xi_batch, y_batch, lambd, r, l2)
        elif self.spin_type == "continuous":
            return self._crossentropy_continuous(xi_batch, y_batch, lambd, r, l2)
        else:
            raise ValueError(f"Unknown spin_type: {self.spin_type}")

    def _crossentropy_vector_d1(self, xi_batch, y_batch, lambd, r, l2):
        """
        Pseudolikelihood for binary spins (d=1).
        Keeps the same logic as the original forward().
        """
        # Ensure no self-interaction
        diagonal = self.J.data.diagonal(dim1=0, dim2=1)
        diagonal.fill_(0)
        J_x = torch.einsum('jab,mjb->ma', self.J, xi_batch)   # [M,N,d]
        y_i_mu = J_x.norm(dim=-1)                                 # [M,N]
        x_J_x = torch.einsum('ma,ma->m', y_batch, J_x)        # [M,N]
        energy_i_mu = -x_J_x + (1.0 / lambd) * torch.log(self.Z_i_mu_func(y_i_mu, lambd, r) + 1e-9)  
        if l2 is None:
            return energy_i_mu.mean()
        else:
            l2_term = l2*(self.J ** 2).mean()
            return energy_i_mu.mean() + l2_term

    def _crossentropy_vector_ddim(self, xi_batch, y_batch, lambd, r, l2):
        """
        Pseudolikelihood for vector spins with fixed norm (d > 1).

        Implements:
            L_PL = - ⟨ sum_i y_i^μ · u_i^μ - (1/λ) log K_d( λ ||u_i^μ|| ) ⟩_{μ}

        where:
            y_i^μ = xi_batch[μ, i, :]
            u_i^μ = (J * xi_batch)_i^μ   (local field, d-dimensional)

        Args:
            xi_batch: [M, N, d] tensor of patterns (batch of P= M patterns)
            lambd:    scalar λ
            alpha:    if not None, replace (1/λ) log K_d(λ||u||) with (1/λ) α ||u||²
            i_rand:   if not None, compute contribution only for that site i (single-site PL)
            r:        radius of spins (if they live on a sphere of radius r)
            l2:       if True, add L2 penalty on J

        Returns:
            Scalar loss (mean over patterns, and over sites when i_rand is None).
        """

        # Local fields for all sites:
        J_x = torch.einsum('jab,mjb->ma', self.J, xi_batch)   # [M, N, d]
        u_norm = J_x.norm(dim=-1)                                # [M, N]
        #print("u_norm/d", u_norm.mean()/self.d)
        y_dot_u = torch.einsum('ma,ma->m', y_batch, J_x)       # [M, N]
        x_arg = lambd * r * u_norm                                # [M]
        normalization = LogKd.apply(x_arg, self.d, True)
        energy_i_mu = -y_dot_u + 1/lambd*normalization
        # Average over sites i, then over patterns mu
        if l2 is None:
            return energy_i_mu.mean()
        else:
            l2_term = l2*(self.J ** 2).mean()
            return energy_i_mu.mean() + l2_term

    def _crossentropy_continuous(self, xi_batch, y_batch, lambd, r=1, l2=None):
        """
        Pseudolikelihood for continuous variables with Gaussian regularization.

        We consider conditionals of the form
            p(y_i | y_{-i}) ∝ exp( λ y_i · u_i - (γ/2) ||y_i||^2 ),
        where u_i is the local field. The exact Gaussian normalizer gives a PL loss
        (up to J–independent constants):
            ℓ_i^μ = - y_i^μ · u_i^μ + (λ / (2γ)) ||u_i^μ||^2.

        Here we implement:
            L = ⟨ ℓ_i^μ ⟩_{μ,i} (+ optional L2 penalty)

        Args:
            xi_batch: [M, N, d] tensor of continuous variables.
            lambd:    scalar λ (float or 0-dim tensor).
            alpha:    unused here (kept for API compatibility).
            i_rand:   if not None, use only site i_rand (single-site PL).
            r:        unused for continuous case (kept for API compatibility).
            l2:       if True, add an L2 penalty on J.

        Returns:
            Scalar loss (mean over patterns μ and sites i).
        """
        gamma = self.gamma  # you must define this in __init__

        J_x = torch.einsum('jab,mjb->ma', self.J, xi_batch)        # [M, N, d]
        y_dot_u  = (y_batch * J_x).sum(dim=-1)                         # [M, N]
        u_norm_sq = (J_x ** 2).sum(dim=-1)                            # [M, N]
        energy_i_mu = -y_dot_u + (lambd / (2.0 * gamma)) * u_norm_sq  # [M, N]
        # Average over sites and patterns
        loss = energy_i_mu.mean()                                     # scalar
        if l2 is not None:
            l2_term = l2*(self.J ** 2).mean()
            loss = loss + l2_term
        return loss
    
    def _bessel_Iv_series(self, x, v: float, n_terms: int = 20):
        """
        Compute modified Bessel function I_v(x) via its power series:

            I_v(x) = sum_{k=0}^∞ (1 / (k! Γ(k+v+1))) (x/2)^{2k+v}

        We truncate the sum at n_terms. Implemented with pure torch ops,
        so it's differentiable by autograd.

        Args:
            x: tensor (any shape)
            v: float, order of Bessel
            n_terms: number of series terms

        Returns:
            Tensor with same shape as x.
        """
        # Work in float64 for numerical stability
        orig_dtype = x.dtype
        x64 = x.to(torch.float64)

        # Avoid log(0) issues
        eps = 1e-12
        x_clamped = torch.clamp(x64, min=eps)

        # k = 0, 1, ..., n_terms-1, with broadcasting over x
        # Shape: [n_terms, 1, 1, ..., 1] to broadcast with x
        k = torch.arange(n_terms, device=x64.device, dtype=x64.dtype)
        k_shape = (n_terms,) + (1,) * x64.ndim
        k = k.view(k_shape)

        v_tensor = torch.tensor(v, dtype=x64.dtype, device=x64.device)

        # log((x/2)^(2k+v)) = (2k+v) * log(x/2)
        log_x_over_2 = torch.log(x_clamped / 2.0)
        log_x_power = (2.0 * k + v_tensor) * log_x_over_2

        # log(1 / (k! Γ(k+v+1))) = -[log(k!) + log Γ(k+v+1)]
        log_factorial_k = torch.lgamma(k + 1.0)           # log(k!)
        log_gamma_kv = torch.lgamma(k + v_tensor + 1.0)   # log Γ(k+v+1)
        log_coeff = -(log_factorial_k + log_gamma_kv)

        # log term_k = log_coeff + log_x_power
        log_terms = log_coeff + log_x_power

        terms = torch.exp(log_terms)   # [n_terms, *x.shape]
        Iv = terms.sum(dim=0)          # sum over k

        # Fix the value at x=0 using known limits:
        # I_v(0) = 0 for v>0; I_0(0) = 1
        if v > 0:
            Iv = torch.where(x64 == 0, torch.zeros_like(Iv), Iv)
        elif abs(v) < 1e-12:
            Iv = torch.where(x64 == 0, torch.ones_like(Iv), Iv)

        return Iv.to(orig_dtype)


    def _K_d(self, x, d=None):
        """
        Compute K_d(x) = (2π)^{d/2} I_{d/2 - 1}(x) / x^{d/2 - 1}
        using a custom torch implementation of I_v(x).

        Args:
            x: Tensor, argument of K_d (usually x = λ r ||u|| ≥ 0)
            d: Optional int; if None, use self.d

        Returns:
            Tensor with same shape as x with K_d(x).
        """
        if d is None:
            d = self.d

        # order of the modified Bessel function
        nu = d / 2.0 - 1.0

        # Compute I_nu(x) via power series (autograd-friendly)
        Iv = self._bessel_Iv_series(x, v=nu, n_terms=20)

        # avoid division by zero at x ~ 0
        eps = 1e-12
        x_clamped = torch.clamp(x, min=eps)

        power = d / 2.0 - 1.0
        prefactor = (2.0 * math.pi) ** (d / 2.0)

        K = prefactor * Iv / (x_clamped ** power)
        return K
    
    def forward(self, xi_batch):
        u_pred = torch.einsum("jab,mjb->ma", self.J, xi_batch)

        if self.spin_type == "vector":
            y_pred = self.normalize_x(u_pred)
        elif self.spin_type == "continuous":
            y_pred = u_pred.clone()

        return y_pred


