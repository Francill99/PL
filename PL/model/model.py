import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class TwoBodiesModel(nn.Module):
    def __init__(self, N, d, gamma=0., r=1, device=None, spin_type: str = "vector"):
        """
        spin_type:
            - 'vector'     : vector spins (binary if d=1, fixed-norm vector if d>1)
            - 'continuous' : continuous spins (independent of d)
        """
        super(TwoBodiesModel, self).__init__()
        self.N = N
        self.d = d
        self.gamma = gamma
        self.r = r
        self.device = device
        self.spin_type = spin_type  # NEW

        self.J = nn.Parameter(torch.randn(N, N, d, d))
        diagonal = self.J.data.diagonal(dim1=0, dim2=1)  # Get diagonal elements
        diagonal.fill_(0)
        #self.normalize_J()

        # in __init__
        mask2d = torch.ones(self.N, self.N, device=self.J.device, dtype=self.J.dtype)
        mask2d.fill_diagonal_(0.0)                 # (N,N) diagonal = 0
        self.register_buffer("mask", mask2d[:, :, None, None])  # (N,N,1,1)

    def normalize_J(self):
        with torch.no_grad():
            self.J.data *= torch.sqrt(torch.tensor(1/(self.N*self.d)))

    def symmetrize_J(self):
        with torch.no_grad():
            self.J.data = (self.J.data + self.J.data.transpose(0,1))/2

    def normalize_x(self, x):
        if self.on_sphere:
            with torch.no_grad():
                norms = x.norm(dim=-1, keepdim=True) + 1e-9
                x = x / norms
        return x

    def Hebb(self, xi, form):
        P = xi.shape[0]  # Number of patterns
        N = self.N
        d = self.d

        if form not in ["Isotropic", "Tensorial"]:
            raise ValueError("Form must be either 'Isotropic' or 'Tensorial'")

        with torch.no_grad():
            self.J.zero_()
            self.J.to(self.device)

            if form == "Isotropic":
                for mu in range(P):
                    for i in range(N):
                        for j in range(N):
                            if i != j:
                                self.J[i, j, :, :] += torch.sum(xi[mu, i, :] * xi[mu, j, :]) / N
            elif form == "Tensorial":
                for mu in range(P):
                    xi_mu = xi[mu].to(self.device)  # Shape: (N, d)
                    outer_products = torch.einsum('ia,jb->ijab', xi_mu, xi_mu) / N  # (N,N,d,d)
                    indices = torch.arange(N)
                    outer_products[indices, indices] = 0
                    self.J += outer_products

                diagonal = self.J.data.diagonal(dim1=0, dim2=1)
                diagonal.fill_(0)

    def dyn_step(self, x, a=None):
        """
        One dynamical update step for the state x.

        Cases:
          - spin_type == "vector", d == 1:
              Ising-like binary spins: x_i ∈ {±r}.
              Update via local field then sign projection.
          - spin_type == "vector", d > 1:
              Vector spins with fixed norm r: ||x_i|| = r.
              Update via local field then projection to sphere.
          - spin_type == "continuous":
              Continuous variables, no projection (pure linear dynamics).

        Args:
            x: [B, N, d] tensor of current spins/variables.
            a: optional step size. If None, x_new = J x.
               If not None, x_new = x + a (J x).

        Returns:
            x_new: [B, N, d] updated configuration.
        """
        B, N, d = x.shape

        # Enforce zero self-coupling (no self-interaction)
        with torch.no_grad():
            diagonal = self.J.diagonal(dim1=0, dim2=1)  # [N, d, d]
            diagonal.zero_()

        # Local field u = J x
        # J: [N, N, d, d]; x: [B, N, d]  ->  J_x: [B, N, d]
        J_x = torch.einsum('ijab,Bjb->Bia', self.J, x)

        with torch.no_grad():
            # Effective pre-projected state
            if a is None:
                x_eff = J_x
            else:
                x_eff = x + a * J_x

            if self.spin_type == "vector" and d == 1:
                x_new = x_eff.sign()
                # just in case r != 1
                x_new = self.r * x_new

            elif self.spin_type == "vector" and d > 1:
                norms = x_eff.norm(dim=-1, keepdim=True) + 1e-9  # [B, N, 1]
                x_new = self.r * x_eff / norms

            elif self.spin_type == "continuous":
                x_new = x_eff - self.gamma*x

            else:
                raise ValueError(f"Unknown/unsupported spin_type={self.spin_type}, d={d}")

        return x_new

    def Z_i_mu_func(self, y_i_mu, lambd, r=1):
        if self.d == 1:
            Z_i_mu = 2 * torch.cosh(lambd * r * y_i_mu)  # [M, N] or [M]
        else:
            print("Z_i_mu_func defined only for d=1")
        return Z_i_mu
    
    def loss(self, xi_batch, loss_type="CE", lambd=1., alpha=1., i_rand=None, r=1, l2=False):
        if loss_type=="CE":
            return self.compute_crossentropy(xi_batch, lambd,  alpha, i_rand, r, l2)
        elif loss_type=="MSE":
            return self.compute_MSE(xi_batch, lambd, alpha, l2)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def compute_MSE(self, xi_batch, lambd, alpha, l2):
        diagonal = self.J.data.diagonal(dim1=0, dim2=1)
        diagonal.fill_(0)
        J_masked = self.J*self.mask
        J_x = torch.einsum('ijab,mjb->mia', J_masked, xi_batch)
        energy_mu = ((xi_batch-lambd*J_x)**2).mean()
        if l2==True:
            l2_term = alpha*(self.J ** 2).mean()
            energy_mu = energy_mu + l2_term
        return energy_mu

    def compute_crossentropy(self, xi_batch, lambd, alpha, i_rand, r, l2):
        """
        Dispatch to the appropriate forward depending on spin_type and d.
        """
        if self.spin_type == "vector":
            if self.d == 1:
                return self._crossentropy_vector_d1(xi_batch, lambd, alpha, i_rand, r, l2)
            else:
                return self._crossentropy_vector_ddim(xi_batch, lambd, alpha, i_rand, r, l2)
        elif self.spin_type == "continuous":
            return self._crossentropy_continuous(xi_batch, lambd, alpha, i_rand, r, l2)
        else:
            raise ValueError(f"Unknown spin_type: {self.spin_type}")

    def _crossentropy_vector_d1(self, xi_batch, lambd, alpha, i_rand, r, l2):
        """
        Pseudolikelihood for binary spins (d=1).
        Keeps the same logic as the original forward().
        """
        # Ensure no self-interaction
        diagonal = self.J.data.diagonal(dim1=0, dim2=1)
        diagonal.fill_(0)
        J_masked = self.J*self.mask  # (N,N,1,1) effectively when d=1

        if i_rand is not None:
            # Single-site contribution for site i_rand
            J_x = torch.einsum('jab,mjb->ma', J_masked[i_rand], xi_batch)  # [M, d]
            y_i_mu = J_x.norm(dim=-1)  # [M]
            x_J_x = torch.einsum('ma,ma->m', xi_batch[:, i_rand], J_x)     # [M]
            energy_i_mu = -x_J_x + (1.0 / lambd) * torch.log(
                    self.Z_i_mu_func(y_i_mu, lambd, r) + 1e-9
                )  # [M]
            if l2 is False:
                return energy_i_mu.mean()
            else:
                l2_term = alpha*(self.J ** 2).mean()
                return energy_i_mu.mean() + l2_term

        else:
            # Full product over all sites
            J_x = torch.einsum('ijab,mjb->mia', J_masked, xi_batch)   # [M,N,d]
            y_i_mu = J_x.norm(dim=-1)                                 # [M,N]
            x_J_x = torch.einsum('mia,mia->mi', xi_batch, J_x)        # [M,N]
            energy_i_mu = -x_J_x + (1.0 / lambd) * torch.log(self.Z_i_mu_func(y_i_mu, lambd, r) + 1e-9)  
            energy_i_mu = energy_i_mu.mean(dim=1)  # average over sites

            if l2 is False:
                return energy_i_mu.mean()
            else:
                l2_term = alpha*(self.J ** 2).mean()
                return energy_i_mu.mean() + l2_term

    def _crossentropy_vector_ddim(self, xi_batch, lambd, alpha=None, i_rand=None, r=1, l2=False):
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
        # Ensure no self-interaction: set diagonal blocks of J to zero (no_grad is fine here)
        #diagonal = self.J.data.diagonal(dim1=0, dim2=1)
        #diagonal.fill_(0)
        J_masked = self.J*self.mask  # (N, N, d, d) with diagonal ~ 0

        if i_rand is not None:
            # Local field at site i_rand
            # J_masked[i_rand] has shape [N, d, d]; xi_batch: [M, N, d]
            J_x = torch.einsum('jab,mjb->ma', J_masked[i_rand], xi_batch)   # [M, d]

            # Spins at site i_rand
            y_i_mu = xi_batch[:, i_rand, :]                                 # [M, d]

            # Norm of the local field 
            u_norm = J_x.norm(dim=-1)                                       # [M]

            # Dot product 
            y_dot_u = torch.einsum('ma,ma->m', y_i_mu, J_x)                 # [M]

            x_arg = lambd * r * u_norm                                  # [M]
            K_vals = self._K_d(x_arg)                                   # [M]
            energy_i_mu = -y_dot_u + (1.0 / lambd) * torch.log(K_vals + 1e-9)

            if not l2:
                return energy_i_mu.mean()
            else:
                l2_term = (self.J ** 2).mean()
                return energy_i_mu.mean() + l2_term
        else:
            # Local fields for all sites:
            J_x = torch.einsum('ijab,mjb->mia', J_masked, xi_batch)   # [M, N, d]
            u_norm = J_x.norm(dim=-1)                                # [M, N]
            y_i_mu = xi_batch                                        # [M, N, d]
            y_dot_u = torch.einsum('mia,mia->mi', y_i_mu, J_x)       # [M, N]
            x_arg = lambd * r * u_norm
            K_vals = self._K_d(x_arg)                                   # [M]
            energy_i_mu = -y_dot_u + (1.0 / lambd) * torch.log(K_vals + 1e-9)
            # Average over sites i, then over patterns mu
            energy_i_mu = energy_i_mu.mean(dim=1)                    # [M]
            if not l2:
                return energy_i_mu.mean()
            else:
                l2_term = (self.J ** 2).mean()
                return energy_i_mu.mean() + l2_term

    def _crossentropy_continuous(self, xi_batch, lambd, alpha=None, i_rand=None, r=1, l2=False):
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

        # Ensure no self-interaction: zero diagonal blocks of J
        with torch.no_grad():
            diagonal = self.J.diagonal(dim1=0, dim2=1)  # [N, d, d]
            diagonal.zero_()
        J_masked = self.J*self.mask  # (N, N, d, d) with zero diagonal

        if i_rand is not None:
            # Local field at site i_rand
            # J_masked[i_rand]: [N, d, d]; xi_batch: [M, N, d]
            J_x = torch.einsum('jab,mjb->ma', J_masked[i_rand], xi_batch)  # [M, d]

            # Data at site i_rand
            y_i_mu = xi_batch[:, i_rand, :]                                # [M, d]

            y_dot_u   = (y_i_mu * J_x).sum(dim=-1)                         # [M]
            u_norm_sq = (J_x ** 2).sum(dim=-1)                             # [M]
            energy_i_mu = -y_dot_u + (lambd / (2.0 * gamma)) * u_norm_sq   # [M]
            loss = energy_i_mu.mean()                                      # scalar

        else:
            J_x = torch.einsum('ijab,mjb->mia', J_masked, xi_batch)        # [M, N, d]
            y_i_mu   = xi_batch                                           # [M, N, d]
            y_dot_u  = (y_i_mu * J_x).sum(dim=-1)                         # [M, N]
            u_norm_sq = (J_x ** 2).sum(dim=-1)                            # [M, N]
            energy_i_mu = -y_dot_u + (lambd / (2.0 * gamma)) * u_norm_sq  # [M, N]
            # Average over sites and patterns
            loss = energy_i_mu.mean()                                     # scalar
        if l2:
            l2_term = (self.J ** 2).mean()
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


