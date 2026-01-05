import math
import torch
import torch.nn as nn

class TwoBodiesModel(nn.Module):
    def __init__(self, N, d, gamma=0., r=1, device='cuda', spin_type: str = "vector", custom_mask=None):
        """
        spin_type:
            - 'vector'     : vector spins (binary if d=1, fixed-norm vector if d>1)
            - 'continuous' : continuous spins (independent of d)
        N: Number of sites
        d: Dimensionality of each site
        custom_mask: Optional [N, N] tensor to mask interactions in J.
        """
        super(TwoBodiesModel, self).__init__()
        self.N = N
        self.d = d
        self.gamma = gamma
        self.r = r
        self.device = device
        self.spin_type = spin_type
        self.custom_mask = custom_mask

        self.J = nn.Parameter(torch.randn(N, N, d, d))  # Interaction tensor
        self.J = self.J.to(device)

        if self.custom_mask is not None:
            if custom_mask.shape != (N, N):
                raise ValueError(f"custom_mask must have shape ({N}, {N})")
            self.mask = custom_mask.to(self.J.device)
            self.mask = self.mask.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, d, d)  # Shape [N, N, d, d]
        else:
            self.mask = torch.ones(N, N, device=self.J.device)  # Shape [N, N]
            self.mask.fill_diagonal_(0)  # Set diagonal to 0
            self.mask = self.mask.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, d, d)  # Shape [N, N, d, d]

        self.J.data *= self.mask  # Apply mask to J


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

        if form not in ["Isotropic", "Tensorial"]:
            raise ValueError("Form must be either 'Isotropic' or 'Tensorial'")

        with torch.no_grad():
            self.J.zero_()

            if form == "Isotropic":
                for mu in range(P):
                    for i in range(N):
                        for j in range(N):
                            if i != j:
                                self.J[i, j, :, :] += torch.sum(xi[mu, i, :] * xi[mu, j, :]) / N
                if self.custom_mask is not None:
                    self.J.data *= self.mask  # Apply custom mask to J
            elif form == "Tensorial":
                for mu in range(P):
                    xi_mu = xi[mu].to(self.device)  # Shape: (N, d)
                    outer_products = torch.einsum('ia,jb->ijab', xi_mu, xi_mu) / N  # (N,N,d,d)
                    indices = torch.arange(N)
                    outer_products[indices, indices] = 0
                    self.J += outer_products

                self.J.data *= self.mask  # Apply mask to J

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
            masked_J = self.J.data * self.mask  # Apply mask to J

        # Local field u = J x
        # J: [N, N, d, d]; x: [B, N, d]  ->  J_x: [B, N, d]
        J_x = torch.einsum('ijab,Bjb->Bia', masked_J, x)
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
            print("To define normalization for d>1")
        return Z_i_mu

    def compute_crossentropy(self, xi_batch, lambd, alpha=None, i_rand=None, r=1, l2=False):
        """
        Dispatch to the appropriate forward depending on spin_type and d.
        """
        if self.spin_type == "vector":
            if self.d == 1:
                return self._crossentropy_vector_d1(xi_batch, lambd, alpha, i_rand, r, l2)
            else:
                return self._crossentropy_vector_ddim(xi_batch, lambd, alpha, i_rand, r, l2)
        elif self.spin_type == "continuous":
            return self._crossentropy_continuous(xi_batch, lambd, i_rand, r, l2)
        else:
            raise ValueError(f"Unknown spin_type: {self.spin_type}")

    def _crossentropy_vector_d1(self, xi_batch, lambd, alpha=None, i_rand=None, r=1, l2=False):
        """
        Pseudolikelihood for binary spins (d=1).
        Keeps the same logic as the original forward().
        """
        J_masked = self.J * self.mask  # Apply mask to J

        if i_rand is not None:
            # Single-site contribution for site i_rand
            J_x = torch.einsum('jab,mjb->ma', J_masked[i_rand], xi_batch)  # [M, d]
            y_i_mu = J_x.norm(dim=-1)  # [M]
            x_J_x = torch.einsum('ma,ma->m', xi_batch[:, i_rand], J_x)     # [M]

            if alpha is None:
                energy_i_mu = -x_J_x + (1.0 / lambd) * torch.log(
                    self.Z_i_mu_func(y_i_mu, lambd, r) + 1e-9
                )  # [M]
            else:
                energy_i_mu = -x_J_x + (1.0 / lambd) * alpha * y_i_mu**2

            if l2 is False:
                return energy_i_mu.mean(dim=0)
            else:
                return -x_J_x.mean() + alpha * (J_masked**2).mean()

        else:
            # Full product over all sites
            J_x = torch.einsum('ijab,mjb->mia', J_masked, xi_batch)   # [M,N,d]
            y_i_mu = J_x.norm(dim=-1)                                 # [M,N]
            x_J_x = torch.einsum('mia,mia->mi', xi_batch, J_x)        # [M,N]

            if alpha is None:
                energy_i_mu = -x_J_x + (1.0 / lambd) * torch.log(
                    self.Z_i_mu_func(y_i_mu, lambd, r) + 1e-9
                )  # [M,N]
            else:
                energy_i_mu = -x_J_x + (1.0 / lambd) * alpha * y_i_mu**2

            energy_i_mu = energy_i_mu.mean(dim=1)  # average over sites

            if l2 is False:
                return energy_i_mu.mean(dim=0)
            else:
                return -x_J_x.mean() + alpha * (J_masked**2).mean()

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
        J_masked = self.J * self.mask  # Apply mask to J


        if i_rand is not None:
            # Local field at site i_rand: u_i^μ = Σ_j J_{ij} y_j^μ
            # J_masked[i_rand] has shape [N, d, d]; xi_batch: [M, N, d]
            J_x = torch.einsum('jab,mjb->ma', J_masked[i_rand], xi_batch)   # [M, d]

            # Spins at site i_rand
            y_i_mu = xi_batch[:, i_rand, :]                                 # [M, d]

            # Norm of the local field ||u_i^μ||
            u_norm = J_x.norm(dim=-1)                                       # [M]

            # Dot product y_i^μ · u_i^μ
            y_dot_u = torch.einsum('ma,ma->m', y_i_mu, J_x)                 # [M]

            if alpha is None:
                # True vector pseudolikelihood: - y·u + (1/λ) log K_d(λ r ||u||)
                x_arg = lambd * r * u_norm                                  # [M]
                K_vals = self._K_d(x_arg)                                   # [M]
                energy_i_mu = -y_dot_u + (1.0 / lambd) * torch.log(K_vals + 1e-9)
            else:
                # Quadratic surrogate: - y·u + (1/λ) α ||u||²
                energy_i_mu = -y_dot_u + (1.0 / lambd) * alpha * (u_norm ** 2)

            if not l2:
                return energy_i_mu.mean()
            else:
                l2_term = (J_masked ** 2).mean()
                return energy_i_mu.mean() + l2_term

        # Local fields for all sites:
        # J_x[m, i, a] = Σ_j,b J[i,j,a,b] * xi[m,j,b]
        J_x = torch.einsum('ijab,mjb->mia', J_masked, xi_batch)   # [M, N, d]

        # Norm ||u_i^μ|| for each (μ, i)
        u_norm = J_x.norm(dim=-1)                                # [M, N]

        # Spins y_i^μ
        y_i_mu = xi_batch                                        # [M, N, d]

        # Dot products y_i^μ · u_i^μ
        y_dot_u = torch.einsum('mia,mia->mi', y_i_mu, J_x)       # [M, N]

        if alpha is None:
            # True vector pseudolikelihood term
            x_arg = lambd * r * u_norm                           # [M, N]
            K_vals = self._K_d(x_arg)                            # [M, N]
            energy_i_mu = -y_dot_u + (1.0 / lambd) * torch.log(K_vals + 1e-9)
        else:
            # Quadratic surrogate
            energy_i_mu = -y_dot_u + (1.0 / lambd) * alpha * (u_norm ** 2)

        # Average over sites i, then over patterns μ
        energy_i_mu = energy_i_mu.mean(dim=1)                    # [M]
        if not l2:
            return energy_i_mu.mean()
        else:
            l2_term = (J_masked.data ** 2).mean()
            return energy_i_mu.mean() + l2_term

    def _crossentropy_continuous(self, xi_batch, lambd, i_rand=None, r=1, l2=False):
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

        J_masked = self.J * self.mask  # Apply mask to J

        if i_rand is not None:
            # Local field at site i_rand: u_i^μ = Σ_j J_{i_rand,j} y_j^μ
            # J_masked[i_rand]: [N, d, d]; xi_batch: [M, N, d]
            J_x = torch.einsum('jab,mjb->ma', J_masked[i_rand], xi_batch)  # [M, d]

            # Data at site i_rand
            y_i_mu = xi_batch[:, i_rand, :]                                # [M, d]

            # y · u and ||u||^2
            y_dot_u   = (y_i_mu * J_x).sum(dim=-1)                         # [M]
            u_norm_sq = (J_x ** 2).sum(dim=-1)                             # [M]

            # ℓ_i^μ = - y·u + (λ / (2γ)) ||u||^2  (up to constants in λ,γ,d)
            energy_i_mu = -y_dot_u + (lambd / (2.0 * gamma)) * u_norm_sq   # [M]

            loss = energy_i_mu.mean()                                      # scalar

        else:
            # Local fields u_i^μ for all sites:
            # J_x[m, i, a] = Σ_j,b J[i,j,a,b] * xi[m,j,b]
            J_x = torch.einsum('ijab,mjb->mia', J_masked, xi_batch)        # [M, N, d]

            y_i_mu   = xi_batch                                           # [M, N, d]
            y_dot_u  = (y_i_mu * J_x).sum(dim=-1)                         # [M, N]
            u_norm_sq = (J_x ** 2).sum(dim=-1)                            # [M, N]

            # ℓ_i^μ = - y·u + (λ / (2γ)) ||u||^2
            energy_i_mu = -y_dot_u + (lambd / (2.0 * gamma)) * u_norm_sq  # [M, N]

            # Average over sites and patterns
            loss = energy_i_mu.mean()                                     # scalar

        # Optional L2 penalty on J
        if l2:
            l2_term = (J_masked ** 2).mean()
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


