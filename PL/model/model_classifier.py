import math
import torch
import torch.nn as nn

from PL.utils.k_d import LogKd


class Classifier(nn.Module):
    def __init__(
        self,
        N,
        d,
        gamma=0.0,
        l=1,
        device=None,
        downf=1.0,
        dtype=torch.float32,
        spin_type: str = "vector",
        label_type: str | None = None,
        dictionary: torch.Tensor | None = None,
    ):
        """
        Supervised teacher-student classifier.

        spin_type controls the input distribution used by the dataset:
            - "vector"      : vector spins, binary if d=1 and spherical if d>1
            - "continuous"  : unconstrained continuous input spins
            - "dictionary"  : input spins sampled from an external dictionary

        label_type controls the conditional output space used in the loss:
            - "vector"      : spherical/binary pseudo-likelihood normalizer
            - "continuous"  : Gaussian/continuous pseudo-likelihood normalizer
            - "dictionary"  : finite dictionary normalizer, i.e. logsumexp over atoms

        For backward compatibility, label_type defaults to spin_type when it is not
        provided. For new dictionary-input experiments it is better to pass
        label_type explicitly.
        """
        super(Classifier, self).__init__()
        self.N = int(N)
        self.d = int(d)
        self.sqrt_d = torch.sqrt(torch.tensor(self.d, dtype=dtype))
        self.gamma = gamma
        self.l = l
        self.device = device
        self.spin_type = spin_type
        self.label_type = spin_type if label_type is None else label_type

        self._validate_types()
        prepared_dictionary = self._prepare_dictionary(dictionary, dtype=dtype, device=device)
        if prepared_dictionary is None:
            self.dictionary = None
        else:
            self.register_buffer("dictionary", prepared_dictionary)

        self.norm0 = downf * math.sqrt(self.d)
        J_ = torch.randn(self.N, self.d, self.d, dtype=dtype, device=device)
        norm_ = torch.norm(J_).clamp_min(1e-12)
        self.J0 = J_ * self.norm0 / norm_
        self.J = nn.Parameter(self.J0)

    def _validate_types(self):
        allowed_spin_types = {"vector", "continuous", "dictionary"}
        allowed_label_types = {"vector", "continuous", "dictionary"}
        if self.spin_type not in allowed_spin_types:
            raise ValueError("spin_type must be 'vector', 'continuous', or 'dictionary'.")
        if self.label_type not in allowed_label_types:
            raise ValueError("label_type must be 'vector', 'continuous', or 'dictionary'.")
        if self.label_type == "dictionary" and self.d <= 1:
            raise ValueError("label_type='dictionary' is only supported for d > 1.")

    def _prepare_dictionary(self, dictionary, dtype, device):
        if self.label_type != "dictionary":
            return None
        if dictionary is None:
            raise ValueError(
                "A dictionary tensor with shape [D, d] must be provided when "
                "label_type='dictionary'."
            )
        if not isinstance(dictionary, torch.Tensor):
            raise TypeError(f"dictionary must be a torch.Tensor, got {type(dictionary)!r}.")
        if dictionary.ndim != 2:
            raise ValueError(f"dictionary must have shape [D, d], got {tuple(dictionary.shape)}.")
        if dictionary.shape[0] <= 0:
            raise ValueError("dictionary must contain at least one atom, got D=0.")
        if dictionary.shape[1] != self.d:
            raise ValueError(
                f"dictionary has last dimension {dictionary.shape[1]}, but classifier d={self.d}."
            )
        if not dictionary.is_floating_point():
            raise TypeError(f"dictionary must be a floating point tensor, got dtype={dictionary.dtype}.")
        return dictionary.detach().clone().to(device=device, dtype=dtype)

    def normalize_J(self):
        norm = torch.norm(self.J.data).clamp_min(1e-12)
        with torch.no_grad():
            self.J.data *= self.norm0 / norm

    @staticmethod
    def _normalize_vectors(x, eps: float = 1e-9):
        norms = x.norm(dim=-1, keepdim=True).clamp_min(eps)
        return x / norms

    def normalize_x(self, x):
        """Backward-compatible helper used by older code paths."""
        if self.spin_type == "vector" or self.label_type == "vector":
            with torch.no_grad():
                x = self._normalize_vectors(x)
        return x

    def Hebb(self, xi, y, form="Tensorial"):
        """
        Supervised Hebbian initialization of J using (xi_mu, y_mu).

        Args:
            xi: [P, N, d] input patterns
            y : [P, d] labels
            form: "Isotropic" or "Tensorial"
        """
        if form not in ["Isotropic", "Tensorial"]:
            raise ValueError("Form must be either 'Isotropic' or 'Tensorial'")

        if xi.ndim != 3 or y.ndim != 2:
            raise ValueError(f"xi must be [P,N,d] and y must be [P,d]. Got xi {xi.shape}, y {y.shape}")

        P, N, d = xi.shape
        if N != self.N or d != self.d:
            raise ValueError(f"Shape mismatch: xi is {xi.shape} but model expects N={self.N}, d={self.d}")
        if y.shape != (P, self.d):
            raise ValueError(f"y must have shape [P,d]=[{P},{self.d}], got {tuple(y.shape)}")

        xi = xi.to(self.J.device)
        y = y.to(self.J.device)

        with torch.no_grad():
            self.J.zero_()

            if form == "Tensorial":
                # J[j,a,b] = sum_mu y[mu,a] xi[mu,j,b]
                self.J.copy_(torch.einsum("pa,pjb->jab", y, xi))
                self.normalize_J()

            else:  # "Isotropic"
                # One scalar per input site, broadcast over the d x d block.
                s = torch.einsum("pa,pja->j", y, xi) / self.N
                self.J.copy_(s[:, None, None].expand(self.N, self.d, self.d))

    def Z_i_mu_func(self, y_i_mu, lambd, r=1):
        if self.d == 1:
            return 2 * torch.cosh(lambd * r * y_i_mu)
        raise ValueError("Z_i_mu_func is defined only for d=1")

    def loss(self, xi_batch, y_batch, loss_type="CE", lambd=1.0, r=1, l2=None):
        if loss_type == "CE":
            return self.compute_crossentropy(xi_batch, y_batch, lambd, r, l2)
        elif loss_type == "MSE":
            return self.compute_MSE(xi_batch, y_batch, lambd, l2)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def _maybe_add_l2(self, loss, l2):
        if l2 is not None and l2 is not False:
            loss = loss + l2 * (self.J ** 2).mean()
        return loss

    def compute_MSE(self, xi_batch, y_batch, lambd, l2):
        J_x = torch.einsum("jab,mjb->ma", self.J, xi_batch)
        energy_mu = ((y_batch - lambd * J_x) ** 2).mean()
        return self._maybe_add_l2(energy_mu, l2)

    def compute_crossentropy(self, xi_batch, y_batch, lambd, r, l2):
        """
        Dispatch according to the output label space.

        The input spin_type affects the empirical distribution of xi_batch, but
        the conditional pseudo-likelihood normalizer is determined by label_type.
        In particular, dictionary labels use a finite logsumexp normalizer over
        dictionary atoms before any fallback to the vector/continuous losses.
        """
        if self.label_type == "dictionary":
            return self._crossentropy_dictionary(xi_batch, y_batch, lambd, l2)

        if self.label_type == "vector":
            if self.d == 1:
                return self._crossentropy_vector_d1(xi_batch, y_batch, lambd, r, l2)
            return self._crossentropy_vector_ddim(xi_batch, y_batch, lambd, r, l2)

        if self.label_type == "continuous":
            return self._crossentropy_continuous(xi_batch, y_batch, lambd, r, l2)

        raise ValueError(f"Unknown label_type: {self.label_type}")

    def _crossentropy_dictionary(self, xi_batch, y_batch, lambd, l2=None):
        """
        Pseudo-likelihood for dictionary-valued labels.

        For u = J x and dictionary D = {v_1, ..., v_D}, this implements

            L = - y·u + (1/lambda) log sum_k exp(lambda v_k·u)

        averaged over the batch. Notice that the whole loss is NOT multiplied by
        lambda; only the finite dictionary normalizer is scaled by 1/lambda, as
        in the spherical pseudo-likelihood convention used elsewhere in this file.
        """
        if self.d <= 1:
            raise ValueError("Dictionary-label crossentropy is only supported for d > 1.")
        if self.dictionary is None:
            raise ValueError("dictionary must be provided for label_type='dictionary'.")

        dictionary = self.dictionary.to(device=xi_batch.device, dtype=xi_batch.dtype)  # [D,d]
        J = self.J.to(device=xi_batch.device, dtype=xi_batch.dtype)

        u = torch.einsum("jab,mjb->ma", J, xi_batch)  # [M,d]
        y_dot_u = torch.einsum("ma,ma->m", y_batch, u)  # [M]

        logits = lambd * (u @ dictionary.T)  # [M,D]
        log_normalization = torch.logsumexp(logits, dim=-1)  # [M]

        loss = (-y_dot_u + (1.0 / lambd) * log_normalization).mean()
        return self._maybe_add_l2(loss, l2)

    def _crossentropy_vector_d1(self, xi_batch, y_batch, lambd, r, l2):
        """Pseudo-likelihood for binary labels, d=1."""
        J_x = torch.einsum("jab,mjb->ma", self.J, xi_batch)  # [M,1]
        y_i_mu = J_x.norm(dim=-1)  # [M]
        y_dot_u = torch.einsum("ma,ma->m", y_batch, J_x)  # [M]
        energy_mu = -y_dot_u + (1.0 / lambd) * torch.log(self.Z_i_mu_func(y_i_mu, lambd, r) + 1e-9)
        return self._maybe_add_l2(energy_mu.mean(), l2)

    def _crossentropy_vector_ddim(self, xi_batch, y_batch, lambd, r, l2):
        """
        Pseudo-likelihood for spherical vector labels with d > 1:

            L = - y·u + (1/lambda) log K_d(lambda r ||u||)
        """
        J_x = torch.einsum("jab,mjb->ma", self.J, xi_batch)  # [M,d]
        u_norm = J_x.norm(dim=-1)  # [M]
        y_dot_u = torch.einsum("ma,ma->m", y_batch, J_x)  # [M]
        x_arg = lambd * r * u_norm
        normalization = LogKd.apply(x_arg, self.d)
        energy_mu = -y_dot_u + (1.0 / lambd) * normalization
        return self._maybe_add_l2(energy_mu.mean(), l2)

    def _crossentropy_continuous(self, xi_batch, y_batch, lambd, r=1, l2=None):
        """Pseudo-likelihood for continuous/Gaussian labels."""
        if self.gamma is None or self.gamma == 0:
            raise ValueError("gamma must be non-zero for label_type='continuous'.")

        J_x = torch.einsum("jab,mjb->ma", self.J, xi_batch)  # [M,d]
        y_dot_u = (y_batch * J_x).sum(dim=-1)  # [M]
        u_norm_sq = (J_x ** 2).sum(dim=-1)  # [M]
        energy_mu = -y_dot_u + (lambd / (2.0 * self.gamma)) * u_norm_sq
        return self._maybe_add_l2(energy_mu.mean(), l2)

    def _nearest_dictionary_elements(self, h):
        if self.dictionary is None:
            raise ValueError("dictionary must be provided for label_type='dictionary'.")
        dictionary = self.dictionary.to(device=h.device, dtype=h.dtype)
        scores = h @ dictionary.T
        indices = scores.argmax(dim=-1)
        return dictionary[indices]

    def forward(self, xi_batch):
        u_pred = torch.einsum("jab,mjb->ma", self.J, xi_batch)

        if self.label_type == "dictionary":
            return self._nearest_dictionary_elements(u_pred)
        if self.label_type == "vector":
            return self._normalize_vectors(u_pred)
        if self.label_type == "continuous":
            return u_pred.clone()

        raise ValueError(f"Unknown label_type: {self.label_type}")
