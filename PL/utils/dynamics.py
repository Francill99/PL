import numpy as np
import torch

def _init_with_overlap_binary(xi: torch.Tensor, rho: float) -> torch.Tensor:
    """
    xi: [B, N, 1] with entries ±1
    returns x0 with mean overlap ~ rho w.r.t. xi by flipping a fraction of sites.
    """
    x0 = xi.clone()
    B, N, d = x0.shape
    assert d == 1

    # For ±1, overlap after flipping k sites is 1 - 2k/N  -> k = N(1-rho)/2
    k = int(round(N * (1.0 - float(rho)) / 2.0))
    if k <= 0:
        return x0
    if k >= N:
        return -x0

    # flip k random sites per pattern
    for b in range(B):
        idx = torch.randperm(N, device=xi.device)[:k]
        x0[b, idx, :] *= -1
    return x0


def _init_with_overlap_sphere(xi: torch.Tensor, rho: float, eps: float = 1e-12) -> torch.Tensor:
    """
    xi: [B, N, d] with per-site norm ~1 (on sphere)
    Build x0 with per-site cosine similarity exactly rho in expectation:
        x0 = rho*xi + sqrt(1-rho^2)*eta_perp
    where eta_perp is random noise orthogonal to xi (per site), then normalize.
    """
    rho = float(rho)
    if rho >= 1.0:
        return xi.clone()
    if rho <= -1.0:
        return (-xi).clone()

    B, N, d = xi.shape
    eta = torch.randn_like(xi)

    # project out component parallel to xi (per site)
    # eta_perp = eta - (eta·xi) xi
    dot = torch.einsum("bnd,bnd->bn", eta, xi).unsqueeze(-1)  # [B,N,1]
    eta_perp = eta - dot * xi

    # normalize eta_perp per site
    eta_norm = torch.linalg.norm(eta_perp, dim=-1, keepdim=True).clamp_min(eps)
    eta_perp = eta_perp / eta_norm

    x0 = rho * xi + np.sqrt(max(0.0, 1.0 - rho**2)) * eta_perp

    # normalize per site (same idea as model.normalize_x) :contentReference[oaicite:2]{index=2}
    x0 = x0 / torch.linalg.norm(x0, dim=-1, keepdim=True).clamp_min(eps)
    return x0


def _overlap_with_initial(x: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
    """
    x, x0: [B, N, d]
    returns overlaps: [B] = mean_i (x_i · x0_i)
    """
    site_dot = torch.einsum("bnd,bnd->bn", x, x0)  # [B,N]
    return site_dot.mean(dim=1)  # [B]


@torch.no_grad()
def compute_retrieval_map(
    xi: torch.Tensor,
    init_overlaps: np.ndarray,
    model,
    steps: int = 1000,
    num_data: int = 10,
    device: torch.device | str | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """
    Args:
        xi: dataset.xi, shape [P, N, d]
        init_overlaps: array of rho values in [0,1] (or [-1,1] for vectors)
        model: must have dyn_step(x) acting on [B,N,d] :contentReference[oaicite:3]{index=3}
        steps: number of dynamics steps
        num_data: number of patterns/runs to average over (B)
        device: where to run (defaults to model params device if possible)
        seed: optional RNG seed

    Returns:
        final_overlaps: shape [len(init_overlaps), num_data]
            final_overlaps[k,b] = overlap(x_T^{(b)}, x0^{(b)}) for rho=init_overlaps[k]
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    if device is None:
        try:
            device = next(model.parameters()).device
        except Exception:
            device = xi.device

    xi = xi.to(device)
    P, N, d = xi.shape

    B = min(int(num_data), int(P))
    # choose B patterns (randomly) to estimate the retrieval map
    idx = torch.randperm(P, device=device)[:B]
    xi_b = xi[idx]  # [B,N,d]

    init_overlaps = np.asarray(init_overlaps, dtype=float)
    out = torch.empty((len(init_overlaps), B), device="cpu", dtype=torch.float32)

    for k, rho in enumerate(init_overlaps):
        if d == 1:
            x0 = _init_with_overlap_binary(xi_b, rho)  # flips (binary) :contentReference[oaicite:4]{index=4}
        else:
            x0 = _init_with_overlap_sphere(xi_b, rho)

        # evolve dynamics
        x = x0
        for _ in range(int(steps)):
            x = model.dyn_step(x)  # :contentReference[oaicite:5]{index=5}

        # overlap with the *initial* vectors
        ov = _overlap_with_initial(x, x0)  # [B]
        out[k] = ov.detach().cpu()

    return out.numpy()
