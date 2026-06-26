import math
from pathlib import Path

import torch


EXACT_D_MAX = 32

_LOG_2PI = math.log(2.0 * math.pi)

_TABLE_CACHE = {}


def _table_path() -> Path:
    return Path(__file__).with_name("kd_lookup_tables.pt")


def _load_tables(device, dtype):
    """
    Load lookup table once and cache it on the requested device/dtype.
    """
    key = (str(device), dtype)

    if key in _TABLE_CACHE:
        return _TABLE_CACHE[key]

    path = _table_path()

    if not path.exists():
        raise FileNotFoundError(
            f"Missing lookup table: {path}. "
            "Generate it once with: python -m PL.utils.build_kd_lookup"
        )

    payload = torch.load(path, map_location="cpu")

    x_grid = payload["x_grid"].to(device=device, dtype=dtype)
    logK = payload["logK"].to(device=device, dtype=dtype)
    A = payload["A"].to(device=device, dtype=dtype)

    x_max = float(payload["x_max"])
    n_grid = int(payload["n_grid"])
    dx = x_max / float(n_grid - 1)

    out = {
        "x_grid": x_grid,
        "logK": logK,
        "A": A,
        "x_max": x_max,
        "n_grid": n_grid,
        "dx": dx,
    }

    _TABLE_CACHE[key] = out
    return out


def _large_x_logK(x: torch.Tensor, d: int) -> torch.Tensor:
    """
    Large-x continuation outside the table.

    For x large:
        K_d(x) ~ exp(x) (2π/x)^((d-1)/2).
    """
    x_safe = x.clamp_min(1e-12)
    m = 0.5 * float(d - 1)
    return x_safe + m * (_LOG_2PI - torch.log(x_safe))


def _large_x_A(x: torch.Tensor, d: int) -> torch.Tensor:
    """
    Derivative of the large-x continuation:

        d/dx [x + m(log(2π)-log x)] = 1 - m/x.
    """
    x_safe = x.clamp_min(1e-12)
    m = 0.5 * float(d - 1)
    return (1.0 - m / x_safe).clamp(min=0.0, max=1.0)


def _high_d_logK(x: torch.Tensor, d: int, eps: float = 1e-12) -> torch.Tensor:
    """
    Approximation used automatically for d > 32.
    """
    d = int(d)
    nu = 0.5 * float(d) - 1.0
    nu_t = torch.tensor(max(nu, 1e-6), device=x.device, dtype=x.dtype)

    x = x.clamp_min(0.0)
    x_safe = x.clamp_min(eps)

    z = x_safe / nu_t
    s = torch.sqrt(1.0 + z * z)
    eta = s + torch.log(z.clamp_min(eps) / (1.0 + s))

    two_pi = torch.tensor(2.0 * math.pi, device=x.device, dtype=x.dtype)

    logI = (
        -0.5 * torch.log(two_pi * nu_t)
        -0.25 * torch.log(1.0 + z * z)
        + nu_t * eta
    )

    return 0.5 * float(d) * _LOG_2PI + logI - nu_t * torch.log(x_safe)


def _high_d_A(x: torch.Tensor, d: int, eps: float = 1e-12) -> torch.Tensor:
    """
    Approximation used automatically for d > 32.
    """
    d = int(d)
    nu = 0.5 * float(d) - 1.0
    nu_t = torch.tensor(max(nu, 1e-6), device=x.device, dtype=x.dtype)

    x = x.clamp_min(0.0)
    z = x / nu_t
    s = torch.sqrt(1.0 + z * z)

    return z / (1.0 + s + eps)


def _hermite_forward(x: torch.Tensor, d: int) -> torch.Tensor:
    """
    Cubic Hermite interpolation of logK_d(x) using precomputed
    logK and derivative A.

    This gives a smooth function, not piecewise-linear kinks.
    """
    tables = _load_tables(x.device, x.dtype)

    x_max = tables["x_max"]
    dx = tables["dx"]
    logK_table = tables["logK"]
    A_table = tables["A"]

    x_pos = x.clamp_min(0.0)
    inside = x_pos <= x_max

    x_in = x_pos.clamp(max=x_max)

    u = x_in / dx
    idx = torch.floor(u).to(torch.long)
    idx = idx.clamp(min=0, max=tables["n_grid"] - 2)

    t = u - idx.to(x.dtype)

    y0 = logK_table[d, idx]
    y1 = logK_table[d, idx + 1]

    m0 = A_table[d, idx]
    m1 = A_table[d, idx + 1]

    t2 = t * t
    t3 = t2 * t

    h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    h10 = t3 - 2.0 * t2 + t
    h01 = -2.0 * t3 + 3.0 * t2
    h11 = t3 - t2

    y_interp = h00 * y0 + h10 * dx * m0 + h01 * y1 + h11 * dx * m1

    y_large = _large_x_logK(x_pos, d)

    return torch.where(inside, y_interp, y_large)


def _hermite_derivative(x: torch.Tensor, d: int) -> torch.Tensor:
    """
    Derivative of the same cubic Hermite interpolant used in forward.
    This makes backward consistent with the interpolated forward.
    """
    tables = _load_tables(x.device, x.dtype)

    x_max = tables["x_max"]
    dx = tables["dx"]
    logK_table = tables["logK"]
    A_table = tables["A"]

    x_pos = x.clamp_min(0.0)
    inside = x_pos <= x_max

    x_in = x_pos.clamp(max=x_max)

    u = x_in / dx
    idx = torch.floor(u).to(torch.long)
    idx = idx.clamp(min=0, max=tables["n_grid"] - 2)

    t = u - idx.to(x.dtype)

    y0 = logK_table[d, idx]
    y1 = logK_table[d, idx + 1]

    m0 = A_table[d, idx]
    m1 = A_table[d, idx + 1]

    t2 = t * t

    dh00 = 6.0 * t2 - 6.0 * t
    dh10 = 3.0 * t2 - 4.0 * t + 1.0
    dh01 = -6.0 * t2 + 6.0 * t
    dh11 = 3.0 * t2 - 2.0 * t

    dy_dt = dh00 * y0 + dh10 * dx * m0 + dh01 * y1 + dh11 * dx * m1
    dy_dx = dy_dt / dx

    y_large_grad = _large_x_A(x_pos, d)

    return torch.where(inside, dy_dx, y_large_grad)


class _LogKdLookup(torch.autograd.Function):
    """
    Lookup-table autograd function.

    Fixed behavior:
        d <= 32 : precomputed lookup table + cubic Hermite interpolation
        d > 32  : high-d approximation

    No external flag. The cutoff is fixed here.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, d: int):
        d = int(d)

        if d < 1:
            raise ValueError(f"d must be >= 1, got {d}")

        if not x.is_floating_point():
            raise TypeError(f"x must be floating point, got dtype={x.dtype}")

        ctx.d = d
        ctx.save_for_backward(x)

        if d <= EXACT_D_MAX:
            return _hermite_forward(x, d)

        return _high_d_logK(x, d)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (x,) = ctx.saved_tensors
        d = ctx.d

        if d <= EXACT_D_MAX:
            grad_x = grad_out * _hermite_derivative(x, d)
        else:
            grad_x = grad_out * _high_d_A(x, d)

        return grad_x, None


class LogKd:
    """
    Public API.

    Use:
        LogKd.apply(x_arg, d)

    Do not pass high_d=True/False.
    """

    @staticmethod
    def apply(x: torch.Tensor, d: int) -> torch.Tensor:
        return _LogKdLookup.apply(x, int(d))


def logK_d(x: torch.Tensor, d: int) -> torch.Tensor:
    return LogKd.apply(x, d)


def A_d(x: torch.Tensor, d: int) -> torch.Tensor:
    d = int(d)

    if d <= EXACT_D_MAX:
        return _hermite_derivative(x, d)

    return _high_d_A(x, d)