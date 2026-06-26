import math
from pathlib import Path

import torch


EXACT_D_MAX = 32
DEFAULT_X_MAX = 200.0
DEFAULT_N_GRID = 16385

_LOG_2PI = math.log(2.0 * math.pi)
_LOG_PI = math.log(math.pi)
_LOG_2 = math.log(2.0)


def _log_surface_area_sphere(d: int, device, dtype):
    half_d = torch.tensor(0.5 * float(d), device=device, dtype=dtype)
    return _LOG_2 + 0.5 * float(d) * _LOG_PI - torch.lgamma(half_d)


def _small_logK(x: torch.Tensor, d: int) -> torch.Tensor:
    d_float = float(d)
    logS = _log_surface_area_sphere(d, x.device, x.dtype)
    return logS + x.pow(2) / (2.0 * d_float) - x.pow(4) / (
        4.0 * d_float * d_float * (d_float + 2.0)
    )


def _small_A(x: torch.Tensor, d: int) -> torch.Tensor:
    d_float = float(d)
    return x / d_float - x.pow(3) / (d_float * d_float * (d_float + 2.0))


def _log_2_cosh(x: torch.Tensor) -> torch.Tensor:
    ax = x.abs()
    return ax + torch.log1p(torch.exp(-2.0 * ax))


def _I_integer_scaled_pair_miller(
    x: torch.Tensor,
    n: int,
    *,
    m_extra: int = 120,
    x_min: float = 1e-6,
):
    """
    Returns exp(-x) I_n(x), exp(-x) I_{n+1}(x)
    for integer n >= 0 using Miller backward recurrence.
    """
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")

    xs = x.clamp_min(x_min)

    if n == 0:
        return torch.special.i0e(xs), torch.special.i1e(xs)

    M = max(int(m_extra), int(n) + 64)

    b_next = torch.zeros_like(xs)
    b_cur = torch.ones_like(xs)

    b_n = None
    b_np1 = None

    for k in range(M, 0, -1):
        b_prev = b_next + (2.0 * float(k) / xs) * b_cur

        if k == n + 1:
            b_np1 = b_cur
            b_n = b_prev

        b_next, b_cur = b_cur, b_prev

    b0 = b_cur
    scale = torch.special.i0e(xs) / b0

    return b_n * scale, b_np1 * scale


def _I_half_scaled_pair_miller(
    x: torch.Tensor,
    n: int,
    *,
    m_extra: int = 120,
    x_min: float = 1e-6,
):
    """
    For odd d = 2n + 1, returns

        exp(-x) I_{n-1/2}(x), exp(-x) I_{n+1/2}(x).
    """
    if n < 1:
        raise ValueError(f"n must be >= 1 for odd d > 1, got {n}")

    xs = x.clamp_min(x_min)

    M = max(int(m_extra), int(n) + 64)

    b_next = torch.zeros_like(xs)
    b_cur = torch.ones_like(xs)

    b_n = None
    b_np1 = None

    for j in range(M, 0, -1):
        b_prev = b_next + (2.0 * (float(j) - 0.5) / xs) * b_cur

        if j == n + 1:
            b_np1 = b_cur
            b_n = b_prev

        b_next, b_cur = b_cur, b_prev

    b0 = b_cur

    two_pi = torch.tensor(2.0 * math.pi, device=x.device, dtype=x.dtype)
    base = torch.rsqrt(two_pi * xs)
    e2 = torch.exp(-2.0 * xs)

    # exp(-x) I_{-1/2}(x)
    E_minus_half = base * (1.0 + e2)

    scale = E_minus_half / b0

    return b_n * scale, b_np1 * scale


def _logK_exact_reference(x: torch.Tensor, d: int) -> torch.Tensor:
    """
    Reference computation used only to build the table.
    """
    if d == 1:
        return _log_2_cosh(x)

    if d == 2:
        tiny = torch.finfo(x.dtype).tiny
        return _LOG_2PI + torch.log(torch.special.i0e(x).clamp_min(tiny)) + x

    small = x < 1e-5
    xs = x.clamp_min(1e-12)
    tiny = torch.finfo(x.dtype).tiny

    if d % 2 == 0:
        n = d // 2 - 1
        E_n, _ = _I_integer_scaled_pair_miller(x, n)
        logK = (
            0.5 * float(d) * _LOG_2PI
            + xs
            + torch.log(E_n.clamp_min(tiny))
            - float(n) * torch.log(xs)
        )
    else:
        n = (d - 1) // 2
        nu = 0.5 * float(d) - 1.0
        E_nu, _ = _I_half_scaled_pair_miller(x, n)
        logK = (
            0.5 * float(d) * _LOG_2PI
            + xs
            + torch.log(E_nu.clamp_min(tiny))
            - nu * torch.log(xs)
        )

    return torch.where(small, _small_logK(x, d), logK)


def _A_exact_reference(x: torch.Tensor, d: int) -> torch.Tensor:
    """
    Reference derivative used only to build the table:

        A_d(x) = I_{d/2}(x) / I_{d/2 - 1}(x).
    """
    if d == 1:
        return torch.tanh(x)

    if d == 2:
        tiny = torch.finfo(x.dtype).tiny
        return torch.special.i1e(x) / torch.special.i0e(x).clamp_min(tiny)

    small = x < 1e-5
    tiny = torch.finfo(x.dtype).tiny

    if d % 2 == 0:
        n = d // 2 - 1
        E_n, E_np1 = _I_integer_scaled_pair_miller(x, n)
        A = E_np1 / E_n.clamp_min(tiny)
    else:
        n = (d - 1) // 2
        E_nu, E_nu1 = _I_half_scaled_pair_miller(x, n)
        A = E_nu1 / E_nu.clamp_min(tiny)

    return torch.where(small, _small_A(x, d), A)


@torch.no_grad()
def build_kd_lookup(
    output_path: str | Path | None = None,
    *,
    x_max: float = DEFAULT_X_MAX,
    n_grid: int = DEFAULT_N_GRID,
    store_dtype: torch.dtype = torch.float32,
):
    """
    Build lookup tables for logK_d(x) and A_d(x), d=1,...,32.

    The saved file contains:
        x_grid : [n_grid]
        logK   : [33, n_grid], row 0 unused
        A      : [33, n_grid], row 0 unused
        x_max
        n_grid
        exact_d_max
    """
    if output_path is None:
        output_path = Path(__file__).with_name("kd_lookup_tables.pt")
    else:
        output_path = Path(output_path)

    x_grid = torch.linspace(0.0, float(x_max), int(n_grid), dtype=torch.float64)

    logK = torch.empty(EXACT_D_MAX + 1, n_grid, dtype=torch.float64)
    A = torch.empty(EXACT_D_MAX + 1, n_grid, dtype=torch.float64)

    logK[0].fill_(float("nan"))
    A[0].fill_(float("nan"))

    for d in range(1, EXACT_D_MAX + 1):
        print(f"building table for d={d}")
        logK[d] = _logK_exact_reference(x_grid, d)
        A[d] = _A_exact_reference(x_grid, d)

    payload = {
        "x_grid": x_grid.to(store_dtype),
        "logK": logK.to(store_dtype),
        "A": A.to(store_dtype),
        "x_max": float(x_max),
        "n_grid": int(n_grid),
        "exact_d_max": int(EXACT_D_MAX),
        "store_dtype": str(store_dtype),
    }

    torch.save(payload, output_path)
    print(f"saved {output_path}")


if __name__ == "__main__":
    build_kd_lookup()