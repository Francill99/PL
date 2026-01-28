import torch, math

_LOG_2PI = math.log(2.0 * math.pi)
_LOG_PI  = math.log(math.pi)


def _log_surface_area_sphere(d: int, device, dtype) -> torch.Tensor:
    # log |S^{d-1}| = log(2) + (d/2) log(pi) - lgamma(d/2)
    dh = torch.tensor(0.5 * d, device=device, dtype=dtype)
    return math.log(2.0) + 0.5 * d * _LOG_PI - torch.lgamma(dh)


def logK_d2(x: torch.Tensor) -> torch.Tensor:
    # x >= 0
    # log I0 = log(i0e) + x
    return _LOG_2PI + torch.log(torch.special.i0e(x)) + x

def A_d2(x: torch.Tensor) -> torch.Tensor:
    # A2 = I1/I0 = i1e/i0e (stable)
    return torch.special.i1e(x) / torch.special.i0e(x)


def A_d_highd(x: torch.Tensor, d: int, eps: float = 1e-12, x_small: float = 1e-3) -> torch.Tensor:
    """
    High-d (large nu) approximation of A_d(x) = d/dx log K_d(x) = I_{nu+1}(x)/I_nu(x),
    with stability patches.

    Args:
        x: tensor >=0
        d: dimension (>1)
        eps: clamp for divisions/logs
        x_small: threshold where we use small-x limit A ~ x/d

    Returns:
        A tensor same shape as x
    """
    d = int(d)
    nu = 0.5 * d - 1.0
    # avoid nu<=0 (d>2 in practice for this approximation, but keep safe)
    nu_t = torch.tensor(max(nu, 1e-6), device=x.device, dtype=x.dtype)

    # x = x.clamp_min(0.0)
    # # small-x exact limit
    # small = x < x_small

    z = x / nu_t
    # A ≈ z/(1+sqrt(1+z^2))
    s = torch.sqrt(1.0 + z * z)
    A = z / (1.0 + s + eps)

    # patch for tiny x (prevents any 0/0 and matches exact limit)
    #A_small = x / float(d)
    return A

def logK_d_highd(x: torch.Tensor, d: int, eps: float = 1e-12, x_small: float = 1e-3) -> torch.Tensor:
    """
    High-d (large nu) approximation of log K_d(x), with a small-x series patch:
        log K_d(x) ~ log|S^{d-1}| + x^2/(2d)

    Args:
        x: tensor >=0
        d: dimension (>1)
        eps: clamp for log/div
        x_small: threshold for small-x patch

    Returns:
        logK tensor same shape as x
    """
    d = int(d)
    nu = 0.5 * d - 1.0
    nu_val = max(nu, 1e-6)
    nu_t = torch.tensor(nu_val, device=x.device, dtype=x.dtype)

    #x = x.clamp_min(0.0)
    #small = x < x_small

    # --- asymptotic branch ---
    z = x.abs() / nu_t
    s = torch.sqrt(1.0 + z * z)  # sqrt(1+z^2)

    # eta(z) = s + log( z / (1+s) )
    # stabilize: clamp z away from 0
    # z_safe = z.clamp_min(eps)
    # denom = (1.0 + s).clamp_min(1.0 + eps)
    eta = s + torch.log(z / (1+s))

    # log I_nu(nu z) ≈ -0.5 log(2πν) - 0.25 log(1+z^2) + ν eta
    logI = (-0.5 * torch.log(torch.tensor(2.0 * math.pi, device=x.device, dtype=x.dtype) * nu_t)
            -0.25 * torch.log(1.0 + z * z)
            + nu_t * eta)

    # log K_d(x) = (d/2) log(2π) + logI - ν log x
    #x_safe = x.clamp_min(eps)
    logK_as = 0.5 * d * _LOG_2PI + logI - nu_t * torch.log(x.abs())

    # --- small-x patch ---
    # logS = _log_surface_area_sphere(d, x.device, x.dtype)
    # logK_small = logS + (x * x) / (2.0 * float(d))

    return logK_as

def _J_half_pair(x: torch.Tensor, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (J_{n-1/2}(x), J_{n+1/2}(x)) where J = exp(-x) I.

    d = 2n+1  with n>=1.
    """
    # protect division; x is >=0 but can hit 0 numerically
    x = x.clamp_min(1e-12)

    # s = sqrt(pi/(2x))
    s = torch.sqrt((math.pi / 2.0) / x)

    e2 = torch.exp(-2.0 * x)
    Jm = s * 0.5 * (1.0 + e2)  # J_{-1/2}
    Jp = s * 0.5 * (1.0 - e2)  # J_{ 1/2}

    # iterate to reach (n-1/2, n+1/2)
    # after k steps: we have (J_{k-1/2}, J_{k+1/2})
    a = Jm  # J_{-1/2}
    b = Jp  # J_{ 1/2}
    for k in range(0, n):
        # compute J_{k+3/2} = J_{k-1/2} - (2k+1)/x * J_{k+1/2}
        c = a - ((2.0 * k + 1.0) / x) * b
        a, b = b, c

    # now a = J_{n-1/2}, b = J_{n+1/2}
    return a, b

def A_odd_d(x: torch.Tensor, d: int) -> torch.Tensor:
    assert d % 2 == 1 and d > 1
    n = (d - 1) // 2  # d=2n+1
    # small-x limit: A_d(x) ~ x/d
    x0 = 1e-4
    small = x < x0

    J_nu, J_nu1 = _J_half_pair(x, n)
    A = J_nu1 / J_nu

    return torch.where(small, x / float(d), A)

def logK_odd_d(x: torch.Tensor, d: int) -> torch.Tensor:
    assert d % 2 == 1 and d > 1
    n  = (d - 1) // 2
    nu = 0.5 * d - 1.0

    x0 = 1e-4
    small = x < x0

    # exact (via J_nu)
    J_nu, _ = _J_half_pair(x, n)
    logI = torch.log(J_nu) + x
    logK = 0.5 * d * _LOG_2PI + logI - nu * torch.log(x.clamp_min(1e-12))

    # small-x expansion: K_d(x) = |S^{d-1}| * (1 + x^2/(2d) + O(x^4))
    # => logK ~ log|S^{d-1}| + x^2/(2d)
    logS = _log_surface_area_sphere(d, x.device, x.dtype)
    logK_small = logS + (x * x) / (2.0 * float(d))

    return torch.where(small, logK_small, logK)


def logK_d_forward_exact(x: torch.Tensor, d: int) -> torch.Tensor:
    if d == 2:
        return logK_d2(x)
    if d % 2 == 1 and d > 1:
        return logK_odd_d(x, d)
    raise ValueError("Exact implementation provided only for d=2 and odd d>1 (half-integer case).")

def A_d_exact(x: torch.Tensor, d: int) -> torch.Tensor:
    if d == 2:
        return A_d2(x)
    if d % 2 == 1 and d > 1:
        return A_odd_d(x, d)
    raise ValueError("Exact implementation provided only for d=2 and odd d>1 (half-integer case).")


class LogKd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, d, high_d=False):
        d = int(d)
        ctx.d = d
        ctx.high_d = bool(high_d)
        ctx.save_for_backward(x)
        if ((d>1 and d%2==1) and high_d == False) or d==2:
            return logK_d_forward_exact(x, d)   
        elif (d>1) and high_d==True:
            return logK_d_highd(x,d)
        

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        d = ctx.d
        high_d = ctx.high_d
        if (d>1 and d%2==1) or d==2:
            grad_x = grad_out * A_d_exact(x, d)    
        elif (d>1) and high_d==True:
            grad_x = grad_out * A_d_highd(x, d)
        return grad_x, None, None
