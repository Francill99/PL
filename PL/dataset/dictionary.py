import torch


VALID_DICTIONARY_TYPES = {"spherical", "gaussian"}


def resolve_dictionary_type(
    dictionary_type: str | None = "spherical",
    *,
    on_the_sphere: bool | None = None,
) -> str:
    """
    Resolve the dictionary distribution while keeping the old boolean API valid.

    Parameters
    ----------
    dictionary_type : {"spherical", "gaussian"} or None
        Explicit dictionary distribution. ``None`` means ``"spherical"``.
    on_the_sphere : bool or None
        Backward-compatible alias used by older code. When provided, it takes
        precedence: ``True`` -> ``"spherical"`` and ``False`` -> ``"gaussian"``.
    """
    if on_the_sphere is not None:
        dictionary_type = "spherical" if on_the_sphere else "gaussian"

    if dictionary_type is None:
        dictionary_type = "spherical"

    dictionary_type = str(dictionary_type).lower()
    if dictionary_type not in VALID_DICTIONARY_TYPES:
        raise ValueError(
            f"dictionary_type must be one of {sorted(VALID_DICTIONARY_TYPES)}, "
            f"got {dictionary_type!r}."
        )
    return dictionary_type


def dictionary_from(dictionary: torch.Tensor) -> torch.Tensor:
    """
    Return an externally provided dictionary tensor without modifying it.

    In particular, externally supplied Gaussian dictionaries are never
    normalized here.
    """
    return dictionary


def power_law_covariance_eigenvalues(
    d: int,
    eta: float = 0.0,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    r"""
    Return the diagonal eigenvalues of the trace-normalized covariance

    .. math::

        \lambda_m = \frac{d\,m^{-\eta}}{\sum_{j=1}^d j^{-\eta}},
        \qquad m=1,\ldots,d.

    Hence ``sum(lambda) = d`` for every ``eta`` and ``eta=0`` gives exactly
    ``lambda_m=1``, i.e. the identity covariance.
    """
    if d <= 0:
        raise ValueError(f"d must be positive, got d={d}.")
    if eta < 0:
        raise ValueError(f"dictionary_eta must be non-negative, got eta={eta}.")
    if not torch.empty((), dtype=dtype).is_floating_point():
        raise TypeError(f"dtype must be floating point, got dtype={dtype}.")

    # Compute the normalized spectrum in float64 for numerical stability and
    # cast only at the end. Subtracting max(log_weight) also avoids underflow
    # for large eta.
    indices = torch.arange(1, d + 1, device=device, dtype=torch.float64)
    log_weights = -float(eta) * torch.log(indices)
    log_weights = log_weights - log_weights.max()
    weights = torch.exp(log_weights)
    eigenvalues = float(d) * weights / weights.sum()
    return eigenvalues.to(dtype=dtype)


def random_dictionary(
    D: int,
    d: int,
    *,
    sigma: float = 1.0,
    dictionary_type: str | None = "spherical",
    dictionary_eta: float = 0.0,
    on_the_sphere: bool | None = None,
    seed: int | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    eps: float = 1e-12,
) -> torch.Tensor:
    r"""
    Create a random dictionary with shape ``[D, d]``.

    ``dictionary_type="spherical"`` (default)
        Draw Gaussian vectors and normalize every row to unit norm. This is the
        original behavior and remains unchanged for backward compatibility.
        ``dictionary_eta`` must remain zero in this case.

    ``dictionary_type="gaussian"``
        Draw independent atoms

        .. math::

            z_k = g_k\,\Sigma_\eta^{1/2},
            \qquad g_k\sim\mathcal N(0,I_d),

        where ``Sigma_eta`` is diagonal and

        .. math::

            (\Sigma_\eta)_{mm}
            = \lambda_m
            = \frac{d\,m^{-\eta}}{\sum_{j=1}^d j^{-\eta}}.

        No atom-wise normalization is applied. Therefore
        ``Tr(Sigma_eta)=d`` and, for ``sigma=1``,
        ``E[||z_k||^2]=d`` for every eta. At ``eta=0`` this reduces exactly to
        the isotropic Gaussian dictionary ``N(0,I_d)``.

    ``sigma`` globally rescales the atoms, so the actual Gaussian covariance is
    ``sigma^2 Sigma_eta``. The default ``sigma=1`` implements the covariance in
    the equations above.

    ``on_the_sphere`` is retained as a backward-compatible alias. Old calls with
    ``on_the_sphere=True/False`` continue to select spherical/Gaussian vectors.
    """
    if D <= 0:
        raise ValueError(f"D must be positive, got D={D}.")
    if d <= 0:
        raise ValueError(f"d must be positive, got d={d}.")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got sigma={sigma}.")
    if dictionary_eta < 0:
        raise ValueError(
            f"dictionary_eta must be non-negative, got dictionary_eta={dictionary_eta}."
        )
    if not torch.empty((), dtype=dtype).is_floating_point():
        raise TypeError(f"dtype must be floating point, got dtype={dtype}.")

    dictionary_type = resolve_dictionary_type(
        dictionary_type,
        on_the_sphere=on_the_sphere,
    )

    if dictionary_type == "spherical" and float(dictionary_eta) != 0.0:
        raise ValueError(
            "dictionary_eta controls the covariance of a Gaussian dictionary. "
            "Use dictionary_type='gaussian' when dictionary_eta is non-zero."
        )

    generator = None
    if seed is not None:
        generator_device = device if device is not None else "cpu"
        generator = torch.Generator(device=generator_device)
        generator.manual_seed(int(seed))

    dictionary = torch.randn(
        D,
        d,
        generator=generator,
        device=device,
        dtype=dtype,
    )

    if dictionary_type == "spherical":
        # Keep the historical operation order: scale first, then normalize.
        # The scale cancels analytically but preserving the old path avoids any
        # change in existing spherical experiments.
        dictionary = float(sigma) * dictionary
        dictionary = dictionary / dictionary.norm(dim=-1, keepdim=True).clamp_min(eps)
        return dictionary

    eigenvalues = power_law_covariance_eigenvalues(
        d,
        eta=dictionary_eta,
        device=device,
        dtype=dtype,
    )
    return float(sigma) * dictionary * eigenvalues.sqrt().unsqueeze(0)
