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
    Return an externally provided dictionary tensor.

    This function is intentionally an identity map. It is kept as a single entry
    point so future dictionary preprocessing can be added without changing the
    dataset API. In particular, externally supplied Gaussian dictionaries are
    not normalized here.
    """
    return dictionary


def random_dictionary(
    D: int,
    d: int,
    *,
    sigma: float = 1.0,
    dictionary_type: str | None = "spherical",
    on_the_sphere: bool | None = None,
    seed: int | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Create a random dictionary with shape ``[D, d]``.

    ``dictionary_type="spherical"`` (default)
        Draw Gaussian vectors and normalize every row to unit norm. This is the
        previous behavior and remains the default for backward compatibility.

    ``dictionary_type="gaussian"``
        Draw independent vectors from ``N(0, sigma^2 I_d)`` and do not normalize
        them. With the default ``sigma=1`` this is exactly a d-dimensional
        Gaussian dictionary with identity covariance.

    ``on_the_sphere`` is retained as a backward-compatible alias. Old calls with
    ``on_the_sphere=True/False`` continue to select spherical/Gaussian vectors.
    """
    if D <= 0:
        raise ValueError(f"D must be positive, got D={D}.")
    if d <= 0:
        raise ValueError(f"d must be positive, got d={d}.")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got sigma={sigma}.")
    if not torch.empty((), dtype=dtype).is_floating_point():
        raise TypeError(f"dtype must be floating point, got dtype={dtype}.")

    dictionary_type = resolve_dictionary_type(
        dictionary_type,
        on_the_sphere=on_the_sphere,
    )

    generator = None
    if seed is not None:
        generator_device = device if device is not None else "cpu"
        generator = torch.Generator(device=generator_device)
        generator.manual_seed(int(seed))

    dictionary = sigma * torch.randn(
        D,
        d,
        generator=generator,
        device=device,
        dtype=dtype,
    )

    if dictionary_type == "spherical":
        dictionary = dictionary / dictionary.norm(dim=-1, keepdim=True).clamp_min(eps)

    return dictionary
