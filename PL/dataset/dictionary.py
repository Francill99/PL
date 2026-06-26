import torch


def dictionary_from(dictionary: torch.Tensor) -> torch.Tensor:
    """
    Return an externally provided dictionary tensor.

    At the moment this function is intentionally the identity map. It is kept as
    a single entry point so that future dictionary preprocessing/validation can
    be added without changing the dataset API.

    Parameters
    ----------
    dictionary : torch.Tensor
        Tensor with shape [D, d], where D is the number of dictionary elements
        and d is the spin dimension.

    Returns
    -------
    torch.Tensor
        The same input tensor.
    """
    return dictionary


def random_dictionary(
    D: int,
    d: int,
    *,
    sigma: float = 1.0,
    on_the_sphere: bool = True,
    seed: int | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Create a random Gaussian dictionary.

    Parameters
    ----------
    D : int
        Number of dictionary elements.
    d : int
        Dimension of each dictionary element.
    sigma : float, optional
        Standard deviation of the Gaussian entries before optional projection.
    on_the_sphere : bool, optional
        If True, project every dictionary vector onto the unit sphere.
    seed : int or None, optional
        Optional random seed for reproducibility.
    device : torch.device, str, or None, optional
        Device on which the dictionary is created.
    dtype : torch.dtype, optional
        Floating point dtype of the dictionary.
    eps : float, optional
        Numerical floor used in the normalization.

    Returns
    -------
    torch.Tensor
        Dictionary tensor with shape [D, d]. If on_the_sphere=True, each row has
        unit norm up to numerical precision.
    """
    if D <= 0:
        raise ValueError(f"D must be positive, got D={D}.")
    if d <= 0:
        raise ValueError(f"d must be positive, got d={d}.")

    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        dictionary = sigma * torch.randn(D, d, generator=generator, device=device, dtype=dtype)
    else:
        dictionary = sigma * torch.randn(D, d, device=device, dtype=dtype)

    if on_the_sphere:
        dictionary = dictionary / dictionary.norm(dim=-1, keepdim=True).clamp_min(eps)

    return dictionary
