import math
import torch
from torch.utils.data import Dataset
from PL.model.model_classifier import Classifier
from PL.dataset.dictionary import dictionary_from

class Dataset_Teacher(Dataset):
    """
    Teacher-driven dataset for generic d, with a local teacher tensor per site:
        T: [N, d, d]
    Labels are produced as a "single spin" object:
        y_mu,i = g( T_i @ xi_mu,i + noise )
    where g enforces the same nature as xi (vector/continuous).
    """

    def __init__(
        self,
        P: int,
        N: int,
        d: int,
        seed: int,
        sigma: float,
        spin_type: str = "continuous",   # "vector", "continuous", or "dictionary"
        label_type: str = "continuous",  # "vector", "continuous", or "dictionary"
        dictionary: torch.Tensor | None = None,
    ):
        self.P = P
        self.N = N
        self.d = d
        self.sigma = sigma
        self.spin_type = spin_type
        self.label_type = label_type
        self.dictionary = self._prepare_dictionary(dictionary)

        torch.manual_seed(seed)

        # Teacher
        self.T = torch.randn(N, d, d)
        teacher_norm2 = float(N * d)
        self.Teacher = self._normalize_frobenius(self.T, target_norm2=teacher_norm2)

        # Training data
        self.xi = self._make_inputs(P)
        self.y = self._make_labels(self.xi)

    def _prepare_dictionary(self, dictionary: torch.Tensor | None) -> torch.Tensor | None:
        """Validate and store an external dictionary when it is requested."""
        uses_dictionary = self.spin_type == "dictionary" or self.label_type == "dictionary"

        allowed_spin_types = {"vector", "continuous", "dictionary"}
        allowed_label_types = {"vector", "continuous", "dictionary"}
        if self.spin_type not in allowed_spin_types:
            raise ValueError("spin_type must be 'vector', 'continuous', or 'dictionary'")
        if self.label_type not in allowed_label_types:
            raise ValueError("label_type must be 'vector', 'continuous', or 'dictionary'")

        if not uses_dictionary:
            return None
        if dictionary is None:
            raise ValueError(
                "A dictionary tensor with shape [D, d] must be provided when "
                "spin_type='dictionary' or label_type='dictionary'."
            )
        if not isinstance(dictionary, torch.Tensor):
            raise TypeError(f"dictionary must be a torch.Tensor, got {type(dictionary)!r}.")
        if dictionary.ndim != 2:
            raise ValueError(f"dictionary must have shape [D, d], got {tuple(dictionary.shape)}.")
        if not dictionary.is_floating_point():
            raise TypeError(f"dictionary must be a floating point tensor, got dtype={dictionary.dtype}.")
        if dictionary.shape[0] <= 0:
            raise ValueError("dictionary must contain at least one element, got D=0.")
        if dictionary.shape[1] != self.d:
            raise ValueError(
                f"dictionary has last dimension {dictionary.shape[1]}, but dataset d={self.d}."
            )

        return dictionary_from(dictionary)

    def _make_inputs(self, P: int) -> torch.Tensor:
        """Generate input spins according to spin_type."""
        if self.spin_type == "dictionary":
            indices = torch.randint(
                0,
                self.dictionary.shape[0],
                (P, self.N),
                device=self.dictionary.device,
            )
            return self.dictionary[indices]
    
        xi = torch.randn(P, self.N, self.d) * self.sigma
    
        if self.spin_type == "vector":
            if self.d == 1:
                # Ising case: avoid xi / |xi|
                xi = torch.where(xi >= 0, torch.ones_like(xi), -torch.ones_like(xi))
            else:
                xi = xi * math.sqrt(self.d) / xi.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    
        elif self.spin_type == "continuous":
            pass
        else:
            raise ValueError("spin_type must be 'vector', 'continuous', or 'dictionary'")
    
        return xi

    def _make_labels(self, xi: torch.Tensor) -> torch.Tensor:
        """
        xi: [P,N,d]
        returns y: [P,d]
        """
        teacher = self.Teacher.to(device=xi.device, dtype=xi.dtype)
        h = torch.einsum("iab,pib->pa", teacher, xi)

        if self.label_type == "vector":
            y = self.normalize(h)
        elif self.label_type == "continuous":
            y = h
        elif self.label_type == "dictionary":
            y = self._nearest_dictionary_elements(h)
        else:
            raise ValueError("label_type must be 'vector', 'continuous', or 'dictionary'")

        return y

    def _nearest_dictionary_elements(self, h: torch.Tensor) -> torch.Tensor:
        """Map each field to argmax_k h·v_k using raw, unnormalized dot products."""
        if self.dictionary is None:
            raise ValueError("dictionary must be provided for label_type='dictionary'.")

        dictionary = self.dictionary.to(device=h.device, dtype=h.dtype)
        scores = h @ dictionary.T
        indices = scores.argmax(dim=-1)
        return dictionary[indices]

    def generate_test_set(self, P_test: int):
        """
        Generate fresh test data from the same teacher.

        Parameters
        ----------
        P_test : int
            Number of test samples.
        seed : int or None
            Optional seed for reproducibility.

        Returns
        -------
        xi_test : torch.Tensor
            Shape [P_test, N, d]
        y_test : torch.Tensor
            Shape [P_test, d]
        """

        xi_test = self._make_inputs(P_test)
        y_test = self._make_labels(xi_test)
        return xi_test, y_test

    @staticmethod
    def normalize(x: torch.Tensor) -> torch.Tensor:
        norms = x.norm(dim=-1, keepdim=True) + 1e-9
        return x / norms

    @staticmethod
    def _normalize_frobenius(T: torch.Tensor, target_norm2: float) -> torch.Tensor:
        current = (T * T).sum()
        scale = math.sqrt(float(target_norm2) / float(current + 1e-12))
        return T * scale

    def __len__(self) -> int:
        return self.P

    def __getitem__(self, index: int):
        return self.xi[index], self.y[index]

def error_generalization_dictionary(
    model: "Classifier",
    dataset: "Dataset_Teacher",
    D: int,
    d: int,
    test_xi: torch.Tensor,
) -> torch.Tensor:
    """
    Dictionary classification error on a test set.

    Both teacher and student classes are selected with the same raw scores

        k*(h) = argmax_k h · v_k,

    using the dictionary already stored in ``dataset``/``model``. No field or
    dictionary vector is normalized, so Gaussian and spherical dictionaries are
    genuinely different.
    """
    if test_xi.ndim != 3:
        raise ValueError(f"test_xi must have shape [P_test, N, d], got {tuple(test_xi.shape)}")
    if test_xi.shape[-1] != d:
        raise ValueError(f"test_xi last dim must be d={d}, got {test_xi.shape[-1]}")

    dataset_dictionary = getattr(dataset, "dictionary", None)
    model_dictionary = getattr(model, "dictionary", None)
    dictionary = dataset_dictionary if dataset_dictionary is not None else model_dictionary
    if dictionary is None:
        raise ValueError("A stored dictionary is required for dictionary generalization error.")
    if dictionary.ndim != 2 or dictionary.shape[1] != d:
        raise ValueError(
            f"dictionary must have shape [D, d] with d={d}, got {tuple(dictionary.shape)}"
        )
    if D is not None and int(D) != dictionary.shape[0]:
        raise ValueError(f"D={D} but the stored dictionary contains {dictionary.shape[0]} atoms.")

    parameter = next(model.parameters(), None)
    model_device = parameter.device if parameter is not None else test_xi.device
    model_dtype = parameter.dtype if parameter is not None else test_xi.dtype

    xi = test_xi.to(device=model_device, dtype=model_dtype)
    dictionary = dictionary.to(device=model_device, dtype=model_dtype)

    # Student field u_s = Jx.
    J = model.J.to(device=model_device, dtype=model_dtype)
    u_student = torch.einsum("jab,mjb->ma", J, xi)

    # Use the normalized teacher tensor actually used by Dataset_Teacher to
    # generate labels, rather than the unnormalized temporary tensor self.T.
    teacher = getattr(dataset, "Teacher", None)
    if teacher is None:
        raise ValueError("dataset.Teacher is required to compute teacher labels.")
    teacher = teacher.to(device=model_device, dtype=model_dtype)
    u_teacher = torch.einsum("jab,mjb->ma", teacher, xi)

    student_idx = (u_student @ dictionary.T).argmax(dim=-1)
    teacher_idx = (u_teacher @ dictionary.T).argmax(dim=-1)
    return (student_idx != teacher_idx).to(torch.long).to(test_xi.device)

