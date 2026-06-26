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
            xi = xi * math.sqrt(self.d) / torch.norm(xi, dim=-1, keepdim=True)
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
        """Map each teacher output to the dictionary atom with largest dot product."""
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
    test_xi: torch.Tensor,   # shape [P_test, N, d]
) -> torch.Tensor:
    """
    Dictionary-based generalization error:
    - build a random dictionary of D vectors in R^d with norm sqrt(d)
    - compute student and teacher labels on test_xi
    - for each test sample, pick the dictionary atom with max cosine similarity
    - return a tensor [P_test] with 0 if argmax matches, 1 otherwise
    """
    if test_xi.ndim != 3:
        raise ValueError(f"test_xi must have shape [P_test, N, d], got {tuple(test_xi.shape)}")
    if test_xi.shape[-1] != d:
        raise ValueError(f"test_xi last dim must be d={d}, got {test_xi.shape[-1]}")
    if D <= 0:
        raise ValueError("D must be > 0")

    device = test_xi.device
    dtype = test_xi.dtype

    # 1) Generate D random vectors in R^d with norm exactly sqrt(d)
    dict_vecs = torch.randn(D, d, device=device, dtype=dtype)
    dict_vecs = dict_vecs / dict_vecs.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    dict_vecs = dict_vecs * (float(d) ** 0.5)  # each row has norm sqrt(d)

    # 2) Student labels y_hat via model.forward
    model_device = next(model.parameters()).device if any(True for _ in model.parameters()) else device
    xi_in = test_xi.to(model_device)
    y_student = model(xi_in)  # expected shape [P_test, d]

    # 3) Teacher labels from dataset.T (same einsum as in forward)
    T = dataset.T.to(model_device)  # expected shape [N, d, d] (i.e., "jab")
    u_teacher = torch.einsum("jab,mjb->ma", T, xi_in)

    if getattr(model, "spin_type", None) == "vector":
        # reuse model's normalization to match its conventions
        y_teacher = model.normalize_x(u_teacher)
    elif getattr(model, "spin_type", None) == "continuous":
        y_teacher = u_teacher.clone()
    else:
        raise ValueError(f"Unknown/unsupported model.spin_type={getattr(model,'spin_type',None)}")

    # Move dictionary to model device too (for the matmuls)
    dict_vecs = dict_vecs.to(model_device)

    # 4) Cosine similarity with all D dictionary vectors
    # cos(y, v) = (y·v) / (||y|| ||v||)
    y_student_n = y_student / y_student.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    y_teacher_n = y_teacher / y_teacher.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    dict_n = dict_vecs / dict_vecs.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    # similarities: [P_test, D]
    sim_student = y_student_n @ dict_n.T
    sim_teacher = y_teacher_n @ dict_n.T

    # 5) Argmax over D for each test point
    arg_student = sim_student.argmax(dim=1)  # [P_test]
    arg_teacher = sim_teacher.argmax(dim=1)  # [P_test]

    # 6) 0 if same argmax, 1 otherwise
    err = (arg_student != arg_teacher).to(torch.long)  # [P_test]

    # return on the same device as test_xi (optional; you can drop this if you want model_device)
    return err.to(device)

