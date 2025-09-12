## Standard libraries
import os
import numpy as np
import random
import math
import time
import copy
import argparse
import torch
import gc

## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class CustomDataset(Dataset):
    def __init__(self, P, N, D, d, sigma, seed=None, on_sphere=True, coefficients="gaussian"):
        """
        P: Number of patterns
        N: Number of sites
        D: Number of features in the random field
        d: Dimensionality of each site
        sigma: Standard deviation of the Gaussian noise
        on_sphere: If True, normalize data to lie on a sphere
        coefficients: Type of coefficients ('binary' or 'gaussian')
        """
        self.P = P
        self.N = N
        self.D = D
        self.d = d
        self.sigma = sigma
        self.on_sphere = on_sphere
        self.coefficients = coefficients

        if seed is not None:
            torch.manual_seed(seed)

        # xi of shape [P, N, d]
        self.xi = torch.randn(P, N, d) * sigma

        # If D > 0, modify xi using the RF method
        if self.D > 0:
            self.RF()

        # Normalize xi along last dimension if on_sphere is True
        if self.on_sphere:
            self.xi = self.normalize(self.xi)

    def RF(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        # Create tensor f of shape [D, N, d] with random Gaussian numbers (std = sigma)
        self.f = torch.randint(0, 2, (self.D, self.N, self.d)).float() * 2 - 1

        # Normalize f along the last dimension if on_sphere is True
        if self.on_sphere:
            self.f = self.normalize(self.f)

        # Create tensor c of shape [P, D] based on coefficient type
        if self.coefficients == "binary":
            self.c = torch.randint(0, 2, (self.P, self.D)).float() * 2 - 1  # Random +1 or -1
        elif self.coefficients == "gaussian":
            self.c = torch.randn(self.P, self.D)  # Gaussian random numbers
        else:
            raise ValueError("coefficients must be 'binary' or 'gaussian'")

        # Divide c by sqrt(D)
        self.c = self.c / math.sqrt(self.D)

        # Update xi: xi[p] = sum_{k=0}^D c[p,k] * f[k]
        self.xi = torch.einsum('pk,kia->pia', self.c, self.f)
        # Create a mask where self.xi == 0
        mask = self.xi == 0

        # Generate random binary values (-1 or +1) for the positions where mask is True
        random_binary_values = torch.randint(0, 2, self.xi.shape, device=self.xi.device) * 2 - 1
        
        # Assign the random binary values to positions where self.xi == 0
        self.xi = torch.where(mask, random_binary_values, self.xi)
        if self.on_sphere:
            self.xi = self.normalize(self.xi)

    def get_generalization(self, P_hat, L=None):  #P_hat number of generalization vectors to get, L number of features to use
        if L==None:
            L = self.D
        if L>self.D:
            raise ValueError("L must be less than or equal to D")

        # Create tensor c of shape [P, N] based on coefficient type
        if self.coefficients == "binary":
            self.c = torch.randint(0, 2, (P_hat, self.D)).float() * 2 - 1  # Random +1 or -1
        elif self.coefficients == "gaussian":
            self.c = torch.randn(P_hat, self.D)  # Gaussian random numbers
        else:
            raise ValueError("coefficients must be 'binary' or 'gaussian'")

        # Divide c by sqrt(D)
        self.c = self.c / math.sqrt(self.D)

        #For each row of self.c, set to zero self.D-L random entries among the self.D. For each row, different random entries to set to zero
        indices_to_zero = torch.rand(self.P, self.D)
        indices_to_zero = indices_to_zero.argsort(dim=1)[:,:self.D-L]
        self.c = self.c.scatter(1, indices_to_zero, 0)

        # xi_new: xi[p] = sum_{k=0}^D c[p,k] * f[k]
        self.xi_new = torch.einsum('pk,kia->pia', self.c, self.f)
        self.xi_new = self.normalize(self.xi_new)
        return self.xi_new


    def normalize(self, x):
        # Normalize each d-dimensional vector in x along the last dimension
        norms = x.norm(dim=-1, keepdim=True)+1e-9
        return x / norms

    def __len__(self):
        # Return the number of patterns P
        return self.P

    def __getitem__(self, index):
        # Return the pattern xi at the given index
        return self.xi[index]