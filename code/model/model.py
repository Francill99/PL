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


class TwoBodiesModel(nn.Module):
    def __init__(self, N, d, on_sphere, r=1):
        super(TwoBodiesModel, self).__init__()
        self.N = N
        self.d = d
        self.on_sphere = on_sphere
        self.r = r

        self.J = nn.Parameter(torch.randn(N, N, d, d))
        diagonal = self.J.data.diagonal(dim1=0, dim2=1)  # Get diagonal elements
        diagonal.fill_(0)
        self.normalize_J()

        self.mask = torch.ones(N, N, device=self.J.device)  # Shape [N, N]
        self.mask.fill_diagonal_(0)  # Set diagonal to 0

        # Expand the mask to match the shape of J (to broadcast correctly)
        self.mask = self.mask.unsqueeze(-1).unsqueeze(-1)  # Shape [N, N, 1, 1]

    def normalize_J(self):
        with torch.no_grad():
            self.J.data *= torch.sqrt(torch.tensor(1/(self.N*self.d)))

    def symmetrize_J(self):
        with torch.no_grad():
            self.J.data = (self.J.data + self.J.data.transpose(0,1))/2

    def normalize_x(self, x):
        """
        Normalize each d-dimensional vector in x on the sphere if self.on_sphere is True.
        """
        if self.on_sphere:
            with torch.no_grad():
                norms = x.norm(dim=-1, keepdim=True)+1e-9
                x = x / norms
        return x

    def Hebb(self, xi, form):
        """
        Calculates J based on the Hebb rule.
        xi -- input tensor of shape [P, N, d]
        form -- a string, either "Isotropic" or "Tensorial"
        """
        P = xi.shape[0]  # Number of patterns
        N = self.N
        d = self.d

        if form not in ["Isotropic", "Tensorial"]:
            raise ValueError("Form must be either 'Isotropic' or 'Tensorial'")

        # Zero out self.J to calculate the new values based on xi
        with torch.no_grad():
            self.J.zero_()

            # Hebbian rule for Isotropic or Tensorial form
            if form == "Isotropic":
                for mu in range(P):
                    # For Isotropic form: J_ij = (1/N) * sum(xi_i^mu * xi_j^mu)
                    for i in range(N):
                        for j in range(N):
                            if i != j:
                                self.J[i, j, :, :] += torch.sum(xi[mu, i, :] * xi[mu, j, :]) / N
            elif form == "Tensorial":
                for mu in range(P):
                    xi_mu = xi[mu].to(device)  # Shape: (N, D)
                    outer_products = torch.einsum('ia,jb->ijab', xi_mu, xi_mu) / N  # Shape: (N, N, D)

                    # Zero out diagonal elements
                    indices = torch.arange(N)
                    outer_products[indices, indices] = 0

                    # Update self.J
                    self.J += outer_products

                diagonal = self.J.data.diagonal(dim1=0, dim2=1)  # Get diagonal elements
                diagonal.fill_(0)

    def dyn_step(self, x, a=None):
        """
        Computes x_{t+1} = sum_j J_ij * x_j and then normalizes it.
        x -- input tensor of shape [N, d]
        """
        B, N, d = x.shape
        diagonal = self.J.data.diagonal(dim1=0, dim2=1)  # Get diagonal elements
        diagonal.fill_(0)
        with torch.no_grad():
            if a is None:
                x_new = torch.einsum('ijab,Bjb->Bia', self.J, x)
            else:
                x_new = x + a*torch.einsum('ijab,Bjb->Bia', self.J, x)
            x_new = self.normalize_x(x_new)
        return x_new

    def dyn_n_step(self, x, n, a=None, bar=False):
        """
        Computes the dynamics for n steps.
        Args:
        x -- input tensor of shape [B, N, d]
        n -- number of steps
        Returns:
        Tensor of shape [B, n, N, d] containing the dynamics over n steps
        """
        B, N, d = x.shape
        # Initialize the tensor to store the states over n steps
        x_new = torch.zeros(n, B, N, d, device=x.device)
        x_new_temp = x.clone()

        # Iterate and evolve the system for n steps
        if bar==True:
            for t in range(n):
                x_new_temp = self.dyn_step(x_new_temp, a)
                x_new[t] = x_new_temp
        else:
            for t in range(n):
                x_new_temp = self.dyn_step(x_new_temp, a)
                x_new[t] = x_new_temp

        return x_new.permute(1, 0, 2, 3)

    def Z_i_mu_func(self, y_i_mu, lambd, r=1):
        if self.d == 1:
            Z_i_mu = 2*torch.cosh(lambd*r*y_i_mu)  # [M, N]

        else:
            # Compute Z_i^mu for spherical spins
            nu = (self.d - 2) / 2  # Order of the Bessel function
            Z_i_mu = (2 * torch.pi ** (self.d / 2) * self.r ** (self.d - 1) /
                sp.special.gamma((self.d - 1) / 2)) * \
                (2 / (lambd * y_i_mu)) ** nu * \
                BesselIvFunction.apply(torch.tensor(nu), self.r * lambd * y_i_mu)  # [M, N]
        return Z_i_mu


    def forward(self, xi_batch, lambd, alpha=None, i_rand=None, r=1, l2=False):
         #* self.mask.to(device)
        diagonal = self.J.data.diagonal(dim1=0, dim2=1)  # Get diagonal elements
        diagonal.fill_(0)
        J_masked = self.J
        if i_rand is not None:
            J_x = torch.einsum('jab,mjb->ma', J_masked[i_rand], xi_batch)  # [M, d]
            y_i_mu = J_x.norm(dim=-1)  # Taking the norm over the last dimension -> [M]
            x_J_x = torch.einsum('ma,ma->m', xi_batch[:,i_rand], J_x)  # [M, N]

            # Compute the energy term for each mu: - dot_product + lam^-1 * log(Z_i_mu)
            if alpha==None:
                energy_i_mu = -x_J_x + (1 / lambd) * torch.log(self.Z_i_mu_func(y_i_mu,lambd,r))  # [M]
            else:
                energy_i_mu = -x_J_x + (1 / lambd) * alpha * y_i_mu**2

        else:
            J_x = torch.einsum('ijab,mjb->mia', J_masked, xi_batch)   # [M, d]
            y_i_mu = J_x.norm(dim=-1)  # Taking the norm over the last dimension -> [M,N]
            x_J_x = torch.einsum('mia,mia->mi', xi_batch, J_x)  # [M, N]
            # Compute the energy term for each mu: - dot_product + lam^-1 * log(Z_i_mu)
                        # Compute the energy term for each mu: - dot_product + lam^-1 * log(Z_i_mu)
            if alpha==None:
                energy_i_mu = -x_J_x + (1 / lambd) * torch.log(self.Z_i_mu_func(y_i_mu,lambd,r)+1e-9)  # [M,N]
            else:
                energy_i_mu = -x_J_x + (1 / lambd) * alpha * y_i_mu**2
            energy_i_mu = energy_i_mu.mean(dim=1)

        if l2==False:
            return energy_i_mu.mean(dim=0)
        else:
            return -x_J_x.mean() + alpha*(self.J.data**2).mean()
