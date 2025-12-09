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

#Checkpoints (to save model parameters during training)
class SaveBestModel:
    def __init__(self, name, print=True, best_valid_loss=-float('inf')): #object initialized with best_loss = +infinite
        self.best_valid_loss = best_valid_loss
        self.print = print
        self.name = name

    def __call__(
        self, current_valid_loss,
        epoch, model, optimizer, criterion, time, asymmetry=None, overlap=None
    ):
        if current_valid_loss > self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            if self.print==True:
                print("Saving best model for epoch: %d, current val loss: %.4e, t: %.1f\n" % (epoch+1, current_valid_loss, time))
            # method to save a model (the state_dict: a python dictionary object that
            # maps each layer to its parameter tensor) and other useful parametrers
            # see: https://pytorch.org/tutorials/beginner/saving_loading_models.html
            if asymmetry is not None and overlap is not None:
                torch.save({
                    'epoch': epoch+1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': criterion,
                    "time": time,
                    "asymmetry": asymmetry,
                    "overlap": overlap,
                    }, self.name)
            else:
                torch.save({
                    'epoch': epoch+1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': criterion,
                    "time": time,
                }, self.name)

#Checkpoints (to save model parameters during training)
class Save_Model:
    def __init__(self, name, print=True): #object initialized with best_loss = +infinite
        self.print = print
        self.name = name

    def __call__(
        self, current_valid_loss,
        epoch, model, optimizer, criterion, time, asymmetry=None, overlap=None
    ):

        if asymmetry is not None and overlap is not None:
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                "time": time,
                "asymmetry": asymmetry,
                "overlap": overlap,
                }, self.name)
        else:
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                "time": time,
            }, self.name)

