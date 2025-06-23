import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import cv2
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import random
import shap
import lime


class Autoencoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),    # (32, 128, 128)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),   # (64, 64, 64)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # (128, 32, 32)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128*32*32, latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128*32*32),
            nn.ReLU(),
            nn.Unflatten(1, (128, 32, 32)),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # (64, 64, 64)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # (32, 128, 128)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),    # (3, 256, 256)
            nn.Tanh()
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
    

class ResNetBlock1D(nn.Module):
    """
    A simple ResNet-like block for 1D feature vectors.
    Consists of two linear layers with BatchNorm and ReLU,
    and a skip connection.
    """
    def __init__(self, in_features, out_features, stride=1):
        super().__init__()
        # If in_features != out_features, we need a projection for the skip connection
        self.downsample = None
        if stride != 1 or in_features != out_features:
            # For 1D, stride isn't directly applicable as it is in Conv2d.
            # Here, 'stride' essentially means a change in feature dimension.
            # If in_features != out_features, we project the skip connection.
            self.downsample = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features)
            )

        self.block = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(inplace=True), # inplace=True can save memory
            nn.Linear(out_features, out_features),
            nn.BatchNorm1d(out_features)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.block(x)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity # Add the skip connection
        out = self.relu(out)
        return out


class TumorClassifierRes(nn.Module):
    def __init__(self, encoder, latent_dim=128):
        super().__init__()
        # Congela l'encoder, tranne ultimo layer (as per your original code)
        for param in encoder.parameters():
            param.requires_grad = False

        for i in range(6, len(encoder)): # Unfreeze from index 6 till the end of encoder
            for param in encoder[i].parameters():
                param.requires_grad = True

        self.encoder = encoder  # encoder già addestrato

        # --- MODIFIED CLASSIFIER HEAD ---
        self.classifier = nn.Sequential(
            # Initial layer to potentially expand features if desired, or just pass latent_dim directly
            # You could start with a block immediately if latent_dim is your desired block input size
            # Here, let's make it more explicitly multi-block
            ResNetBlock1D(latent_dim, 128), # First block: latent_dim -> 128
            nn.Dropout(0.3), # Dropout can be placed between blocks or after them

            ResNetBlock1D(128, 64),  # Second block: 128 -> 64
            nn.Dropout(0.3),

            # Final classification layer (output a single logit for binary classification)
            nn.Linear(64, 1) # Output 1 for BCEWithLogitsLoss
        )

    def forward(self, x):
        z = self.encoder(x)  # get compressed features from encoder
        return self.classifier(z)