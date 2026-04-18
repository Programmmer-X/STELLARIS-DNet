"""
module1/model.py
STELLARIS-DNet — Module 1 Architectures
Contains: MLP (HTRU2) + 1D CNN (pulse profiles) + Autoencoder (magnetar anomaly)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from module1.config import *


# ─────────────────────────────────────────────
# 1. MLP — HTRU2 Binary Classifier
# Input:  8 statistical features
# Output: 1 logit (sigmoid → pulsar probability)
# ─────────────────────────────────────────────
class PulsarMLP(nn.Module):
    def __init__(
        self,
        input_dim:   int  = MLP_INPUT_DIM,
        hidden_dims: list = MLP_HIDDEN_DIMS,
        dropout:     float = MLP_DROPOUT
    ):
        super().__init__()

        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers += [
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
            in_dim = h_dim

        self.encoder = nn.Sequential(*layers)          # reused in unified
        self.head    = nn.Linear(in_dim, 1)            # binary output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.head(features)                     # raw logit

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Returns encoder output for unified model."""
        return self.encoder(x)


# ─────────────────────────────────────────────
# 2. 1D CNN — Pulsar Subtype Classifier
# Input:  (batch, 1, SIGNAL_LENGTH)
# Output: (batch, NUM_PULSAR_CLASSES) logits
# Encoder output saved separately for unified
# ─────────────────────────────────────────────
class PulsarCNN(nn.Module):
    def __init__(
        self,
        in_channels:  int   = CNN_IN_CHANNELS,
        channels:     list  = CNN_CHANNELS,
        kernel_sizes: list  = CNN_KERNEL_SIZES,
        dropout:      float = CNN_DROPOUT,
        num_classes:  int   = NUM_PULSAR_CLASSES,
        encoder_dim:  int   = CNN_ENCODER_DIM
    ):
        super().__init__()
        assert len(channels) == len(kernel_sizes), \
            "channels and kernel_sizes must have same length"

        # Build conv encoder
        conv_layers = []
        c_in = in_channels
        for c_out, k in zip(channels, kernel_sizes):
            conv_layers += [
                nn.Conv1d(c_in, c_out, kernel_size=k,
                          padding=k // 2),             # same padding
                nn.BatchNorm1d(c_out),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2)
            ]
            c_in = c_out

        self.conv_encoder = nn.Sequential(*conv_layers)
        self.pool         = nn.AdaptiveAvgPool1d(1)    # → (batch, C, 1)

        # Project to fixed encoder_dim for unified model
        self.project = nn.Sequential(
            nn.Linear(channels[-1], encoder_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.head = nn.Linear(encoder_dim, num_classes)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Shared encoder — used by unified model."""
        x = self.conv_encoder(x)                       # (B, C, L')
        x = self.pool(x).squeeze(-1)                   # (B, C)
        return self.project(x)                         # (B, encoder_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(x))               # (B, num_classes)


# ─────────────────────────────────────────────
# 3. AUTOENCODER — Magnetar Anomaly Detector
# Trained ONLY on normal pulsar profiles
# High reconstruction error → magnetar candidate
# Input/Output: (batch, AE_INPUT_DIM)
# ─────────────────────────────────────────────
class PulsarAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim:   int   = AE_INPUT_DIM,
        hidden_dims: list  = AE_HIDDEN_DIMS,
        latent_dim:  int   = AE_LATENT_DIM,
        dropout:     float = AE_DROPOUT
    ):
        super().__init__()

        # ── Encoder ──────────────────────────
        enc_layers = []
        in_dim = input_dim
        for h in hidden_dims:
            enc_layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
            in_dim = h
        enc_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # ── Decoder ──────────────────────────
        dec_layers = []
        in_dim = latent_dim
        for h in reversed(hidden_dims):
            dec_layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
            in_dim = h
        dec_layers.append(nn.Linear(in_dim, input_dim))
        dec_layers.append(nn.Sigmoid())                # output in [0,1]
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns per-sample MSE reconstruction error.
        High error = anomalous = potential magnetar candidate.
        """
        with torch.no_grad():
            x_hat = self.forward(x)
            error = F.mse_loss(x_hat, x, reduction="none")
            return error.mean(dim=1)                   # (batch,)

    def is_magnetar_candidate(
        self,
        x:         torch.Tensor,
        threshold: float
    ) -> torch.Tensor:
        """
        Returns boolean tensor: True = magnetar candidate.
        Threshold set during evaluate.py (95th percentile of val errors).
        """
        errors = self.reconstruction_error(x)
        return errors > threshold                      # (batch,) bool


# ─────────────────────────────────────────────
# SANITY CHECK
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("Module 1 Model Sanity Check")
    print("=" * 50)
    device = torch.device("cpu")

    # ── MLP ──
    mlp   = PulsarMLP().to(device)
    x_mlp = torch.randn(16, MLP_INPUT_DIM)
    out   = mlp(x_mlp)
    feats = mlp.get_features(x_mlp)
    print(f"MLP     | input: {x_mlp.shape} → output: {out.shape} "
          f"| features: {feats.shape}")
    assert out.shape   == (16, 1),              "MLP output shape wrong"
    assert feats.shape == (16, MLP_HIDDEN_DIMS[-1]), "MLP features shape wrong"

    # ── 1D CNN ──
    cnn   = PulsarCNN().to(device)
    x_cnn = torch.randn(16, 1, SIGNAL_LENGTH)
    out   = cnn(x_cnn)
    enc   = cnn.encode(x_cnn)
    print(f"CNN     | input: {x_cnn.shape} → output: {out.shape} "
          f"| encoder: {enc.shape}")
    assert out.shape == (16, NUM_PULSAR_CLASSES), "CNN output shape wrong"
    assert enc.shape == (16, CNN_ENCODER_DIM),    "CNN encoder shape wrong"

    # ── Autoencoder ──
    ae    = PulsarAutoencoder().to(device)
    x_ae  = torch.randn(16, AE_INPUT_DIM)
    recon = ae(x_ae)
    errs  = ae.reconstruction_error(x_ae)
    flags = ae.is_magnetar_candidate(x_ae, threshold=0.05)
    print(f"AE      | input: {x_ae.shape} → recon: {recon.shape} "
          f"| errors: {errs.shape} | flags: {flags.sum().item()} flagged")
    assert recon.shape == (16, AE_INPUT_DIM), "AE recon shape wrong"
    assert errs.shape  == (16,),              "AE error shape wrong"

    print()
    print("✅ All model shapes verified")

    # Parameter counts
    print()
    from core.utils import count_parameters
    print("MLP parameters:")
    count_parameters(mlp)
    print("CNN parameters:")
    count_parameters(cnn)
    print("AE parameters:")
    count_parameters(ae)