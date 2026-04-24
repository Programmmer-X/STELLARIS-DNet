"""
module1/model.py
STELLARIS-DNet — Module 1 Architectures

Models:
  1. PulsarMLP          — HTRU2 binary classifier (8 features → 1 logit)
  2. PulsarCNN          — Pulsar subtype classifier (1D signal → 4 classes)
  3. PulsarAutoencoder  — Magnetar anomaly detector (reconstruction error)

Upgrades:
  - Self-attention in CNN encoder (USE_ATTENTION)
  - Time-domain + frequency-domain feature fusion (USE_FREQ_FUSION)
  - Physics-aware energy proxy input (USE_PHYSICS_FEATURES)
  - Exposed latent representations for analysis
  - Improved AE latent structure with latent norm
  - All upgrades gated by config.py toggles
  - Encoder dims guaranteed correct for unified model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from module1.config import *


# ═════════════════════════════════════════════
# SECTION 1 — SHARED BUILDING BLOCKS
# ═════════════════════════════════════════════

class ChannelAttention1D(nn.Module):
    """
    Squeeze-and-Excitation style channel attention for 1D conv features.
    Input:  (B, C, L)
    Output: (B, C, L)  — channel-wise recalibrated
    """
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Global average pool → (B, C)
        w = x.mean(dim=-1)
        w = self.fc(w).unsqueeze(-1)   # (B, C, 1)
        return x * w                   # broadcast over L


class MultiHeadTemporalAttention(nn.Module):
    """
    Multi-head self-attention over the temporal (L) dimension.
    Applied after conv encoder — captures long-range pulse dependencies.
    Input:  (B, C, L)
    Output: (B, C, L)
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,          # expects (B, L, C)
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L) → transpose to (B, L, C) for MHA
        x_t  = x.permute(0, 2, 1)            # (B, L, C)
        attn_out, _ = self.attn(x_t, x_t, x_t)
        out  = self.norm(attn_out + x_t)      # residual + norm
        return out.permute(0, 2, 1)           # back to (B, C, L)


class FreqFusionHead(nn.Module):
    """
    Projects FFT features to match CNN encoder dim, then fuses via concat+project.
    Input:  time_feat (B, T), freq_feat (B, F)
    Output: fused (B, CNN_ENCODER_DIM)
    """
    def __init__(
        self,
        time_dim: int = CNN_ENCODER_DIM,
        freq_dim: int = FFT_BINS,
        out_dim:  int = CNN_ENCODER_DIM,
    ):
        super().__init__()
        self.freq_proj = nn.Sequential(
            nn.Linear(freq_dim, CNN_FREQ_DIM),
            nn.ReLU(),
        )
        self.fuse = nn.Sequential(
            nn.Linear(time_dim + CNN_FREQ_DIM, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(
        self,
        time_feat: torch.Tensor,
        freq_feat: torch.Tensor,
    ) -> torch.Tensor:
        f = self.freq_proj(freq_feat)          # (B, CNN_FREQ_DIM)
        return self.fuse(torch.cat([time_feat, f], dim=-1))  # (B, out_dim)


# ═════════════════════════════════════════════
# SECTION 2 — MLP (HTRU2 Binary Classifier)
# Input:  (B, 8) statistical features
# Output: (B, 1) raw logit
# Optionally appends energy proxy → (B, 9)
# ═════════════════════════════════════════════

class PulsarMLP(nn.Module):
    """
    Binary pulsar classifier on HTRU2 8-feature vectors.
    USE_PHYSICS_FEATURES=True → accepts an optional energy scalar
    appended to features, expanding input_dim by 1.
    """
    def __init__(
        self,
        input_dim:   int   = MLP_INPUT_DIM,
        hidden_dims: list  = MLP_HIDDEN_DIMS,
        dropout:     float = MLP_DROPOUT,
        use_energy:  bool  = False,            # set True if energy appended
    ):
        super().__init__()
        # If energy proxy is appended externally, input_dim += 1
        actual_dim = input_dim + (1 if use_energy else 0)

        layers = []
        in_dim = actual_dim
        for h_dim in hidden_dims:
            layers += [
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = h_dim

        self.encoder = nn.Sequential(*layers)
        self.head    = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))      # (B, 1) raw logit

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Returns encoder output — used by unified model."""
        return self.encoder(x)                 # (B, hidden_dims[-1]) = (B, 32)


# ═════════════════════════════════════════════
# SECTION 3 — 1D CNN (Pulsar Subtype Classifier)
# Input:  (B, 1, SIGNAL_LENGTH) time-domain
# Optional: (B, FFT_BINS) freq features
# Output: (B, NUM_PULSAR_CLASSES) logits
# Encoder: (B, CNN_ENCODER_DIM=256) guaranteed
# ═════════════════════════════════════════════

class PulsarCNN(nn.Module):
    """
    1D CNN pulsar subtype classifier with optional:
      - Channel attention (USE_ATTENTION)
      - Temporal self-attention (USE_ATTENTION)
      - FFT frequency feature fusion (USE_FREQ_FUSION)

    Encoder always outputs CNN_ENCODER_DIM (256) — unified-compatible.
    """
    def __init__(
        self,
        in_channels:  int   = CNN_IN_CHANNELS,
        channels:     list  = CNN_CHANNELS,
        kernel_sizes: list  = CNN_KERNEL_SIZES,
        dropout:      float = CNN_DROPOUT,
        num_classes:  int   = NUM_PULSAR_CLASSES,
        encoder_dim:  int   = CNN_ENCODER_DIM,
    ):
        super().__init__()
        assert len(channels) == len(kernel_sizes), \
            "channels and kernel_sizes must match in length"

        # ── Conv blocks with optional channel attention ──
        conv_layers = []
        c_in = in_channels
        for c_out, k in zip(channels, kernel_sizes):
            conv_layers.append(
                nn.Conv1d(c_in, c_out, kernel_size=k, padding=k // 2)
            )
            conv_layers.append(nn.BatchNorm1d(c_out))
            conv_layers.append(nn.ReLU())
            if USE_ATTENTION:
                conv_layers.append(ChannelAttention1D(c_out))
            conv_layers.append(nn.MaxPool1d(kernel_size=2))
            c_in = c_out

        self.conv_encoder = nn.Sequential(*conv_layers)
        self.pool         = nn.AdaptiveAvgPool1d(1)    # → (B, C, 1)

        # ── Temporal self-attention (after conv, before pool) ──
        self.use_temporal_attn = USE_ATTENTION
        if USE_ATTENTION:
            self.temporal_attn = MultiHeadTemporalAttention(
                embed_dim=channels[-1],
                num_heads=CNN_ATTN_HEADS,
                dropout=CNN_ATTN_DROPOUT,
            )

        # ── Time-domain projection → encoder_dim ──
        self.time_proj = nn.Sequential(
            nn.Linear(channels[-1], encoder_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ── Frequency fusion ──
        self.use_freq_fusion = USE_FREQ_FUSION
        if USE_FREQ_FUSION:
            self.freq_fusion = FreqFusionHead(
                time_dim=encoder_dim,
                freq_dim=FFT_BINS,
                out_dim=encoder_dim,
            )

        self.head = nn.Linear(encoder_dim, num_classes)

    def encode(
        self,
        x_time: torch.Tensor,
        x_freq: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Returns unified-compatible encoder output: (B, CNN_ENCODER_DIM).
        x_time: (B, 1, L)
        x_freq: (B, FFT_BINS)  — required if USE_FREQ_FUSION=True
        """
        # Conv feature extraction
        x = self.conv_encoder(x_time)          # (B, C_last, L')

        # Temporal attention (before pooling — operates on sequence)
        if self.use_temporal_attn:
            x = self.temporal_attn(x)          # (B, C_last, L')

        x = self.pool(x).squeeze(-1)           # (B, C_last)
        x = self.time_proj(x)                  # (B, encoder_dim)

        # Frequency fusion
        if self.use_freq_fusion and x_freq is not None:
            x = self.freq_fusion(x, x_freq)    # (B, encoder_dim)

        return x

    def forward(
        self,
        x_time: torch.Tensor,
        x_freq: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        x_time: (B, 1, L)
        x_freq: (B, FFT_BINS) — optional frequency features
        Returns: (B, num_classes) logits
        """
        return self.head(self.encode(x_time, x_freq))

    def get_features(
        self,
        x_time: torch.Tensor,
        x_freq: torch.Tensor = None,
    ) -> torch.Tensor:
        """Alias for encode() — used by unified model."""
        return self.encode(x_time, x_freq)


# ═════════════════════════════════════════════
# SECTION 4 — AUTOENCODER (Magnetar Anomaly)
# Trained ONLY on normal pulsar profiles
# High reconstruction error → magnetar candidate
# Input/Output: (B, AE_INPUT_DIM)
# Latent: (B, AE_LATENT_DIM=16)
# ═════════════════════════════════════════════

class PulsarAutoencoder(nn.Module):
    """
    Symmetric autoencoder for pulsar anomaly detection.

    Upgrades over original:
      - LayerNorm on latent vector (stabilizes latent space)
      - `get_latent()` exposed for PCA / latent visualization
      - `encode()` / `decode()` split for flexible usage
      - Threshold computed via percentile OR z-score (per config)
      - reconstruction_error() remains unchanged API
    """
    def __init__(
        self,
        input_dim:   int   = AE_INPUT_DIM,
        hidden_dims: list  = AE_HIDDEN_DIMS,
        latent_dim:  int   = AE_LATENT_DIM,
        dropout:     float = AE_DROPOUT,
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
                nn.Dropout(dropout),
            ]
            in_dim = h
        enc_layers.append(nn.Linear(in_dim, latent_dim))

        # Latent normalization — stabilizes anomaly score distribution
        enc_layers.append(nn.LayerNorm(latent_dim))

        self.encoder = nn.Sequential(*enc_layers)

        # ── Decoder ──────────────────────────
        dec_layers = []
        in_dim = latent_dim
        for h in reversed(hidden_dims):
            dec_layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = h
        dec_layers.append(nn.Linear(in_dim, input_dim))
        dec_layers.append(nn.Sigmoid())        # output in [0, 1]

        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Returns latent representation z: (B, AE_LATENT_DIM)."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstructs input from latent z: (B, AE_INPUT_DIM)."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full encode-decode pass. Returns reconstruction."""
        return self.decode(self.encode(x))

    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns latent vectors for visualization (PCA, t-SNE).
        Runs with no_grad for inference-only usage.
        """
        with torch.no_grad():
            return self.encode(x)              # (B, AE_LATENT_DIM)

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        Per-sample MSE reconstruction error.
        High error → anomalous → potential magnetar candidate.
        Returns: (B,) tensor — API unchanged.
        """
        with torch.no_grad():
            x_hat = self.forward(x)
            error = F.mse_loss(x_hat, x, reduction="none")
            return error.mean(dim=1)           # (B,)

    def compute_threshold(
        self,
        errors: torch.Tensor,
        method: str   = AE_THRESHOLD_METHOD,
    ) -> float:
        """
        Compute anomaly threshold from a set of validation errors.
        method='percentile' → AE_ANOMALY_PERCENTILE-th percentile
        method='zscore'     → mean + AE_ZSCORE_SIGMA * std

        Called during evaluate.py — not during training.
        Returns: float threshold value
        """
        errs_np = errors.cpu().numpy() if isinstance(errors, torch.Tensor) \
                  else errors
        import numpy as np
        if method == "percentile":
            return float(np.percentile(errs_np, AE_ANOMALY_PERCENTILE))
        elif method == "zscore":
            return float(errs_np.mean() + AE_ZSCORE_SIGMA * errs_np.std())
        else:
            raise ValueError(f"Unknown threshold method: {method}. "
                             f"Use 'percentile' or 'zscore'.")

    def is_magnetar_candidate(
        self,
        x:         torch.Tensor,
        threshold: float,
    ) -> torch.Tensor:
        """
        Returns boolean tensor: True = magnetar candidate.
        Threshold set during evaluate.py.
        Returns: (B,) bool — API unchanged.
        """
        return self.reconstruction_error(x) > threshold


# ═════════════════════════════════════════════
# SECTION 5 — SANITY CHECK
# ═════════════════════════════════════════════
if __name__ == "__main__":
    import numpy as np
    print("=" * 60)
    print("Module 1 Model Sanity Check — Enhanced")
    print("=" * 60)
    device = torch.device("cpu")

    # ── MLP — standard ──
    mlp    = PulsarMLP().to(device)
    x_mlp  = torch.randn(16, MLP_INPUT_DIM)
    out    = mlp(x_mlp)
    feats  = mlp.get_features(x_mlp)
    print(f"MLP     | in: {x_mlp.shape} → out: {out.shape} "
          f"| encoder: {feats.shape}")
    assert out.shape   == (16, 1),                   "MLP output shape wrong"
    assert feats.shape == (16, MLP_HIDDEN_DIMS[-1]), "MLP features shape wrong"

    # ── MLP — with energy proxy ──
    mlp_e  = PulsarMLP(use_energy=True).to(device)
    x_mlp_e = torch.randn(16, MLP_INPUT_DIM + 1)
    out_e   = mlp_e(x_mlp_e)
    print(f"MLP+En  | in: {x_mlp_e.shape} → out: {out_e.shape}")
    assert out_e.shape == (16, 1), "MLP+energy output shape wrong"

    # ── CNN — time only (standard) ──
    cnn    = PulsarCNN().to(device)
    x_time = torch.randn(16, 1, SIGNAL_LENGTH)
    out    = cnn(x_time)
    enc    = cnn.encode(x_time)
    print(f"CNN     | in: {x_time.shape} → out: {out.shape} "
          f"| encoder: {enc.shape}")
    assert out.shape == (16, NUM_PULSAR_CLASSES), "CNN output shape wrong"
    assert enc.shape == (16, CNN_ENCODER_DIM),    "CNN encoder shape wrong"

    # ── CNN — time + freq fusion ──
    if USE_FREQ_FUSION:
        x_freq = torch.randn(16, FFT_BINS)
        out_f  = cnn(x_time, x_freq)
        enc_f  = cnn.encode(x_time, x_freq)
        print(f"CNN+FFT | in: {x_time.shape}+{x_freq.shape} "
              f"→ out: {out_f.shape} | encoder: {enc_f.shape}")
        assert out_f.shape == (16, NUM_PULSAR_CLASSES), "CNN+FFT output wrong"
        assert enc_f.shape == (16, CNN_ENCODER_DIM),    "CNN+FFT encoder wrong"

    # ── Autoencoder ──
    ae     = PulsarAutoencoder().to(device)
    x_ae   = torch.randn(16, AE_INPUT_DIM)
    recon  = ae(x_ae)
    errs   = ae.reconstruction_error(x_ae)
    z      = ae.get_latent(x_ae)
    thresh = ae.compute_threshold(errs, method="percentile")
    thresh_z = ae.compute_threshold(errs, method="zscore")
    flags  = ae.is_magnetar_candidate(x_ae, threshold=thresh)
    print(f"AE      | in: {x_ae.shape} → recon: {recon.shape} "
          f"| latent: {z.shape} | errors: {errs.shape}")
    print(f"AE      | threshold (pct): {thresh:.6f} "
          f"| threshold (zscore): {thresh_z:.6f} "
          f"| flagged: {flags.sum().item()}")
    assert recon.shape == (16, AE_INPUT_DIM), "AE recon shape wrong"
    assert errs.shape  == (16,),              "AE error shape wrong"
    assert z.shape     == (16, AE_LATENT_DIM), "AE latent shape wrong"

    print()
    print("✅ All model shapes verified")

    # ── Parameter counts ──
    print()
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.utils import count_parameters
    print("MLP parameters:");       count_parameters(mlp)
    print("CNN parameters:");       count_parameters(cnn)
    print("AE parameters:");        count_parameters(ae)

    # ── Encoder dim summary (for unified model) ──
    print()
    print("─" * 40)
    print("Encoder Dims (Unified Model Registry)")
    print("─" * 40)
    print(f"  M1-MLP encoder : {MLP_HIDDEN_DIMS[-1]:>4d}  (mlp_encoder.pt)")
    print(f"  M1-CNN encoder : {CNN_ENCODER_DIM:>4d}  (cnn_encoder.pt)")
    print(f"  M1-AE  latent  : {AE_LATENT_DIM:>4d}  (ae_encoder.pt)")
    print("─" * 40)