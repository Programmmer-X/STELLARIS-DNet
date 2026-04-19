"""
module2/model.py
STELLARIS-DNet — Module 2 Architectures
2A: EfficientNet-B0 for Radio Galaxy Classification (FRI/FRII)
2B: 1D CNN for Gravitational Wave Detection (Signal/Noise)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from module2.config import *


# ─────────────────────────────────────────────
# 1. RADIO GALAXY CLASSIFIER (Sub-task 2A)
# EfficientNet-B0 with transfer learning
# Input:  (B, 3, 224, 224)
# Output: (B, 2) logits — FRI / FRII
# ─────────────────────────────────────────────
class RadioGalaxyClassifier(nn.Module):
    def __init__(
        self,
        num_classes:  int   = RGZ_NUM_CLASSES,
        encoder_dim:  int   = RGZ_ENCODER_DIM,
        dropout:      float = RGZ_DROPOUT,
        pretrained:   bool  = True
    ):
        super().__init__()

        # ── Backbone ─────────────────────────
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = efficientnet_b0(weights=weights)

        # Remove original classifier
        # EfficientNet-B0 feature dim = 1280
        self.backbone     = backbone.features   # conv layers only
        self.pool         = nn.AdaptiveAvgPool2d(1)

        # ── Encoder projection ────────────────
        # Projects 1280 → encoder_dim for unified model
        self.encoder = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(1280, encoder_dim),
            nn.BatchNorm1d(encoder_dim),
            nn.ReLU()
        )

        # ── Classification head ───────────────
        self.head = nn.Linear(encoder_dim, num_classes)

        # ── Freeze backbone initially ─────────
        self.freeze_backbone()

    def freeze_backbone(self):
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("🔒 Backbone frozen")

    def unfreeze_last_blocks(self, n_blocks: int = 2):
        """
        Unfreeze last n blocks of EfficientNet backbone.
        Called after RGZ_FREEZE_EPOCHS epochs.
        """
        # EfficientNet-B0 has 9 feature blocks (0-8)
        blocks = list(self.backbone.children())
        for block in blocks[-n_blocks:]:
            for param in block.parameters():
                param.requires_grad = True
        print(f"🔓 Unfroze last {n_blocks} backbone blocks")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Returns encoder features for unified model."""
        x = self.backbone(x)            # (B, 1280, 7, 7)
        x = self.pool(x).flatten(1)    # (B, 1280)
        return self.encoder(x)          # (B, encoder_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(x))  # (B, num_classes)


# ─────────────────────────────────────────────
# 2. GRAVITATIONAL WAVE DETECTOR (Sub-task 2B)
# 1D CNN on multi-detector strain data
# Input:  (B, 3, 4096) — 3 detectors × 4096 samples
# Output: (B, 2) logits — Noise / Signal
# ─────────────────────────────────────────────
class GravWaveDetector(nn.Module):
    def __init__(
        self,
        n_detectors:  int   = LIGO_N_DETECTORS,
        signal_len:   int   = LIGO_SIGNAL_LEN,
        channels:     list  = LIGO_CHANNELS,
        kernel_sizes: list  = LIGO_KERNEL_SIZES,
        dropout:      float = LIGO_DROPOUT,
        num_classes:  int   = LIGO_NUM_CLASSES,
        encoder_dim:  int   = LIGO_ENCODER_DIM
    ):
        super().__init__()
        assert len(channels) == len(kernel_sizes)

        # ── 1D Conv Encoder ───────────────────
        conv_layers = []
        in_ch = n_detectors
        for out_ch, k in zip(channels, kernel_sizes):
            conv_layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=k,
                          padding=k//2),             # same padding
                nn.BatchNorm1d(out_ch),
                nn.GELU(),
                nn.MaxPool1d(kernel_size=2),
                nn.Dropout(p=dropout * 0.5)
            ]
            in_ch = out_ch

        self.conv_encoder = nn.Sequential(*conv_layers)
        self.pool         = nn.AdaptiveAvgPool1d(1)  # (B, C, 1)

        # ── Encoder projection ────────────────
        self.encoder = nn.Sequential(
            nn.Linear(channels[-1], encoder_dim),
            nn.BatchNorm1d(encoder_dim),
            nn.GELU(),
            nn.Dropout(p=dropout)
        )

        # ── Classification head ───────────────
        self.head = nn.Linear(encoder_dim, num_classes)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Returns encoder features for unified model."""
        x = self.conv_encoder(x)        # (B, C, L')
        x = self.pool(x).squeeze(-1)   # (B, C)
        return self.encoder(x)          # (B, encoder_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(x))  # (B, num_classes)


# ─────────────────────────────────────────────
# SANITY CHECK
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("Module 2 Model Sanity Check")
    print("=" * 50)
    device = torch.device("cpu")

    # ── 2A: Radio Galaxy Classifier ──
    print("\n── Sub-task 2A: RadioGalaxyClassifier ──")
    rgc     = RadioGalaxyClassifier(pretrained=False).to(device)
    x_rgc   = torch.randn(4, 3, RGZ_IMG_SIZE, RGZ_IMG_SIZE)
    out_rgc = rgc(x_rgc)
    enc_rgc = rgc.encode(x_rgc)
    print(f"Input:   {x_rgc.shape}")
    print(f"Output:  {out_rgc.shape}")
    print(f"Encoder: {enc_rgc.shape}")
    assert out_rgc.shape == (4, RGZ_NUM_CLASSES),  "2A output wrong"
    assert enc_rgc.shape == (4, RGZ_ENCODER_DIM),  "2A encoder wrong"

    # Test freeze/unfreeze
    rgc.unfreeze_last_blocks(2)

    # ── 2B: GW Detector ──
    print("\n── Sub-task 2B: GravWaveDetector ──")
    gwd     = GravWaveDetector().to(device)
    x_gwd   = torch.randn(4, LIGO_N_DETECTORS, LIGO_SIGNAL_LEN)
    out_gwd = gwd(x_gwd)
    enc_gwd = gwd.encode(x_gwd)
    print(f"Input:   {x_gwd.shape}")
    print(f"Output:  {out_gwd.shape}")
    print(f"Encoder: {enc_gwd.shape}")
    assert out_gwd.shape == (4, LIGO_NUM_CLASSES), "2B output wrong"
    assert enc_gwd.shape == (4, LIGO_ENCODER_DIM), "2B encoder wrong"

    print("\n✅ All shapes verified")

    from core.utils import count_parameters
    print("\nRadioGalaxyClassifier:")
    count_parameters(rgc)
    print("\nGravWaveDetector:")
    count_parameters(gwd)