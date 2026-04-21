"""
module2/model.py
STELLARIS-DNet — Module 2 Architectures
BOTH 2A and 2B use EfficientNet-B0
2A: Radio Galaxy images (150x150 → 224x224)
2B: GW CQT spectrograms (3 x CQT_BINS x CQT_STEPS → 224x224)
Shared architecture — different heads only
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
# SHARED EFFICIENTNET-B0 BACKBONE
# Used by BOTH 2A and 2B
# ─────────────────────────────────────────────
class AstroEfficientNet(nn.Module):
    """
    Shared EfficientNet-B0 base for both sub-tasks.
    Input:  (B, 3, H, W) — any size, resized internally
    Output: (B, encoder_dim) encoder features
            (B, num_classes) classification logits
    """
    def __init__(
        self,
        num_classes:  int,
        encoder_dim:  int   = RGZ_ENCODER_DIM,
        dropout:      float = RGZ_DROPOUT,
        pretrained:   bool  = True,
        img_size:     int   = 224
    ):
        super().__init__()
        self.img_size = img_size

        # ── EfficientNet-B0 backbone ──────────
        weights  = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = efficientnet_b0(weights=weights)

        self.backbone = backbone.features  # output: (B, 1280, 7, 7)
        self.pool     = nn.AdaptiveAvgPool2d(1)

        # ── Encoder projection → unified dim ──
        self.encoder = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(1280, encoder_dim),
            nn.BatchNorm1d(encoder_dim),
            nn.GELU()
        )

        # ── Classification head ───────────────
        self.head = nn.Linear(encoder_dim, num_classes)

        # Start frozen
        self.freeze_backbone()

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        print("🔒 Backbone frozen")

    def unfreeze_last_blocks(self, n: int = 2):
        blocks = list(self.backbone.children())
        for block in blocks[-n:]:
            for p in block.parameters():
                p.requires_grad = True
        print(f"🔓 Unfroze last {n} backbone blocks")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encoder output — used by unified model."""
        # Resize to EfficientNet input size if needed
        if x.shape[-1] != self.img_size or x.shape[-2] != self.img_size:
            x = F.interpolate(x, size=(self.img_size, self.img_size),
                              mode="bilinear", align_corners=False)
        x = self.backbone(x)            # (B, 1280, 7, 7)
        x = self.pool(x).flatten(1)     # (B, 1280)
        return self.encoder(x)          # (B, encoder_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(x))


# ─────────────────────────────────────────────
# 2A: RADIO GALAXY CLASSIFIER
# ─────────────────────────────────────────────
class RadioGalaxyClassifier(AstroEfficientNet):
    """
    EfficientNet-B0 for FRI/FRII radio galaxy classification.
    Input: (B, 3, 224, 224) — MiraBest radio images
    """
    def __init__(self, pretrained: bool = True):
        super().__init__(
            num_classes = RGZ_NUM_CLASSES,
            encoder_dim = RGZ_ENCODER_DIM,
            dropout     = RGZ_DROPOUT,
            pretrained  = pretrained,
            img_size    = RGZ_IMG_SIZE
        )


# ─────────────────────────────────────────────
# 2B: GRAVITATIONAL WAVE DETECTOR
# ─────────────────────────────────────────────
class GravWaveDetector(AstroEfficientNet):
    """
    EfficientNet-B0 for GW Signal/Noise classification.
    Input: (B, 3, LIGO_CQT_BINS, LIGO_CQT_STEPS) — CQT spectrograms
    Same architecture as 2A — just different input size + classes.
    """
    def __init__(self, pretrained: bool = True):
        super().__init__(
            num_classes = LIGO_NUM_CLASSES,
            encoder_dim = LIGO_ENCODER_DIM,
            dropout     = LIGO_DROPOUT,
            pretrained  = pretrained,
            img_size    = LIGO_IMG_SIZE
        )


# ─────────────────────────────────────────────
# SANITY CHECK
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("Module 2 Model Sanity Check")
    print("=" * 50)
    device = torch.device("cpu")

    # ── 2A ──
    print("\n── 2A: RadioGalaxyClassifier ──")
    rgc   = RadioGalaxyClassifier(pretrained=False).to(device)
    x_2a  = torch.randn(4, 3, RGZ_IMG_SIZE, RGZ_IMG_SIZE)
    out   = rgc(x_2a)
    enc   = rgc.encode(x_2a)
    print(f"Input:   {x_2a.shape}")
    print(f"Output:  {out.shape}")
    print(f"Encoder: {enc.shape}")
    assert out.shape == (4, RGZ_NUM_CLASSES)
    assert enc.shape == (4, RGZ_ENCODER_DIM)
    rgc.unfreeze_last_blocks(2)

    # ── 2B ──
    print("\n── 2B: GravWaveDetector ──")
    gwd   = GravWaveDetector(pretrained=False).to(device)
    x_2b  = torch.randn(4, LIGO_N_DETECTORS, LIGO_CQT_BINS, LIGO_CQT_STEPS)
    out   = gwd(x_2b)
    enc   = gwd.encode(x_2b)
    print(f"Input:   {x_2b.shape}")
    print(f"Output:  {out.shape}")
    print(f"Encoder: {enc.shape}")
    assert out.shape == (4, LIGO_NUM_CLASSES)
    assert enc.shape == (4, LIGO_ENCODER_DIM)

    print("\n✅ All shapes verified")

    from core.utils import count_parameters
    print("\nRadioGalaxyClassifier:")
    count_parameters(rgc)
    print("\nGravWaveDetector:")
    count_parameters(gwd)