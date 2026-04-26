"""
module2/model.py
STELLARIS-DNet — Module 2 Architectures (Upgraded)

Shared upgrades:
  EfficientNet-B0 → B2   (1280 → 1408 features)
  AdaptiveAvgPool → GeM  (Generalised Mean Pooling)
  + CBAM attention on feature map
  + Optional Transformer on spatial sequence

2A: RadioGalaxyClassifier  — B2 + CBAM + GeM + jet power aux head
2B: GravWaveDetector       — B2 + CBAM + GeM + Transformer
    GravWave1DCNN          — unchanged (for CQT vs Raw comparison)

API compatibility:
  model.backbone[-1]         → GradCAM target layer (unchanged)
  model.encode(x)            → feature vector
  model.unfreeze_last_blocks(n)
  RadioGalaxyClassifier.forward(x) → (logits, jet_power)
  GravWaveDetector.forward(x)      → logits
  GravWave1DCNN.forward(x)         → logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from module2.config import *


# ─────────────────────────────────────────────
# BACKBONE FACTORY
# Loads EfficientNet-B2 from torchvision
# Falls back to B0 gracefully for older installs
# ─────────────────────────────────────────────
def _load_efficientnet(pretrained: bool = True):
    """
    Returns (features_module, feat_dim).
    B2: feat_dim=1408  |  B0 fallback: feat_dim=1280
    """
    try:
        from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
        weights  = EfficientNet_B2_Weights.IMAGENET1K_V1 if pretrained else None
        net      = efficientnet_b2(weights=weights)
        feat_dim = 1408
        print(f"✅ EfficientNet-B2 loaded (feat_dim={feat_dim})")
    except (ImportError, AttributeError):
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
        weights  = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        net      = efficientnet_b0(weights=weights)
        feat_dim = 1280
        print(f"⚠️  B2 unavailable → EfficientNet-B0 (feat_dim={feat_dim})")
    return net.features, feat_dim


# ─────────────────────────────────────────────
# CBAM — Convolutional Block Attention Module
# ─────────────────────────────────────────────
class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = CBAM_REDUCTION):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = self.mlp(self.avg_pool(x))
        mx  = self.mlp(self.max_pool(x))
        return torch.sigmoid(avg + mx).unsqueeze(-1).unsqueeze(-1)  # (B,C,1,1)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                              padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx  = x.max(dim=1, keepdim=True)[0]
        return torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))  # (B,1,H,W)


class CBAM(nn.Module):
    """
    Applies channel attention then spatial attention.
    2A: highlights jet structures and hotspot regions
    2B: highlights chirp time-frequency signatures
    """
    def __init__(self, channels: int, reduction: int = CBAM_REDUCTION):
        super().__init__()
        self.channel = ChannelAttention(channels, reduction)
        self.spatial = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel(x)   # recalibrate channels
        x = x * self.spatial(x)   # recalibrate spatial locations
        return x


# ─────────────────────────────────────────────
# GeM POOLING — Generalised Mean Pooling
# ─────────────────────────────────────────────
class GeMPooling2D(nn.Module):
    """
    Replaces AdaptiveAvgPool2d on 2D feature maps.
    p=1 → average pooling  |  p→∞ → max pooling
    Learnable p finds the optimal point between these extremes.
    """
    def __init__(self, p: float = GEM_P, eps: float = GEM_EPS,
                 learnable: bool = GEM_LEARNABLE):
        super().__init__()
        self.eps = eps
        self.p   = nn.Parameter(torch.tensor(float(p))) if learnable else float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.p.clamp(min=1.0) if isinstance(self.p, torch.Tensor) else self.p
        return F.adaptive_avg_pool2d(
            x.clamp(min=self.eps).pow(p), (1, 1)
        ).pow(1.0 / p).flatten(1)                    # (B, C)


class GeMPooling1D(nn.Module):
    """GeM pooling over sequence dimension: (B, L, D) → (B, D)."""
    def __init__(self, p: float = GEM_P, eps: float = GEM_EPS,
                 learnable: bool = GEM_LEARNABLE):
        super().__init__()
        self.eps = eps
        self.p   = nn.Parameter(torch.tensor(float(p))) if learnable else float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.p.clamp(min=1.0) if isinstance(self.p, torch.Tensor) else self.p
        return x.clamp(min=self.eps).pow(p).mean(dim=1).pow(1.0 / p)


# ─────────────────────────────────────────────
# POSITIONAL ENCODING (for Transformer)
# ─────────────────────────────────────────────
class SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 256,
                 dropout: float = TRANSFORMER_DROPOUT):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1)])


# ─────────────────────────────────────────────
# ENHANCED ENCODER HEAD
# Linear→BN→GELU→Dropout→Linear→BN→GELU
# Better than single linear: more expressive,
# avoids representation collapse at small batch sizes
# ─────────────────────────────────────────────
class EncoderHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────
# AstroEfficientNet — UPGRADED BASE CLASS
# ─────────────────────────────────────────────
class AstroEfficientNet(nn.Module):
    """
    Upgraded shared base for 2A and 2B:

    Path A (use_transformer=False, default for 2A):
        Input → Backbone → CBAM → GeMPooling2D → EncoderHead → Head

    Path B (use_transformer=True, default for 2B):
        Input → Backbone → CBAM → flatten spatial
        → Linear projection → Transformer → GeMPooling1D
        → EncoderHead → Head

    GradCAM compatibility: model.backbone[-1] still works as target.
    Progressive unfreezing: model.unfreeze_last_blocks(n) unchanged.
    """
    def __init__(
        self,
        num_classes:     int,
        encoder_dim:     int   = ENCODER_DIM,
        dropout:         float = 0.5,
        pretrained:      bool  = True,
        img_size:        int   = 224,
        use_cbam:        bool  = USE_CBAM,
        use_gem:         bool  = USE_GEM,
        use_transformer: bool  = False,
    ):
        super().__init__()
        self.img_size        = img_size
        self.use_transformer = use_transformer

        # ── Backbone ──────────────────────────
        self.backbone, self.feat_dim = _load_efficientnet(pretrained)

        # ── CBAM ──────────────────────────────
        self.cbam = CBAM(self.feat_dim) if use_cbam else nn.Identity()

        # ── Pooling / Transformer path ─────────
        if use_transformer:
            # Spatial sequence → Transformer → GeM1D
            self.spatial_proj = nn.Linear(self.feat_dim, TRANSFORMER_DIM)
            self.norm_proj    = nn.LayerNorm(TRANSFORMER_DIM)
            self.pe           = SinusoidalPE(TRANSFORMER_DIM)
            enc_layer         = nn.TransformerEncoderLayer(
                d_model=TRANSFORMER_DIM,
                nhead=TRANSFORMER_HEADS,
                dim_feedforward=TRANSFORMER_FF_DIM,
                dropout=TRANSFORMER_DROPOUT,
                batch_first=True,
                norm_first=True    # Pre-LN: more stable training
            )
            self.transformer  = nn.TransformerEncoder(
                enc_layer, num_layers=TRANSFORMER_LAYERS
            )
            self.gem_pool = GeMPooling1D()
            pool_out_dim  = TRANSFORMER_DIM
        else:
            # Feature map → GeM2D (or AvgPool)
            if use_gem:
                self.pool = GeMPooling2D()
            else:
                self.pool = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1), nn.Flatten()
                )
            pool_out_dim = self.feat_dim

        # ── Encoder Head ──────────────────────
        self.encoder = EncoderHead(pool_out_dim, encoder_dim, dropout)

        # ── Classification Head ────────────────
        self.head = nn.Linear(encoder_dim, num_classes)

        self.freeze_backbone()

    # ── Freeze / Unfreeze ─────────────────────
    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        print("🔒 Backbone frozen")

    def unfreeze_last_blocks(self, n: int = 2):
        for block in list(self.backbone.children())[-n:]:
            for p in block.parameters():
                p.requires_grad = True
        print(f"🔓 Unfroze last {n} backbone blocks")

    # ── Feature Extraction ────────────────────
    def _pool_features(self, x: torch.Tensor) -> torch.Tensor:
        """Backbone → CBAM → pool → (B, pool_out_dim)."""
        if x.shape[-1] != self.img_size or x.shape[-2] != self.img_size:
            x = F.interpolate(x, (self.img_size, self.img_size),
                              mode="bilinear", align_corners=False)
        fm = self.backbone(x)    # (B, feat_dim, H', W')
        fm = self.cbam(fm)

        if self.use_transformer:
            B, C, H, W = fm.shape
            seq = fm.flatten(2).transpose(1, 2)           # (B, H*W, C)
            seq = self.norm_proj(self.spatial_proj(seq))  # (B, H*W, TRANSFORMER_DIM)
            seq = self.pe(seq)
            seq = self.transformer(seq)                    # (B, H*W, TRANSFORMER_DIM)
            return self.gem_pool(seq)                      # (B, TRANSFORMER_DIM)
        else:
            return self.pool(fm)                           # (B, feat_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Returns encoder features. Used by unified model."""
        return self.encoder(self._pool_features(x))       # (B, encoder_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(x))


# ─────────────────────────────────────────────
# 2A: RadioGalaxyClassifier
# FRI vs FRII — with jet power physics head
# ─────────────────────────────────────────────
class RadioGalaxyClassifier(AstroEfficientNet):
    """
    Upgraded: B2 + CBAM + GeM (Transformer OFF — small dataset)
    forward() returns (logits, jet_power) for physics loss integration.
    jet_power: predicted log10(P / W·Hz⁻¹) — used to enforce FR boundary.
    """
    def __init__(self, pretrained: bool = True):
        super().__init__(
            num_classes=RGZ_NUM_CLASSES,
            encoder_dim=RGZ_ENCODER_DIM,
            dropout=RGZ_DROPOUT,
            pretrained=pretrained,
            img_size=RGZ_IMG_SIZE,
            use_transformer=USE_TRANSFORMER_2A
        )
        # Auxiliary physics head — predicts log10(jet power)
        self.jet_power_head = nn.Sequential(
            nn.Linear(RGZ_ENCODER_DIM, 64),
            nn.GELU(),
            nn.Linear(64, 1)                 # scalar: log10(P / W·Hz⁻¹)
        ) if USE_JET_POWER_HEAD else None

    def forward(self, x: torch.Tensor):
        """
        Returns: (logits, jet_power)
          logits:    (B, 2)  — FRI/FRII
          jet_power: (B, 1)  — log10(P) or None
        """
        features  = self.encode(x)
        logits    = self.head(features)
        jet_power = self.jet_power_head(features) if self.jet_power_head else None
        return logits, jet_power


# ─────────────────────────────────────────────
# 2B: GravWaveDetector
# Signal vs Noise — Transformer ON for 4000+ samples
# ─────────────────────────────────────────────
class GravWaveDetector(AstroEfficientNet):
    """
    Upgraded: B2 + CBAM + GeM + Transformer (ON for G2Net).
    forward() returns plain logits for drop-in compatibility with evaluate.py.
    """
    def __init__(self, pretrained: bool = True):
        super().__init__(
            num_classes=LIGO_NUM_CLASSES,
            encoder_dim=LIGO_ENCODER_DIM,
            dropout=LIGO_DROPOUT,
            pretrained=pretrained,
            img_size=LIGO_IMG_SIZE,
            use_transformer=USE_TRANSFORMER_2B
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(x))


# ─────────────────────────────────────────────
# 2B-RAW: GravWave1DCNN
# Unchanged — for CQT vs Raw comparison in evaluate.py
# ─────────────────────────────────────────────
class GravWave1DCNN(nn.Module):
    """Simple 1D CNN for raw signal comparison (no CQT)."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(LIGO_N_DETECTORS, 32,  15, padding=7),
            nn.BatchNorm1d(32),  nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(32, 64,  11, padding=5),
            nn.BatchNorm1d(64),  nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(64, 128,  7, padding=3),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(128, 256,  5, padding=2),
            nn.BatchNorm1d(256), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256, LIGO_NUM_CLASSES)
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x).squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(x))


# ─────────────────────────────────────────────
# SANITY CHECK
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("Module 2 Model Sanity Check (Upgraded)")
    print("=" * 55)
    device = torch.device("cpu")

    print("\n── 2A: RadioGalaxyClassifier ──")
    rgc = RadioGalaxyClassifier(pretrained=False).to(device)
    x   = torch.randn(4, 3, RGZ_IMG_SIZE, RGZ_IMG_SIZE)
    logits, jet_power = rgc(x)
    feats = rgc.encode(x)
    print(f"Input:     {x.shape}")
    print(f"Logits:    {logits.shape}     expected (4, {RGZ_NUM_CLASSES})")
    print(f"Jet power: {jet_power.shape}  expected (4, 1)")
    print(f"Encoder:   {feats.shape}     expected (4, {RGZ_ENCODER_DIM})")
    assert logits.shape    == (4, RGZ_NUM_CLASSES)
    assert jet_power.shape == (4, 1)
    assert feats.shape     == (4, RGZ_ENCODER_DIM)

    print("\n── 2B: GravWaveDetector ──")
    gwd = GravWaveDetector(pretrained=False).to(device)
    x   = torch.randn(4, LIGO_N_DETECTORS, LIGO_CQT_BINS, LIGO_CQT_STEPS)
    out = gwd(x)
    enc = gwd.encode(x)
    print(f"Input:   {x.shape}")
    print(f"Output:  {out.shape}   expected (4, {LIGO_NUM_CLASSES})")
    print(f"Encoder: {enc.shape}  expected (4, {LIGO_ENCODER_DIM})")
    assert out.shape == (4, LIGO_NUM_CLASSES)
    assert enc.shape == (4, LIGO_ENCODER_DIM)

    print("\n── 2B-RAW: GravWave1DCNN ──")
    cnn = GravWave1DCNN().to(device)
    x   = torch.randn(4, LIGO_N_DETECTORS, LIGO_SIGNAL_LEN)
    print(f"Input:  {x.shape}")
    print(f"Output: {cnn(x).shape}   expected (4, {LIGO_NUM_CLASSES})")
    assert cnn(x).shape == (4, LIGO_NUM_CLASSES)

    print("\n✅ All shapes verified")
    from core.utils import count_parameters
    print("\nRadioGalaxyClassifier:");  count_parameters(rgc)
    print("\nGravWaveDetector:");       count_parameters(gwd)
    print("\nGravWave1DCNN:");          count_parameters(cnn)