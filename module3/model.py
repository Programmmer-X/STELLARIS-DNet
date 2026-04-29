"""
module3/model.py
STELLARIS-DNet — Module 3 Architecture
FT-Transformer (Feature Tokenizer + Transformer)
Dual-head: 5-class stellar classification + 4-param physical regression
Encoder output (CLS token, dim=256) saved separately for unified model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from module3.config import *


# ─────────────────────────────────────────────
# 1. FEATURE TOKENIZER
# One independent Linear projection per feature.
# No weight sharing — each feature gets its own
# embedding space in d_token dimensions.
# Input:  (B, NUM_FEATURES)
# Output: (B, NUM_FEATURES, d_token)
# ─────────────────────────────────────────────
class FeatureTokenizer(nn.Module):
    def __init__(self, num_features: int, d_token: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_features, d_token))
        self.bias   = nn.Parameter(torch.zeros(num_features, d_token))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F) → (B, F, 1) * (F, d_token) → (B, F, d_token)
        return x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


# ─────────────────────────────────────────────
# 2. TRANSFORMER BLOCK
# Pre-LayerNorm variant — more stable than
# post-LN for tabular data.
# ─────────────────────────────────────────────
class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model:  int,
        n_heads:  int,
        ffn_mult: int   = 4,
        dropout:  float = 0.1
    ):
        super().__init__()
        assert d_model % n_heads == 0, \
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(
            d_model, n_heads,
            dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ffn_mult, d_model),
            nn.Dropout(dropout)
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.drop(attn_out)
        # Pre-LN FFN
        x = x + self.ffn(self.norm2(x))
        return x


# ─────────────────────────────────────────────
# 3. FT-TRANSFORMER — MAIN MODEL
# ─────────────────────────────────────────────
class StellarFTTransformer(nn.Module):
    """
    Feature Tokenizer + Transformer for stellar classification + regression.

    Input:   (B, NUM_FEATURES=8)  — normalized observable features
    Outputs:
        class_logits: (B, NUM_STELLAR_CLASSES=5)
        reg_out:      (B, NUM_REGRESSION=4)  — sigmoid-bounded log-scale params
        encoder_feat: (B, ENCODER_DIM=256)   — CLS token for unified model
    """

    def __init__(
        self,
        num_features:   int   = NUM_FEATURES,
        d_token:        int   = TRANSFORMER_DIM,
        n_heads:        int   = TRANSFORMER_HEADS,
        n_layers:       int   = TRANSFORMER_LAYERS,
        ffn_mult:       int   = TRANSFORMER_FFN_MULT,
        dropout:        float = TRANSFORMER_DROPOUT,
        head_hidden:    list  = HEAD_HIDDEN_DIMS,
        head_dropout:   float = HEAD_DROPOUT,
        encoder_dim:    int   = ENCODER_DIM,
        num_classes:    int   = NUM_STELLAR_CLASSES,
        num_regression: int   = NUM_REGRESSION
    ):
        super().__init__()

        # ── Learnable feature scaling ─────────
        # Allows model to suppress low-signal features
        # (e.g. abs_mag=0 for NS, period_ms=0 for non-pulsars).
        self.feature_scale = nn.Parameter(torch.ones(num_features))

        # ── Feature Tokenizer ─────────────────
        self.tokenizer = FeatureTokenizer(num_features, d_token)

        # ── [CLS] token ───────────────────────
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
        nn.init.normal_(self.cls_token, std=0.02)

        # ── Transformer Encoder ───────────────
        self.transformer = nn.ModuleList([
            TransformerBlock(d_token, n_heads, ffn_mult, dropout)
            for _ in range(n_layers)
        ])
        self.norm_out = nn.LayerNorm(d_token)

        # ── CLS → encoder_dim projection ──────
        # This output is what the unified model uses.
        self.encoder_proj = nn.Sequential(
            nn.Linear(d_token, encoder_dim),
            nn.LayerNorm(encoder_dim),
            nn.GELU(),
            nn.Dropout(head_dropout)
        )

        # ── Classification Head ───────────────
        cls_layers = []
        in_dim = encoder_dim
        for h in head_hidden:
            cls_layers += [
                nn.Linear(in_dim, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(head_dropout)
            ]
            in_dim = h
        cls_layers.append(nn.Linear(in_dim, num_classes))
        self.class_head = nn.Sequential(*cls_layers)

        # ── Regression Head ───────────────────
        reg_layers = []
        in_dim = encoder_dim
        for h in head_hidden:
            reg_layers += [
                nn.Linear(in_dim, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(head_dropout)
            ]
            in_dim = h
        reg_layers.append(nn.Linear(in_dim, num_regression))
        self.reg_head = nn.Sequential(*reg_layers)

        # ── Regression bounds tensor ──────────
        # Registered as buffer — moves with .to(device) automatically.
        bounds = torch.tensor([
            [LOG_MASS_MIN,   LOG_MASS_MAX],
            [LOG_LUM_MIN,    LOG_LUM_MAX],
            [LOG_TEFF_MIN,   LOG_TEFF_MAX],
            [LOG_RADIUS_MIN, LOG_RADIUS_MAX],
        ], dtype=torch.float32)
        self.register_buffer('reg_bounds', bounds)

        self._init_weights()

    def _init_weights(self):
        """Xavier uniform for all Linear layers in heads and projections."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _bounded_output(self, raw: torch.Tensor) -> torch.Tensor:
        """
        Constrains each regression output to its physical range.
        output = lo + (hi - lo) * sigmoid(raw)
        Sigmoid-scaling keeps gradients flowing at boundaries.
        Clamp does not — sigmoid is preferred here.
        """
        lo = self.reg_bounds[:, 0]   # (4,)
        hi = self.reg_bounds[:, 1]   # (4,)
        return lo + (hi - lo) * torch.sigmoid(raw)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Shared encoder — used by unified model after freezing.
        x: (B, NUM_FEATURES)
        returns: (B, ENCODER_DIM=256)
        """
        B = x.size(0)

        # Learnable feature scaling
        x = x * self.feature_scale.unsqueeze(0)   # (B, F)

        # Tokenize: (B, F, d_token)
        tokens = self.tokenizer(x)

        # Prepend CLS token: (B, F+1, d_token)
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)

        # Transformer layers
        for block in self.transformer:
            tokens = block(tokens)

        tokens = self.norm_out(tokens)

        # Extract CLS token (position 0) → project to encoder_dim
        cls_out = tokens[:, 0, :]           # (B, d_token)
        return self.encoder_proj(cls_out)   # (B, encoder_dim)

    def forward(
        self,
        x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: (B, NUM_FEATURES)
        returns:
            class_logits: (B, NUM_STELLAR_CLASSES)
            reg_out:      (B, NUM_REGRESSION) — physically bounded
            encoder_feat: (B, ENCODER_DIM)
        """
        encoder_feat = self.encode(x)
        class_logits = self.class_head(encoder_feat)
        reg_raw      = self.reg_head(encoder_feat)
        reg_out      = self._bounded_output(reg_raw)
        return class_logits, reg_out, encoder_feat

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for encode() — consistent naming across all modules."""
        return self.encode(x)


# ─────────────────────────────────────────────
# SANITY CHECK
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("Module 3 Model Sanity Check")
    print("=" * 55)

    device = torch.device("cpu")
    model  = StellarFTTransformer().to(device)

    B = 16
    x = torch.randn(B, NUM_FEATURES)

    class_logits, reg_out, enc_feat = model(x)

    print(f"\nInput shape:        {x.shape}")
    print(f"Class logits shape: {class_logits.shape}")
    print(f"Reg output shape:   {reg_out.shape}")
    print(f"Encoder feat shape: {enc_feat.shape}")

    # Shape assertions
    assert class_logits.shape == (B, NUM_STELLAR_CLASSES), \
        f"Class head wrong: {class_logits.shape}"
    assert reg_out.shape == (B, NUM_REGRESSION), \
        f"Reg head wrong: {reg_out.shape}"
    assert enc_feat.shape == (B, ENCODER_DIM), \
        f"Encoder wrong: {enc_feat.shape}"
    print("\n✅ Shape assertions passed")

    # encode() and get_features() consistency
    # Must use eval() — dropout makes two separate calls differ in train mode
    model.eval()
    with torch.no_grad():
        enc1 = model.encode(x)
        enc2 = model.get_features(x)
    assert enc1.shape == (B, ENCODER_DIM)
    assert torch.allclose(enc1, enc2), "encode() and get_features() mismatch"
    model.train()
    print("✅ encode() == get_features() confirmed")

    # NaN check
    assert not torch.isnan(class_logits).any(), "NaN in class logits"
    assert not torch.isnan(reg_out).any(),      "NaN in reg output"
    assert not torch.isnan(enc_feat).any(),     "NaN in encoder feat"
    print("✅ No NaNs in any output")

    # Regression bounds check
    bounds = model.reg_bounds
    for i, name in enumerate(REGRESSION_TARGETS):
        lo = bounds[i, 0].item()
        hi = bounds[i, 1].item()
        pred_min = reg_out[:, i].min().item()
        pred_max = reg_out[:, i].max().item()
        assert pred_min >= lo - 1e-5, \
            f"{name} below lower bound: {pred_min:.4f} < {lo}"
        assert pred_max <= hi + 1e-5, \
            f"{name} above upper bound: {pred_max:.4f} > {hi}"
    print("✅ All regression outputs within physical bounds")

    # Learnable feature scale check
    assert model.feature_scale.shape == (NUM_FEATURES,), \
        "feature_scale shape wrong"
    assert model.feature_scale.requires_grad, \
        "feature_scale must be trainable"
    print("✅ Learnable feature scaling confirmed")

    # GPU test
    if torch.cuda.is_available():
        model_gpu = StellarFTTransformer().cuda()
        x_gpu     = x.cuda()
        logits_g, reg_g, enc_g = model_gpu(x_gpu)
        assert logits_g.device.type == "cuda"
        print("✅ GPU forward pass confirmed")
    else:
        print("⚠️  No GPU — skipping CUDA check (will run on Kaggle)")

    # Parameter count
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters()
                    if p.requires_grad)
    print(f"\n📊 Total params:     {total:,}")
    print(f"📊 Trainable params: {trainable:,}")

    # Bounds summary
    print(f"\nRegression output bounds:")
    for i, name in enumerate(REGRESSION_TARGETS):
        lo = bounds[i, 0].item()
        hi = bounds[i, 1].item()
        print(f"  {name:12s}: [{lo:.1f}, {hi:.1f}]")

    # Feature scale init
    print(f"\nFeature scale init (all 1.0 at start):")
    for name, val in zip(FEATURE_NAMES, model.feature_scale.tolist()):
        print(f"  {name:12s}: {val:.1f}")

    print("\n✅ module3/model.py — all checks passed")