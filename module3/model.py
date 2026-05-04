"""
module3/model.py
STELLARIS-DNet — Module 3 Architecture (v2)
FT-Transformer (Feature Tokenizer + Transformer)
Dual-head: 5-class classification + 4-param regression
Encoder output (CLS token, dim=256) saved separately for unified model.

v2 changes:
  - Input is now 14 features (7 physical + 7 validity flags)
  - Validity flags allow model to know which features are real vs filled
  - Architecture itself unchanged — feature scaling layer still present
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
# Per-feature linear projection (no weight sharing).
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
        return x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


# ─────────────────────────────────────────────
# 2. TRANSFORMER BLOCK (Pre-LayerNorm)
# ─────────────────────────────────────────────
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ffn_mult=4, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
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

    def forward(self, x):
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.drop(attn_out)
        x = x + self.ffn(self.norm2(x))
        return x


# ─────────────────────────────────────────────
# 3. FT-TRANSFORMER — MAIN MODEL
# ─────────────────────────────────────────────
class StellarFTTransformer(nn.Module):
    """
    Input:  (B, NUM_FEATURES=14) — 7 physical (standardised) + 7 validity flags
    Output: (class_logits, reg_out, encoder_feat)
        class_logits: (B, 5)
        reg_out:      (B, 4) sigmoid-bounded log-scale params
        encoder_feat: (B, 256) CLS token for unified model
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

        # Learnable feature scaling — applied before tokenization
        self.feature_scale = nn.Parameter(torch.ones(num_features))

        self.tokenizer = FeatureTokenizer(num_features, d_token)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
        nn.init.normal_(self.cls_token, std=0.02)

        self.transformer = nn.ModuleList([
            TransformerBlock(d_token, n_heads, ffn_mult, dropout)
            for _ in range(n_layers)
        ])
        self.norm_out = nn.LayerNorm(d_token)

        self.encoder_proj = nn.Sequential(
            nn.Linear(d_token, encoder_dim),
            nn.LayerNorm(encoder_dim),
            nn.GELU(),
            nn.Dropout(head_dropout)
        )

        # Classification head
        cls_layers, in_dim = [], encoder_dim
        for h in head_hidden:
            cls_layers += [
                nn.Linear(in_dim, h), nn.LayerNorm(h),
                nn.GELU(), nn.Dropout(head_dropout)
            ]
            in_dim = h
        cls_layers.append(nn.Linear(in_dim, num_classes))
        self.class_head = nn.Sequential(*cls_layers)

        # Regression head
        reg_layers, in_dim = [], encoder_dim
        for h in head_hidden:
            reg_layers += [
                nn.Linear(in_dim, h), nn.LayerNorm(h),
                nn.GELU(), nn.Dropout(head_dropout)
            ]
            in_dim = h
        reg_layers.append(nn.Linear(in_dim, num_regression))
        self.reg_head = nn.Sequential(*reg_layers)

        # Regression bounds (registered as buffer — moves with .to(device))
        bounds = torch.tensor([
            [LOG_MASS_MIN,   LOG_MASS_MAX],
            [LOG_LUM_MIN,    LOG_LUM_MAX],
            [LOG_TEFF_MIN,   LOG_TEFF_MAX],
            [LOG_RADIUS_MIN, LOG_RADIUS_MAX],
        ], dtype=torch.float32)
        self.register_buffer('reg_bounds', bounds)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _bounded_output(self, raw: torch.Tensor) -> torch.Tensor:
        """sigmoid-scaled output: lo + (hi - lo) * sigmoid(raw)"""
        lo = self.reg_bounds[:, 0]
        hi = self.reg_bounds[:, 1]
        return lo + (hi - lo) * torch.sigmoid(raw)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Shared encoder for unified model. Input (B,14) → output (B,256)."""
        B = x.size(0)
        x = x * self.feature_scale.unsqueeze(0)
        tokens = self.tokenizer(x)

        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)

        for block in self.transformer:
            tokens = block(tokens)

        tokens = self.norm_out(tokens)
        cls_out = tokens[:, 0, :]
        return self.encoder_proj(cls_out)

    def forward(self, x):
        encoder_feat = self.encode(x)
        class_logits = self.class_head(encoder_feat)
        reg_raw      = self.reg_head(encoder_feat)
        reg_out      = self._bounded_output(reg_raw)
        return class_logits, reg_out, encoder_feat

    def get_features(self, x):
        return self.encode(x)


# ─────────────────────────────────────────────
# SANITY CHECK
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("Module 3 Model Sanity Check (v2)")
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

    assert class_logits.shape == (B, NUM_STELLAR_CLASSES)
    assert reg_out.shape      == (B, NUM_REGRESSION)
    assert enc_feat.shape     == (B, ENCODER_DIM)
    print("\n✅ Shape assertions passed")

    # encode/get_features consistency (eval mode for dropout)
    model.eval()
    with torch.no_grad():
        e1 = model.encode(x)
        e2 = model.get_features(x)
    assert torch.allclose(e1, e2), "encode/get_features mismatch"
    model.train()
    print("✅ encode() == get_features() confirmed")

    assert not torch.isnan(class_logits).any()
    assert not torch.isnan(reg_out).any()
    assert not torch.isnan(enc_feat).any()
    print("✅ No NaNs in output")

    # Bounds check
    for i in range(NUM_REGRESSION):
        lo = model.reg_bounds[i, 0].item()
        hi = model.reg_bounds[i, 1].item()
        assert reg_out[:, i].min().item() >= lo - 1e-5
        assert reg_out[:, i].max().item() <= hi + 1e-5
    print("✅ All regression outputs within physical bounds")

    assert model.feature_scale.shape == (NUM_FEATURES,)
    assert model.feature_scale.requires_grad
    print(f"✅ Learnable feature scaling: {NUM_FEATURES} dims")

    if torch.cuda.is_available():
        gpu_model = StellarFTTransformer().cuda()
        _ = gpu_model(x.cuda())
        print("✅ GPU forward pass confirmed")
    else:
        print("⚠️  No GPU — skipping CUDA check")

    total = sum(p.numel() for p in model.parameters())
    print(f"\n📊 Total params: {total:,}")
    print(f"\n✅ module3/model.py v2 — all checks passed")