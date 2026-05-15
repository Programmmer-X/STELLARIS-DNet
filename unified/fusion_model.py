"""
unified/fusion_model.py
STELLARIS-DNet — Unified Multi-Modal Fusion Model

Architecture:
  6 frozen encoders → per-encoder projection (→256) → LayerNorm
  → concat(6×256 + 6-dim mask) = 1542
  → FusionMLP (1542 → 768 → 512)
  → 7 independent output heads + physics consistency (computed)

All heads read from the same 512-dim fused embedding.
No head depends on another head's prediction.
Missing modalities handled via modality mask + zero-fill.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unified.config import (
    ENCODER_DIMS, PROJ_DIM, NUM_ENCODERS, MODALITY_ORDER,
    FUSION_INPUT_DIM, FUSION_HIDDEN_DIM, FUSED_DIM,
    FUSION_DROPOUT_1, FUSION_DROPOUT_2,
    PROJ_DROPOUT_EXPAND, PROJ_DROPOUT_PASS,
    NUM_STELLAR_CLASSES, NUM_PULSAR_SUBTYPES, NUM_RADIO_CLASSES,
    NUM_REG_TARGETS, REG_BOUNDS, HEAD_HIDDEN_DIM,
    ENCODER_PATHS, SCALER_PATHS, AE_THRESHOLD_PATH,
)


# ═════════════════════════════════════════════════════════════
# 1. ENCODER PROJECTION
# ═════════════════════════════════════════════════════════════

class EncoderProjection(nn.Module):
    """
    Projects a single encoder output to PROJ_DIM with LayerNorm.

    For encoders already at PROJ_DIM (256), this is LayerNorm only.
    For smaller encoders (32, 16), this is Linear → LayerNorm → GELU → Dropout.
    """
    def __init__(self, in_dim: int, out_dim: int = PROJ_DIM):
        super().__init__()
        if in_dim == out_dim:
            # Identity projection — just normalize
            self.proj = nn.LayerNorm(out_dim)
            self.is_expansion = False
        else:
            # Expansion projection — learned + regularized
            self.proj = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
                nn.Dropout(PROJ_DROPOUT_EXPAND),
            )
            self.is_expansion = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# ═════════════════════════════════════════════════════════════
# 2. FUSION MLP
# ═════════════════════════════════════════════════════════════

class FusionMLP(nn.Module):
    """
    Fuses projected encoder embeddings + modality mask into a
    single 512-dim representation.

    Input:  (B, 1542) = concat(6×256 projected + 6-dim mask)
    Output: (B, 512)  = fused embedding
    """
    def __init__(
        self,
        input_dim:  int = FUSION_INPUT_DIM,
        hidden_dim: int = FUSION_HIDDEN_DIM,
        output_dim: int = FUSED_DIM,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(FUSION_DROPOUT_1),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(FUSION_DROPOUT_2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


# ═════════════════════════════════════════════════════════════
# 3. OUTPUT HEADS
# ═════════════════════════════════════════════════════════════

class ClassificationHead(nn.Module):
    """Generic classification head: fused_dim → hidden → num_classes."""
    def __init__(self, input_dim: int, num_classes: int,
                 hidden_dim: int = HEAD_HIDDEN_DIM):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)  # (B, num_classes) logits


class BinaryHead(nn.Module):
    """Binary detection head: fused_dim → hidden → 1 → sigmoid."""
    def __init__(self, input_dim: int, hidden_dim: int = HEAD_HIDDEN_DIM):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)  # (B, 1) raw logit — apply sigmoid at loss/inference


class RegressionHead(nn.Module):
    """
    Physical parameter regression with sigmoid-bounded output.
    Same formulation as Module 3: sigmoid(raw) * (max - min) + min
    """
    def __init__(self, input_dim: int, num_targets: int = NUM_REG_TARGETS,
                 hidden_dim: int = HEAD_HIDDEN_DIM):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_targets),
        )

        # Register bounds as buffers (not parameters)
        bounds = list(REG_BOUNDS.values())
        self.register_buffer(
            "reg_min", torch.tensor([b[0] for b in bounds], dtype=torch.float32)
        )
        self.register_buffer(
            "reg_max", torch.tensor([b[1] for b in bounds], dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.head(x)                          # (B, 4)
        return torch.sigmoid(raw) * (self.reg_max - self.reg_min) + self.reg_min


class AnomalyHead(nn.Module):
    """Pulse profile anomaly score: fused_dim → hidden → 1 → sigmoid."""
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)  # (B, 1) raw logit — sigmoid at loss/inference


# ═════════════════════════════════════════════════════════════
# 4. ENCODER LOADER
# ═════════════════════════════════════════════════════════════

def _load_m1_mlp(path: str, device: torch.device) -> nn.Module:
    """Load PulsarMLP and return the encoder portion."""
    from module1.model import PulsarMLP
    model = PulsarMLP()
    # mlp_encoder.pt contains encoder state_dict only
    model.encoder.load_state_dict(torch.load(path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def _load_m1_cnn(path: str, device: torch.device) -> nn.Module:
    """Load full PulsarCNN (fixed: includes temporal_attn + freq_fusion)."""
    from module1.model import PulsarCNN
    model = PulsarCNN()
    # cnn_encoder.pt contains full model state_dict
    model.load_state_dict(torch.load(path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def _load_m1_ae(path: str, device: torch.device) -> nn.Module:
    """Load PulsarAutoencoder and return the encoder portion."""
    from module1.model import PulsarAutoencoder
    model = PulsarAutoencoder()
    # ae_encoder.pt contains encoder state_dict only
    model.encoder.load_state_dict(torch.load(path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def _load_m2_rgc(path: str, device: torch.device) -> nn.Module:
    """Load RadioGalaxyClassifier from full checkpoint."""
    from module2.model import RadioGalaxyClassifier
    model = RadioGalaxyClassifier()
    ckpt = torch.load(path, map_location=device)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def _load_m2_gwd(path: str, device: torch.device) -> nn.Module:
    """Load GravWaveDetector from full checkpoint."""
    from module2.model import GravWaveDetector
    model = GravWaveDetector()
    ckpt = torch.load(path, map_location=device)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def _load_m3(path: str, device: torch.device) -> nn.Module:
    """Load StellarFTTransformer from full checkpoint."""
    from module3.model import StellarFTTransformer
    model = StellarFTTransformer()
    ckpt = torch.load(path, map_location=device)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


ENCODER_LOADERS = {
    "m1_mlp": _load_m1_mlp,
    "m1_cnn": _load_m1_cnn,
    "m1_ae":  _load_m1_ae,
    "m2_rgc": _load_m2_rgc,
    "m2_gwd": _load_m2_gwd,
    "m3":     _load_m3,
}


# ═════════════════════════════════════════════════════════════
# 5. ENCODER FORWARD FUNCTIONS
# ═════════════════════════════════════════════════════════════

def _encode_m1_mlp(model, batch: dict) -> torch.Tensor:
    """PulsarMLP encoder: (B, 8) → (B, 32)"""
    return model.get_features(batch["m1_mlp"])


def _encode_m1_cnn(model, batch: dict) -> torch.Tensor:
    """PulsarCNN encoder: (B, 1, 64) + optional (B, 32) → (B, 256)"""
    x_time = batch["m1_cnn_time"]
    x_freq = batch.get("m1_cnn_freq", None)
    return model.encode(x_time, x_freq)


def _encode_m1_ae(model, batch: dict) -> torch.Tensor:
    """PulsarAutoencoder encoder: (B, 64) → (B, 16)"""
    return model.encode(batch["m1_ae"])


def _encode_m2_rgc(model, batch: dict) -> torch.Tensor:
    """RadioGalaxyClassifier encoder: (B, 3, 224, 224) → (B, 256)"""
    return model.encode(batch["m2_rgc"])


def _encode_m2_gwd(model, batch: dict) -> torch.Tensor:
    """GravWaveDetector encoder: (B, 3, 128, 128) → (B, 256)"""
    return model.encode(batch["m2_gwd"])


def _encode_m3(model, batch: dict) -> torch.Tensor:
    """StellarFTTransformer encoder: (B, 7) → (B, 256)"""
    return model.encode(batch["m3"])


ENCODER_FORWARDS = {
    "m1_mlp": _encode_m1_mlp,
    "m1_cnn": _encode_m1_cnn,
    "m1_ae":  _encode_m1_ae,
    "m2_rgc": _encode_m2_rgc,
    "m2_gwd": _encode_m2_gwd,
    "m3":     _encode_m3,
}


# ═════════════════════════════════════════════════════════════
# 6. UNIFIED MODEL
# ═════════════════════════════════════════════════════════════

class UnifiedModel(nn.Module):
    """
    STELLARIS-DNet Unified Multi-Modal Fusion Model.

    Loads 6 frozen encoders, projects all to 256-dim, fuses via MLP,
    and routes to 7 independent output heads.

    Forward input:  batch dict with modality tensors + mask
    Forward output: dict of head predictions + fused embedding
    """
    def __init__(self, device: torch.device, load_encoders: bool = True):
        super().__init__()
        self.device = device

        # ── Projection layers (trainable) ────────────────────────────
        self.projections = nn.ModuleDict({
            name: EncoderProjection(dim, PROJ_DIM)
            for name, dim in ENCODER_DIMS.items()
        })

        # ── Fusion MLP (trainable) ───────────────────────────────────
        self.fusion = FusionMLP()

        # ── Output heads (trainable) ─────────────────────────────────
        self.stellar_cls_head     = ClassificationHead(FUSED_DIM, NUM_STELLAR_CLASSES)
        self.pulsar_det_head      = BinaryHead(FUSED_DIM)
        self.pulsar_subtype_head  = ClassificationHead(FUSED_DIM, NUM_PULSAR_SUBTYPES)
        self.radio_morphology_head = ClassificationHead(FUSED_DIM, NUM_RADIO_CLASSES)
        self.gw_det_head          = BinaryHead(FUSED_DIM)
        self.anomaly_head         = AnomalyHead(FUSED_DIM)
        self.regression_head      = RegressionHead(FUSED_DIM)

        # ── Encoders (frozen, not nn.Module children) ────────────────
        # Stored in a plain dict to prevent optimizer from seeing them
        self._encoders = {}
        if load_encoders:
            self._load_all_encoders()

    def _load_all_encoders(self):
        """Load and freeze all module encoders."""
        for name in MODALITY_ORDER:
            path = ENCODER_PATHS[name]
            if not os.path.exists(path):
                print(f"⚠️  Encoder not found: {name} → {path}")
                continue
            loader = ENCODER_LOADERS[name]
            model = loader(path, self.device)
            # Freeze all parameters
            for param in model.parameters():
                param.requires_grad = False
            self._encoders[name] = model
            size_kb = os.path.getsize(path) / 1024
            print(f"🔒 {name:<8s} encoder loaded + frozen  ({size_kb:.1f} KB)")

    def unfreeze_encoder(self, name: str):
        """Unfreeze a specific encoder for fine-tuning."""
        if name not in self._encoders:
            raise ValueError(f"Encoder '{name}' not loaded")
        for param in self._encoders[name].parameters():
            param.requires_grad = True
        print(f"🔓 {name} encoder unfrozen")

    def freeze_encoder(self, name: str):
        """Re-freeze a specific encoder."""
        if name not in self._encoders:
            raise ValueError(f"Encoder '{name}' not loaded")
        for param in self._encoders[name].parameters():
            param.requires_grad = False
        print(f"🔒 {name} encoder re-frozen")

    def get_encoder_params(self) -> list:
        """Returns list of unfrozen encoder parameters (for optimizer)."""
        params = []
        for name, model in self._encoders.items():
            for p in model.parameters():
                if p.requires_grad:
                    params.append(p)
        return params

    def _run_encoders(
        self,
        batch: dict,
        mask: torch.Tensor,
    ) -> list:
        """
        Run each active encoder and return projected embeddings.

        Args:
            batch: dict with modality tensors (keys match ENCODER_FORWARDS)
            mask:  (B, 6) binary modality mask

        Returns:
            list of 6 projected embeddings, each (B, PROJ_DIM)
        """
        projected = []
        for i, name in enumerate(MODALITY_ORDER):
            dim = ENCODER_DIMS[name]

            if name in self._encoders and mask[:, i].any():
                # Run encoder
                encoder = self._encoders[name]
                forward_fn = ENCODER_FORWARDS[name]
                with torch.set_grad_enabled(
                    any(p.requires_grad for p in encoder.parameters())
                ):
                    raw_emb = forward_fn(encoder, batch)  # (B, encoder_dim)

                # Zero-fill samples where this modality is absent
                sample_mask = mask[:, i].unsqueeze(1).float()  # (B, 1)
                raw_emb = raw_emb * sample_mask

                # Project
                proj_emb = self.projections[name](raw_emb)  # (B, PROJ_DIM)
                proj_emb = proj_emb * sample_mask  # re-apply mask after projection
            else:
                # Encoder not loaded or entirely absent in this batch
                B = mask.shape[0]
                proj_emb = torch.zeros(B, PROJ_DIM, device=self.device)

            projected.append(proj_emb)

        return projected

    def forward(
        self,
        batch: dict,
        mask: torch.Tensor,
    ) -> dict:
        """
        Full forward pass.

        Args:
            batch: dict with modality input tensors
            mask:  (B, 6) binary tensor — which encoders are active

        Returns:
            dict with keys:
                stellar_cls:      (B, 5)   logits
                pulsar_det:       (B, 1)   logit
                pulsar_subtype:   (B, 4)   logits
                radio_morphology: (B, 2)   logits
                gw_det:           (B, 1)   logit
                anomaly:          (B, 1)   logit
                regression:       (B, 4)   bounded values
                fused:            (B, 512) fused embedding
                mask:             (B, 6)   modality mask (passthrough)
        """
        mask = mask.to(self.device).float()

        # ── Encode + project ─────────────────────────────────────────
        projected = self._run_encoders(batch, mask)

        # ── Concatenate + append mask ────────────────────────────────
        fused_input = torch.cat(projected + [mask], dim=1)  # (B, 1542)

        # ── Fusion ───────────────────────────────────────────────────
        fused = self.fusion(fused_input)  # (B, 512)

        # ── Output heads ─────────────────────────────────────────────
        outputs = {
            "stellar_cls":      self.stellar_cls_head(fused),
            "pulsar_det":       self.pulsar_det_head(fused),
            "pulsar_subtype":   self.pulsar_subtype_head(fused),
            "radio_morphology": self.radio_morphology_head(fused),
            "gw_det":           self.gw_det_head(fused),
            "anomaly":          self.anomaly_head(fused),
            "regression":       self.regression_head(fused),
            "fused":            fused,
            "mask":             mask,
        }

        return outputs

    def count_params(self) -> dict:
        """Count parameters by component."""
        counts = {}

        # Projections
        proj_total = sum(p.numel() for p in self.projections.parameters())
        proj_train = sum(p.numel() for p in self.projections.parameters()
                        if p.requires_grad)
        counts["projections"] = {"total": proj_total, "trainable": proj_train}

        # Fusion MLP
        fus_total = sum(p.numel() for p in self.fusion.parameters())
        counts["fusion_mlp"] = {"total": fus_total, "trainable": fus_total}

        # Heads
        head_names = ["stellar_cls_head", "pulsar_det_head",
                      "pulsar_subtype_head", "radio_morphology_head",
                      "gw_det_head", "anomaly_head", "regression_head"]
        head_total = 0
        for hname in head_names:
            head = getattr(self, hname)
            n = sum(p.numel() for p in head.parameters())
            head_total += n
        counts["all_heads"] = {"total": head_total, "trainable": head_total}

        # Encoders (frozen)
        enc_total = sum(
            sum(p.numel() for p in m.parameters())
            for m in self._encoders.values()
        )
        enc_train = sum(
            sum(p.numel() for p in m.parameters() if p.requires_grad)
            for m in self._encoders.values()
        )
        counts["encoders"] = {"total": enc_total, "trainable": enc_train}

        # Grand total
        trainable_all = (proj_train + fus_total + head_total + enc_train)
        total_all = (proj_total + fus_total + head_total + enc_total)
        counts["grand_total"] = {"total": total_all, "trainable": trainable_all}

        return counts


# ═════════════════════════════════════════════════════════════
# 7. SANITY CHECK
# ═════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("Unified Fusion Model — Architecture Sanity Check")
    print("=" * 60)

    device = torch.device("cpu")
    B = 4

    # Build model WITHOUT loading real encoders
    model = UnifiedModel(device=device, load_encoders=False)
    model.eval()

    # ── Verify projection dimensions ─────────────────────────────
    print("\n── Projection Layers ──")
    for name, proj in model.projections.items():
        in_dim = ENCODER_DIMS[name]
        x = torch.randn(B, in_dim)
        out = proj(x)
        print(f"  {name:<8s}: {in_dim:>4d} → {out.shape[1]:>4d}"
              f"  {'(expansion)' if proj.is_expansion else '(LayerNorm only)'}")
        assert out.shape == (B, PROJ_DIM)

    # ── Verify fusion MLP ────────────────────────────────────────
    print("\n── Fusion MLP ──")
    x_fused = torch.randn(B, FUSION_INPUT_DIM)
    out_fused = model.fusion(x_fused)
    print(f"  Input:  {x_fused.shape}")
    print(f"  Output: {out_fused.shape}")
    assert out_fused.shape == (B, FUSED_DIM)

    # ── Verify all heads ─────────────────────────────────────────
    print("\n── Output Heads ──")
    x_head = torch.randn(B, FUSED_DIM)

    heads = {
        "stellar_cls":      (model.stellar_cls_head, (B, NUM_STELLAR_CLASSES)),
        "pulsar_det":       (model.pulsar_det_head, (B, 1)),
        "pulsar_subtype":   (model.pulsar_subtype_head, (B, NUM_PULSAR_SUBTYPES)),
        "radio_morphology": (model.radio_morphology_head, (B, NUM_RADIO_CLASSES)),
        "gw_det":           (model.gw_det_head, (B, 1)),
        "anomaly":          (model.anomaly_head, (B, 1)),
        "regression":       (model.regression_head, (B, NUM_REG_TARGETS)),
    }

    for hname, (head, expected_shape) in heads.items():
        out = head(x_head)
        status = "✅" if out.shape == expected_shape else "❌"
        print(f"  {status} {hname:<20s}: {out.shape}")
        assert out.shape == expected_shape

    # ── Verify regression bounds ─────────────────────────────────
    print("\n── Regression Bounds ──")
    reg_out = model.regression_head(x_head)
    reg_min = model.regression_head.reg_min
    reg_max = model.regression_head.reg_max
    for i, (name, (lo, hi)) in enumerate(REG_BOUNDS.items()):
        vals = reg_out[:, i]
        print(f"  {name:<12s}: [{vals.min():.3f}, {vals.max():.3f}]"
              f"  bounds: [{lo}, {hi}]")
        assert vals.min() >= lo - 0.01 and vals.max() <= hi + 0.01

    # ── Verify full forward (mock) ───────────────────────────────
    print("\n── Full Forward (no encoders, zero-fill) ──")
    mock_batch = {}
    mock_mask = torch.zeros(B, NUM_ENCODERS)
    outputs = model(mock_batch, mock_mask)

    for key, val in outputs.items():
        if isinstance(val, torch.Tensor):
            print(f"  {key:<20s}: {list(val.shape)}")

    # ── Parameter counts ─────────────────────────────────────────
    print("\n── Parameter Counts ──")
    counts = model.count_params()
    for comp, c in counts.items():
        print(f"  {comp:<16s}: {c['total']:>10,} total"
              f"  |  {c['trainable']:>10,} trainable")

    print("\n" + "=" * 60)
    print("✅ All architecture checks passed")
    print("=" * 60)