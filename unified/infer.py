"""
unified/infer.py
STELLARIS-DNet — Unified Single-Observation Inference

Usage:
    from unified.infer import UnifiedInference

    engine = UnifiedInference(checkpoint_path="checkpoints/unified/unified_final.pt")

    result = engine.predict(
        m1_mlp_features=np.array([...]),  # (8,) HTRU2 raw features
        m3_features=np.array([...]),      # (7,) stellar raw features
    )

    print(result["stellar_cls"])      # {"class": "Neutron_Star", "prob": 0.97}
    print(result["pulsar_det"])       # {"detected": True, "prob": 0.89}
    print(result["regression"])       # {"log_mass": 0.34, "log_lum": ...}
    print(result["head_validity"])    # {"stellar_cls": True, "pulsar_det": True, ...}

All preprocessing (scaling) handled internally.
Missing modalities handled via mask — pass only what you have.
"""

import os
import sys
import pickle
import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unified.config import (
    MODALITY_ORDER, MODALITY_INDEX, NUM_ENCODERS,
    STELLAR_CLASS_NAMES, PULSAR_SUBTYPE_NAMES, RADIO_CLASS_NAMES,
    REG_TARGET_NAMES, SCALER_PATHS, AE_THRESHOLD_PATH,
    HEAD_VALIDITY, ENCODER_PATHS,
)
from unified.dataset import INPUT_SHAPES
from unified.fusion_model import UnifiedModel
from core.utils import get_device


class UnifiedInference:
    """
    Self-contained inference engine for STELLARIS-DNet unified model.

    Loads model + scalers once at init. Call predict() per observation.
    Handles all preprocessing internally.
    """

    def __init__(self, checkpoint_path: str, device: torch.device = None):
        """
        Args:
            checkpoint_path: path to unified .pt checkpoint
            device: torch device (auto-detected if None)
        """
        self.device = device or get_device()

        # ── Load model ───────────────────────────────────────────
        print(f"Loading unified model from {checkpoint_path}...")
        self.model = UnifiedModel(device=self.device, load_encoders=True)
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"✅ Model loaded (epoch {ckpt.get('epoch', '?')})")

        # ── Load scalers ─────────────────────────────────────────
        self.scalers = {}
        for key, path in SCALER_PATHS.items():
            if os.path.exists(path):
                with open(path, "rb") as f:
                    self.scalers[key] = pickle.load(f)
                print(f"✅ Scaler loaded: {key}")
            else:
                print(f"⚠️  Scaler not found: {key} → {path}")

        # ── Load AE threshold ────────────────────────────────────
        self.ae_threshold = None
        if os.path.exists(AE_THRESHOLD_PATH):
            self.ae_threshold = float(np.load(AE_THRESHOLD_PATH)[0])
            print(f"✅ AE threshold: {self.ae_threshold:.6f}")

    @torch.no_grad()
    def predict(
        self,
        m1_mlp_features: np.ndarray = None,
        m1_pulse_profile: np.ndarray = None,
        m1_fft_features: np.ndarray = None,
        m2_radio_image: np.ndarray = None,
        m2_ligo_cqt: np.ndarray = None,
        m3_features: np.ndarray = None,
    ) -> dict:
        """
        Run unified inference on a single observation.

        Pass only the modalities you have. Missing → zero-fill + mask.
        Scalers applied internally for M1-MLP and M3.

        Args:
            m1_mlp_features:  (8,) raw HTRU2 statistical features
            m1_pulse_profile: (64,) or (N,) raw pulse profile (will pad/trim)
            m1_fft_features:  (32,) FFT magnitude features (optional for CNN)
            m2_radio_image:   (3, 224, 224) preprocessed radio galaxy image
            m2_ligo_cqt:      (3, 128, 128) preprocessed CQT spectrogram
            m3_features:      (7,) raw stellar features [teff, log_g, ...]

        Returns:
            dict with per-head predictions, confidence, and validity flags
        """
        inputs = {}
        mask = [0.0] * NUM_ENCODERS

        # ── M1-MLP ───────────────────────────────────────────────
        if m1_mlp_features is not None:
            feat = np.asarray(m1_mlp_features, dtype=np.float32).reshape(1, -1)
            if "m1_mlp" in self.scalers:
                feat = self.scalers["m1_mlp"].transform(feat)
            inputs["m1_mlp"] = torch.tensor(feat[0], dtype=torch.float32)
            mask[MODALITY_INDEX["m1_mlp"]] = 1.0

        # ── M1-CNN + M1-AE ──────────────────────────────────────
        if m1_pulse_profile is not None:
            profile = np.asarray(m1_pulse_profile, dtype=np.float32)
            # Pad/trim to 64
            if len(profile) < 64:
                profile = np.pad(profile, (0, 64 - len(profile)))
            elif len(profile) > 64:
                profile = profile[:64]
            # Min-max normalize
            pmin, pmax = profile.min(), profile.max()
            if pmax > pmin:
                profile = (profile - pmin) / (pmax - pmin)

            inputs["m1_cnn_time"] = torch.tensor(
                profile.reshape(1, 64), dtype=torch.float32
            )
            inputs["m1_ae"] = torch.tensor(profile, dtype=torch.float32)
            mask[MODALITY_INDEX["m1_cnn"]] = 1.0
            mask[MODALITY_INDEX["m1_ae"]] = 1.0

            # FFT features
            if m1_fft_features is not None:
                inputs["m1_cnn_freq"] = torch.tensor(
                    np.asarray(m1_fft_features, dtype=np.float32),
                    dtype=torch.float32,
                )
            else:
                # Compute FFT on the fly
                fft_mag = np.abs(np.fft.rfft(profile))[:32]
                fft_feat = np.log1p(fft_mag).astype(np.float32)
                fmax = fft_feat.max()
                if fmax > 0:
                    fft_feat /= fmax
                inputs["m1_cnn_freq"] = torch.tensor(fft_feat)

        # ── M2-RGC ──────────────────────────────────────────────
        if m2_radio_image is not None:
            img = np.asarray(m2_radio_image, dtype=np.float32)
            inputs["m2_rgc"] = torch.tensor(img)
            mask[MODALITY_INDEX["m2_rgc"]] = 1.0

        # ── M2-GWD ──────────────────────────────────────────────
        if m2_ligo_cqt is not None:
            cqt = np.asarray(m2_ligo_cqt, dtype=np.float32)
            inputs["m2_gwd"] = torch.tensor(cqt)
            mask[MODALITY_INDEX["m2_gwd"]] = 1.0

        # ── M3 ──────────────────────────────────────────────────
        if m3_features is not None:
            feat = np.asarray(m3_features, dtype=np.float32).reshape(1, -1)
            if "m3" in self.scalers:
                feat = self.scalers["m3"].transform(feat)
            inputs["m3"] = torch.tensor(feat[0], dtype=torch.float32)
            mask[MODALITY_INDEX["m3"]] = 1.0

        # ── Zero-fill absent modalities ──────────────────────────
        for key, shape in INPUT_SHAPES.items():
            if key not in inputs:
                inputs[key] = torch.zeros(shape, dtype=torch.float32)

        # ── Batch dimension (B=1) ────────────────────────────────
        batch_inputs = {k: v.unsqueeze(0).to(self.device) for k, v in inputs.items()}
        batch_mask = torch.tensor([mask], dtype=torch.float32).to(self.device)

        # ── Forward ──────────────────────────────────────────────
        outputs = self.model(batch_inputs, batch_mask)

        # ── Parse outputs ────────────────────────────────────────
        result = self._parse_outputs(outputs, mask)
        return result

    def _parse_outputs(self, outputs: dict, mask: list) -> dict:
        """Convert raw model outputs to interpretable predictions."""
        result = {}

        # ── Head validity ────────────────────────────────────────
        validity = {}
        for head, required_mods in HEAD_VALIDITY.items():
            valid = all(mask[MODALITY_INDEX[m]] > 0 for m in required_mods)
            validity[head] = valid
        result["head_validity"] = validity

        # ── Stellar classification ───────────────────────────────
        probs = F.softmax(outputs["stellar_cls"][0], dim=0).cpu().numpy()
        pred_idx = int(probs.argmax())
        result["stellar_cls"] = {
            "class": STELLAR_CLASS_NAMES[pred_idx],
            "class_idx": pred_idx,
            "prob": float(probs[pred_idx]),
            "all_probs": {n: float(p) for n, p in
                         zip(STELLAR_CLASS_NAMES, probs)},
            "valid": validity.get("stellar_cls", False),
        }

        # ── Pulsar detection ─────────────────────────────────────
        prob = float(torch.sigmoid(outputs["pulsar_det"][0, 0]).cpu())
        result["pulsar_det"] = {
            "detected": prob > 0.5,
            "prob": prob,
            "valid": validity.get("pulsar_det", False),
        }

        # ── Pulsar subtype ───────────────────────────────────────
        probs = F.softmax(outputs["pulsar_subtype"][0], dim=0).cpu().numpy()
        pred_idx = int(probs.argmax())
        result["pulsar_subtype"] = {
            "subtype": PULSAR_SUBTYPE_NAMES[pred_idx],
            "subtype_idx": pred_idx,
            "prob": float(probs[pred_idx]),
            "valid": validity.get("pulsar_subtype", False),
        }

        # ── Radio morphology ────────────────────────────────────
        probs = F.softmax(outputs["radio_morphology"][0], dim=0).cpu().numpy()
        pred_idx = int(probs.argmax())
        result["radio_morphology"] = {
            "class": RADIO_CLASS_NAMES[pred_idx],
            "prob": float(probs[pred_idx]),
            "valid": validity.get("radio_morphology", False),
        }

        # ── GW detection ────────────────────────────────────────
        prob = float(torch.sigmoid(outputs["gw_det"][0, 0]).cpu())
        result["gw_det"] = {
            "detected": prob > 0.5,
            "prob": prob,
            "valid": validity.get("gw_det", False),
        }

        # ── Anomaly score ────────────────────────────────────────
        score = float(torch.sigmoid(outputs["anomaly"][0, 0]).cpu())
        is_anomaly = None
        if self.ae_threshold is not None:
            is_anomaly = score > self.ae_threshold
        result["anomaly"] = {
            "score": score,
            "is_anomaly": is_anomaly,
            "threshold": self.ae_threshold,
            "valid": validity.get("anomaly", False),
        }

        # ── Regression ───────────────────────────────────────────
        reg_vals = outputs["regression"][0].cpu().numpy()
        result["regression"] = {
            name: float(val)
            for name, val in zip(REG_TARGET_NAMES, reg_vals)
        }
        result["regression"]["valid"] = validity.get("regression", False)

        # ── Fused embedding ──────────────────────────────────────
        result["fused_embedding"] = outputs["fused"][0].cpu().numpy()

        # ── Active modalities ────────────────────────────────────
        result["active_modalities"] = [
            MODALITY_ORDER[i] for i in range(NUM_ENCODERS) if mask[i] > 0
        ]

        return result

    def predict_batch(self, observations: list) -> list:
        """Run predict() on a list of observations. Returns list of results."""
        return [self.predict(**obs) for obs in observations]

    def summary(self, result: dict) -> str:
        """Pretty-print a prediction result."""
        lines = ["─── STELLARIS-DNet Prediction ───"]
        lines.append(f"Active modalities: {result['active_modalities']}")

        v = result["head_validity"]

        if v.get("stellar_cls"):
            sc = result["stellar_cls"]
            lines.append(f"Stellar class: {sc['class']} ({sc['prob']:.3f})")

        if v.get("pulsar_det"):
            pd = result["pulsar_det"]
            det = "DETECTED" if pd["detected"] else "not detected"
            lines.append(f"Pulsar: {det} ({pd['prob']:.3f})")

        if v.get("pulsar_subtype"):
            ps = result["pulsar_subtype"]
            lines.append(f"Pulsar subtype: {ps['subtype']} ({ps['prob']:.3f})")

        if v.get("radio_morphology"):
            rm = result["radio_morphology"]
            lines.append(f"Radio morphology: {rm['class']} ({rm['prob']:.3f})")

        if v.get("gw_det"):
            gw = result["gw_det"]
            det = "SIGNAL" if gw["detected"] else "noise"
            lines.append(f"GW: {det} ({gw['prob']:.3f})")

        if v.get("anomaly"):
            an = result["anomaly"]
            flag = " ⚠️ ANOMALY" if an.get("is_anomaly") else ""
            lines.append(f"Anomaly score: {an['score']:.4f}{flag}")

        if v.get("regression"):
            reg = result["regression"]
            reg_str = " | ".join(
                f"{n}: {reg[n]:.3f}" for n in REG_TARGET_NAMES
            )
            lines.append(f"Regression: {reg_str}")

        lines.append("────────────────────────────────")
        return "\n".join(lines)


# ═════════════════════════════════════════════════════════════
# DEMO / SANITY CHECK
# ═════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("Unified Inference — Sanity Check")
    print("=" * 60)
    print("\nThis script requires a trained unified checkpoint.")
    print("Run unified/train.py first, then:")
    print("  python unified/infer.py")
    print()

    ckpt_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "checkpoints", "unified", "unified_final.pt"
    )

    if not os.path.exists(ckpt_path):
        print(f"❌ Checkpoint not found: {ckpt_path}")
        print("   Train the unified model first.")
        sys.exit(0)

    engine = UnifiedInference(checkpoint_path=ckpt_path)

    # Mock M1-MLP input (8 HTRU2 features)
    result = engine.predict(
        m1_mlp_features=np.random.randn(8).astype(np.float32),
    )
    print("\n" + engine.summary(result))

    # Mock M3 input (7 stellar features)
    result = engine.predict(
        m3_features=np.random.randn(7).astype(np.float32),
    )
    print("\n" + engine.summary(result))

    # Mock cross-modal M1+M3
    result = engine.predict(
        m1_mlp_features=np.random.randn(8).astype(np.float32),
        m3_features=np.random.randn(7).astype(np.float32),
    )
    print("\n" + engine.summary(result))

    print("\n✅ Inference sanity check complete")