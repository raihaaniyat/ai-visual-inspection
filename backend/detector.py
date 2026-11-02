"""
Deterministic anomaly-detection layer.

This is the ONLY component allowed to decide "defective / not defective".
Everything downstream (the VLM explainer) describes that decision in
words - it never overrides it. Same "don't trust the agent's self-report,
verify independently" principle used in AI-Dev-Crew's test verification.

Uses Anomalib's Engine.predict() + PredictDataset, NOT TorchInferencer.

Confirmed by a real run on Ali's machine (2026-09-10): TorchInferencer
requires a checkpoint saved via anomalib's own exporter (a .pt/.pth file
whose top-level dict has a "model" key) - it can NOT load the raw
Lightning .ckpt files that training/common.py produces directly from
`Engine.fit()`/`ModelCheckpoint` (raised
`ValueError: Unknown PyTorch checkpoint format .ckpt`). Engine.predict()
loads those same raw .ckpt files natively via Lightning's own checkpoint
loading, so it works with what training already produces - no separate
export step needed. This was flagged as the documented fallback in
SETUP_GUIDE.md before it was confirmed necessary; it's now the only path.

PredictDataset reads images from disk rather than an in-memory array, so
each call writes the uploaded image to a throwaway temp file first.
"""
from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from backend.config import settings


class ModelNotTrainedError(RuntimeError):
    """Raised when no checkpoint exists yet for a given category/backend."""


@dataclass
class DetectionResult:
    is_defective: bool
    anomaly_score: float
    heatmap: np.ndarray | None   # float array, same H x W as input, 0..1
    latency_ms: float


def _checkpoint_path(category: str, backend: str) -> Path:
    # Expected layout produced by training/train_patchcore.py and
    # training/train_efficientad.py:
    #   backend/data/checkpoints/<category>/<backend>/model.ckpt
    return settings.checkpoint_dir_path / category / backend / "model.ckpt"


# Maps our own "backend" name (used in file paths / API params) to the
# anomalib model class name - same mapping training/common.py uses.
_MODEL_CLASS_NAMES = {
    "patchcore": "Patchcore",
    "efficient_ad": "EfficientAd",
}


class AnomalyDetector:
    """Loads one trained Anomalib model for one (category, backend) pair."""

    def __init__(self, category: str, backend: str | None = None):
        self.category = category
        self.backend = backend or settings.ad_backend
        self._model = None
        self._engine = None
        self._ckpt_path: Path | None = None
        self._load()

    def _load(self) -> None:
        ckpt = _checkpoint_path(self.category, self.backend)
        if not ckpt.exists():
            raise ModelNotTrainedError(
                f"No trained checkpoint at {ckpt}. Run training/train_{self.backend}.py "
                f"for category='{self.category}' first (see SETUP_GUIDE.md)."
            )
        if self.backend not in _MODEL_CLASS_NAMES:
            msg = f"Unknown backend '{self.backend}'. Known: {list(_MODEL_CLASS_NAMES)}"
            raise ValueError(msg)

        # ANOMALIB-API: imports kept local so the whole module still imports
        # (and the rest of the app still runs) even before anomalib/torch
        # are installed, e.g. while just exploring the API endpoints.
        import anomalib.models as anomalib_models
        from anomalib.engine import Engine

        model_cls = getattr(anomalib_models, _MODEL_CLASS_NAMES[self.backend])

        # The model instance's weights are a placeholder - engine.predict()
        # below loads the real trained weights from ckpt_path on every
        # call, the same way training/common.py's engine.fit()/test() did.
        self._model = model_cls()
        self._engine = Engine()
        self._ckpt_path = ckpt

    def predict(self, image: Image.Image) -> DetectionResult:
        from anomalib.data import PredictDataset

        start = time.perf_counter()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / "input.png"
            image.convert("RGB").save(tmp_path)

            dataset = PredictDataset(path=tmp_path)
            predictions = self._engine.predict(
                model=self._model,
                dataset=dataset,
                ckpt_path=str(self._ckpt_path),
            )

        if not predictions:
            msg = (
                f"engine.predict() returned no results for "
                f"category='{self.category}', backend='{self.backend}'."
            )
            raise RuntimeError(msg)

        # engine.predict() returns one ImageBatch per dataloader batch;
        # PredictDataset + a single image -> one batch of size 1.
        batch = predictions[0]

        score = float(batch.pred_score[0])

        anomaly_map = getattr(batch, "anomaly_map", None)
        if anomaly_map is not None:
            anomaly_map = np.asarray(anomaly_map[0].detach().cpu(), dtype=np.float32)
            anomaly_map = np.squeeze(anomaly_map)  # drop any leading channel dim -> (H, W)
            m_min, m_max = float(anomaly_map.min()), float(anomaly_map.max())
            if m_max > m_min:
                anomaly_map = (anomaly_map - m_min) / (m_max - m_min)

        is_defective = score >= settings.anomaly_threshold

        latency_ms = (time.perf_counter() - start) * 1000.0
        return DetectionResult(
            is_defective=is_defective,
            anomaly_score=score,
            heatmap=anomaly_map,
            latency_ms=latency_ms,
        )


def make_heatmap_overlay(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Blend a 0..1 anomaly heatmap onto the original image using a JET colormap."""
    import cv2

    base = np.array(image.convert("RGB"))
    h, w = base.shape[:2]

    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * np.clip(heatmap_resized, 0, 1))
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = np.uint8(base * (1 - alpha) + heatmap_color * alpha)
    return Image.fromarray(overlay)


# Simple process-wide cache so the API doesn't reload a checkpoint from
# disk on every request.
_detector_cache: dict[tuple[str, str], AnomalyDetector] = {}


def get_detector(category: str, backend: str | None = None) -> AnomalyDetector:
    backend = backend or settings.ad_backend
    key = (category, backend)
    if key not in _detector_cache:
        _detector_cache[key] = AnomalyDetector(category, backend)
    return _detector_cache[key]
