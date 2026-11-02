"""score_thumbnail: the thumbnail pair's implementation of the scoring contract."""

from pathlib import Path
from typing import Optional

import numpy as np

from config.schema import CONTENT_TYPES, MODELS_DIR
from src.modelling.model import ReachModel
from src.scoring.interface import ScoreResult
from src.scoring.suggestions import thumbnail_notes, visible_drivers
from src.thumbnail.features import (
    FEATURE_VERSION,
    extract_image_features,
    extract_thumbnail_features,
    load_image,
)

MODEL_NAME = "thumbnail"
_cache = {}


def load_model(models_dir: Path = MODELS_DIR) -> ReachModel:
    key = str(models_dir)
    if key not in _cache:
        _cache[key] = ReachModel.load(MODEL_NAME, models_dir)
    return _cache[key]


def clear_cache():
    _cache.clear()


def _finish(feats, content_type, subscriber_count, model, source) -> ScoreResult:
    if feats.get("image_missing", 0):
        return ScoreResult(score=0.0, notes=["The image could not be read. Upload a JPEG or PNG."],
                           meta={"model": MODEL_NAME, "image_missing": True, "source": source})
    scored = model.score(feats, content_type)
    score, raw, all_drivers = scored["score"], scored["raw"], scored["drivers"]
    drivers = visible_drivers(all_drivers)
    notes = thumbnail_notes(feats, all_drivers, content_type, lambda name: model.benchmark(content_type, name))
    report = model.report
    return ScoreResult(
        score=float(round(score)),
        drivers={k: float(round(v)) for k, v in drivers.items()},
        notes=notes,
        features={k: v for k, v in feats.items() if not k.startswith("ct_")},
        meta={
            "model": MODEL_NAME,
            "backend": "model",
            "feature_version": FEATURE_VERSION,
            "raw_prediction": raw,
            "adjusted_prediction": scored["adjusted"],
            "context_points": scored["context_points"],
            "data_source": report.data_source if report else "unknown",
            "trained_at": report.trained_at if report else None,
            "spearman": report.spearman if report else None,
            "content_type_known": content_type in CONTENT_TYPES,
            "subscriber_count": subscriber_count,
            "source": source,
        },
    )


def score_thumbnail(
    image_path: str,
    content_type: str,
    subscriber_count: Optional[float] = None,
    model: Optional[ReachModel] = None,
    models_dir: Path = MODELS_DIR,
) -> ScoreResult:
    model = model or load_model(models_dir)
    feats = extract_thumbnail_features(image_path, content_type, subscriber_count)
    return _finish(feats, content_type, subscriber_count, model, str(image_path))


def score_thumbnail_image(
    image_bgr: np.ndarray,
    content_type: str,
    subscriber_count: Optional[float] = None,
    model: Optional[ReachModel] = None,
    models_dir: Path = MODELS_DIR,
) -> ScoreResult:
    """Score an image already in memory (the app's upload path)."""
    import math

    model = model or load_model(models_dir)
    feats = extract_image_features(image_bgr)
    feats["image_missing"] = 0.0
    feats["log_subscribers"] = math.log10(subscriber_count) if subscriber_count and subscriber_count > 0 else float("nan")
    for ct in CONTENT_TYPES:
        feats["ct_" + ct] = float(content_type == ct)
    return _finish(feats, content_type, subscriber_count, model, "memory")


def score_thumbnail_bytes(data: bytes, content_type: str, subscriber_count: Optional[float] = None, **kw) -> ScoreResult:
    import cv2

    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR) if arr.size else None
    if img is None:
        return ScoreResult(score=0.0, notes=["The image could not be read. Upload a JPEG or PNG."],
                           meta={"model": MODEL_NAME, "image_missing": True})
    return score_thumbnail_image(img, content_type, subscriber_count, **kw)
