"""score_title: the title pair's implementation of the scoring contract."""

from pathlib import Path
from typing import Optional

from config.schema import CONTENT_TYPES, MODELS_DIR
from src.modelling.model import ReachModel
from src.scoring.interface import ScoreResult
from src.scoring.suggestions import title_notes, visible_drivers
from src.title.features import FEATURE_VERSION, extract_title_features

MODEL_NAME = "title"
_cache = {}


def load_model(models_dir: Path = MODELS_DIR) -> ReachModel:
    key = str(models_dir)
    if key not in _cache:
        _cache[key] = ReachModel.load(MODEL_NAME, models_dir)
    return _cache[key]


def clear_cache():
    _cache.clear()


def score_title(
    title: str,
    content_type: str,
    subscriber_count: Optional[float] = None,
    model: Optional[ReachModel] = None,
    models_dir: Path = MODELS_DIR,
) -> ScoreResult:
    model = model or load_model(models_dir)
    if not (title or "").strip():
        return ScoreResult(score=0.0, notes=["Enter a title to score."], meta={"model": MODEL_NAME, "empty": True})
    feats = extract_title_features(title, content_type, subscriber_count)
    scored = model.score(feats, content_type)
    score, raw, all_drivers = scored["score"], scored["raw"], scored["drivers"]
    drivers = visible_drivers(all_drivers)
    notes = title_notes(feats, all_drivers, content_type, lambda name: model.benchmark(content_type, name))
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
        },
    )
