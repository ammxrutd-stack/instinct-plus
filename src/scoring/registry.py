"""Resolve the scoring contract to whatever is actually available.

The app asks for a Scorer and gets one that uses the trained title and
thumbnail models when their artefacts exist under models/, and the placeholder
otherwise. The Scorer reports which it is doing and on what data the models
were trained, so the interface can show a banner instead of a silent fake.

It also combines the two results into one overall score. The weights are each
model's out-of-fold Spearman correlation: a model that has demonstrated more
skill on held-out channels gets more say. A placeholder gets the floor weight.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from config.schema import MODELS_DIR
from src.scoring import placeholder
from src.scoring.interface import ScoreResult

FLOOR_WEIGHT = 0.05


@dataclass
class ScorerStatus:
    title_backend: str          # "model" | "placeholder"
    thumbnail_backend: str
    data_source: str            # "synthetic" | "collected" | "mixed" | "none"
    title_card: Optional[Dict[str, Any]] = None
    thumbnail_card: Optional[Dict[str, Any]] = None
    problems: Dict[str, str] = field(default_factory=dict)

    @property
    def all_real(self) -> bool:
        return self.title_backend == "model" and self.thumbnail_backend == "model"

    @property
    def synthetic(self) -> bool:
        return self.data_source in ("synthetic", "mixed")


class Scorer:
    def __init__(self, title_fn, thumbnail_fn, thumbnail_bytes_fn, status: ScorerStatus):
        self._title = title_fn
        self._thumb = thumbnail_fn
        self._thumb_bytes = thumbnail_bytes_fn
        self.status = status

    # the contract
    def score_title(self, title: str, content_type: str, subscriber_count: Optional[float] = None) -> ScoreResult:
        return self._title(title, content_type, subscriber_count=subscriber_count)

    def score_thumbnail(self, image_path: str, content_type: str, subscriber_count: Optional[float] = None) -> ScoreResult:
        return self._thumb(image_path, content_type, subscriber_count=subscriber_count)

    def score_thumbnail_bytes(self, data: bytes, content_type: str, subscriber_count: Optional[float] = None) -> ScoreResult:
        return self._thumb_bytes(data, content_type, subscriber_count=subscriber_count)

    # combination
    def weights(self) -> Dict[str, float]:
        def skill(card):
            if not card:
                return FLOOR_WEIGHT
            s = card.get("spearman")
            return max(FLOOR_WEIGHT, float(s)) if s is not None and s == s else FLOOR_WEIGHT

        wt = skill(self.status.title_card if self.status.title_backend == "model" else None)
        wi = skill(self.status.thumbnail_card if self.status.thumbnail_backend == "model" else None)
        total = wt + wi
        return {"title": wt / total, "thumbnail": wi / total}

    def combine(self, title: Optional[ScoreResult], thumbnail: Optional[ScoreResult]) -> ScoreResult:
        """Weighted blend. Either half may be missing (the creator has only a
        title so far); then the overall is that half, flagged in meta."""
        w = self.weights()
        parts = {}
        if title is not None and not title.meta.get("empty"):
            parts["title"] = title
        if thumbnail is not None and not thumbnail.meta.get("image_missing"):
            parts["thumbnail"] = thumbnail
        if not parts:
            return ScoreResult(score=0.0, notes=["Nothing to score yet."], meta={"parts": []})
        total_w = sum(w[k] for k in parts)
        score = sum(w[k] / total_w * parts[k].score for k in parts)
        drivers = {}
        for k, res in parts.items():
            for feature, pts in res.drivers.items():
                drivers[f"{k}: {feature}"] = float(round(pts * w[k] / total_w))
        drivers = dict(sorted(drivers.items(), key=lambda kv: -abs(kv[1]))[:10])
        notes = []
        for k, res in parts.items():
            notes.extend(f"{k.capitalize()}: {n}" for n in res.notes[:2])
        return ScoreResult(
            score=float(round(score)),
            drivers=drivers,
            notes=notes,
            meta={"parts": list(parts), "weights": {k: round(w[k] / total_w, 3) for k in parts}},
        )

    def score_video(self, title: str, image_path: Optional[str], content_type: str,
                    subscriber_count: Optional[float] = None) -> Dict[str, ScoreResult]:
        t = self.score_title(title, content_type, subscriber_count)
        i = self.score_thumbnail(image_path, content_type, subscriber_count) if image_path else None
        return {"title": t, "thumbnail": i, "overall": self.combine(t, i)}


_scorers: Dict[str, Scorer] = {}


def get_scorer(models_dir: Path = MODELS_DIR, allow_placeholder: bool = True) -> Scorer:
    key = str(models_dir)
    if key in _scorers:
        return _scorers[key]
    from src.modelling.model import ReachModel

    problems = {}
    title_fn, thumb_fn, thumb_bytes_fn = None, None, None
    title_card = ReachModel.card("title", models_dir)
    thumb_card = ReachModel.card("thumbnail", models_dir)

    try:
        from src.title.model import load_model, score_title

        load_model(models_dir)
        title_fn = lambda *a, **k: score_title(*a, models_dir=models_dir, **k)  # noqa: E731
    except Exception as e:  # missing artefact or incompatible pickle
        problems["title"] = str(e)
    try:
        from src.thumbnail.model import load_model as load_thumb, score_thumbnail, score_thumbnail_bytes

        load_thumb(models_dir)
        thumb_fn = lambda *a, **k: score_thumbnail(*a, models_dir=models_dir, **k)  # noqa: E731
        thumb_bytes_fn = lambda *a, **k: score_thumbnail_bytes(*a, models_dir=models_dir, **k)  # noqa: E731
    except Exception as e:
        problems["thumbnail"] = str(e)

    if (title_fn is None or thumb_fn is None) and not allow_placeholder:
        raise RuntimeError(f"trained models missing: {problems}")

    sources = set()
    if title_fn is not None and title_card:
        sources.add(title_card.get("data_source", "unknown"))
    if thumb_fn is not None and thumb_card:
        sources.add(thumb_card.get("data_source", "unknown"))
    data_source = "none" if not sources else (sources.pop() if len(sources) == 1 else "mixed")

    status = ScorerStatus(
        title_backend="model" if title_fn else "placeholder",
        thumbnail_backend="model" if thumb_fn else "placeholder",
        data_source=data_source,
        title_card=title_card if title_fn else None,
        thumbnail_card=thumb_card if thumb_fn else None,
        problems=problems,
    )
    scorer = Scorer(
        title_fn or placeholder.score_title,
        thumb_fn or placeholder.score_thumbnail,
        thumb_bytes_fn or (lambda data, ct, **k: placeholder.score_thumbnail("<bytes>", ct, **k)),
        status,
    )
    _scorers[key] = scorer
    return scorer


def reset():
    """Forget cached scorers (after retraining, or in tests)."""
    _scorers.clear()
    try:
        from src.title.model import clear_cache as c1
        from src.thumbnail.model import clear_cache as c2

        c1()
        c2()
    except Exception:
        pass
