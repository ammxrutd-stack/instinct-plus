"""Fake implementations of the scoring contract.

This exists so the app can be built and tested before either model is
trained, and so the app still starts on a machine with no models/ directory.
The numbers mean nothing. src/scoring/registry.py prefers the real models and
only falls back to this when an artefact is missing, and says so.
"""

import random
from typing import Optional

from src.scoring.interface import ScoreResult

PLACEHOLDER_NOTE = "This output is a placeholder, not a real prediction. Train the models: python -m src.modelling.train"


def score_title(title: str, content_type: str, subscriber_count: Optional[float] = None, **_) -> ScoreResult:
    """Return a random score for a title, in the shape of the real thing."""
    return ScoreResult(
        score=random.uniform(20, 90),
        drivers={
            "title_length": random.uniform(-10, 10),
            "has_number": random.uniform(-10, 10),
            "sentiment": random.uniform(-10, 10),
        },
        notes=[PLACEHOLDER_NOTE],
        meta={"model": "title", "backend": "placeholder", "data_source": "none"},
    )


def score_thumbnail(image_path: str, content_type: str, subscriber_count: Optional[float] = None, **_) -> ScoreResult:
    """Return a random score for a thumbnail, in the shape of the real thing."""
    return ScoreResult(
        score=random.uniform(20, 90),
        drivers={
            "face_present": random.uniform(-10, 10),
            "brightness": random.uniform(-10, 10),
            "text_area": random.uniform(-10, 10),
        },
        notes=[PLACEHOLDER_NOTE],
        meta={"model": "thumbnail", "backend": "placeholder", "data_source": "none"},
    )
