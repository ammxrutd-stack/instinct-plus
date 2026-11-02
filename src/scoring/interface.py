"""The scoring contract.

The title model and the thumbnail model are built separately by different
pairs, and the app consumes both. Both must return a ScoreResult, so the app
can be built and tested against this interface before either model exists.

Changes to this file require agreement from all three pairs.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ScoreResult:
    """The result of scoring one title or one thumbnail.

    score: predicted reach, 0 to 100. A percentile against the model's own
        predictions for training videos of the same content type: 70 means
        "better than 70% of them".
    drivers: feature name -> its contribution to the score, in points.
        Positive helped, negative hurt. Only the features that mattered.
    notes: human readable suggestions for the creator.
    features: every measured feature, for the app's detail view.
    meta: provenance (model name, data source, raw prediction, ...).
    """

    score: float = 0.0
    drivers: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    features: Dict[str, float] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)


def score_title(title: str, content_type: str) -> ScoreResult:
    """Score a proposed video title. Implemented by the title pair in
    src/title/model.py; resolved at runtime by src/scoring/registry.py."""
    raise NotImplementedError


def score_thumbnail(image_path: str, content_type: str) -> ScoreResult:
    """Score a proposed thumbnail image. Implemented by the thumbnail pair in
    src/thumbnail/model.py; resolved at runtime by src/scoring/registry.py."""
    raise NotImplementedError
