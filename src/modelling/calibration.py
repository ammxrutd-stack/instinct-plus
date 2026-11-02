"""Map a model's raw prediction (reach, in log units) onto the 0-100 score.

The score is a percentile: "this prediction is higher than X% of the
predictions the model made for training videos of the same content type".
Using the distribution of *predictions* rather than of true reach matters:
a regressor's outputs are compressed toward the mean, so ranking against true
reach would pin every score near 50. Ranking against out-of-fold predictions
spreads scores across the whole range while keeping the ordering honest.

Per content type, because a prediction that is good for finance may be
mediocre for gaming, and the creator wants to know where they stand against
their own category.
"""

from typing import Dict, Optional

import numpy as np

from config.schema import SCORE_MAX, SCORE_MIN

GRID = np.linspace(0, 100, 201)
ALL = "__all__"


class Calibrator:
    def __init__(self, quantiles: Dict[str, np.ndarray]):
        # quantiles[content_type] is the prediction value at each GRID percentile
        self.quantiles = {k: np.asarray(v, dtype=float) for k, v in quantiles.items()}

    @classmethod
    def fit(cls, predictions: np.ndarray, content_types, min_rows: int = 30) -> "Calibrator":
        predictions = np.asarray(predictions, dtype=float)
        content_types = np.asarray(content_types)
        quantiles = {ALL: np.percentile(predictions, GRID)}
        for ct in np.unique(content_types):
            mask = content_types == ct
            if mask.sum() >= min_rows:
                quantiles[str(ct)] = np.percentile(predictions[mask], GRID)
        return cls(quantiles)

    def table(self, content_type: Optional[str]) -> np.ndarray:
        return self.quantiles.get(str(content_type), self.quantiles[ALL])

    def score(self, prediction: float, content_type: Optional[str]) -> float:
        q = self.table(content_type)
        s = float(np.interp(prediction, q, GRID))
        return float(min(SCORE_MAX, max(SCORE_MIN, s)))

    def points_per_unit(self, content_type: Optional[str]) -> float:
        """Local slope of the score map, in points per unit of prediction.

        Measured between the 20th and 80th percentiles so a few extreme
        predictions do not set it. Used to express SHAP values in points.
        """
        q = self.table(content_type)
        lo, hi = np.interp([20, 80], GRID, q)
        if hi - lo < 1e-9:
            return 0.0
        return 60.0 / float(hi - lo)

    def to_dict(self) -> Dict[str, list]:
        return {k: v.tolist() for k, v in self.quantiles.items()}

    @classmethod
    def from_dict(cls, d: Dict[str, list]) -> "Calibrator":
        return cls({k: np.asarray(v) for k, v in d.items()})
