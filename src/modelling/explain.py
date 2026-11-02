"""Turn a prediction into drivers: feature -> contribution in score points.

SHAP TreeExplainer gives exact, additive attributions for tree ensembles. If it
is unavailable (import failure, unsupported estimator) we fall back to a
one-at-a-time perturbation: replace each feature with its training median and
record how much the prediction moves. Less principled, but model-agnostic and
never silently absent.

Both return contributions in the model's prediction units; the caller scales
them to score points with Calibrator.points_per_unit.
"""

from typing import Dict, Optional, Sequence

import numpy as np


class Explainer:
    def __init__(self, estimator, feature_names: Sequence[str], medians: np.ndarray):
        self.estimator = estimator
        self.feature_names = list(feature_names)
        self.medians = np.asarray(medians, dtype=float)
        self._shap = None
        self._shap_failed = False

    def _tree_explainer(self):
        if self._shap is None and not self._shap_failed:
            try:
                import shap

                self._shap = shap.TreeExplainer(self.estimator)
            except Exception:
                self._shap_failed = True
        return self._shap

    def contributions(self, x: np.ndarray) -> Dict[str, float]:
        """x is one row of features, shape (n_features,)."""
        x = np.asarray(x, dtype=float).reshape(1, -1)
        explainer = self._tree_explainer()
        if explainer is not None:
            try:
                values = np.asarray(explainer.shap_values(x)).reshape(-1)
                if values.shape[0] == len(self.feature_names):
                    return dict(zip(self.feature_names, values.astype(float)))
            except Exception:
                self._shap_failed = True
        return self._perturbation(x)

    def _perturbation(self, x: np.ndarray) -> Dict[str, float]:
        base = float(self.estimator.predict(x)[0])
        out = {}
        for i, name in enumerate(self.feature_names):
            if x[0, i] == self.medians[i]:
                out[name] = 0.0
                continue
            x_alt = x.copy()
            x_alt[0, i] = self.medians[i]
            out[name] = base - float(self.estimator.predict(x_alt)[0])
        return out

    def expected_value(self) -> Optional[float]:
        explainer = self._tree_explainer()
        if explainer is None:
            return None
        ev = explainer.expected_value
        return float(np.asarray(ev).reshape(-1)[0])
