"""Gradient boosted trees with a fallback.

XGBoost is the preferred backend. On macOS it needs the OpenMP runtime
(`brew install libomp`), which not every teammate has, so when the import
fails we fall back to scikit-learn's HistGradientBoostingRegressor: also
gradient boosted trees, also supported by SHAP's TreeExplainer. The saved
artefact records which backend produced it.
"""

from typing import Any, Dict

import numpy as np

DEFAULT_PARAMS: Dict[str, Any] = {
    "n_estimators": 500,
    "learning_rate": 0.03,
    "max_depth": 4,
    "min_child_weight": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 2.0,
}


def xgboost_available() -> bool:
    try:
        import xgboost  # noqa: F401
        return True
    except Exception:  # ImportError, or XGBoostError when libomp is missing
        return False


def make_regressor(params: Dict[str, Any] = None, random_state: int = 0):
    """Return (estimator, backend_name)."""
    params = {**DEFAULT_PARAMS, **(params or {})}
    if xgboost_available():
        from xgboost import XGBRegressor

        return (
            XGBRegressor(
                n_estimators=params["n_estimators"],
                learning_rate=params["learning_rate"],
                max_depth=params["max_depth"],
                min_child_weight=params["min_child_weight"],
                subsample=params["subsample"],
                colsample_bytree=params["colsample_bytree"],
                reg_lambda=params["reg_lambda"],
                objective="reg:squarederror",
                random_state=random_state,
                n_jobs=4,
                verbosity=0,
            ),
            "xgboost",
        )

    from sklearn.ensemble import HistGradientBoostingRegressor

    return (
        HistGradientBoostingRegressor(
            max_iter=params["n_estimators"],
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            min_samples_leaf=max(5, int(params["min_child_weight"])),
            l2_regularization=params["reg_lambda"],
            random_state=random_state,
        ),
        "sklearn",
    )


def feature_importance(estimator, feature_names) -> Dict[str, float]:
    """Normalised global importance, whichever backend. Used for the model card."""
    if hasattr(estimator, "feature_importances_"):
        raw = np.asarray(estimator.feature_importances_, dtype=float)
    else:
        # HistGradientBoosting has no native importances; permutation is done
        # by the trainer, which passes them in. Fall back to uniform here.
        raw = np.ones(len(feature_names))
    total = raw.sum() or 1.0
    return {name: float(v / total) for name, v in zip(feature_names, raw)}
