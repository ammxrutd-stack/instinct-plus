"""ReachModel: a trained reach predictor with everything scoring needs.

One object holds the estimator, the ordered feature list, the calibrator that
turns predictions into 0-100 scores, the explainer that turns them into
drivers, per-content-type feature benchmarks for suggestions, and metadata
about how it was trained. It is pickled with joblib under models/<name>/ with
a JSON sidecar that humans and the app can read without unpickling.
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

from config.schema import CHANNEL_ID, CONTENT_TYPE, CONTENT_TYPES, MODELS_DIR, REACH
from src.modelling.backend import feature_importance, make_regressor
from src.modelling.calibration import Calibrator
from src.modelling.explain import Explainer

CONTEXT_FEATURES = ("log_subscribers",)  # given to the model, kept out of the score

ARTEFACT = "model.joblib"
SIDECAR = "model_card.json"


@dataclass
class TrainingReport:
    rows: int
    channels: int
    folds: int
    backend: str
    spearman: float
    r2: float
    mae: float
    baseline_mae: float
    per_content_type: Dict[str, Dict[str, float]]
    importance: Dict[str, float]
    data_source: str
    data_hash: str
    trained_at: str
    age_slope: float
    feature_version: str
    notes: List[str] = field(default_factory=list)


def content_type_onehot(content_type: str) -> Dict[str, float]:
    return {"ct_" + ct: float(content_type == ct) for ct in CONTENT_TYPES}


def frame_hash(table: pd.DataFrame) -> str:
    h = hashlib.sha256(pd.util.hash_pandas_object(table, index=False).values.tobytes())
    return h.hexdigest()[:12]


class ReachModel:
    """Fit on a feature frame + reach target; score single items."""

    def __init__(self, name: str):
        self.name = name
        self.estimator = None
        self.features: List[str] = []
        self.calibrator: Optional[Calibrator] = None
        self.medians: Optional[np.ndarray] = None
        self.benchmarks: Dict[str, Dict[str, Dict[str, float]]] = {}
        self.report: Optional[TrainingReport] = None
        self._explainer: Optional[Explainer] = None

    # ----------------------------------------------------------------- fit --

    def fit(
        self,
        features: pd.DataFrame,
        table: pd.DataFrame,
        data_source: str,
        feature_version: str,
        age_slope: float,
        folds: int = 5,
        params: Optional[Dict[str, Any]] = None,
        random_state: int = 0,
    ) -> TrainingReport:
        """`features` is aligned row-for-row with `table` (which carries REACH,
        CHANNEL_ID and CONTENT_TYPE). Cross-validation groups by channel so a
        channel's house style cannot leak between train and validation folds.
        """
        if len(features) != len(table):
            raise ValueError("features and table must align")
        self.features = list(features.columns)
        X = features.to_numpy(dtype=float)
        col_medians = np.nanmedian(X, axis=0)
        X = np.where(np.isnan(X), col_medians, X)
        y = table[REACH].to_numpy(dtype=float)
        groups = table[CHANNEL_ID].to_numpy()
        cts = table[CONTENT_TYPE].astype(str).to_numpy()

        n_groups = len(np.unique(groups))
        folds = max(2, min(folds, n_groups))
        oof = np.full(len(y), np.nan)
        backend = None
        for train_idx, val_idx in GroupKFold(n_splits=folds).split(X, y, groups):
            est, backend = make_regressor(params, random_state)
            est.fit(X[train_idx], y[train_idx])
            oof[val_idx] = est.predict(X[val_idx])

        self.estimator, backend = make_regressor(params, random_state)
        self.estimator.fit(X, y)
        self.medians = np.nanmedian(X, axis=0)
        self.calibrator = Calibrator.fit(oof, cts)
        self.benchmarks = self._benchmarks(features, cts)
        self._explainer = None

        per_ct = {}
        for ct in np.unique(cts):
            m = cts == ct
            if m.sum() >= 10:
                per_ct[str(ct)] = {
                    "rows": int(m.sum()),
                    "spearman": _spearman(y[m], oof[m]),
                    "mae": float(mean_absolute_error(y[m], oof[m])),
                }

        importance = feature_importance(self.estimator, self.features)
        self.report = TrainingReport(
            rows=int(len(y)),
            channels=int(n_groups),
            folds=folds,
            backend=backend,
            spearman=_spearman(y, oof),
            r2=float(r2_score(y, oof)),
            mae=float(mean_absolute_error(y, oof)),
            baseline_mae=float(mean_absolute_error(y, np.full_like(y, y.mean()))),
            per_content_type=per_ct,
            importance=dict(sorted(importance.items(), key=lambda kv: -kv[1])),
            data_source=data_source,
            data_hash=frame_hash(table[[c for c in table.columns if c != REACH]]),
            trained_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            age_slope=float(age_slope),
            feature_version=feature_version,
        )
        return self.report

    @staticmethod
    def _benchmarks(features: pd.DataFrame, cts: np.ndarray) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Per content type, the 10/25/50/75/90th percentile of every feature.
        Suggestions use these to say "longer than 80% of cooking titles"."""
        out: Dict[str, Dict[str, Dict[str, float]]] = {}
        for ct in np.unique(cts):
            sub = features[cts == ct]
            out[str(ct)] = {
                col: {
                    "p10": float(sub[col].quantile(0.10)),
                    "p25": float(sub[col].quantile(0.25)),
                    "p50": float(sub[col].quantile(0.50)),
                    "p75": float(sub[col].quantile(0.75)),
                    "p90": float(sub[col].quantile(0.90)),
                    "mean": float(sub[col].mean()),
                }
                for col in features.columns
            }
        return out

    # --------------------------------------------------------------- score --

    def vector(self, feats: Dict[str, float]) -> np.ndarray:
        """Order features; a NaN (optional context not supplied) becomes the
        training median so the tree sees a typical value, not a zero."""
        v = np.array([float(feats.get(name, np.nan)) for name in self.features], dtype=float)
        missing = np.isnan(v)
        if missing.any():
            v[missing] = self.medians[missing] if self.medians is not None else 0.0
        return v

    def predict_raw(self, feats: Dict[str, float]) -> float:
        return float(self.estimator.predict(self.vector(feats).reshape(1, -1))[0])

    def contributions(self, feats: Dict[str, float]) -> Dict[str, float]:
        """Per-feature contributions in prediction units (SHAP or fallback)."""
        if self._explainer is None:
            self._explainer = Explainer(self.estimator, self.features, self.medians)
        return self._explainer.contributions(self.vector(feats))

    def explain(self, feats: Dict[str, float], content_type: str) -> Dict[str, float]:
        """Drivers in score points, every feature (caller filters)."""
        scale = self.calibrator.points_per_unit(content_type)
        return {k: float(v * scale) for k, v in self.contributions(feats).items()}

    def score(self, feats: Dict[str, float], content_type: str, neutralise: Sequence[str] = CONTEXT_FEATURES) -> Dict[str, Any]:
        """Score with context features neutralised.

        Channel size is given to the model because views-per-subscriber
        depends on it, but the score should answer "how good is this title",
        not "how small is this channel". The contribution of each context
        feature is removed from the raw prediction before calibration and
        reported separately, so a 100K channel and a 10M channel can compare
        the same title on equal terms.
        """
        raw = self.predict_raw(feats)
        contributions = self.contributions(feats)
        context = {k: float(contributions.get(k, 0.0)) for k in neutralise if k in contributions}
        adjusted = raw - sum(context.values())
        scale = self.calibrator.points_per_unit(content_type)
        drivers = {k: float(v * scale) for k, v in contributions.items() if k not in context}
        return {
            "score": self.calibrator.score(adjusted, content_type),
            "raw": raw,
            "adjusted": adjusted,
            "drivers": drivers,
            "context_points": {k: float(v * scale) for k, v in context.items()},
        }

    def benchmark(self, content_type: str, feature: str) -> Optional[Dict[str, float]]:
        return self.benchmarks.get(content_type, {}).get(feature)

    # ------------------------------------------------------------- persist --

    def directory(self, root: Path = MODELS_DIR) -> Path:
        return Path(root) / self.name

    def save(self, root: Path = MODELS_DIR) -> Path:
        d = self.directory(root)
        d.mkdir(parents=True, exist_ok=True)
        explainer, self._explainer = self._explainer, None  # not picklable, rebuilt lazily
        joblib.dump(self, d / ARTEFACT)
        self._explainer = explainer
        card = {"name": self.name, "features": self.features, **asdict(self.report)}
        (d / SIDECAR).write_text(json.dumps(card, indent=2), encoding="utf-8")
        return d

    @classmethod
    def load(cls, name: str, root: Path = MODELS_DIR) -> "ReachModel":
        path = Path(root) / name / ARTEFACT
        if not path.exists():
            raise FileNotFoundError(f"no trained {name} model at {path}")
        model = joblib.load(path)
        model._explainer = None
        return model

    @classmethod
    def exists(cls, name: str, root: Path = MODELS_DIR) -> bool:
        return (Path(root) / name / ARTEFACT).exists()

    @classmethod
    def card(cls, name: str, root: Path = MODELS_DIR) -> Optional[Dict[str, Any]]:
        path = Path(root) / name / SIDECAR
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))


def _spearman(a, b) -> float:
    if len(a) < 3:
        return float("nan")
    r = spearmanr(a, b).correlation
    return float(r) if r == r else float("nan")


def build_feature_frame(
    table: pd.DataFrame,
    extract: Callable[[pd.Series], Dict[str, float]],
    feature_names: Sequence[str],
) -> pd.DataFrame:
    """Apply a per-row extractor and return a frame with exactly feature_names."""
    rows = [extract(row) for _, row in table.iterrows()]
    frame = pd.DataFrame(rows)
    for name in feature_names:
        if name not in frame.columns:
            frame[name] = 0.0
    frame = frame[list(feature_names)].astype(float)
    non_context = [c for c in feature_names if c != "log_subscribers"]
    frame[non_context] = frame[non_context].fillna(0.0)
    return frame
