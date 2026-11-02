"""Run the validation protocol (docs/validation_protocol.md) and write a report.

    python -m src.modelling.evaluate                 # both models
    python -m src.modelling.evaluate --out docs/model_report.md

For each model: re-run channel-grouped cross-validation on the current table,
compare against two baselines (predict the mean; predict from content type
alone), report Spearman, R2 and MAE overall and per content type, check that
the 0-100 calibration is flat (a score of 70 should beat about 70% of the
category), and list the top features. The point is a number the team can
compare across weeks and across the two models, computed the same way every
time.
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

from config.schema import CHANNEL_ID, CONTENT_TYPE, PROCESSED_DIR, REACH, THUMBNAIL_FILE
from src.modelling.backend import make_regressor
from src.modelling.calibration import Calibrator
from src.modelling.train import data_source_label, load_table, thumbnail_features, title_features
from src.scoring.target import add_target


def _spear(a, b):
    return float(spearmanr(a, b).correlation)


def cross_validate(features: pd.DataFrame, table: pd.DataFrame, folds: int = 5) -> np.ndarray:
    X = features.to_numpy(dtype=float)
    X = np.where(np.isnan(X), np.nanmedian(X, axis=0), X)
    y = table[REACH].to_numpy(dtype=float)
    groups = table[CHANNEL_ID].to_numpy()
    oof = np.full(len(y), np.nan)
    for tr, va in GroupKFold(n_splits=min(folds, len(np.unique(groups)))).split(X, y, groups):
        est, _ = make_regressor()
        est.fit(X[tr], y[tr])
        oof[va] = est.predict(X[va])
    return oof


def content_type_baseline(table: pd.DataFrame, folds: int = 5) -> np.ndarray:
    """Predict each video's content-type mean, learned on the training folds."""
    y = table[REACH].to_numpy(dtype=float)
    cts = table[CONTENT_TYPE].to_numpy()
    groups = table[CHANNEL_ID].to_numpy()
    oof = np.full(len(y), np.nan)
    for tr, va in GroupKFold(n_splits=min(folds, len(np.unique(groups)))).split(y, y, groups):
        means = pd.Series(y[tr]).groupby(cts[tr]).mean()
        oof[va] = [means.get(c, y[tr].mean()) for c in cts[va]]
    return oof


def calibration_check(oof: np.ndarray, y: np.ndarray, cts: np.ndarray) -> List[Dict[str, float]]:
    """Bucket scores into deciles; within each, what share of the category's
    true reach does the median video beat? Flat means honest."""
    cal = Calibrator.fit(oof, cts)
    scores = np.array([cal.score(p, c) for p, c in zip(oof, cts)])
    true_pct = pd.Series(y).groupby(cts).rank(pct=True).to_numpy() * 100
    rows = []
    for lo in range(0, 100, 20):
        m = (scores >= lo) & (scores < lo + 20 if lo < 80 else scores <= 100)
        if m.sum():
            rows.append({"score_bucket": f"{lo}-{lo + 20}", "videos": int(m.sum()),
                         "mean_true_percentile": float(true_pct[m].mean())})
    return rows


def evaluate_model(name: str, features: pd.DataFrame, table: pd.DataFrame, folds: int) -> Dict:
    y = table[REACH].to_numpy(dtype=float)
    cts = table[CONTENT_TYPE].astype(str).to_numpy()
    oof = cross_validate(features, table, folds)
    base_mean = np.full_like(y, y.mean())
    base_ct = content_type_baseline(table, folds)
    out = {
        "name": name,
        "rows": int(len(y)),
        "channels": int(table[CHANNEL_ID].nunique()),
        "model": {"spearman": _spear(y, oof), "r2": float(r2_score(y, oof)), "mae": float(mean_absolute_error(y, oof))},
        "baseline_mean": {"spearman": float("nan"), "r2": float(r2_score(y, base_mean)), "mae": float(mean_absolute_error(y, base_mean))},
        "baseline_content_type": {"spearman": _spear(y, base_ct), "r2": float(r2_score(y, base_ct)), "mae": float(mean_absolute_error(y, base_ct))},
        "per_content_type": {},
        "calibration": calibration_check(oof, y, cts),
    }
    for ct in np.unique(cts):
        m = cts == ct
        out["per_content_type"][str(ct)] = {"rows": int(m.sum()), "spearman": _spear(y[m], oof[m]), "mae": float(mean_absolute_error(y[m], oof[m]))}
    return out


def render(results: List[Dict], source: str, slope: float) -> str:
    lines = [
        "# Model evaluation report",
        "",
        f"Generated {datetime.now(timezone.utc).isoformat(timespec='minutes')} by `python -m src.modelling.evaluate`.",
        f"Data source: **{source}**. Age slope removed from the target: {slope:.3f}.",
        "",
        "Metrics are out-of-fold with channels grouped, so no channel appears in both the training and the validation fold.",
        "",
    ]
    for r in results:
        lines += [f"## {r['name']} model", "", f"{r['rows']:,} videos from {r['channels']} channels.", "",
                  "| | Spearman ρ | R² | MAE |", "|---|---|---|---|"]
        for label, key in (("Model", "model"), ("Baseline: content-type mean", "baseline_content_type"), ("Baseline: global mean", "baseline_mean")):
            m = r[key]
            sp = "–" if m["spearman"] != m["spearman"] else f"{m['spearman']:.3f}"
            lines.append(f"| {label} | {sp} | {m['r2']:.3f} | {m['mae']:.3f} |")
        lines += ["", "| Content type | videos | Spearman ρ | MAE |", "|---|---|---|---|"]
        for ct, m in r["per_content_type"].items():
            lines.append(f"| {ct} | {m['rows']} | {m['spearman']:.3f} | {m['mae']:.3f} |")
        lines += ["", "Calibration (a flat, rising line means the 0-100 score is honest):", "",
                  "| Score bucket | videos | mean true percentile |", "|---|---|---|"]
        for c in r["calibration"]:
            lines.append(f"| {c['score_bucket']} | {c['videos']} | {c['mean_true_percentile']:.0f} |")
        lines.append("")
    if source == "synthetic":
        lines += ["> These numbers come from synthetic data and describe how well the pipeline recovers the generator's rules, not how well it predicts YouTube.", ""]
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("which", nargs="?", choices=["title", "thumbnail", "both"], default="both")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--out", default=None, help="write a markdown report here")
    args = p.parse_args(argv)

    table = load_table()
    source = data_source_label()
    table, slope = add_target(table)
    results = []
    if args.which in ("title", "both"):
        results.append(evaluate_model("title", title_features(table), table, args.folds))
    if args.which in ("thumbnail", "both"):
        sub = table[table[THUMBNAIL_FILE].astype(str).str.len() > 0].reset_index(drop=True)
        feats = thumbnail_features(sub, cache=PROCESSED_DIR / "features_thumbnail.parquet", log=lambda *a: None)
        ok = feats["image_missing"] == 0
        results.append(evaluate_model("thumbnail", feats[ok].reset_index(drop=True), sub[ok].reset_index(drop=True), args.folds))
    report = render(results, source, slope)
    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")
        print(f"wrote {args.out}")
    print(report)


if __name__ == "__main__":
    main()
