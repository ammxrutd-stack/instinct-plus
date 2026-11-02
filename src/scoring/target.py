"""The target variable: what "reach" means.

Both models predict the same quantity, defined once here so the title pair and
the thumbnail pair cannot drift apart.

reach = log10(views / subscribers), adjusted for the video's age.

Why views per subscriber: a million views is routine for a 30M-subscriber
channel and extraordinary for a 100K one. Dividing by subscribers asks "how far
did this video travel relative to the audience that already existed", which is
the thing a title and thumbnail actually influence. Log because the ratio spans
four orders of magnitude and is roughly log-normal.

Why age adjusted: the collection window admits videos 30 to 365 days old, and
older videos have had longer to accumulate views. A pooled linear fit of reach
on log10(age) is removed so that age is not the dominant predictor. The slope
is returned so it can be stored with the model.

reach_score is the percentile of reach within its content type, 0 to 100. It
is a reporting convenience (benchmarks, the app), not the training target:
trees train on the continuous reach value and the model's own predictions are
mapped onto a 0-100 score at inference (see src/modelling/calibration).
"""

from datetime import datetime, timezone
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from config.schema import (
    CONTENT_TYPE,
    PUBLISHED_AT,
    REACH,
    REACH_SCORE,
    SUBSCRIBER_COUNT,
    VIEW_COUNT,
)

REFERENCE_AGE_DAYS = 90  # reach is expressed as if every video were this old


def age_in_days(published_at: pd.Series, now: Optional[datetime] = None) -> pd.Series:
    """Days between publish time and `now` (UTC)."""
    now = now or datetime.now(timezone.utc)
    published = pd.to_datetime(published_at, utc=True)
    return (now - published).dt.total_seconds() / 86400.0


def raw_reach(view_count: pd.Series, subscriber_count: pd.Series) -> pd.Series:
    views = pd.to_numeric(view_count, errors="coerce").clip(lower=0)
    subs = pd.to_numeric(subscriber_count, errors="coerce").clip(lower=1)
    return np.log10((views + 1.0) / subs)


def fit_age_slope(reach: pd.Series, age_days: pd.Series) -> float:
    """Slope of reach on log10(age), pooled. Zero if there is nothing to fit."""
    x = np.log10(np.clip(age_days.to_numpy(dtype=float), 1.0, None))
    y = reach.to_numpy(dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 10 or np.std(x[ok]) < 1e-9:
        return 0.0
    slope, _ = np.polyfit(x[ok], y[ok], 1)
    return float(slope)


def adjust_for_age(reach: pd.Series, age_days: pd.Series, slope: float) -> pd.Series:
    offset = np.log10(np.clip(age_days.to_numpy(dtype=float), 1.0, None)) - np.log10(REFERENCE_AGE_DAYS)
    return reach - slope * offset


def add_target(
    table: pd.DataFrame,
    now: Optional[datetime] = None,
    age_slope: Optional[float] = None,
) -> Tuple[pd.DataFrame, float]:
    """Return a copy of the table with REACH and REACH_SCORE columns.

    `age_slope` fixes the adjustment to a previously fitted value (for scoring
    a new table with an existing model); when None it is fitted here.
    """
    out = table.copy()
    reach = raw_reach(out[VIEW_COUNT], out[SUBSCRIBER_COUNT])
    age = age_in_days(out[PUBLISHED_AT], now)
    slope = fit_age_slope(reach, age) if age_slope is None else float(age_slope)
    out[REACH] = adjust_for_age(reach, age, slope)
    out[REACH_SCORE] = (
        out.groupby(CONTENT_TYPE)[REACH].rank(pct=True, method="average") * 100.0
    )
    return out, slope
