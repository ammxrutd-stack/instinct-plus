import numpy as np
import pandas as pd

from config.schema import CONTENT_TYPE, PUBLISHED_AT, REACH, REACH_SCORE, SUBSCRIBER_COUNT, VIEW_COUNT
from src.scoring.target import REFERENCE_AGE_DAYS, add_target, fit_age_slope, raw_reach
from tests.conftest import NOW


def test_raw_reach_is_log_ratio():
    r = raw_reach(pd.Series([1000, 10_000]), pd.Series([1000, 1000]))
    assert abs(r.iloc[0] - 0.0) < 0.01
    assert abs(r.iloc[1] - 1.0) < 0.01


def test_age_slope_is_recovered_and_removed():
    rng = np.random.default_rng(0)
    n = 400
    age = rng.uniform(30, 365, n)
    reach = 0.3 * np.log10(age) + rng.normal(0, 0.01, n)
    slope = fit_age_slope(pd.Series(reach), pd.Series(age))
    assert abs(slope - 0.3) < 0.02
    table = pd.DataFrame({
        VIEW_COUNT: (1000 * 10 ** reach).astype(int),
        SUBSCRIBER_COUNT: 1000,
        PUBLISHED_AT: [(NOW - pd.Timedelta(days=a)).strftime("%Y-%m-%dT%H:%M:%SZ") for a in age],
        CONTENT_TYPE: "gaming",
    })
    out, s = add_target(table, now=NOW)
    # after adjustment, reach no longer tracks age
    corr = np.corrcoef(np.log10(age), out[REACH])[0, 1]
    assert abs(corr) < 0.15
    assert out[REACH_SCORE].between(0, 100).all()


def test_fixed_slope_is_respected():
    table = pd.DataFrame({VIEW_COUNT: [100, 200], SUBSCRIBER_COUNT: [1000, 1000], PUBLISHED_AT: ["2026-06-01T00:00:00Z"] * 2, CONTENT_TYPE: ["a", "a"]})
    _, s = add_target(table, now=NOW, age_slope=0.5)
    assert s == 0.5
