"""Both models must satisfy the scoring contract.

These run against the placeholder (always available) and against real models
trained on a tiny synthetic dataset (tests/conftest.py). The same assertions
hold for both: that is the point of the contract.
"""

import pytest

from src.scoring import placeholder
from src.scoring.interface import ScoreResult

CONTENT_TYPE = "gaming"


def _check(result: ScoreResult):
    assert isinstance(result, ScoreResult)
    assert isinstance(result.score, float)
    assert 0 <= result.score <= 100
    assert isinstance(result.drivers, dict)
    assert all(isinstance(v, float) for v in result.drivers.values())
    assert isinstance(result.notes, list)
    assert all(isinstance(n, str) for n in result.notes)


def test_placeholder_satisfies_contract():
    _check(placeholder.score_title("How I Fixed My Sleep In 30 Days", CONTENT_TYPE))
    _check(placeholder.score_thumbnail("data/thumbnails/example.jpg", CONTENT_TYPE))


def test_real_models_satisfy_contract(scorer, tiny_data):
    thumb = next((tiny_data / "thumbnails").glob("*.jpg"))
    _check(scorer.score_title("How I Fixed My Sleep In 30 Days", CONTENT_TYPE))
    _check(scorer.score_thumbnail(str(thumb), CONTENT_TYPE))
    _check(scorer.score_thumbnail_bytes(thumb.read_bytes(), CONTENT_TYPE))
    assert scorer.status.all_real
    assert scorer.status.data_source == "synthetic"


def test_combined_satisfies_contract(scorer, tiny_data):
    thumb = next((tiny_data / "thumbnails").glob("*.jpg"))
    out = scorer.score_video("How I Fixed My Sleep In 30 Days", str(thumb), CONTENT_TYPE, 50_000)
    for key in ("title", "thumbnail", "overall"):
        _check(out[key])
    assert set(out["overall"].meta["parts"]) == {"title", "thumbnail"}
    assert abs(sum(out["overall"].meta["weights"].values()) - 1) < 1e-6


def test_interface_is_not_implemented():
    from src.scoring import interface

    with pytest.raises(NotImplementedError):
        interface.score_title("anything", CONTENT_TYPE)
    with pytest.raises(NotImplementedError):
        interface.score_thumbnail("anything.jpg", CONTENT_TYPE)
