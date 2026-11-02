import json

import numpy as np

from config.schema import CONTENT_TYPES
from src.modelling.calibration import Calibrator
from src.modelling.model import ReachModel


def test_calibrator_maps_predictions_to_percentiles():
    preds = np.linspace(-1, 1, 1001)
    cts = np.array(["gaming"] * 1001)
    c = Calibrator.fit(preds, cts)
    assert abs(c.score(0.0, "gaming") - 50) < 1
    assert c.score(-5, "gaming") == 0.0
    assert c.score(5, "gaming") == 100.0
    assert c.points_per_unit("gaming") > 0
    # unknown content type falls back to the pooled table
    assert abs(c.score(0.0, "podcast") - 50) < 1
    rt = Calibrator.from_dict(json.loads(json.dumps(c.to_dict())))
    assert rt.score(0.3, "gaming") == c.score(0.3, "gaming")


def test_model_card_and_artefact(tiny_models):
    for name in ("title", "thumbnail"):
        assert ReachModel.exists(name, tiny_models)
        card = ReachModel.card(name, tiny_models)
        assert card["data_source"] == "synthetic"
        assert card["rows"] > 0 and card["channels"] >= 6
        assert set(card["per_content_type"]) == set(CONTENT_TYPES)
        assert abs(sum(card["importance"].values()) - 1) < 1e-6
        model = ReachModel.load(name, tiny_models)
        assert model.features == card["features"]


def test_scores_are_deterministic_and_drivers_exclude_context(scorer):
    a = scorer.score_title("The Truth About Index Funds", "finance", 100_000)
    b = scorer.score_title("The Truth About Index Funds", "finance", 100_000)
    assert a.score == b.score and a.drivers == b.drivers
    assert not any(k.startswith("ct_") or k == "log_subscribers" for k in a.drivers)
    assert "log_subscribers" in a.meta["context_points"]


def test_channel_size_is_neutralised(scorer, tiny_models):
    """Removing the channel-size contribution should close most of the gap
    between a small and a huge channel scoring the same title."""
    from src.modelling.model import ReachModel

    model = ReachModel.load("title", tiny_models)
    small = scorer.score_title("The Truth About Index Funds", "finance", 20_000)
    huge = scorer.score_title("The Truth About Index Funds", "finance", 30_000_000)
    raw_gap = abs(model.calibrator.score(small.meta["raw_prediction"], "finance")
                  - model.calibrator.score(huge.meta["raw_prediction"], "finance"))
    neutral_gap = abs(small.score - huge.score)
    assert small.meta["raw_prediction"] != huge.meta["raw_prediction"]
    assert raw_gap > 5  # the context does move the raw prediction
    assert neutral_gap < raw_gap * 0.6


def test_drivers_are_in_points_and_roughly_additive(scorer):
    r = scorer.score_title("7 Mistakes Everyone Makes With Sourdough (and how to fix them)", "cooking")
    assert r.drivers
    assert all(abs(v) >= 0.5 for v in r.drivers.values())
    assert len(r.drivers) <= 8


def test_empty_title_scores_zero(scorer):
    r = scorer.score_title("   ", "gaming")
    assert r.score == 0.0 and r.meta.get("empty")
    assert scorer.combine(r, None).score == 0.0


def test_missing_image_scores_zero(scorer, tmp_path):
    r = scorer.score_thumbnail(str(tmp_path / "missing.jpg"), "gaming")
    assert r.score == 0.0 and r.meta.get("image_missing")
    bad = scorer.score_thumbnail_bytes(b"not an image", "gaming")
    assert bad.meta.get("image_missing")


def test_combine_weights_follow_skill(scorer):
    w = scorer.weights()
    assert abs(sum(w.values()) - 1) < 1e-9
    t = scorer.score_title("Why Sourdough Fails", "cooking")
    only_title = scorer.combine(t, None)
    assert only_title.score == t.score and only_title.meta["parts"] == ["title"]


def test_registry_falls_back_to_placeholder(tmp_path):
    from src.scoring.registry import get_scorer, reset

    reset()
    s = get_scorer(tmp_path / "no_models")
    assert s.status.title_backend == "placeholder" and s.status.thumbnail_backend == "placeholder"
    assert not s.status.all_real and s.status.data_source == "none"
    r = s.score_title("anything", "gaming")
    assert 0 <= r.score <= 100 and "placeholder" in r.notes[0].lower()
    reset()


def test_notes_are_actionable(scorer):
    r = scorer.score_title("RAMEN", "cooking")
    assert any("short" in n.lower() for n in r.notes)
    assert all(not n.endswith("points)") or "(" in n for n in r.notes)
