"""Shared fixtures: a tiny synthetic dataset and models trained on it.

Training is session-scoped and takes a few seconds, so every test that needs
a real ReachModel shares one. Nothing here touches the project's data/ or
models/ directories.
"""

from datetime import datetime, timezone
from pathlib import Path

import pytest

NOW = datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc)


@pytest.fixture(scope="session")
def tiny_data(tmp_path_factory) -> Path:
    from src.data.synthetic import make_synthetic_dataset

    data_dir = tmp_path_factory.mktemp("data")
    make_synthetic_dataset(channels_per_type=3, videos_per_channel=25, seed=1, data_dir=data_dir, now=NOW)
    return data_dir


@pytest.fixture(scope="session")
def tiny_models(tiny_data) -> Path:
    from src.modelling.train import load_table, thumbnail_features, title_features, train_one
    from src.scoring.target import add_target
    from src.thumbnail.features import FEATURE_VERSION as TV
    from src.title.features import FEATURE_VERSION as TTV

    models_dir = tiny_data.parent / "models"
    table = load_table(tiny_data / "processed")
    table, slope = add_target(table, now=NOW)
    train_one("title", table, title_features(table), TTV, slope, "synthetic", folds=3, models_dir=models_dir, log=lambda *a: None)
    feats = thumbnail_features(table, thumbnails_dir=tiny_data / "thumbnails", cache=None, log=lambda *a: None)
    train_one("thumbnail", table, feats, TV, slope, "synthetic", folds=3, models_dir=models_dir, log=lambda *a: None)
    return models_dir


@pytest.fixture(scope="session")
def scorer(tiny_models):
    from src.scoring.registry import get_scorer, reset

    reset()
    s = get_scorer(tiny_models)
    yield s
    reset()
