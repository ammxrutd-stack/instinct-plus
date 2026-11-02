import pandas as pd

from config.schema import CHANNEL_ID, COLUMNS, CONTENT_TYPE, CONTENT_TYPES, SUBSCRIBER_COUNT, THUMBNAIL_FILE, VIEW_COUNT
from src.data.synthetic import MARKER, is_synthetic


def test_synthetic_table_matches_schema(tiny_data):
    table = pd.read_parquet(tiny_data / "processed" / "videos.parquet")
    assert list(table.columns) == COLUMNS
    assert set(table[CONTENT_TYPE]) == set(CONTENT_TYPES)
    assert (table[SUBSCRIBER_COUNT] >= 10_000).all()
    assert (table[VIEW_COUNT] > 0).all()
    assert table[CHANNEL_ID].nunique() == 12
    assert all((tiny_data / "thumbnails" / f).exists() for f in table[THUMBNAIL_FILE])
    assert is_synthetic(tiny_data)
    assert (tiny_data / "processed" / MARKER).exists()


def test_synthetic_is_reproducible(tmp_path):
    from src.data.synthetic import make_synthetic_dataset
    from tests.conftest import NOW

    a = make_synthetic_dataset(1, 5, seed=3, data_dir=tmp_path / "a", thumbnails=False, now=NOW)
    b = make_synthetic_dataset(1, 5, seed=3, data_dir=tmp_path / "b", thumbnails=False, now=NOW)
    assert a.equals(b)
