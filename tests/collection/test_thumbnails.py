"""thumbnails: filename pattern, skip-existing, failure keeps the row."""

import pandas as pd
import pytest
import requests

from config.schema import THUMBNAIL_FILE, THUMBNAIL_PATTERN, VIDEO_ID
from src.collection.build import run_build, to_frame, write_table
from src.collection.client import CollectionError
from src.collection.thumbnails import THUMB_URL, download_thumbnails, run_thumbnails, thumbnail_path

JPEG = b"\xff\xd8\xff\xe0fake-jpeg-bytes"


class FakeResponse:
    def __init__(self, status_code, content=b""):
        self.status_code = status_code
        self.content = content


class FakeImageSession:
    """200 with bytes by default; per-id overrides to simulate 404s and dropped connections."""

    def __init__(self, failures=None):
        self.failures = failures or {}
        self.urls = []

    def get(self, url, timeout=None):
        self.urls.append(url)
        video_id = url.rsplit("/", 2)[-2]
        outcome = self.failures.get(video_id)
        if isinstance(outcome, Exception):
            raise outcome
        if outcome is not None:
            return FakeResponse(outcome)
        return FakeResponse(200, JPEG)


def frame_for(*video_ids):
    rows = [{
        VIDEO_ID: v, "title": "t", "channel_id": "UCx", "channel_title": "X",
        "subscriber_count": 20000, "view_count": 5, "published_at": "2026-01-01T00:00:00Z",
        "duration_seconds": 120, "content_type": "gaming",
        THUMBNAIL_FILE: THUMBNAIL_PATTERN.format(video_id=v),
    } for v in video_ids]
    return to_frame(rows)


def test_filenames_follow_the_schema_pattern(thumbnails_dir):
    assert thumbnail_path(thumbnails_dir, "abcDEF12345") == thumbnails_dir / "abcDEF12345.jpg"
    assert thumbnail_path(thumbnails_dir, "abcDEF12345").name == THUMBNAIL_PATTERN.format(video_id="abcDEF12345")


def test_downloads_hqdefault_for_every_row(thumbnails_dir):
    session = FakeImageSession()
    frame, counts = download_thumbnails(frame_for("aaaaaaaaaaa", "bbbbbbbbbbb"), thumbnails_dir, session)
    assert counts == {"downloaded": 2, "already_present": 0, "failed": 0}
    assert session.urls == [THUMB_URL.format(video_id=v) for v in ("aaaaaaaaaaa", "bbbbbbbbbbb")]
    assert (thumbnails_dir / "aaaaaaaaaaa.jpg").read_bytes() == JPEG
    assert list(frame[THUMBNAIL_FILE]) == ["aaaaaaaaaaa.jpg", "bbbbbbbbbbb.jpg"]


def test_existing_files_are_skipped_without_a_request(thumbnails_dir):
    thumbnails_dir.mkdir()
    (thumbnails_dir / "aaaaaaaaaaa.jpg").write_bytes(JPEG)
    session = FakeImageSession()
    _, counts = download_thumbnails(frame_for("aaaaaaaaaaa", "bbbbbbbbbbb"), thumbnails_dir, session)
    assert counts == {"downloaded": 1, "already_present": 1, "failed": 0}
    assert len(session.urls) == 1


def test_failures_keep_the_row_with_empty_thumbnail_file(thumbnails_dir):
    session = FakeImageSession({
        "notfound001": 404,
        "dropped0001": requests.ConnectionError("reset"),
    })
    frame, counts = download_thumbnails(
        frame_for("aaaaaaaaaaa", "notfound001", "dropped0001"), thumbnails_dir, session)
    assert counts == {"downloaded": 1, "already_present": 0, "failed": 2}
    assert len(frame) == 3
    assert list(frame[THUMBNAIL_FILE]) == ["aaaaaaaaaaa.jpg", "", ""]
    assert not (thumbnails_dir / "notfound001.jpg").exists()


def test_previously_failed_row_is_retried_and_restored(thumbnails_dir):
    frame = frame_for("aaaaaaaaaaa")
    frame[THUMBNAIL_FILE] = ""
    frame, counts = download_thumbnails(frame, thumbnails_dir, FakeImageSession())
    assert counts["downloaded"] == 1
    assert list(frame[THUMBNAIL_FILE]) == ["aaaaaaaaaaa.jpg"]


def test_run_thumbnails_rewrites_the_table(populated_raw, processed_dir, thumbnails_dir, now):
    run_build(populated_raw, processed_dir, csv=True, now=now)
    session = FakeImageSession({"alpha000004": 404})
    summary = run_thumbnails(processed_dir, thumbnails_dir, session)

    assert summary["rows"] == 5
    assert summary["downloaded"] == 4
    assert summary["failed"] == 1
    table = pd.read_parquet(processed_dir / "videos.parquet").set_index(VIDEO_ID)
    assert table.loc["alpha000004", THUMBNAIL_FILE] == ""
    assert table.loc["alpha000001", THUMBNAIL_FILE] == "alpha000001.jpg"
    csv_table = pd.read_csv(processed_dir / "videos.csv", keep_default_na=False).set_index(VIDEO_ID)
    assert csv_table.loc["alpha000004", THUMBNAIL_FILE] == "", "the csv copy is kept in step"
    assert {p.name for p in thumbnails_dir.iterdir()} == {
        "alpha000001.jpg", "alpha000002.jpg", "beta0000001.jpg", "shared00001.jpg",
    }


def test_run_thumbnails_without_a_table_is_a_collection_error(processed_dir, thumbnails_dir):
    with pytest.raises(CollectionError, match="run build first"):
        run_thumbnails(processed_dir, thumbnails_dir, FakeImageSession())


def test_write_table_round_trips_thumbnail_file(processed_dir):
    write_table(frame_for("aaaaaaaaaaa"), processed_dir)
    assert pd.read_parquet(processed_dir / "videos.parquet")[THUMBNAIL_FILE][0] == "aaaaaaaaaaa.jpg"
