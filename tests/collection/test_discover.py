"""discover: resolution, early stopping, dedupe, checkpoint resumption."""

import json
from datetime import timedelta

import pytest

from src.collection.checkpoint import checkpoint_path, load_checkpoint
from src.collection.discover import (
    discover_videos,
    make_run_id,
    read_video_ids,
    run_discover,
    window_start,
    write_video_ids,
)

from .conftest import ALPHA, BETA, FakeClient

ALPHA_IN_WINDOW = ["alpha000001", "alpha000002", "alpha000003", "alpha000004", "shared00001"]
BETA_IN_WINDOW = ["beta0000001", "beta0000002", "shared00001"]
SEEDS = {"gaming": ["@alpha", "@beta", "@ghost"]}


def channels_file(raw_dir, run_id):
    path = raw_dir / f"channels_{run_id}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_run_id_format(now):
    assert make_run_id(now) == "20260916T120000Z"


def test_writes_sorted_deduped_ids_and_verbatim_channel_records(fake_client, raw_dir, now):
    summary = run_discover(fake_client, SEEDS, raw_dir, run_id="RUN1", now=now)

    assert read_video_ids(raw_dir) == sorted(set(ALPHA_IN_WINDOW + BETA_IN_WINDOW))
    assert summary["video_ids"] == read_video_ids(raw_dir)
    assert summary["channels_processed"] == 2
    assert summary["unresolved"] == ["@ghost"]
    assert summary["quota_exceeded"] is False
    assert summary["units"] == fake_client.units

    records = channels_file(raw_dir, "RUN1")
    assert [r["id"] for r in records] == [ALPHA, BETA]
    for record in records:
        assert record["_content_type"] == "gaming"
        assert record["_handle"] in ("@alpha", "@beta")
        # verbatim: the API item's own keys are all still there
        assert {"kind", "etag", "id", "snippet", "contentDetails", "statistics"} <= set(record)
    assert records[0]["statistics"]["subscriberCount"] == "2500000"


def test_shared_id_is_written_once(fake_client, raw_dir, now):
    run_discover(fake_client, SEEDS, raw_dir, now=now)
    ids = read_video_ids(raw_dir)
    assert ids.count("shared00001") == 1


def test_early_stopping_does_not_page_past_the_window(fake_client, raw_dir, now):
    run_discover(fake_client, {"gaming": ["@alpha"]}, raw_dir, now=now)
    tokens = [p.get("pageToken") for p in fake_client.calls_to("playlistItems")]
    assert tokens == [None, "ALPHA_PAGE2"], "page 3 is entirely before the window"
    assert "alpha000005" not in read_video_ids(raw_dir)
    assert "alpha000004" in read_video_ids(raw_dir), "exactly the window start is kept"


def test_private_item_without_publish_time_is_skipped(fake_client, raw_dir, now):
    run_discover(fake_client, {"gaming": ["@beta"]}, raw_dir, now=now)
    assert "beta0000003" not in read_video_ids(raw_dir)


def test_cap_limits_ids_per_channel(fake_client, now):
    ids = discover_videos(fake_client, "UU" + ALPHA[2:], window_start(now), cap=2)
    assert ids == ["alpha000001", "alpha000002"]
    assert len(fake_client.calls_to("playlistItems")) == 1


def test_unknown_content_type_warns_but_runs(fake_client, raw_dir, now):
    with pytest.warns(UserWarning, match="vlogs"):
        summary = run_discover(fake_client, {"vlogs": ["@alpha"]}, raw_dir, now=now)
    assert summary["channels_processed"] == 1
    assert channels_file(raw_dir, summary["run_id"])[0]["_content_type"] == "vlogs"


def test_merges_with_existing_video_ids(fake_client, raw_dir, now):
    write_video_ids(raw_dir, ["zzzzzzzzzzz"])
    run_discover(fake_client, {"gaming": ["@beta"]}, raw_dir, now=now)
    assert "zzzzzzzzzzz" in read_video_ids(raw_dir)
    assert read_video_ids(raw_dir)[-1] == "zzzzzzzzzzz"


def test_quota_exhaustion_writes_checkpoint_and_keeps_partial_ids(
        channels_fixture, playlists_fixture, raw_dir, now):
    # @alpha costs 1 (resolve) + 2 (pages) = 3 units; resolving @beta would be the 4th
    client = FakeClient(channels_fixture, playlists_fixture, budget=3)
    summary = run_discover(client, SEEDS, raw_dir, run_id="RUN1", now=now)

    assert summary["quota_exceeded"] is True
    assert summary["channels_processed"] == 1
    assert read_video_ids(raw_dir) == sorted(ALPHA_IN_WINDOW)
    assert load_checkpoint(raw_dir) == {
        "completed_channel_ids": [ALPHA],
        "completed_handles": ["@alpha"],
    }


def test_resumption_skips_checkpointed_channels_and_clears_checkpoint(
        channels_fixture, playlists_fixture, raw_dir, now):
    first = FakeClient(channels_fixture, playlists_fixture, budget=3)
    run_discover(first, SEEDS, raw_dir, run_id="RUN1", now=now)
    assert checkpoint_path(raw_dir).exists()

    second = FakeClient(channels_fixture, playlists_fixture)
    summary = run_discover(second, SEEDS, raw_dir, run_id="RUN2", now=now)

    handles = [p["forHandle"] for p in second.calls_to("channels")]
    assert "@alpha" not in handles, "a completed channel costs nothing on the resumed run"
    assert handles == ["@beta", "@ghost"]
    assert summary["channels_skipped"] == 1
    assert summary["channels_processed"] == 1
    assert not checkpoint_path(raw_dir).exists(), "a full run clears the checkpoint"
    assert read_video_ids(raw_dir) == sorted(set(ALPHA_IN_WINDOW + BETA_IN_WINDOW))
    # the resumed run's channels file holds only the channels it processed
    assert [r["id"] for r in channels_file(raw_dir, "RUN2")] == [BETA]


def test_window_start_is_max_days_before_now(now):
    assert now - window_start(now) == timedelta(days=365)
