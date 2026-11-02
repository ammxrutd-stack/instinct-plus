"""fetch: batching, dedupe against earlier runs, verbatim append."""

import json

from src.collection.discover import write_video_ids
from src.collection.fetch import already_fetched, batched, run_fetch

from .conftest import FakeClient, write_jsonl


def requested_batches(client):
    return [params["id"].split(",") for params in client.calls_to("videos")]


def test_batched_splits_137_into_50_50_37():
    ids = [f"id{i:09d}" for i in range(137)]
    sizes = [len(batch) for batch in batched(ids)]
    assert sizes == [50, 50, 37]
    assert sum(batched(ids), []) == ids


def test_run_fetch_makes_three_calls_for_137_ids(raw_dir):
    ids = [f"id{i:09d}" for i in range(137)]
    write_video_ids(raw_dir, ids)
    client = FakeClient()
    summary = run_fetch(client, raw_dir, run_id="RUN1")
    assert [len(b) for b in requested_batches(client)] == [50, 50, 37]
    assert summary["calls"] == 3
    assert summary["units"] == 3
    assert summary["not_returned"] == 137, "the fake has no items for these ids"


def test_skips_ids_already_in_any_raw_file(raw_dir, videos_fixture):
    write_video_ids(raw_dir, ["alpha000001", "alpha000002", "beta0000001"])
    write_jsonl(raw_dir / "videos_20250101T000000Z.jsonl", [videos_fixture["alpha000001"]])
    write_jsonl(raw_dir / "videos_20250102T000000Z.jsonl", [videos_fixture["beta0000001"]])
    assert already_fetched(raw_dir) == {"alpha000001", "beta0000001"}

    client = FakeClient(videos=videos_fixture)
    summary = run_fetch(client, raw_dir, run_id="RUN1")
    assert requested_batches(client) == [["alpha000002"]]
    assert summary["already_fetched"] == 2
    assert summary["written"] == 1


def test_appends_items_verbatim_one_per_line(raw_dir, videos_fixture):
    write_video_ids(raw_dir, ["alpha000001", "shared00001"])
    client = FakeClient(videos=videos_fixture)
    summary = run_fetch(client, raw_dir, run_id="RUN1")

    out = raw_dir / "videos_RUN1.jsonl"
    assert summary["output"] == out
    lines = [json.loads(line) for line in out.read_text().splitlines()]
    assert lines == [videos_fixture["alpha000001"], videos_fixture["shared00001"]]


def test_rerun_never_double_spends(raw_dir, videos_fixture):
    write_video_ids(raw_dir, ["alpha000001"])
    run_fetch(FakeClient(videos=videos_fixture), raw_dir, run_id="RUN1")
    client = FakeClient(videos=videos_fixture)
    summary = run_fetch(client, raw_dir, run_id="RUN2")
    assert client.units == 0
    assert summary["output"] is None


def test_quota_mid_fetch_keeps_what_was_paid_for(raw_dir, videos_fixture):
    ids = ["alpha000001"] + [f"id{i:09d}" for i in range(60)]
    write_video_ids(raw_dir, ids)
    client = FakeClient(videos=videos_fixture, budget=1)
    summary = run_fetch(client, raw_dir, run_id="RUN1")
    assert summary["quota_exceeded"] is True
    assert summary["calls"] == 1
    assert already_fetched(raw_dir) == {"alpha000001"}


def test_explicit_ids_and_limit(raw_dir, videos_fixture):
    write_video_ids(raw_dir, ["beta0000001"])  # ignored when ids are given explicitly
    client = FakeClient(videos=videos_fixture)
    run_fetch(client, raw_dir, run_id="RUN1", ids=["alpha000001", "alpha000002", "shared00001"], limit=2)
    assert requested_batches(client) == [["alpha000001", "alpha000002"]]
