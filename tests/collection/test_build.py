"""build: filters at their boundaries, exclusion counts, exact column projection."""

import copy
import csv
from datetime import timedelta

import pandas as pd
import pytest

from config.schema import (
    CHANNEL_TITLE,
    COLUMNS,
    CONTENT_TYPE,
    DURATION_SECONDS,
    MAX_DAYS_SINCE_PUBLISH,
    MIN_DAYS_SINCE_PUBLISH,
    MIN_DURATION_SECONDS,
    MIN_SUBSCRIBERS,
    SUBSCRIBER_COUNT,
    THUMBNAIL_FILE,
    THUMBNAIL_PATTERN,
    VIDEO_ID,
    VIEW_COUNT,
)
from src.collection.build import (
    EXCLUSIONS,
    build_rows,
    load_raw_channels,
    load_raw_videos,
    parse_duration,
    run_build,
)

from .conftest import ALPHA, BETA, TINY, channel_record, write_jsonl


@pytest.mark.parametrize("value,expected", [
    ("PT1H2M3S", 3723),
    ("PT60S", 60),
    ("PT1M", 60),
    ("P1DT2H", 93600),
    ("PT45S", 45),
    ("PT12M34S", 754),
    ("P0D", 0),
    ("PT0S", 0),
    (None, None),
    ("", None),
    ("P", None),
    ("forty seconds", None),
    ("PT1H2M3S extra", None),
])
def test_parse_duration(value, expected):
    assert parse_duration(value) == expected


def stamp(when):
    return when.strftime("%Y-%m-%dT%H:%M:%SZ")


@pytest.fixture
def alpha_channels(channels_fixture):
    return {ALPHA: channel_record(channels_fixture, "@alpha")}


@pytest.fixture
def make_video(videos_fixture, now):
    """A clean in-window video on @alpha, with any field overridden for the case."""
    def _make(video_id="vid00000001", days_old=100, duration="PT10M", views="500",
              channel_id=ALPHA, live="none"):
        item = copy.deepcopy(videos_fixture["alpha000001"])
        item["id"] = video_id
        item["snippet"]["channelId"] = channel_id
        item["snippet"]["publishedAt"] = stamp(now - timedelta(days=days_old))
        item["snippet"]["liveBroadcastContent"] = live
        item["contentDetails"]["duration"] = duration
        if views is None:
            item["statistics"].pop("viewCount", None)
        else:
            item["statistics"]["viewCount"] = views
        return item
    return _make


def kept(items, channels, now):
    rows, dropped = build_rows(items, channels, now)
    return [row[VIDEO_ID] for row in rows], {k: v for k, v in dropped.items() if v}


def test_subscriber_boundary(channels_fixture, make_video, now):
    at_min = {BETA: channel_record(channels_fixture, "@beta")}      # exactly 10,000
    below = {TINY: channel_record(channels_fixture, "@tiny")}       # 9,999
    assert at_min[BETA]["statistics"]["subscriberCount"] == str(MIN_SUBSCRIBERS)

    assert kept([make_video(channel_id=BETA)], at_min, now) == (["vid00000001"], {})
    assert kept([make_video(channel_id=TINY)], below, now) == ([], {"low_subs": 1})


def test_age_boundaries(alpha_channels, make_video, now):
    cases = {
        "exact_min": make_video("exactmin001", days_old=MIN_DAYS_SINCE_PUBLISH),
        "too_new": make_video("toonew00001", days_old=MIN_DAYS_SINCE_PUBLISH - 1),
        "exact_max": make_video("exactmax001", days_old=MAX_DAYS_SINCE_PUBLISH),
        "too_old": make_video("tooold00001", days_old=MAX_DAYS_SINCE_PUBLISH + 1),
    }
    ids, dropped = kept(list(cases.values()), alpha_channels, now)
    assert ids == ["exactmin001", "exactmax001"]
    assert dropped == {"out_of_window": 2}


def test_duration_boundary(alpha_channels, make_video, now):
    items = [
        make_video("exact60s001", duration=f"PT{MIN_DURATION_SECONDS}S"),
        make_video("oneminute01", duration="PT1M"),
        make_video("fiftynine01", duration="PT59S"),
    ]
    ids, dropped = kept(items, alpha_channels, now)
    assert ids == ["exact60s001", "oneminute01"]
    assert dropped == {"short": 1}


def test_every_exclusion_cause_is_counted_separately(populated_raw, processed_dir, now):
    summary = run_build(populated_raw, processed_dir, now=now)
    assert list(summary["dropped"]) == EXCLUSIONS, "every cause reported, in filter order"
    assert summary["dropped"] == {
        "unknown_channel": 1,    # orphan00001
        "low_subs": 1,           # tiny0000001
        "live_or_upcoming": 1,   # live0000001
        "no_views": 1,           # noviews0001
        "bad_duration": 1,       # badduration
        "short": 1,              # alpha000003
        "out_of_window": 1,      # beta0000002
    }
    assert summary["rows"] == 5


def test_output_columns_equal_schema_columns_exactly_in_order(populated_raw, processed_dir, now):
    summary = run_build(populated_raw, processed_dir, now=now)
    frame = pd.read_parquet(summary["written"][0])
    assert list(frame.columns) == COLUMNS
    assert summary["written"][0] == processed_dir / "videos.parquet"


def test_row_values_and_dtypes(populated_raw, processed_dir, now):
    run_build(populated_raw, processed_dir, now=now)
    frame = pd.read_parquet(processed_dir / "videos.parquet").set_index(VIDEO_ID)

    row = frame.loc["alpha000002"]
    assert row[CHANNEL_TITLE] == "Alpha Gaming"
    assert row[SUBSCRIBER_COUNT] == 2_500_000
    assert row[VIEW_COUNT] == 820_000
    assert row[DURATION_SECONDS] == 3723
    assert row[CONTENT_TYPE] == "gaming"
    assert row[THUMBNAIL_FILE] == THUMBNAIL_PATTERN.format(video_id="alpha000002")
    for column in (SUBSCRIBER_COUNT, VIEW_COUNT, DURATION_SECONDS):
        assert str(frame[column].dtype) == "int64"
    assert sorted(frame.index) == [
        "alpha000001", "alpha000002", "alpha000004", "beta0000001", "shared00001",
    ]


def test_csv_flag_writes_csv_with_same_header(populated_raw, processed_dir, now):
    summary = run_build(populated_raw, processed_dir, csv=True, now=now)
    csv_path = processed_dir / "videos.csv"
    assert csv_path in summary["written"]
    with csv_path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == COLUMNS
    assert len(rows) == 1 + summary["rows"]


def test_latest_channel_record_wins(raw_dir, channels_fixture, videos_fixture, now):
    old = channel_record(channels_fixture, "@tiny")
    new = channel_record(channels_fixture, "@tiny")
    new["statistics"]["subscriberCount"] = "50000"   # grew past MIN_SUBSCRIBERS since
    new["snippet"]["title"] = "Not So Tiny"
    write_jsonl(raw_dir / "channels_20260101T000000Z.jsonl", [old])
    write_jsonl(raw_dir / "channels_20260901T000000Z.jsonl", [new])
    write_jsonl(raw_dir / "videos_20260901T000000Z.jsonl", [videos_fixture["tiny0000001"]])

    assert load_raw_channels(raw_dir)[TINY]["snippet"]["title"] == "Not So Tiny"
    rows, dropped = build_rows(load_raw_videos(raw_dir), load_raw_channels(raw_dir), now)
    assert [r[CHANNEL_TITLE] for r in rows] == ["Not So Tiny"]
    assert dropped["low_subs"] == 0


def test_same_video_in_two_raw_files_yields_one_row(raw_dir, channels_fixture, videos_fixture):
    write_jsonl(raw_dir / "videos_20260101T000000Z.jsonl", [videos_fixture["alpha000001"]])
    write_jsonl(raw_dir / "videos_20260201T000000Z.jsonl", [videos_fixture["alpha000001"]])
    assert len(load_raw_videos(raw_dir)) == 1


def test_empty_raw_dir_writes_an_empty_table_with_all_columns(raw_dir, processed_dir, now):
    summary = run_build(raw_dir, processed_dir, now=now)
    frame = pd.read_parquet(processed_dir / "videos.parquet")
    assert summary["rows"] == 0
    assert list(frame.columns) == COLUMNS
