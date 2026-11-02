"""Raw JSONL to the processed table. No network.

Reads every videos_*.jsonl and channels_*.jsonl under data/raw/, joins each
video to its channel, applies the filters from config.schema and projects
onto schema.COLUMNS. Because the raw files hold API items verbatim, a filter
or feature bug is fixed by re-running this stage rather than by spending
quota, and the publish window can be re-cut the same way.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from config.schema import (
    CHANNEL_ID,
    CHANNEL_TITLE,
    COLUMNS,
    CONTENT_TYPE,
    DURATION_SECONDS,
    MAX_DAYS_SINCE_PUBLISH,
    MIN_DAYS_SINCE_PUBLISH,
    MIN_DURATION_SECONDS,
    MIN_SUBSCRIBERS,
    PROCESSED_DIR,
    PUBLISHED_AT,
    RAW_DIR,
    SUBSCRIBER_COUNT,
    THUMBNAIL_FILE,
    THUMBNAIL_PATTERN,
    TITLE,
    VIDEO_ID,
    VIDEOS_TABLE,
    VIDEOS_TABLE_CSV,
    VIEW_COUNT,
)

from .discover import CHANNELS_GLOB, parse_timestamp, subscriber_count
from .fetch import VIDEOS_GLOB

# ISO 8601 as the API emits it: PT1H2M3S, PT60S, P1DT2H, and P0D for live events
DURATION_RE = re.compile(r"^P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?)?$")

# every reason a raw video can be excluded, in the order the filters run
EXCLUSIONS = [
    "unknown_channel",
    "low_subs",
    "live_or_upcoming",
    "no_views",
    "bad_duration",
    "short",
    "out_of_window",
]

INT_COLUMNS = {SUBSCRIBER_COUNT: "int64", VIEW_COUNT: "int64", DURATION_SECONDS: "int64"}


def parse_duration(value: Optional[str]) -> Optional[int]:
    """ISO 8601 duration to whole seconds. None when unparseable."""
    match = DURATION_RE.match(value or "")
    if not match or not any(group is not None for group in match.groups()):
        return None
    days, hours, minutes, seconds = (int(group or 0) for group in match.groups())
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def load_jsonl(path: Path) -> list:
    with Path(path).open(encoding="utf-8") as lines:
        return [json.loads(line) for line in lines if line.strip()]


def load_raw_videos(raw_dir=RAW_DIR) -> list:
    """Every raw video item, one per id. The newest file wins when an id repeats."""
    by_id = {}
    for path in sorted(Path(raw_dir).glob(VIDEOS_GLOB)):  # run_id sorts chronologically
        for item in load_jsonl(path):
            by_id[item["id"]] = item
    return list(by_id.values())


def load_raw_channels(raw_dir=RAW_DIR) -> dict:
    """channel id -> newest channel record, carrying _content_type and _handle."""
    by_id = {}
    for path in sorted(Path(raw_dir).glob(CHANNELS_GLOB)):
        for item in load_jsonl(path):
            by_id[item["id"]] = item
    return by_id


def age_days(published_at: str, now: datetime) -> int:
    return (now - parse_timestamp(published_at)).days


def build_rows(items: list, channels: dict, now: datetime) -> tuple:
    """Apply the filters and project onto schema.COLUMNS. Returns (rows, dropped)."""
    rows = []
    dropped = {cause: 0 for cause in EXCLUSIONS}

    for item in items:
        snippet = item.get("snippet", {})
        channel = channels.get(snippet.get("channelId"))
        if not channel:
            dropped["unknown_channel"] += 1
            continue
        if subscriber_count(channel) < MIN_SUBSCRIBERS:
            dropped["low_subs"] += 1
            continue

        # Live broadcasts and scheduled premieres report duration "P0D" and have
        # view dynamics unlike on-demand video. Excluded for the same reason as
        # Shorts. Checked before parsing so a genuine parse bug stays visible.
        if snippet.get("liveBroadcastContent", "none") != "none":
            dropped["live_or_upcoming"] += 1
            continue

        views = item.get("statistics", {}).get("viewCount")
        if views is None:
            dropped["no_views"] += 1
            continue

        seconds = parse_duration(item.get("contentDetails", {}).get("duration"))
        if seconds is None:
            dropped["bad_duration"] += 1
            continue
        if seconds < MIN_DURATION_SECONDS:
            dropped["short"] += 1
            continue

        age = age_days(snippet["publishedAt"], now)
        if not MIN_DAYS_SINCE_PUBLISH <= age <= MAX_DAYS_SINCE_PUBLISH:
            dropped["out_of_window"] += 1
            continue

        video_id = item["id"]
        rows.append({
            VIDEO_ID: video_id,
            TITLE: snippet["title"],
            CHANNEL_ID: channel["id"],
            CHANNEL_TITLE: channel.get("snippet", {}).get("title", ""),
            SUBSCRIBER_COUNT: subscriber_count(channel),
            VIEW_COUNT: int(views),
            PUBLISHED_AT: snippet["publishedAt"],
            DURATION_SECONDS: seconds,
            CONTENT_TYPE: channel["_content_type"],
            THUMBNAIL_FILE: THUMBNAIL_PATTERN.format(video_id=video_id),
        })
    return rows, dropped


def to_frame(rows: list) -> pd.DataFrame:
    """Exactly schema.COLUMNS in order, with integer dtypes that survive Parquet."""
    return pd.DataFrame(rows, columns=COLUMNS).astype(INT_COLUMNS)


def write_table(frame: pd.DataFrame, processed_dir=PROCESSED_DIR, csv: bool = False) -> list:
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    frame = frame[COLUMNS]
    parquet_path = processed_dir / VIDEOS_TABLE.name
    frame.to_parquet(parquet_path, index=False)
    written = [parquet_path]
    if csv:
        csv_path = processed_dir / VIDEOS_TABLE_CSV.name
        frame.to_csv(csv_path, index=False)
        written.append(csv_path)
    return written


def run_build(raw_dir=RAW_DIR, processed_dir=PROCESSED_DIR, csv: bool = False,
              now: Optional[datetime] = None) -> dict:
    """Join, filter, project and write. Returns the run summary."""
    now = now or datetime.now(timezone.utc)
    items = load_raw_videos(raw_dir)
    channels = load_raw_channels(raw_dir)
    print(f"{len(items):,} raw videos across {len(channels):,} channels")

    rows, dropped = build_rows(items, channels, now)
    written = write_table(to_frame(rows), processed_dir, csv=csv)
    print(f"{len(rows):,} rows kept, {sum(dropped.values()):,} excluded")
    return {
        "raw_videos": len(items),
        "raw_channels": len(channels),
        "rows": len(rows),
        "dropped": dropped,
        "written": written,
    }
