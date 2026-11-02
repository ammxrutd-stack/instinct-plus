"""Seed channels to video ids.

Resolves each handle in config.channels.CHANNELS, then walks the channel's
uploads playlist newest-first and stops at the first item older than the
publish window, so a channel with ten years of uploads costs only the pages
that intersect the window. build is the authoritative filter; the early stop
here is a quota optimisation using the same window, not a second source of
truth.

Outputs under data/raw/:
    video_ids.txt              one id per line, deduped, sorted, merged with
                               whatever an earlier run left there
    channels_{run_id}.jsonl    channels.list items verbatim, plus _content_type
                               and _handle added by us
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional

from config.channels import CHANNELS
from config.schema import (
    CONTENT_TYPES,
    MAX_DAYS_SINCE_PUBLISH,
    MIN_DAYS_SINCE_PUBLISH,
    RAW_DIR,
)

from .checkpoint import (
    CHECKPOINT_FILE,
    clear_checkpoint,
    load_checkpoint,
    quota_reset_message,
    write_checkpoint,
)
from .client import QuotaExceeded

VIDEO_IDS_FILE = "video_ids.txt"
CHANNELS_FILE = "channels_{run_id}.jsonl"
CHANNELS_GLOB = "channels_*.jsonl"
PAGE_SIZE = 50  # the maximum playlistItems.list accepts


def make_run_id(now: Optional[datetime] = None) -> str:
    """UTC timestamp, YYYYMMDDTHHMMSSZ. Discover and fetch share one per run."""
    return (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")


def parse_timestamp(value: str) -> datetime:
    """API timestamps are RFC 3339 with a Z suffix, which fromisoformat rejects on 3.9."""
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def window_start(now: datetime) -> datetime:
    return now - timedelta(days=MAX_DAYS_SINCE_PUBLISH)


def window_end(now: datetime) -> datetime:
    return now - timedelta(days=MIN_DAYS_SINCE_PUBLISH)


def read_video_ids(raw_dir=RAW_DIR) -> list:
    path = Path(raw_dir) / VIDEO_IDS_FILE
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def write_video_ids(raw_dir, ids: Iterable[str]) -> list:
    """Merge with the existing file, dedupe, sort. Returns the ids written."""
    merged = sorted(set(read_video_ids(raw_dir)) | set(ids))
    path = Path(raw_dir) / VIDEO_IDS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{video_id}\n" for video_id in merged), encoding="utf-8")
    return merged


def check_content_types(seeds: dict) -> list:
    """Warn, not fail, so the seed list can be developed ahead of CONTENT_TYPES."""
    unknown = [key for key in seeds if key not in CONTENT_TYPES]
    for key in unknown:
        warnings.warn(
            f"content type {key!r} in CHANNELS is not in schema.CONTENT_TYPES",
            stacklevel=2,
        )
    return unknown


def resolve_handle(client, handle: str) -> Optional[dict]:
    """channels.list forHandle. None when the handle does not resolve."""
    payload = client.get(
        "channels", part="snippet,contentDetails,statistics", forHandle=handle
    )
    items = (payload or {}).get("items") or []
    return items[0] if items else None


def subscriber_count(channel: dict) -> int:
    """0 when the channel hides its count; build then drops it as below MIN_SUBSCRIBERS."""
    value = channel.get("statistics", {}).get("subscriberCount")
    return int(value) if value else 0


def uploads_playlist(channel: dict) -> Optional[str]:
    return channel.get("contentDetails", {}).get("relatedPlaylists", {}).get("uploads")


def discover_videos(client, playlist_id: str, start: datetime, cap: Optional[int] = None) -> list:
    """Page an uploads playlist newest-first, stopping once past the window start."""
    video_ids, page_token = [], None
    while True:
        payload = client.get(
            "playlistItems",
            part="contentDetails",
            playlistId=playlist_id,
            maxResults=PAGE_SIZE,
            pageToken=page_token,
        )
        if not payload:
            break

        stop = False
        for item in payload.get("items", []):
            details = item.get("contentDetails", {})
            published = details.get("videoPublishedAt")
            if not published:
                continue  # private or deleted uploads carry no publish time
            if parse_timestamp(published) < start:
                stop = True  # newest-first, so everything after is older too
                break
            video_ids.append(details["videoId"])

        if cap and len(video_ids) >= cap:
            return video_ids[:cap]
        page_token = payload.get("nextPageToken")
        if stop or not page_token:
            break
    return video_ids


def run_discover(client, seeds: Optional[dict] = None, raw_dir=RAW_DIR,
                 run_id: Optional[str] = None, now: Optional[datetime] = None,
                 cap: Optional[int] = None) -> dict:
    """Resolve, page and record every channel in ``seeds``. Returns the run summary.

    Channels are resolved and paged one at a time so that a quota failure
    leaves a consistent state: every channel marked complete has all its
    in-window ids in video_ids.txt and its record in the channels file.
    """
    seeds = CHANNELS if seeds is None else seeds
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    now = now or datetime.now(timezone.utc)
    run_id = run_id or make_run_id(now)
    start = window_start(now)
    check_content_types(seeds)

    checkpoint = load_checkpoint(raw_dir)
    done_ids = set(checkpoint["completed_channel_ids"])
    done_handles = set(checkpoint["completed_handles"])
    completed_ids, completed_handles = list(done_ids), list(done_handles)

    channels_path = raw_dir / CHANNELS_FILE.format(run_id=run_id)
    found = set()
    summary = {
        "run_id": run_id,
        "window_start": start,
        "window_end": window_end(now),
        "channels_processed": 0,
        "channels_skipped": 0,
        "unresolved": [],
        "per_channel": [],  # (content_type, handle, title, subscribers, ids found)
        "quota_exceeded": False,
    }

    print(f"Window: {start.date()} to {window_end(now).date()} "
          f"({MIN_DAYS_SINCE_PUBLISH}-{MAX_DAYS_SINCE_PUBLISH} days old)")
    if done_handles:
        print(f"Resuming from checkpoint: {len(done_handles)} channels already complete")

    try:
        for content_type, handles in seeds.items():
            print(f"\n[{content_type}]")
            for handle in handles:
                if handle in done_handles:
                    summary["channels_skipped"] += 1
                    continue
                channel = resolve_handle(client, handle)
                if channel is None:
                    print(f"  ! {handle} did not resolve, skipping")
                    summary["unresolved"].append(handle)
                    continue
                if channel["id"] in done_ids:  # same channel under another spelling
                    summary["channels_skipped"] += 1
                    continue

                record = dict(channel, _content_type=content_type, _handle=handle)
                with channels_path.open("a", encoding="utf-8") as handle_file:
                    handle_file.write(json.dumps(record) + "\n")

                playlist = uploads_playlist(channel)
                ids = discover_videos(client, playlist, start, cap) if playlist else []
                found.update(ids)
                completed_ids.append(channel["id"])
                completed_handles.append(handle)

                title = channel.get("snippet", {}).get("title", "")
                subs = subscriber_count(channel)
                summary["channels_processed"] += 1
                summary["per_channel"].append((content_type, handle, title, subs, len(ids)))
                print(f"  {handle:<28} {title[:32]:<34} {subs:>12,} subs {len(ids):>5} in window")
    except QuotaExceeded as error:
        write_checkpoint(raw_dir, completed_ids, completed_handles, now)
        summary["quota_exceeded"] = True
        print(f"\n! {error}. Checkpoint written to {CHECKPOINT_FILE}; rerun to continue. "
              f"{quota_reset_message()}")
    else:
        clear_checkpoint(raw_dir)

    all_ids = write_video_ids(raw_dir, found)
    summary["video_ids"] = sorted(found)
    summary["total_video_ids"] = len(all_ids)
    summary["units"] = client.units
    return summary

