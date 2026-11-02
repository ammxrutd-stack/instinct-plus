"""Resumption after quota exhaustion.

Discover is the stage that can run out of budget part-way through the
channel list, so it records which channels finished and the next run skips
them. Fetch needs no checkpoint: it skips ids already present in raw JSONL.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional

from config.schema import RAW_DIR

CHECKPOINT_FILE = "checkpoint.json"


def checkpoint_path(raw_dir=RAW_DIR) -> Path:
    return Path(raw_dir) / CHECKPOINT_FILE


def load_checkpoint(raw_dir=RAW_DIR) -> dict:
    """Completed channel ids and handles, as empty lists when there is no checkpoint."""
    path = checkpoint_path(raw_dir)
    if not path.exists():
        return {"completed_channel_ids": [], "completed_handles": []}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "completed_channel_ids": list(data.get("completed_channel_ids", [])),
        "completed_handles": list(data.get("completed_handles", [])),
    }


def write_checkpoint(raw_dir, completed_channel_ids: Iterable[str],
                     completed_handles: Iterable[str], now: Optional[datetime] = None) -> Path:
    """Handles are stored alongside ids so the next run skips without re-resolving."""
    path = checkpoint_path(raw_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "written_at": (now or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "completed_channel_ids": sorted(set(completed_channel_ids)),
        "completed_handles": sorted(set(completed_handles)),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def clear_checkpoint(raw_dir=RAW_DIR) -> None:
    path = checkpoint_path(raw_dir)
    if path.exists():
        path.unlink()


def next_quota_reset(now: Optional[datetime] = None) -> Optional[datetime]:
    """Midnight Pacific after ``now``. None if the zone database is unavailable."""
    try:
        from zoneinfo import ZoneInfo
        pacific = ZoneInfo("America/Los_Angeles")
    except Exception:  # no tzdata on this machine; the message degrades gracefully
        return None
    local = (now or datetime.now(timezone.utc)).astimezone(pacific)
    return (local + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)


def quota_reset_message(now: Optional[datetime] = None) -> str:
    reset = next_quota_reset(now)
    if reset is None:
        return "Quota resets at midnight Pacific."
    utc = reset.astimezone(timezone.utc)
    return (f"Quota resets at midnight Pacific "
            f"({reset:%Y-%m-%d %H:%M %Z}, {utc:%Y-%m-%d %H:%M} UTC).")
