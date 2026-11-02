"""Processed table to image files. Plain HTTPS against i.ytimg.com, no API quota.

hqdefault exists for every video ever uploaded; maxresdefault does not, and
its absence correlates with the target variable. A failed download keeps its
row with an empty thumbnail_file because the title model can still train on
it; dropping the row would discard usable data over a missing image.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from config.schema import (
    PROCESSED_DIR,
    THUMBNAIL_FILE,
    THUMBNAIL_PATTERN,
    THUMBNAILS_DIR,
    VIDEO_ID,
    VIDEOS_TABLE,
    VIDEOS_TABLE_CSV,
)

from .build import write_table
from .client import CollectionError

THUMB_URL = "https://i.ytimg.com/vi/{video_id}/hqdefault.jpg"
TIMEOUT_SECONDS = 30
PROGRESS_EVERY = 250


def thumbnail_path(thumbnails_dir, video_id: str) -> Path:
    return Path(thumbnails_dir) / THUMBNAIL_PATTERN.format(video_id=video_id)


def download_thumbnail(session, video_id: str, target: Path) -> bool:
    try:
        response = session.get(THUMB_URL.format(video_id=video_id), timeout=TIMEOUT_SECONDS)
    except requests.RequestException:
        return False
    if response.status_code != 200 or not response.content:
        return False
    target.write_bytes(response.content)
    return True


def download_thumbnails(frame: pd.DataFrame, thumbnails_dir=THUMBNAILS_DIR,
                        session=None) -> tuple:
    """Fetch one image per row. Returns (updated frame, counts).

    Rows whose image already exists are skipped, so reruns are cheap. Rows that
    failed on an earlier run (empty thumbnail_file) are retried and restored.
    """
    thumbnails_dir = Path(thumbnails_dir)
    thumbnails_dir.mkdir(parents=True, exist_ok=True)
    session = session or requests.Session()
    frame = frame.copy()
    counts = {"downloaded": 0, "already_present": 0, "failed": 0}

    for position, (index, video_id) in enumerate(frame[VIDEO_ID].items(), 1):
        target = thumbnail_path(thumbnails_dir, video_id)
        if target.exists():
            counts["already_present"] += 1
            frame.at[index, THUMBNAIL_FILE] = target.name
        elif download_thumbnail(session, video_id, target):
            counts["downloaded"] += 1
            frame.at[index, THUMBNAIL_FILE] = target.name
        else:
            counts["failed"] += 1
            frame.at[index, THUMBNAIL_FILE] = ""
        if position % PROGRESS_EVERY == 0:
            print(f"  {position:,}/{len(frame):,} thumbnails")
    return frame, counts


def run_thumbnails(processed_dir=PROCESSED_DIR, thumbnails_dir=THUMBNAILS_DIR,
                   session=None) -> dict:
    """Download for every row in the processed table and rewrite it. Returns the summary."""
    processed_dir = Path(processed_dir)
    table_path = processed_dir / VIDEOS_TABLE.name
    if not table_path.exists():
        raise CollectionError(f"{table_path} does not exist; run build first")

    frame = pd.read_parquet(table_path)
    print(f"{len(frame):,} rows in {table_path.name}")
    frame, counts = download_thumbnails(frame, thumbnails_dir, session)

    # the CSV is a convenience copy; keep it in step only if build produced one
    csv = (processed_dir / VIDEOS_TABLE_CSV.name).exists()
    written = write_table(frame, processed_dir, csv=csv)
    return {"rows": len(frame), "written": written, **counts}
