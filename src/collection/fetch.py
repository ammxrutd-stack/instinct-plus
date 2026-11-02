"""Video ids to raw JSONL.

videos.list accepts 50 ids per call and costs 1 unit however many parts are
requested, so contentDetails (for duration) rides along free. Items are
appended verbatim, one per line, to a per-run file; ids already present in
any earlier videos_*.jsonl are skipped so a rerun never double-spends.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator, Optional

from config.schema import RAW_DIR

from .checkpoint import quota_reset_message
from .client import QuotaExceeded
from .discover import make_run_id, read_video_ids

BATCH_SIZE = 50  # the maximum videos.list accepts
PARTS = "snippet,statistics,contentDetails"
VIDEOS_FILE = "videos_{run_id}.jsonl"
VIDEOS_GLOB = "videos_*.jsonl"
PROGRESS_EVERY = 20  # calls


def batched(ids: Iterable[str], size: int = BATCH_SIZE) -> Iterator[list]:
    ids = list(ids)
    for start in range(0, len(ids), size):
        yield ids[start:start + size]


def already_fetched(raw_dir=RAW_DIR) -> set:
    """Ids present in any raw videos file, whichever run wrote it."""
    seen = set()
    for path in sorted(Path(raw_dir).glob(VIDEOS_GLOB)):
        with path.open(encoding="utf-8") as lines:
            for line in lines:
                if line.strip():
                    seen.add(json.loads(line)["id"])
    return seen


def fetch_batch(client, batch: list) -> Optional[list]:
    """One videos.list call. None when the client gave up on it."""
    payload = client.get("videos", part=PARTS, id=",".join(batch))
    if payload is None:
        return None
    return payload.get("items", [])


def run_fetch(client, raw_dir=RAW_DIR, run_id: Optional[str] = None,
              ids: Optional[Iterable[str]] = None, limit: Optional[int] = None) -> dict:
    """Fetch every pending id in video_ids.txt (or ``ids``). Returns the run summary.

    Each batch is appended as soon as it returns, so a run cut short by quota
    keeps what it paid for and the next run picks up from the remainder.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    run_id = run_id or make_run_id()

    wanted = list(ids) if ids is not None else read_video_ids(raw_dir)
    done = already_fetched(raw_dir)
    pending = [video_id for video_id in wanted if video_id not in done]
    skipped = len(wanted) - len(pending)
    if limit is not None:
        pending = pending[:limit]

    out_path = raw_dir / VIDEOS_FILE.format(run_id=run_id)
    summary = {
        "run_id": run_id,
        "requested": len(wanted),
        "already_fetched": skipped,
        "pending": len(pending),
        "calls": 0,
        "written": 0,
        "not_returned": 0,  # deleted, private, or a batch the client gave up on
        "quota_exceeded": False,
    }
    print(f"{len(pending):,} ids to fetch ({summary['already_fetched']:,} already in raw)")

    try:
        for batch in batched(pending):
            items = fetch_batch(client, batch)
            summary["calls"] += 1
            if items is None:
                summary["not_returned"] += len(batch)
                continue
            with out_path.open("a", encoding="utf-8") as out:
                for item in items:
                    out.write(json.dumps(item) + "\n")
            summary["written"] += len(items)
            summary["not_returned"] += len(batch) - len(items)
            if summary["calls"] % PROGRESS_EVERY == 0:
                print(f"  {summary['written']:,} videos fetched")
    except QuotaExceeded as error:
        summary["quota_exceeded"] = True
        print(f"\n! {error}. Fetched items are saved; rerun to continue. "
              f"{quota_reset_message()}")

    summary["units"] = client.units
    summary["output"] = out_path if out_path.exists() else None
    return summary
