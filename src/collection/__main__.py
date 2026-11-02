"""Command line entry point for the collection pipeline.

    python -m src.collection discover     seed channels -> data/raw/video_ids.txt
    python -m src.collection fetch        video ids -> data/raw/videos_{run_id}.jsonl
    python -m src.collection build        raw JSONL -> data/processed/videos.parquet
    python -m src.collection thumbnails   processed table -> data/thumbnails/
    python -m src.collection all          the four stages in order
    python -m src.collection verify       resolve every handle in CHANNELS and
                                          flag any unresolved or below MIN_SUBSCRIBERS

Flags: --budget N (default 9000), --smoke (2 channels of the first content
type, 50 videos each), --csv (build also writes videos.csv), --no-thumbs
(all skips the thumbnail stage).

Exit codes: 0 on success and on quota exhaustion (the run is resumable), 1 on
a fatal error such as a wrong key.
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

from config.channels import CHANNELS
from config.schema import MIN_SUBSCRIBERS

from .build import run_build
from .checkpoint import quota_reset_message
from .client import DEFAULT_BUDGET, Client, CollectionError, QuotaExceeded, load_api_key
from .discover import make_run_id, resolve_handle, run_discover, subscriber_count
from .fetch import run_fetch
from .thumbnails import run_thumbnails

SMOKE_CHANNELS = 2
SMOKE_CAP = 50


def smoke_seeds(seeds: Optional[dict] = None) -> dict:
    """The first two channels of the first content type: enough to prove the key works."""
    seeds = CHANNELS if seeds is None else seeds
    first = next(iter(seeds))
    return {first: seeds[first][:SMOKE_CHANNELS]}


def print_summary(stage: str, lines: list) -> None:
    """Uniform closing block so every run ends the same way, whatever the stage."""
    print(f"\n== {stage} summary ==")
    width = max(len(label) for label, _ in lines)
    for label, value in lines:
        print(f"  {label:<{width}}  {value}")


def discover_lines(summary: dict, budget: int) -> list:
    by_type = {}
    for content_type, _handle, _title, _subs, count in summary["per_channel"]:
        by_type[content_type] = by_type.get(content_type, 0) + count
    lines = [
        ("units spent", f"{summary['units']:,} of {budget:,}"),
        ("channels processed", summary["channels_processed"]),
        ("channels skipped (checkpoint)", summary["channels_skipped"]),
        ("handles unresolved", ", ".join(summary["unresolved"]) or "none"),
        ("videos discovered this run", f"{len(summary['video_ids']):,}"),
        ("video ids in video_ids.txt", f"{summary['total_video_ids']:,}"),
    ]
    lines += [(f"  {content_type}", f"{count:,}") for content_type, count in by_type.items()]
    if summary["quota_exceeded"]:
        lines.append(("status", "stopped on quota, checkpoint written"))
    return lines


def fetch_lines(summary: dict, budget: int) -> list:
    lines = [
        ("units spent", f"{summary['units']:,} of {budget:,}"),
        ("ids requested", f"{summary['requested']:,}"),
        ("already in raw", f"{summary['already_fetched']:,}"),
        ("videos written", f"{summary['written']:,}"),
        ("ids not returned", f"{summary['not_returned']:,}"),
        ("output", summary["output"] or "nothing new"),
    ]
    if summary["quota_exceeded"]:
        lines.append(("status", "stopped on quota, rerun to continue"))
    return lines


def build_lines(summary: dict) -> list:
    lines = [
        ("raw videos", f"{summary['raw_videos']:,}"),
        ("raw channels", f"{summary['raw_channels']:,}"),
        ("rows written", f"{summary['rows']:,}"),
    ]
    lines += [(f"  excluded: {cause}", f"{count:,}") for cause, count in summary["dropped"].items()]
    lines += [("output", path) for path in summary["written"]]
    return lines


def thumbnail_lines(summary: dict) -> list:
    return [
        ("rows", f"{summary['rows']:,}"),
        ("downloaded", f"{summary['downloaded']:,}"),
        ("already present", f"{summary['already_present']:,}"),
        ("failed (row kept, thumbnail_file empty)", f"{summary['failed']:,}"),
    ] + [("output", path) for path in summary["written"]]


def cmd_verify(client, seeds: Optional[dict] = None) -> dict:
    """Resolve every handle so squatters and small channels are caught before a full run."""
    seeds = CHANNELS if seeds is None else seeds
    flagged = []
    checked = 0
    print(f"{'handle':<30} {'channel':<34} {'subscribers':>12}  flag")
    for content_type, handles in seeds.items():
        print(f"[{content_type}]")
        for handle in handles:
            checked += 1
            channel = resolve_handle(client, handle)
            if channel is None:
                print(f"{handle:<30} {'(unresolved)':<34} {'-':>12}  UNRESOLVED")
                flagged.append((handle, "unresolved"))
                continue
            title = channel.get("snippet", {}).get("title", "")
            subs = subscriber_count(channel)
            flag = "BELOW MIN_SUBSCRIBERS" if subs < MIN_SUBSCRIBERS else ""
            if flag:
                flagged.append((handle, flag))
            print(f"{handle:<30} {title[:34]:<34} {subs:>12,}  {flag}")
    return {"checked": checked, "flagged": flagged, "units": client.units}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m src.collection",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("stage", choices=["discover", "fetch", "build", "thumbnails", "all", "verify"])
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET,
                        help=f"quota units this run may spend (default {DEFAULT_BUDGET})")
    parser.add_argument("--smoke", action="store_true",
                        help=f"{SMOKE_CHANNELS} channels of the first content type, "
                             f"{SMOKE_CAP} videos each")
    parser.add_argument("--csv", action="store_true",
                        help="build also writes videos.csv")
    parser.add_argument("--no-thumbs", action="store_true",
                        help="all skips the thumbnail stage")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    seeds = smoke_seeds() if args.smoke else None
    cap = SMOKE_CAP if args.smoke else None
    fetch_limit = SMOKE_CHANNELS * SMOKE_CAP if args.smoke else None

    if args.stage == "build":
        print_summary("build", build_lines(run_build(csv=args.csv)))
        return 0
    if args.stage == "thumbnails":
        print_summary("thumbnails", thumbnail_lines(run_thumbnails()))
        return 0

    client = Client(load_api_key(), budget=args.budget)

    if args.stage == "verify":
        summary = cmd_verify(client, seeds)
        print_summary("verify", [
            ("units spent", f"{summary['units']:,} of {args.budget:,}"),
            ("handles checked", summary["checked"]),
            ("flagged", "; ".join(f"{h} ({why})" for h, why in summary["flagged"]) or "none"),
        ])
        return 0

    run_id = make_run_id()

    if args.stage == "discover":
        summary = run_discover(client, seeds=seeds, run_id=run_id, cap=cap)
        print_summary("discover", discover_lines(summary, args.budget))
        return 0

    if args.stage == "fetch":
        summary = run_fetch(client, run_id=run_id, limit=fetch_limit)
        print_summary("fetch", fetch_lines(summary, args.budget))
        return 0

    # all: each stage prints its own block; stop at the first stage cut short by quota
    discovered = run_discover(client, seeds=seeds, run_id=run_id, cap=cap)
    print_summary("discover", discover_lines(discovered, args.budget))
    if discovered["quota_exceeded"]:
        return 0

    # in smoke mode fetch only what this run discovered, not a full video_ids.txt
    fetched = run_fetch(client, run_id=run_id,
                        ids=discovered["video_ids"] if args.smoke else None)
    print_summary("fetch", fetch_lines(fetched, args.budget))
    if fetched["quota_exceeded"]:
        return 0

    built = run_build(csv=args.csv)
    print_summary("build", build_lines(built))

    if not args.no_thumbs and built["rows"]:
        print_summary("thumbnails", thumbnail_lines(run_thumbnails()))

    print_summary("run", [
        ("units spent", f"{client.units:,} of {args.budget:,}"),
        ("channels processed", discovered["channels_processed"]),
        ("videos discovered", f"{len(discovered['video_ids']):,}"),
        ("videos fetched", f"{fetched['written']:,}"),
        ("rows written", f"{built['rows']:,}"),
        ("excluded", ", ".join(f"{c} {n}" for c, n in built["dropped"].items() if n) or "none"),
    ])
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except QuotaExceeded as error:
        # only verify can reach here; the stages handle their own checkpointing
        print(f"\n! {error}. {quota_reset_message()}")
        return 0
    except CollectionError as error:
        print(f"\nerror: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
