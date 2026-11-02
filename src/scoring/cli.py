"""Score from the command line, without the app.

    python -m src.scoring.cli "7 Ramen Mistakes" --type cooking
    python -m src.scoring.cli "7 Ramen Mistakes" --type cooking --thumb thumb.jpg --subs 250000
    python -m src.scoring.cli --batch titles.csv --type finance --out scored.csv
    python -m src.scoring.cli "RAMEN" --type cooking --suggest       # generate + rescore
    python -m src.scoring.cli "RAMEN" --type cooking --json

The batch CSV needs a `title` column; `content_type` and `subscriber_count`
columns are optional and override --type / --subs per row.
"""

import argparse
import json
import sys
from dataclasses import asdict

import pandas as pd

from config.schema import CONTENT_TYPES
from src.scoring.registry import get_scorer


def _print_result(label, r):
    print(f"{label:<10} {r.score:5.0f}")
    for k, v in r.drivers.items():
        print(f"    {k:<24} {v:+6.0f}")
    for n in r.notes:
        print(f"  - {n}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("title", nargs="?", help="the title to score")
    p.add_argument("--type", dest="content_type", default=CONTENT_TYPES[0], help=f"one of {', '.join(CONTENT_TYPES)}")
    p.add_argument("--thumb", help="path to a thumbnail image")
    p.add_argument("--subs", type=float, default=None, help="your subscriber count (optional context)")
    p.add_argument("--batch", help="CSV with a title column")
    p.add_argument("--out", help="where to write the scored batch CSV")
    p.add_argument("--suggest", action="store_true", help="generate alternatives and score them")
    p.add_argument("--no-claude", action="store_true", help="rule-based suggestions only")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    args = p.parse_args(argv)

    scorer = get_scorer()
    if not scorer.status.all_real:
        print("warning: using placeholder scores; train the models first (python -m src.modelling.train)", file=sys.stderr)
    elif scorer.status.synthetic:
        print("warning: models were trained on synthetic data", file=sys.stderr)

    if args.batch:
        frame = pd.read_csv(args.batch)
        if "title" not in frame.columns:
            raise SystemExit("the CSV needs a `title` column")
        rows = []
        for _, r in frame.iterrows():
            ct = str(r["content_type"]) if "content_type" in frame.columns and pd.notna(r["content_type"]) else args.content_type
            subs = float(r["subscriber_count"]) if "subscriber_count" in frame.columns and pd.notna(r["subscriber_count"]) else args.subs
            res = scorer.score_title(str(r["title"]), ct, subs)
            rows.append({"title": r["title"], "content_type": ct, "score": res.score,
                         "drivers": json.dumps(res.drivers), "notes": " | ".join(res.notes)})
        out = pd.DataFrame(rows).sort_values("score", ascending=False)
        if args.out:
            out.to_csv(args.out, index=False)
            print(f"wrote {len(out)} rows to {args.out}")
        else:
            print(out[["score", "title"]].to_string(index=False))
        return

    if not args.title and not args.thumb:
        p.error("give a title, a --thumb, or a --batch CSV")

    result = scorer.score_video(args.title or "", args.thumb, args.content_type, args.subs)
    payload = {k: (asdict(v) if v is not None else None) for k, v in result.items()}
    if args.suggest and args.title:
        from src.generation.titles import improve_title

        sugg = improve_title(args.title, args.content_type, scorer, args.subs, use_claude=not args.no_claude)
        payload["suggestions"] = {"source": sugg["source"], "error": sugg["error"], "candidates": [c.to_dict() for c in sugg["candidates"]]}

    if args.json:
        print(json.dumps(payload, indent=2, default=float))
        return
    if result["title"] is not None and args.title:
        _print_result("title", result["title"])
    if result["thumbnail"] is not None:
        _print_result("thumbnail", result["thumbnail"])
    if result["title"] is not None and result["thumbnail"] is not None:
        print(f"{'overall':<10} {result['overall'].score:5.0f}   weights {result['overall'].meta.get('weights')}")
    if "suggestions" in payload:
        s = payload["suggestions"]
        print(f"\nsuggestions ({s['source']}{': ' + s['error'] if s['error'] else ''})")
        for c in s["candidates"]:
            print(f"  {c['score']:5.0f} {c['delta']:+5.0f}  {c['title']}   ({c['rationale']})")


if __name__ == "__main__":
    main()
