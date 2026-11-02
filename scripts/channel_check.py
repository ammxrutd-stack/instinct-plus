"""Score a real channel's videos and compare with their actual views.

    python scripts/channel_check.py @projectvolta --subs 150 --type education

Needs no API key: the channel's public Videos tab lists long-form uploads
with durations and view counts, the Shorts tab lists Shorts, and thumbnails
are public images. Each video is tagged `long` or `short`; Shorts are kept
apart because the product (and the training data) is about long-form video,
where a title and thumbnail have to earn the click. Writes data/channel_demo/<handle>/ with the thumbnails, a
scored CSV (loadable in the app's Batch tab) and prints a table of model score
against actual views with the rank correlation between them.
"""

import argparse
import hashlib
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.schema import CONTENT_TYPES, DATA_DIR  # noqa: E402
from src.scoring.registry import get_scorer  # noqa: E402

NS = {"a": "http://www.w3.org/2005/Atom", "yt": "http://www.youtube.com/xml/schemas/2015", "media": "http://search.yahoo.com/mrss/"}
UA = {"User-Agent": "Mozilla/5.0 (Macintosh) AppleWebKit/537.36 Chrome/120 Safari/537.36", "Accept-Language": "en-AU,en;q=0.8"}


def resolve_channel_id(handle: str) -> str:
    handle = handle if handle.startswith("@") else "@" + handle
    html = requests.get(f"https://www.youtube.com/{handle}", headers=UA, timeout=30).text
    m = re.search(r'"channelId":"(UC[\w-]{22})"', html) or re.search(r'youtube\.com/channel/(UC[\w-]{22})', html)
    if not m:
        raise SystemExit(f"could not find a channel id on youtube.com/{handle}")
    return m.group(1)


def _initial_data(url: str):
    html = requests.get(url, headers=UA, timeout=30).text
    m = re.search(r"ytInitialData\s*=\s*(\{.+?\})\s*;\s*</script>", html, re.S)
    return json.loads(m.group(1)) if m else None


def _walk(o, keys, out):
    if isinstance(o, dict):
        for k, v in o.items():
            if k in keys:
                out.append(v)
            _walk(v, keys, out)
    elif isinstance(o, list):
        for v in o:
            _walk(v, keys, out)


_SCALE = {"": 1, "k": 1e3, "thousand": 1e3, "m": 1e6, "million": 1e6, "b": 1e9, "billion": 1e9}


def _views(text: str):
    m = re.search(r"([\d.,]+)\s*(thousand|million|billion|[KMB])?\s*views", text or "", re.I)
    if not m:
        return 0 if re.search(r"no views", text or "", re.I) else None
    n = float(m.group(1).replace(",", ""))
    return int(n * _SCALE[(m.group(2) or "").lower()])


def _duration(text: str):
    parts = [int(x) for x in re.findall(r"\d+", text or "")]
    if not parts or not re.fullmatch(r"[\d:]+", (text or "").strip()):
        return None
    secs = 0
    for x in parts:
        secs = secs * 60 + x
    return secs


def channel_tabs(handle: str, cache_dir: Path) -> pd.DataFrame:
    """Long-form uploads from /videos and Shorts from /shorts, as the public
    page renders them (about 30 and 48 items respectively before scrolling)."""
    handle = handle if handle.startswith("@") else "@" + handle
    rows = []
    for tab, kind in (("videos", "long"), ("streams", "live"), ("shorts", "short")):
        cache = cache_dir / f"{tab}.json"
        data = _initial_data(f"https://www.youtube.com/{handle}/{tab}")
        if data is None and cache.exists():
            data = json.loads(cache.read_text(encoding="utf-8"))
        elif data is not None:
            cache.write_text(json.dumps(data), encoding="utf-8")
        if data is None:
            continue
        items = []
        _walk(data, {"lockupViewModel", "videoRenderer", "gridVideoRenderer", "shortsLockupViewModel", "reelItemRenderer"}, items)
        for it in items:
            blob = json.dumps(it)
            ids = re.findall(r'"videoId": "([\w-]{11})"', blob) or re.findall(r"/vi/([\w-]{11})/", blob)
            if not ids:
                continue
            vid = ids[0]
            title = None
            if "accessibilityText" in it:          # shorts: "TITLE, 35 views – play Short"
                title = re.sub(r",\s*(?:[\d.,]+\s*(?:thousand|million|billion|[KMB])?\s*views|No views).*$", "", it["accessibilityText"], flags=re.I).strip()
            else:
                titles = []
                _walk(it.get("metadata", it), {"title"}, titles)
                for t in titles:
                    if isinstance(t, dict):
                        title = t.get("content") or t.get("simpleText") or "".join(r.get("text", "") for r in t.get("runs", []))
                        if title:
                            break
                if not title and "title" in it and isinstance(it["title"], dict):
                    title = it["title"].get("simpleText") or "".join(r.get("text", "") for r in it["title"].get("runs", []))
            if not title:
                continue
            badge = re.findall(r'"text": "(\d+:\d{2}(?::\d{2})?)"', blob) or re.findall(r'"lengthText".{0,120}?"simpleText": "([\d:]+)"', blob)
            rows.append({"video_id": vid, "title": title, "views": _views(blob if kind != "short" else it.get("accessibilityText", "")),
                         "duration_seconds": _duration(badge[0]) if badge else (None if kind != "short" else 30), "kind": kind})
    table = pd.DataFrame(rows).drop_duplicates("video_id")
    return table.reset_index(drop=True)


def feed(channel_id: str, cache: Path) -> pd.DataFrame:
    """The public Atom feed. YouTube sometimes answers with an HTML page
    instead; retry, and fall back to the last good copy on disk."""
    import time

    xml = None
    for attempt in range(3):
        text = requests.get(f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}", headers=UA, timeout=30).text
        if text.lstrip().startswith("<?xml") or text.lstrip().startswith("<feed"):
            xml = text
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(xml, encoding="utf-8")
            break
        time.sleep(1.5 * (attempt + 1))
    if xml is None:
        if cache.exists():
            print("feed fetch failed; using the cached copy")
            xml = cache.read_text(encoding="utf-8")
        else:
            raise SystemExit("YouTube did not return the feed; try again in a minute")
    root = ET.fromstring(xml)
    rows = []
    for e in root.findall("a:entry", NS):
        stats = e.find("media:group/media:community/media:statistics", NS)
        rows.append({
            "video_id": e.find("yt:videoId", NS).text,
            "title": e.find("a:title", NS).text,
            "published_at": e.find("a:published", NS).text,
            "view_count": int(stats.get("views")) if stats is not None else None,
        })
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("handle")
    p.add_argument("--subs", type=float, required=True)
    p.add_argument("--type", dest="content_type", default="education", choices=CONTENT_TYPES)
    p.add_argument("--copy-to", help="also copy thumbnails here (e.g. ~/Desktop/instinct-demo)")
    args = p.parse_args()

    out_dir = DATA_DIR / "channel_demo" / args.handle.lstrip("@")
    out_dir.mkdir(parents=True, exist_ok=True)
    videos = channel_tabs(args.handle, out_dir)
    if len(videos) == 0:  # the tab pages did not render; the feed still gives the last 15
        cid = resolve_channel_id(args.handle)
        videos = feed(cid, out_dir / "feed.xml")
        videos["kind"] = "unknown"
        videos["duration_seconds"] = None
    videos["view_count"] = videos["views"] if "views" in videos.columns else videos["view_count"]
    n_long, n_live, n_short = (int((videos["kind"] == k).sum()) for k in ("long", "live", "short"))
    print(f"{args.handle}: {len(videos)} videos ({n_long} long-form uploads, {n_live} live streams, {n_short} Shorts)\n")
    copy_dir = Path(args.copy_to).expanduser() if args.copy_to else None
    if copy_dir:
        copy_dir.mkdir(parents=True, exist_ok=True)

    scorer = get_scorer()
    rows = []
    manifest = []
    for _, v in videos.iterrows():
        path = out_dir / f"{v.video_id}.jpg"
        if not path.exists():
            r = requests.get(f"https://i.ytimg.com/vi/{v.video_id}/hqdefault.jpg", headers=UA, timeout=30)
            if r.status_code == 200:
                path.write_bytes(r.content)
        safe = re.sub(r"[^\w]+", "_", v.title)[:40] + ".jpg"
        sub = {"long": "long-form", "live": "long-form", "short": "shorts"}.get(v.kind, "other")
        if copy_dir and path.exists():
            (copy_dir / sub).mkdir(exist_ok=True)
            (copy_dir / sub / safe).write_bytes(path.read_bytes())
        if path.exists():
            manifest.append({"channel": args.handle, "video_id": v.video_id, "title": v.title, "views": v.view_count,
                             "kind": v.kind, "duration_seconds": v.duration_seconds,
                             "md5": hashlib.md5(path.read_bytes()).hexdigest(), "filename": safe, "path": str(path)})
        res = scorer.score_video(v.title, str(path) if path.exists() else None, args.content_type, args.subs)
        rows.append({
            "title": v.title, "kind": v.kind, "views": v.view_count,
            "duration_seconds": v.duration_seconds,
            "title_score": res["title"].score,
            "thumb_score": res["thumbnail"].score if res["thumbnail"] else None,
            "overall": res["overall"].score,
            "top_note": res["title"].notes[0] if res["title"].notes else "",
        })
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "scored.csv", index=False)
    pd.DataFrame(manifest).to_csv(out_dir / "manifest.csv", index=False)  # the app reads this to auto-fill titles
    for kinds, name in ((("long", "live"), "batch_long_form.csv"), (("short",), "batch_shorts.csv")):
        part = table[table.kind.isin(kinds)]
        if len(part):
            pd.DataFrame({"title": part.title, "content_type": args.content_type, "subscriber_count": int(args.subs)}).to_csv(out_dir / name, index=False)
            if copy_dir:
                (copy_dir / name).write_bytes((out_dir / name).read_bytes())

    from scipy.stats import spearmanr

    for kinds, label in ((("long", "live"), "LONG-FORM (uploads + live streams)"), (("short",), "SHORTS")):
        part = table[table.kind.isin(kinds)].sort_values("overall", ascending=False)
        if not len(part):
            continue
        print(f"== {label} ({len(part)}) ==")
        with pd.option_context("display.width", 200, "display.max_colwidth", 60):
            print(part[["overall", "title_score", "thumb_score", "views", "duration_seconds", "title"]].to_string(index=False))
        known = part.dropna(subset=["views"])
        if len(known) >= 4:
            rho = spearmanr(known["overall"], known["views"]).correlation
            rho_t = spearmanr(known["title_score"], known["views"]).correlation
            print(f"Spearman model vs actual views: {rho:+.2f}  (title alone {rho_t:+.2f}, n={len(known)})\n")
    print(f"\nwrote {out_dir / 'scored.csv'} and {out_dir / 'batch.csv'}")
    if scorer.status.synthetic:
        print("note: the models are trained on synthetic data; agreement with real views is a bonus, not a claim.")


if __name__ == "__main__":
    main()
