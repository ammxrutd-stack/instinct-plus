"""Generate a synthetic Instinct+ dataset so the pipeline runs without API keys.

This exists for three reasons: to develop and test the app and the training
code before collection has finished, to run the test suite in CI without
quota or a Drive download, and to give a reviewer a one-command demo.

The titles are template-filled per content type, the thumbnails are drawn with
PIL, and views are produced by a hidden "reach" function of a few title and
image properties plus channel and noise terms. The hidden function is written
down in reach_from_properties() so that anyone can check what the models
should be able to recover.

**Models trained on this data know nothing about YouTube.** They learn the
rules in this file. The model card records data_source="synthetic" and the
app shows a banner. Replace with collected data before trusting a score.

Usage:
    python -m src.data.synthetic                  # ~3,400 videos
    python -m src.data.synthetic --channels 6     # smaller
"""

import argparse
import math
import random
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from config.schema import (
    CHANNEL_ID,
    CHANNEL_TITLE,
    COLUMNS,
    CONTENT_TYPE,
    CONTENT_TYPES,
    DATA_DIR,
    DURATION_SECONDS,
    MAX_DAYS_SINCE_PUBLISH,
    MIN_DAYS_SINCE_PUBLISH,
    PROCESSED_DIR,
    PUBLISHED_AT,
    SUBSCRIBER_COUNT,
    THUMBNAIL_FILE,
    THUMBNAIL_PATTERN,
    THUMBNAILS_DIR,
    TITLE,
    VIDEO_ID,
    VIDEOS_TABLE,
    VIDEOS_TABLE_CSV,
    VIEW_COUNT,
)

MARKER = "SYNTHETIC"  # written next to the table so training can label the source

VOCAB: Dict[str, Dict[str, List[str]]] = {
    "gaming": {
        "topic": ["Elden Ring", "Minecraft", "Zelda", "Fortnite", "Baldur's Gate 3", "Hollow Knight",
                  "Stardew Valley", "GTA 6", "Helldivers 2", "Dark Souls", "Mario Kart", "Silksong",
                  "Pokemon", "Cyberpunk 2077", "Skyrim", "Valorant", "Palworld", "the new Zelda"],
        "noun": ["boss", "speedrun", "build", "mod", "glitch", "ending", "update", "DLC", "challenge", "run"],
        "verb": ["beat", "broke", "finished", "ruined", "speedran", "survived", "100%ed", "modded"],
        "adj": ["hardest", "weirdest", "most broken", "impossible", "cursed", "forgotten", "underrated"],
    },
    "cooking": {
        "topic": ["sourdough", "ramen", "carbonara", "fried chicken", "croissants", "smash burgers",
                  "birria tacos", "pad thai", "brisket", "cheesecake", "pizza dough", "butter chicken",
                  "dumplings", "steak", "focaccia", "mac and cheese", "pho", "bagels"],
        "noun": ["recipe", "technique", "method", "secret", "mistake", "shortcut", "version", "test"],
        "verb": ["perfected", "ruined", "tested", "fixed", "made", "cooked", "mastered", "remade"],
        "adj": ["crispiest", "easiest", "cheapest", "fastest", "most expensive", "lazy", "restaurant-style"],
    },
    "finance": {
        "topic": ["index funds", "the housing market", "credit cards", "Bitcoin", "inflation", "your 401k",
                  "dividend stocks", "the Fed", "student loans", "real estate", "side hustles", "retirement",
                  "the stock market", "Roth IRAs", "a recession", "car loans", "Tesla stock", "budgeting"],
        "noun": ["mistake", "rule", "strategy", "trap", "plan", "truth", "myth", "lesson", "habit"],
        "verb": ["explained", "ranked", "tested", "broke down", "analyzed", "exposed", "reviewed"],
        "adj": ["biggest", "dumbest", "simplest", "most dangerous", "underrated", "overrated", "honest"],
    },
    "education": {
        "topic": ["black holes", "the Roman Empire", "nuclear fusion", "the immune system", "quantum computers",
                  "the Bronze Age collapse", "evolution", "the speed of light", "volcanoes", "prime numbers",
                  "the ocean floor", "ancient Egypt", "antibiotics", "jet engines", "the brain", "sleep",
                  "earthquakes", "the Moon landing"],
        "noun": ["mystery", "problem", "experiment", "paradox", "discovery", "theory", "question", "mistake"],
        "verb": ["explained", "solved", "tested", "visualized", "simulated", "debunked", "measured"],
        "adj": ["strangest", "most important", "hardest", "most misunderstood", "oldest", "deadliest"],
    },
}

CHANNEL_NAMES = {
    "gaming": ["PixelPete", "LootGoblin", "QuestLog", "RespawnRadio", "GlitchGirl", "DungeonDan",
               "ControllerChaos", "SpeedrunSam", "LagSpike", "BossRushBen", "ModMaven", "IndieIvy",
               "FrameDrop", "SaveScum", "NoScopeNina"],
    "cooking": ["ButterAndSalt", "ChefTomasz", "WokTalk", "TheLazyBaker", "FlourPower", "GrillMaster Gus",
                "SpoonAndStory", "KitchenKat", "DoughBoyDev", "PanSearPaul", "UmamiUma", "BrothBros",
                "CrumbAndCo", "SizzleSid", "ProofedPriya"],
    "finance": ["MoneyWithMaya", "TheFrugalFox", "DividendDave", "LedgerLuke", "CashflowCara", "IndexIan",
                "BrokeToBoring", "CompoundCorner", "FedWatch", "RentVsBuy", "TaxTalkTina", "BullBearBob",
                "PennyPilot", "SideHustleSue", "YieldYard"],
    "education": ["CuriousCosmos", "HistoryHal", "LabCoatLena", "AtomicAnna", "DeepTimeDoc", "BrainBytes",
                  "RocketRhea", "MapRoom", "ThinkTankTheo", "CellCity", "StrataStan", "OrbitOllie",
                  "WhyMachine", "FactoryFloor", "MathMinute"],
}

MIN_SUBS, MAX_SUBS = 15_000, 40_000_000


def pick(rng: random.Random, items: List[str]) -> str:
    return rng.choice(items)


def make_title(rng: random.Random, ct: str) -> str:
    v = VOCAB[ct]
    topic, noun, verb, adj = pick(rng, v["topic"]), pick(rng, v["noun"]), pick(rng, v["verb"]), pick(rng, v["adj"])
    n = rng.choice([3, 5, 7, 10, 12, 15, 20, 24, 30, 50, 100])
    templates = [
        f"I {verb} {topic} and it changed everything",
        f"The {adj} {topic} {noun} nobody talks about",
        f"{n} {topic} {noun}s you need to know",
        f"Why {topic} is harder than you think",
        f"How to master {topic} in {n} minutes",
        f"We {verb} {topic} for {n} days. Here's what happened",
        f"{topic}: the {noun} everyone gets wrong",
        f"Is {topic} actually worth it?",
        f"{topic} {noun} ({adj} version)",
        f"The truth about {topic}",
        f"{topic} vs {pick(rng, v['topic'])}: which is better?",
        f"I spent ${n * 100} on {topic} so you don't have to",
        f"{topic} {noun}",
        f"My {adj} {topic} {noun} yet",
        f"Stop making this {topic} {noun}",
        f"{topic} explained in {n} minutes",
        f"What {n} years of {topic} taught me",
        f"The {n}-minute {topic} {noun} | full guide",
        f"This {topic} {noun} should be illegal",
        f"{topic} {noun} tier list [{adj}]",
        f"a quiet video about {topic}",
        f"{topic} - {noun} - part {rng.randint(1, 9)}",
    ]
    t = pick(rng, templates)
    style = rng.random()
    if style < 0.12:
        t = t.upper()
    elif style < 0.45:
        words = t.split()
        # shout one or two words, a common thumbnail-title habit
        for _ in range(rng.choice([1, 1, 2])):
            i = rng.randrange(len(words))
            if words[i].isalpha():
                words[i] = words[i].upper()
        t = " ".join(words)
    elif style < 0.75:
        t = t.title().replace("'S", "'s").replace("Vs", "vs")
    if rng.random() < 0.08:
        t += "!"
    if rng.random() < 0.05:
        t += " 😱" if rng.random() < 0.5 else " 🔥"
    return t


# ----------------------------------------------------------------- images --

def _font(size: int):
    try:
        return ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", size)
    except Exception:
        try:
            return ImageFont.load_default(size=size)
        except TypeError:  # older Pillow
            return ImageFont.load_default()


def draw_face(d: ImageDraw.ImageDraw, cx: int, cy: int, r: int, rng: random.Random):
    """A cartoon face shaped for the Haar cascade: a dark eye band above
    lighter cheeks, a bright nose bridge, a dark mouth. Detection on drawings
    is still far below real photographs; that is fine for a stand-in."""
    skin = rng.choice([(236, 188, 160), (205, 150, 110), (160, 110, 80), (250, 215, 190), (120, 80, 55)])
    shade = tuple(max(0, c - 70) for c in skin)
    d.ellipse((cx - r, cy - int(r * 1.25), cx + r, cy + int(r * 1.25)), fill=skin, outline=(60, 40, 30), width=2)
    hair = rng.choice([(40, 30, 20), (90, 60, 30), (20, 20, 20), (200, 170, 90)])
    d.chord((cx - r, cy - int(r * 1.25), cx + r, cy + int(r * 0.05)), 180, 360, fill=hair)
    ey = cy - int(r * 0.28)
    ex = int(r * 0.42)
    er = max(4, int(r * 0.2))
    # dark eye sockets (the band the cascade expects), with a small highlight
    for sx in (-1, 1):
        d.ellipse((cx + sx * ex - er - 4, ey - er, cx + sx * ex + er + 4, ey + er), fill=shade)
        d.ellipse((cx + sx * ex - er + 1, ey - er + 2, cx + sx * ex + er - 1, ey + er - 2), fill=(30, 25, 30))
        d.ellipse((cx + sx * ex - 2, ey - er // 2 - 2, cx + sx * ex + 2, ey - er // 2 + 2), fill=(240, 240, 240))
        d.line((cx + sx * ex - er - 6, ey - er - 5, cx + sx * ex + er + 6, ey - er - 9), fill=hair, width=max(3, r // 9))
    # bright nose bridge between the eyes, shadow under the nose
    d.line((cx, ey, cx, cy + int(r * 0.25)), fill=tuple(min(255, c + 25) for c in skin), width=max(3, r // 8))
    d.ellipse((cx - int(r * 0.18), cy + int(r * 0.18), cx + int(r * 0.18), cy + int(r * 0.38)), fill=shade)
    my = cy + int(r * 0.65)
    d.ellipse((cx - int(r * 0.42), my - int(r * 0.16), cx + int(r * 0.42), my + int(r * 0.16)), fill=(70, 25, 35))
    d.rectangle((cx - int(r * 0.3), my - int(r * 0.12), cx + int(r * 0.3), my - int(r * 0.02)), fill=(235, 235, 235))


def make_thumbnail(rng: random.Random, props: Dict[str, float], path: Path):
    """Render a 480x360 hqdefault-style JPEG. props are the hidden properties."""
    W, H = 480, 360
    img = Image.new("RGB", (W, H), (0, 0, 0))
    frame_h = 270 if props["letterbox"] else H
    top = (H - frame_h) // 2
    hue = rng.random()
    sat = props["saturation"]
    val = props["brightness"]
    base = _hsv(hue, sat, val)
    frame = Image.new("RGB", (W, frame_h), base)
    d = ImageDraw.Draw(frame)
    # background texture: a few blobs, more when "busy"
    for _ in range(int(props["busyness"] * 14)):
        bx, by = rng.randrange(W), rng.randrange(frame_h)
        br = rng.randrange(20, 140)
        d.ellipse((bx - br, by - br, bx + br, by + br), fill=_hsv((hue + rng.uniform(-0.2, 0.2)) % 1, sat * rng.uniform(0.5, 1.0), min(1.0, val * rng.uniform(0.6, 1.3))))
    cx = W // 2
    if props["face"]:
        r = int(rng.uniform(0.11, 0.2) * W * props["face_size"])
        cx = int(W * rng.choice([0.25, 0.3, 0.7, 0.75]))
        cy = int(frame_h * rng.uniform(0.45, 0.6))
        draw_face(d, cx, cy, r, rng)
    if props["text_words"]:
        words = [w.upper() for w in rng.sample(["BEST", "WORST", "NEW", "WHY", "HOW", "FREE", "FAST", "EASY", "SECRET", "TRUTH", "100%", "NEVER", "WOW", "BIG", "TINY"], props["text_words"])]
        size = int(props["text_size"] * 110)
        f = _font(size)
        y = int(frame_h * rng.uniform(0.08, 0.2)) if props["face"] else int(frame_h * 0.3)
        face_right = props["face"] and cx > W / 2
        for w in words:
            x = int(W * rng.uniform(0.04, 0.1)) if (not props["face"] or face_right) else int(W * 0.55)
            fill = (255, 255, 255) if val < 0.6 else rng.choice([(255, 230, 0), (255, 255, 255), (20, 20, 20)])
            d.text((x + 3, y + 3), w, font=f, fill=(0, 0, 0))
            d.text((x, y), w, font=f, fill=fill)
            y += int(size * 1.1)
    img.paste(frame, (0, top))
    img.save(path, "JPEG", quality=85)


def _hsv(h: float, s: float, v: float) -> Tuple[int, int, int]:
    import colorsys

    r, g, b = colorsys.hsv_to_rgb(h % 1.0, max(0, min(1, s)), max(0, min(1, v)))
    return int(r * 255), int(g * 255), int(b * 255)


# ------------------------------------------------------------- the rules --

def reach_from_properties(title: str, img: Dict[str, float], ct: str) -> float:
    """The hidden function. Reach in log10(views/subs) units before noise."""
    from src.title.features import extract_title_features

    f = extract_title_features(title, ct)
    r = {"gaming": -0.55, "cooking": -0.75, "finance": -0.6, "education": -0.45}[ct]
    r -= 0.008 * max(0.0, abs(f["char_count"] - 50) - 10)
    r += 0.22 * f["has_number"]
    r += 0.15 * f["is_question"]
    r += 0.25 * min(f["curiosity_phrase"], 1)
    r += 0.12 * min(f["power_word_count"], 2)
    r += 0.12 * min(f["second_person"], 1)
    r -= 0.10 * max(0, f["exclamations"] - 1)
    if ct == "gaming":
        r += 0.2 if 0.1 <= f["caps_word_ratio"] <= 0.5 else 0.0
    if ct == "finance":
        r -= 0.22 if f["caps_word_ratio"] > 0.3 else 0.0
        r += 0.15 * f["has_money"]
    if ct in ("cooking", "education"):
        r += 0.18 * f["how_to"]
    r += 0.10 * f["has_brackets"]
    # image rules
    r += 0.28 * img["face"] * (0.6 + 0.4 * img["face_size"])
    text = img["text_words"] * img["text_size"]
    r += 0.14 if 0 < text <= 1.6 else (-0.12 if text > 1.6 else 0.0)
    r -= 0.5 * (img["brightness"] - 0.6) ** 2
    r += 0.15 * (img["saturation"] - 0.5)
    r -= 0.18 * (1.0 - img["letterbox"])  # no bars in hqdefault = a 4:3 upload
    r -= 0.10 * max(0.0, img["busyness"] - 0.6)
    return r


def sample_image_props(rng: random.Random, ct: str) -> Dict[str, float]:
    face_rate = {"gaming": 0.55, "cooking": 0.35, "finance": 0.7, "education": 0.3}[ct]
    return {
        "face": float(rng.random() < face_rate),
        "face_size": rng.uniform(0.6, 1.4),
        "text_words": rng.choice([0, 0, 1, 1, 2, 2, 3]),
        "text_size": rng.uniform(0.4, 1.0),
        "brightness": min(1.0, max(0.12, rng.gauss(0.6, 0.2))),
        "saturation": min(1.0, max(0.05, rng.gauss(0.55, 0.22))),
        "letterbox": float(rng.random() < 0.7),  # most uploads are 16:9, so most hqdefaults have bars
        "busyness": rng.random(),
    }


def make_synthetic_dataset(
    channels_per_type: int = 14,
    videos_per_channel: int = 60,
    seed: int = 0,
    data_dir: Path = DATA_DIR,
    thumbnails: bool = True,
    now: datetime = None,
) -> pd.DataFrame:
    rng = random.Random(seed)
    now = now or datetime.now(timezone.utc)
    processed = data_dir / "processed"
    thumbs = data_dir / "thumbnails"
    processed.mkdir(parents=True, exist_ok=True)
    thumbs.mkdir(parents=True, exist_ok=True)

    rows = []
    counter = 0
    for ct in CONTENT_TYPES:
        names = CHANNEL_NAMES[ct][:channels_per_type]
        for ci, name in enumerate(names):
            # log-uniform subscriber counts so small and huge channels both appear
            subs = int(10 ** rng.uniform(math.log10(MIN_SUBS), math.log10(MAX_SUBS)))
            subs = int(float(f"{subs:.3g}"))  # the API rounds to 3 significant figures
            channel_id = f"UCsyn{ct[:3]}{ci:02d}" + "x" * 14
            channel_effect = rng.gauss(0, 0.2)
            # bigger channels reach proportionally fewer of their subscribers
            size_effect = -0.18 * (math.log10(subs) - 5.5)
            for _ in range(videos_per_channel):
                counter += 1
                video_id = f"syn{counter:07d}xx"
                title = make_title(rng, ct)
                props = sample_image_props(rng, ct)
                age = rng.uniform(MIN_DAYS_SINCE_PUBLISH, MAX_DAYS_SINCE_PUBLISH)
                latent = reach_from_properties(title, props, ct) + channel_effect + size_effect
                latent += 0.3 * (math.log10(age) - math.log10(90))  # older = more views
                latent += rng.gauss(0, 0.25)
                views = int(max(50, subs * 10 ** latent))
                thumb_name = THUMBNAIL_PATTERN.format(video_id=video_id)
                if thumbnails:
                    make_thumbnail(rng, props, thumbs / thumb_name)
                rows.append({
                    VIDEO_ID: video_id,
                    TITLE: title,
                    CHANNEL_ID: channel_id,
                    CHANNEL_TITLE: name,
                    SUBSCRIBER_COUNT: subs,
                    VIEW_COUNT: views,
                    PUBLISHED_AT: (now - timedelta(days=age)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    DURATION_SECONDS: int(rng.uniform(120, 1800)),
                    CONTENT_TYPE: ct,
                    THUMBNAIL_FILE: thumb_name if thumbnails else "",
                })
    table = pd.DataFrame(rows, columns=COLUMNS)
    table.to_parquet(processed / VIDEOS_TABLE.name, index=False)
    table.to_csv(processed / VIDEOS_TABLE_CSV.name, index=False)
    (processed / MARKER).write_text(
        f"generated by src/data/synthetic.py seed={seed} rows={len(table)} at {now.isoformat()}\n"
    )
    return table


def is_synthetic(data_dir: Path = DATA_DIR) -> bool:
    return (data_dir / "processed" / MARKER).exists()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--channels", type=int, default=14, help="channels per content type (max 15)")
    parser.add_argument("--videos", type=int, default=60, help="videos per channel")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-thumbs", action="store_true")
    parser.add_argument("--force", action="store_true", help="overwrite an existing non-synthetic table")
    args = parser.parse_args()

    if VIDEOS_TABLE.exists() and not is_synthetic() and not args.force:
        raise SystemExit(
            f"{VIDEOS_TABLE} exists and is not marked synthetic. Refusing to overwrite "
            "collected data; pass --force if you really mean it."
        )
    if is_synthetic():
        for p in THUMBNAILS_DIR.glob("syn*.jpg"):
            p.unlink()
    table = make_synthetic_dataset(min(args.channels, 15), args.videos, args.seed, thumbnails=not args.no_thumbs)
    print(f"wrote {len(table):,} synthetic videos across {table[CHANNEL_ID].nunique()} channels")
    print(f"  {VIDEOS_TABLE.relative_to(VIDEOS_TABLE.parents[2])}")
    if not args.no_thumbs:
        print(f"  {len(table):,} thumbnails in {THUMBNAILS_DIR.relative_to(THUMBNAILS_DIR.parents[1])}/")
    print("This data is SYNTHETIC: models trained on it demonstrate the pipeline, not YouTube.")


if __name__ == "__main__":
    main()
