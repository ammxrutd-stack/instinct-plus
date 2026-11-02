"""Turn drivers and features into sentences a creator can act on.

A driver says "has_number cost you 4 points". A note says "Add a concrete
number: a count, a price or a timespan. Titles with one score higher in
finance." The mapping from feature to advice lives here, in one place, so the
app and the generation prompt tell the same story.

Benchmarks (per-content-type feature percentiles stored in the model) let a
note say where the creator stands against their category rather than quoting
an absolute threshold.
"""

from typing import Callable, Dict, List, Optional

TITLE_LABELS: Dict[str, str] = {
    "char_count": "title length",
    "word_count": "word count",
    "avg_word_len": "average word length",
    "longest_word_len": "longest word",
    "has_number": "a number in the title",
    "starts_with_number": "leading with a number",
    "number_count": "how many numbers",
    "has_year": "mentioning a year",
    "has_money": "mentioning money",
    "is_question": "question phrasing",
    "question_marks": "question marks",
    "exclamations": "exclamation marks",
    "starts_with_question_word": "opening with why/how/what",
    "caps_word_count": "ALL-CAPS words",
    "caps_word_ratio": "share of ALL-CAPS words",
    "upper_letter_ratio": "share of capital letters",
    "title_case_ratio": "Title Case",
    "has_brackets": "a [bracketed] tag",
    "has_parentheses": "a (parenthetical)",
    "has_pipe": "a | separator",
    "has_colon": "a colon",
    "has_dash": "a dash",
    "has_ellipsis": "an ellipsis",
    "has_quotes": "quotation marks",
    "punctuation_count": "punctuation",
    "emoji_count": "emoji",
    "first_person": "first person (I / my)",
    "second_person": "speaking to the viewer (you / your)",
    "power_word_count": "power words",
    "curiosity_phrase": "a curiosity hook",
    "urgency_word_count": "urgency words",
    "negation_count": "negative words",
    "superlative_count": "superlatives",
    "how_to": "how-to framing",
    "is_listicle": "list format",
    "is_versus": "a comparison (vs)",
    "stopword_ratio": "filler words",
    "unique_word_ratio": "word variety",
    "sentiment_compound": "sentiment",
    "sentiment_pos": "positive tone",
    "sentiment_neg": "negative tone",
    "log_subscribers": "channel size",
}

THUMBNAIL_LABELS: Dict[str, str] = {
    "brightness_mean": "brightness",
    "brightness_std": "range of light and dark",
    "contrast": "contrast",
    "dark_fraction": "dark area",
    "bright_fraction": "blown-out highlights",
    "saturation_mean": "colour saturation",
    "saturation_std": "saturation variety",
    "colorfulness": "colourfulness",
    "red_share": "red",
    "orange_yellow_share": "orange and yellow",
    "green_share": "green",
    "blue_share": "blue",
    "purple_pink_share": "purple and pink",
    "warm_share": "warm colours",
    "dominant_colours": "number of distinct colours",
    "edge_density": "visual busyness",
    "centre_edge_density": "detail in the centre",
    "sharpness": "sharpness",
    "entropy": "tonal range",
    "symmetry": "symmetry",
    "face_count": "number of faces",
    "face_present": "a visible face",
    "largest_face_area": "face size",
    "face_centre_x": "face position (left/right)",
    "face_centre_y": "face position (up/down)",
    "text_area": "area covered by text",
    "text_block_count": "text blocks",
    "letterbox_fraction": "letterbox bars",
    "aspect_ratio": "aspect ratio",
    "image_missing": "missing image",
    "log_subscribers": "channel size",
}


def label(feature: str, labels: Dict[str, str]) -> str:
    return labels.get(feature, feature.replace("_", " "))


CONTEXT = {"log_subscribers"}

FAMILIES: Dict[str, str] = {
    "has_number": "number", "number_count": "number", "starts_with_number": "number", "has_year": "number",
    "char_count": "length", "word_count": "length", "avg_word_len": "length", "longest_word_len": "length",
    "stopword_ratio": "length", "unique_word_ratio": "length",
    "caps_word_count": "caps", "caps_word_ratio": "caps", "upper_letter_ratio": "caps", "title_case_ratio": "caps",
    "is_question": "question", "question_marks": "question", "starts_with_question_word": "question",
    "sentiment_compound": "tone", "sentiment_pos": "tone", "sentiment_neg": "tone",
    "has_brackets": "tag", "has_parentheses": "tag", "has_pipe": "tag", "has_colon": "tag", "has_dash": "tag",
    "punctuation_count": "tag", "has_quotes": "tag", "has_ellipsis": "tag",
    "face_count": "face", "face_present": "face", "largest_face_area": "face", "face_centre_x": "face", "face_centre_y": "face",
    "text_area": "text", "text_block_count": "text",
    "brightness_mean": "light", "brightness_std": "light", "dark_fraction": "light", "bright_fraction": "light", "contrast": "light",
    "saturation_mean": "colour", "saturation_std": "colour", "colorfulness": "colour", "red_share": "colour",
    "orange_yellow_share": "colour", "green_share": "colour", "blue_share": "colour", "purple_pink_share": "colour",
    "warm_share": "colour", "dominant_colours": "colour",
    "edge_density": "busy", "centre_edge_density": "busy", "entropy": "busy", "sharpness": "busy", "symmetry": "busy",
}


def family(feature: str) -> str:
    return FAMILIES.get(feature, feature)


def visible_drivers(drivers: Dict[str, float], k: int = 8, min_points: float = 0.5) -> Dict[str, float]:
    """The drivers worth showing: not the content-type one-hots, not channel
    size (context the creator cannot change), not noise."""
    items = [(f, v) for f, v in drivers.items() if not f.startswith("ct_") and f not in CONTEXT and abs(v) >= min_points]
    items.sort(key=lambda kv: -abs(kv[1]))
    return dict(items[:k])


def _pct(value: float, bench: Optional[Dict[str, float]]) -> Optional[int]:
    """Rough percentile of value against a benchmark dict (p10..p90)."""
    if not bench:
        return None
    pts = [(10, bench["p10"]), (25, bench["p25"]), (50, bench["p50"]), (75, bench["p75"]), (90, bench["p90"])]
    pct = 5
    for p, cut in pts:
        if value >= cut:
            pct = p
    return pct


Benchmark = Callable[[str], Optional[Dict[str, float]]]


def title_notes(
    features: Dict[str, float],
    drivers: Dict[str, float],
    content_type: str,
    benchmark: Benchmark,
    max_notes: int = 5,
) -> List[str]:
    notes: List[str] = []
    f = features
    shown = visible_drivers(drivers)
    negatives = [(k, v) for k, v in shown.items() if v < 0]
    positives = [(k, v) for k, v in shown.items() if v > 0]

    chars = int(f.get("char_count", 0))
    length_bench = benchmark("char_count")
    pct = _pct(chars, length_bench)
    if length_bench and pct is not None and pct >= 90:
        notes.append(
            f"At {chars} characters this is longer than 90% of {content_type} titles "
            f"(typical: {length_bench['p50']:.0f}). YouTube cuts titles at roughly 70 characters in search and suggested videos."
        )
    elif length_bench and pct is not None and pct <= 10 and chars > 0:
        notes.append(
            f"At {chars} characters this is shorter than 90% of {content_type} titles "
            f"(typical: {length_bench['p50']:.0f}). Add the specific detail that makes it worth clicking."
        )

    hints = {
        "has_number": lambda: "Add a concrete number: a count, a price, a timespan or a result." if f.get("has_number", 0) == 0 else "The number is not helping here; try the title without it.",
        "is_question": lambda: "Try phrasing it as a question the video answers." if f.get("is_question", 0) == 0 else "Question phrasing is costing points in this category; state the payoff instead.",
        "curiosity_phrase": lambda: "Open a loop: what happened, what went wrong, what nobody tells you." if f.get("curiosity_phrase", 0) == 0 else "The curiosity hook reads as clickbait for this category; be more concrete.",
        "caps_word_ratio": lambda: "Fewer ALL-CAPS words. One emphasised word reads as energy; several read as shouting." if f.get("caps_word_ratio", 0) > 0 else "One ALL-CAPS word for emphasis is normal in this category.",
        "caps_word_count": lambda: "Fewer ALL-CAPS words. One emphasised word reads as energy; several read as shouting.",
        "upper_letter_ratio": lambda: "Tone down the capitalisation.",
        "second_person": lambda: "Speak to the viewer: 'you' or 'your' tells them the video is about them." if f.get("second_person", 0) == 0 else "Less 'you'; make the subject the thing itself.",
        "power_word_count": lambda: "Add one strong word (secret, mistake, truth, best, never), not three." if f.get("power_word_count", 0) == 0 else "Too many power words; keep the strongest one.",
        "exclamations": lambda: "Drop the exclamation marks.",
        "how_to": lambda: "'How to' framing works in this category; lead with the outcome the viewer gets." if f.get("how_to", 0) == 0 else "'How to' is underperforming here; try a claim or a result instead.",
        "has_money": lambda: "Name the amount: a price or a sum anchors a finance title.",
        "has_brackets": lambda: "A short [bracketed] tag (format, part number, year) can help." if f.get("has_brackets", 0) == 0 else "Drop the bracketed tag.",
        "char_count": lambda: "Shorten it; keep the hook in the first 50 characters." if f.get("char_count", 0) > 60 else "It is short; add the specific detail that makes it worth clicking.",
        "word_count": lambda: "Fewer words; every word should earn its place." if f.get("word_count", 0) > 10 else "Add a few more specific words.",
        "stopword_ratio": lambda: "Cut filler words (the, of, a, and).",
        "emoji_count": lambda: "Remove the emoji.",
        "sentiment_neg": lambda: "The tone is negative; this category rewards a positive framing." ,
        "sentiment_compound": lambda: "Shift the tone; a stronger emotional angle helps here." ,
        "is_listicle": lambda: "A numbered list ('7 ways...') tends to do well here." if f.get("is_listicle", 0) == 0 else "List format is underperforming; try a single strong claim.",
        "is_versus": lambda: "A head-to-head ('X vs Y') comparison performs in this category." if f.get("is_versus", 0) == 0 else "Comparison framing is underperforming here.",
        "log_subscribers": lambda: "Larger channels reach a smaller share of their subscribers; that is the channel size effect, not the title.",
    }
    used = {"length"} if notes else set()
    for feature, points in negatives:
        fam = family(feature)
        if fam in used:
            continue
        used.add(fam)
        hint = hints.get(feature)
        text = hint() if hint else f"{label(feature, TITLE_LABELS).capitalize()} is holding the score back."
        notes.append(f"{text} ({points:+.0f} points)")
        if len([n for n in notes if "points)" in n]) >= 3:
            break
    if positives:
        best = ", ".join(f"{label(k, TITLE_LABELS)} ({v:+.0f})" for k, v in positives[:2])
        notes.append(f"Working for you: {best}.")
    if not negatives and not positives:
        notes.append("No single feature stands out; the title is close to typical for its category.")
    return notes[:max_notes]


def thumbnail_notes(
    features: Dict[str, float],
    drivers: Dict[str, float],
    content_type: str,
    benchmark: Benchmark,
    max_notes: int = 5,
) -> List[str]:
    notes: List[str] = []
    f = features
    if f.get("image_missing", 0):
        return ["The image could not be read. Upload a JPEG or PNG."]
    shown = visible_drivers(drivers)
    negatives = [(k, v) for k, v in shown.items() if v < 0]
    positives = [(k, v) for k, v in shown.items() if v > 0]

    ar = f.get("aspect_ratio", 1.78)
    if abs(ar - 1.78) > 0.12:
        notes.append(f"The picture is {ar:.2f}:1; YouTube displays 16:9 (1.78:1) and pads or crops anything else. Export at 1280x720.")
    text_bench = benchmark("text_area")
    text_pct = _pct(f.get("text_area", 0), text_bench)
    if text_bench and text_pct is not None and text_pct >= 90 and f.get("text_area", 0) > 0.15:
        notes.append("More text than 90% of thumbnails in this category. Keep it to two or three large words; the title carries the rest.")

    hints = {
        "face_present": lambda: "Add a face. A clear, expressive face close to the camera is the strongest single thumbnail signal." if f.get("face_present", 0) == 0 else "The face is not helping here; try the subject itself larger in frame.",
        "face_count": lambda: "One face, not several: the viewer's eye needs a single subject." if f.get("face_count", 0) > 1 else "Add a face.",
        "largest_face_area": lambda: "Make the face bigger: crop in so it fills a quarter of the frame or more." if f.get("largest_face_area", 0) < 0.08 else "The face dominates; pull back slightly so the context shows.",
        "text_area": lambda: ("A two or three word caption in large type would help." if f.get("text_area", 0) == 0 else "Cut the text to two or three words in large type."),
        "text_block_count": lambda: "Fewer separate text blocks; one caption.",
        "brightness_mean": lambda: "Brighten it: dark thumbnails disappear in the feed." if f.get("brightness_mean", 0) < 0.45 else "It is very bright; add a darker element for the eye to land on.",
        "dark_fraction": lambda: "Too much of the frame is dark; lift the exposure or add a lighter background.",
        "bright_fraction": lambda: "Large blown-out areas; bring the highlights down.",
        "contrast": lambda: "Increase contrast between the subject and the background." if f.get("contrast", 0) < 0.25 else "Very high contrast; soften it slightly.",
        "colorfulness": lambda: "Add a saturated accent colour; muted thumbnails get overlooked." if f.get("colorfulness", 0) < 40 else "Very colourful; one dominant colour with a single accent reads cleaner.",
        "saturation_mean": lambda: "Boost saturation a little." if f.get("saturation_mean", 0) < 0.35 else "Pull saturation back; it reads as garish.",
        "edge_density": lambda: "Too busy: simplify the background so the subject stands out." if f.get("edge_density", 0) > 0.1 else "Very plain; add a focal element.",
        "centre_edge_density": lambda: "Put the subject in the centre third; the middle of the frame is empty." if f.get("centre_edge_density", 0) < 0.03 else "The centre is cluttered; give the subject room.",
        "dominant_colours": lambda: "Too many competing colours; limit the palette to two or three." if f.get("dominant_colours", 0) > 3 else "Add a second colour for contrast.",
        "warm_share": lambda: "Warm colours (red, orange, yellow) tend to perform in this category." if f.get("warm_share", 0) < 0.3 else "Try a cooler palette; warm tones are overused here.",
        "red_share": lambda: "Try a red accent; it tends to perform here." if f.get("red_share", 0) < 0.1 else "Less red; it is overused in this category.",
        "orange_yellow_share": lambda: "Try a yellow or orange accent." if f.get("orange_yellow_share", 0) < 0.1 else "Less orange and yellow; try a cooler dominant colour.",
        "green_share": lambda: "A green element could help here." if f.get("green_share", 0) < 0.1 else "Less green; it is underperforming in this category.",
        "blue_share": lambda: "A blue element could help here." if f.get("blue_share", 0) < 0.1 else "Less blue; try a warmer dominant colour.",
        "purple_pink_share": lambda: "A purple or pink accent could help here." if f.get("purple_pink_share", 0) < 0.1 else "Less purple and pink.",
        "saturation_std": lambda: "Mix a saturated subject with a muted background so it pops.",
        "brightness_std": lambda: "Add tonal contrast: a bright subject on a darker background or the reverse.",
        "entropy": lambda: "Flat tones; add light and shadow." if f.get("entropy", 0) < 4 else "Very busy tonally; simplify.",
        "face_centre_x": lambda: "Move the face off-centre, leaving room for a caption on the other side.",
        "face_centre_y": lambda: "Move the face up so the eyes sit in the top third.",
        "sharpness": lambda: "The image is soft; use a sharper source.",
        "symmetry": lambda: "Break the symmetry: place the subject off-centre." if f.get("symmetry", 0) > 0.85 else "",
        "log_subscribers": lambda: "Larger channels reach a smaller share of their subscribers; that is the channel size effect, not the image.",
    }
    used = set()
    if any("text" in n for n in notes):
        used.add("text")
    if abs(ar - 1.78) > 0.12:
        used.add("aspect_ratio")  # already explained above
    for feature, points in negatives:
        fam = family(feature)
        if fam in used:
            continue
        used.add(fam)
        hint = hints.get(feature)
        text = hint() if hint else ""
        if not text:
            text = f"{label(feature, THUMBNAIL_LABELS).capitalize()} is holding the score back."
        notes.append(f"{text} ({points:+.0f} points)")
        if len([n for n in notes if "points)" in n]) >= 3:
            break
    if positives:
        best = ", ".join(f"{label(k, THUMBNAIL_LABELS)} ({v:+.0f})" for k, v in positives[:2])
        notes.append(f"Working for you: {best}.")
    if not negatives and not positives and len(notes) == 0:
        notes.append("No single feature stands out; the thumbnail is close to typical for its category.")
    return notes[:max_notes]
