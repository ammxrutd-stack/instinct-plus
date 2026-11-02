"""Title features: a proposed title -> dict of named floats.

Everything here is cheap, deterministic and explainable: each feature has a
name a creator can act on ("your title has no number", "it is longer than 90%
of cooking titles"). See docs/feature_dictionary_title.md for definitions.

Sentiment uses VADER when installed and is zero otherwise, so the extractor
never fails on a machine missing the optional dependency; the feature simply
carries no signal there. Training and inference must agree, which is why the
model card records whether VADER was present (via FEATURE_VERSION).
"""

import math
import re
from typing import Dict, List, Optional

from config.schema import CONTENT_TYPES

FEATURE_VERSION = "title-v2"

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    _VADER = SentimentIntensityAnalyzer()
except Exception:  # pragma: no cover - optional dependency
    _VADER = None

POWER_WORDS = {
    "secret", "secrets", "ultimate", "best", "worst", "never", "insane", "crazy",
    "finally", "truth", "mistake", "mistakes", "every", "nobody", "everyone",
    "proven", "perfect", "easy", "simple", "fast", "free", "new", "real",
    "actually", "honest", "brutal", "shocking", "hidden", "banned", "illegal",
    "impossible", "genius", "dangerous", "rare", "forbidden", "exposed",
    "revealed", "wrong", "right", "only", "last", "first", "biggest", "smallest",
    "cheapest", "expensive", "rich", "broke", "million", "billion", "legendary",
    "epic", "world", "ever", "huge", "tiny", "strange", "weird", "mystery",
}
CURIOSITY_PHRASES = [
    "what happened", "you won't believe", "you wont believe", "the truth about",
    "nobody tells you", "no one tells you", "went wrong", "until", "here's why",
    "heres why", "this is why", "the reason", "what they don't", "what nobody",
    "changed everything", "i was wrong", "i tried", "i tested", "we tested",
    "i spent", "i built", "i made", "i bought", "turns out", "the problem with",
    "don't do this", "stop doing", "before you", "watch this",
]
URGENCY_WORDS = {"now", "today", "before", "last", "finally", "immediately", "hurry", "soon", "new", "2024", "2025", "2026"}
NEGATION_WORDS = {"never", "no", "not", "don't", "dont", "stop", "without", "worst", "can't", "cant", "won't", "wont", "nobody", "none"}
SUPERLATIVE_WORDS = {"best", "worst", "biggest", "smallest", "most", "least", "greatest", "fastest", "slowest", "cheapest", "hardest", "easiest", "ultimate", "perfect"}
FIRST_PERSON = {"i", "i'm", "im", "my", "me", "we", "our", "us", "i've", "ive", "mine"}
SECOND_PERSON = {"you", "your", "you're", "youre", "yours", "u", "you'll", "you've"}
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "of", "to", "in", "on", "for", "with",
    "at", "by", "from", "is", "it", "its", "this", "that", "as", "be", "are",
    "was", "were", "how", "what", "why", "when", "who", "do", "does", "did",
    "i", "you", "we", "my", "your", "so", "if", "vs", "than", "into", "about",
}

WORD_RE = re.compile(r"[A-Za-z0-9'’]+")
NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
MONEY_RE = re.compile(r"[$£€¥]\s?\d|\d\s?(?:dollars|usd|k\b|m\b)", re.I)
LISTICLE_RE = re.compile(r"^\s*\d+\s+\w+|\b\d+\s+(?:things|ways|tips|reasons|mistakes|secrets|rules|facts|ideas|hacks|steps|recipes|games|lessons|signs|habits)\b", re.I)
EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF\U00002B00-\U00002BFF]"
)
HOW_TO_RE = re.compile(r"\bhow\s+(?:to|i|we|you)\b", re.I)
VS_RE = re.compile(r"\bvs\.?\b|\bversus\b", re.I)


def _words(title: str) -> List[str]:
    return WORD_RE.findall(title)


def extract_title_features(title: str, content_type: str, subscriber_count: Optional[float] = None) -> Dict[str, float]:
    """subscriber_count is optional context: views per subscriber falls as
    channels grow, so the model needs it to place a title fairly. When None the
    scorer substitutes the training median (see ReachModel.vector)."""
    title = (title or "").strip()
    words = _words(title)
    lower_words = [w.lower() for w in words]
    n_words = len(words)
    n_chars = len(title)
    alpha_words = [w for w in words if any(c.isalpha() for c in w)]
    caps_words = [w for w in alpha_words if len(w) >= 2 and w.isupper()]
    title_case_words = [w for w in alpha_words if w[:1].isupper()]
    letters = [c for c in title if c.isalpha()]
    upper_letters = [c for c in letters if c.isupper()]
    lowered = title.lower()

    f: Dict[str, float] = {
        "char_count": float(n_chars),
        "word_count": float(n_words),
        "avg_word_len": float(sum(len(w) for w in words) / n_words) if n_words else 0.0,
        "longest_word_len": float(max((len(w) for w in words), default=0)),
        "has_number": float(bool(NUMBER_RE.search(title))),
        "starts_with_number": float(bool(re.match(r"^\s*\d", title))),
        "number_count": float(len(NUMBER_RE.findall(title))),
        "has_year": float(bool(YEAR_RE.search(title))),
        "has_money": float(bool(MONEY_RE.search(title))),
        "is_question": float(title.rstrip().endswith("?")),
        "question_marks": float(title.count("?")),
        "exclamations": float(title.count("!")),
        "starts_with_question_word": float(bool(re.match(r"^\s*(why|how|what|when|who|which|where|can|should|is|do|does|did)\b", title, re.I))),
        "caps_word_count": float(len(caps_words)),
        "caps_word_ratio": float(len(caps_words) / len(alpha_words)) if alpha_words else 0.0,
        "upper_letter_ratio": float(len(upper_letters) / len(letters)) if letters else 0.0,
        "title_case_ratio": float(len(title_case_words) / len(alpha_words)) if alpha_words else 0.0,
        "has_brackets": float("[" in title or "]" in title),
        "has_parentheses": float("(" in title or ")" in title),
        "has_pipe": float("|" in title),
        "has_colon": float(":" in title),
        "has_dash": float(" - " in title or " – " in title or " — " in title),
        "has_ellipsis": float("..." in title or "…" in title),
        "has_quotes": float('"' in title or "“" in title or "”" in title),
        "punctuation_count": float(sum(1 for c in title if c in "!?.,:;|()[]\"'-–—…")),
        "emoji_count": float(len(EMOJI_RE.findall(title))),
        "first_person": float(sum(1 for w in lower_words if w in FIRST_PERSON)),
        "second_person": float(sum(1 for w in lower_words if w in SECOND_PERSON)),
        "power_word_count": float(sum(1 for w in lower_words if w in POWER_WORDS)),
        "curiosity_phrase": float(sum(1 for p in CURIOSITY_PHRASES if p in lowered)),
        "urgency_word_count": float(sum(1 for w in lower_words if w in URGENCY_WORDS)),
        "negation_count": float(sum(1 for w in lower_words if w in NEGATION_WORDS)),
        "superlative_count": float(sum(1 for w in lower_words if w in SUPERLATIVE_WORDS or (w.endswith("est") and len(w) > 4))),
        "how_to": float(bool(HOW_TO_RE.search(title))),
        "is_listicle": float(bool(LISTICLE_RE.search(title))),
        "is_versus": float(bool(VS_RE.search(title))),
        "stopword_ratio": float(sum(1 for w in lower_words if w in STOPWORDS) / n_words) if n_words else 0.0,
        "unique_word_ratio": float(len(set(lower_words)) / n_words) if n_words else 0.0,
        "sentiment_compound": 0.0,
        "sentiment_pos": 0.0,
        "sentiment_neg": 0.0,
    }
    if _VADER is not None and title:
        s = _VADER.polarity_scores(title)
        f["sentiment_compound"] = float(s["compound"])
        f["sentiment_pos"] = float(s["pos"])
        f["sentiment_neg"] = float(s["neg"])

    f["log_subscribers"] = math.log10(subscriber_count) if subscriber_count and subscriber_count > 0 else float("nan")
    for ct in CONTENT_TYPES:
        f["ct_" + ct] = float(content_type == ct)
    return f


TITLE_FEATURES: List[str] = list(extract_title_features("probe title", CONTENT_TYPES[0]).keys())
