"""Title generation: ask Claude for alternatives, then score them.

The loop is the product's second half. Scoring tells a creator what is weak
about a title; generation hands them candidates that fix it, and scoring
again tells them which candidate the model rates highest. The prompt carries
the scoring model's own diagnosis (drivers and notes) so Claude writes toward
what the model rewards rather than guessing.

Without an API key (or when the API fails), rule_based_variants() applies the
same advice mechanically: add a number, ask a question, trim, address the
viewer. It is deliberately simple; its job is to keep the app useful offline
and to give the tests something deterministic.

Configuration:
    ANTHROPIC_API_KEY        in .env (or an `ant auth login` profile)
    ANTHROPIC_WORKSPACE_ID   only for keys that are not scoped to a workspace;
                             sent as the anthropic-workspace-id header
    INSTINCT_CLAUDE_MODEL    defaults to claude-opus-5-5
"""

import os
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from config.schema import PROJECT_ROOT
from src.scoring.interface import ScoreResult
from src.scoring.suggestions import TITLE_LABELS, label

load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_MODEL = os.getenv("INSTINCT_CLAUDE_MODEL", "claude-opus-5-5")


class GenerationError(RuntimeError):
    """The API path failed; the caller may fall back to rule_based_variants."""


class TitleIdea(BaseModel):
    title: str = Field(description="The proposed YouTube title, under 70 characters.")
    rationale: str = Field(description="One sentence on what this version changes and why.")


class TitleIdeas(BaseModel):
    ideas: List[TitleIdea]


@dataclass
class Candidate:
    title: str
    rationale: str
    source: str              # "claude" | "rules" | "original"
    score: Optional[float] = None
    delta: Optional[float] = None
    drivers: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def make_client():
    """An Anthropic client that sends the workspace header when one is configured."""
    import anthropic

    workspace = os.getenv("ANTHROPIC_WORKSPACE_ID")
    headers = {"anthropic-workspace-id": workspace} if workspace else None
    return anthropic.Anthropic(default_headers=headers)


def claude_available() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN") or os.getenv("ANTHROPIC_PROFILE"))


SYSTEM_PROMPT = """You write YouTube titles for a creator. You are given their draft title, the
content category, and a diagnosis from a model that predicts reach from title
features. Produce distinct alternatives that keep the video's actual subject
and promise (never invent claims the video cannot deliver), address the
weaknesses in the diagnosis, and preserve anything the diagnosis says is
working. Vary the angle across the set: a number, a question, a curiosity
hook, a plain specific statement, a how-to, a comparison. Keep every title
under 70 characters, with at most one word in capitals, and no emoji.
Avoid misleading or exaggerated claims."""


def _diagnosis(title: str, content_type: str, result: Optional[ScoreResult]) -> str:
    lines = [f"Draft title: {title}", f"Category: {content_type}"]
    if result is not None and result.drivers:
        lines.append(f"Model score: {result.score:.0f}/100 (percentile within {content_type})")
        helping = [(k, v) for k, v in result.drivers.items() if v > 0]
        hurting = [(k, v) for k, v in result.drivers.items() if v < 0]
        if helping:
            lines.append("Working: " + "; ".join(f"{label(k, TITLE_LABELS)} ({v:+.1f})" for k, v in helping))
        if hurting:
            lines.append("Hurting: " + "; ".join(f"{label(k, TITLE_LABELS)} ({v:+.1f})" for k, v in hurting))
        if result.notes:
            lines.append("Suggestions:\n" + "\n".join(f"- {n}" for n in result.notes))
    return "\n".join(lines)


def generate_titles(
    title: str,
    content_type: str,
    result: Optional[ScoreResult] = None,
    n: int = 6,
    model: str = DEFAULT_MODEL,
    client=None,
) -> List[Candidate]:
    """Ask Claude for `n` alternatives. Raises GenerationError on any failure."""
    try:
        import anthropic
    except ImportError as e:  # pragma: no cover
        raise GenerationError("the anthropic package is not installed") from e

    client = client or make_client()
    prompt = _diagnosis(title, content_type, result) + f"\n\nWrite {n} alternatives."
    try:
        response = client.messages.parse(
            model=model,
            max_tokens=4000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
            output_format=TitleIdeas,
        )
    except anthropic.AuthenticationError as e:
        raise GenerationError("Anthropic rejected the API key. Check ANTHROPIC_API_KEY in .env.") from e
    except anthropic.RateLimitError as e:
        raise GenerationError("Anthropic rate limit hit; try again in a minute.") from e
    except anthropic.APIStatusError as e:
        raise GenerationError(f"Anthropic API error {e.status_code}: {e.message}") from e
    except anthropic.APIConnectionError as e:
        raise GenerationError("Could not reach the Anthropic API (network).") from e

    if response.stop_reason == "refusal":
        raise GenerationError("Claude declined this request; using rule-based variants instead.")
    parsed = getattr(response, "parsed_output", None)
    if parsed is None:
        raise GenerationError("Claude returned no structured output.")
    seen = {_norm(title)}
    out = []
    for idea in parsed.ideas:
        t = idea.title.strip()
        if _norm(t) in seen or not t:
            continue
        seen.add(_norm(t))
        out.append(Candidate(title=t, rationale=idea.rationale.strip(), source="claude"))
    if not out:
        raise GenerationError("Claude returned no new titles.")
    return out[:n]


# ----------------------------------------------------------- offline path --

STOP = {"the", "a", "an", "of", "to", "and", "that", "this", "very", "really", "just", "actually"}
QUESTION_OPENERS = {"why", "how", "what", "when", "who", "which", "where", "is", "are", "can", "should", "do", "does", "did"}


def _norm(t: str) -> str:
    return re.sub(r"\W+", " ", t.lower()).strip()


def rule_based_variants(title: str, content_type: str, result: Optional[ScoreResult] = None, n: int = 6) -> List[Candidate]:
    """Mechanical rewrites that apply the scoring model's advice."""
    base = title.strip().rstrip("!.?")
    words = base.split()
    opens_with_question = bool(words) and words[0].lower() in QUESTION_OPENERS
    # the subject of the title, without a leading question word, for wrapping
    subject_words = words[1:] if opens_with_question and len(words) > 2 else words
    core = " ".join(w for w in subject_words if w.lower() not in STOP) or base
    f = result.features if result is not None else {}
    cands: List[Candidate] = []

    def add(t: str, why: str):
        t = re.sub(r"\s+", " ", t).strip()
        if _norm(t) != _norm(title) and all(_norm(t) != _norm(c.title) for c in cands):
            cands.append(Candidate(title=t[:90], rationale=why, source="rules"))

    if not f.get("has_number", 0):
        add(f"5 Things I Learned About {core}", "adds a concrete number")
        add(f"{core} in 10 Minutes", "adds a timespan")
    if not f.get("is_question", 0) and not opens_with_question:
        add(f"Is {core} Worth It?", "asks the question the video answers")
        add(f"Why {core}?", "opens with why")
    elif opens_with_question and not f.get("is_question", 0):
        add(f"{base}?", "makes the question explicit")
    if not f.get("curiosity_phrase", 0):
        add(f"{core}: What Nobody Tells You", "adds a curiosity hook")
        if not opens_with_question:
            add(f"I Tried {core} (Here's What Happened)", "first person + open loop")
    if not f.get("second_person", 0):
        add(f"{core}: What You Need to Know", "speaks to the viewer")
    if f.get("caps_word_ratio", 0) > 0.3:
        add(base.title(), "drops the shouting caps")
    if f.get("char_count", 0) > 65:
        add(" ".join(words[:8]), "shortens to the hook")
    if not f.get("how_to", 0) and content_type in ("cooking", "education"):
        add(f"How to {core}", "how-to framing")
    if not f.get("has_brackets", 0):
        add(f"{base} [Full Guide]", "adds a format tag")
    add(f"The Truth About {core}", "plain curiosity framing")
    add(f"{core} vs Everything Else", "comparison framing")
    return cands[:n]


def improve_title(
    title: str,
    content_type: str,
    scorer,
    subscriber_count: Optional[float] = None,
    n: int = 6,
    use_claude: bool = True,
    model: str = DEFAULT_MODEL,
    client=None,
) -> Dict[str, Any]:
    """Score the draft, generate alternatives, score those, rank.

    Returns {"original": Candidate, "candidates": [Candidate...], "source":
    "claude"|"rules", "error": str|None}. Never raises for API trouble: the
    error is reported and the rule-based variants are used.
    """
    original_result = scorer.score_title(title, content_type, subscriber_count)
    original = Candidate(title=title, rationale="your draft", source="original",
                         score=original_result.score, delta=0.0, drivers=original_result.drivers)
    error = None
    cands: List[Candidate] = []
    source = "rules"
    if use_claude and claude_available():
        try:
            cands = generate_titles(title, content_type, original_result, n=n, model=model, client=client)
            source = "claude"
        except GenerationError as e:
            error = str(e)
    elif use_claude:
        error = "No ANTHROPIC_API_KEY set; showing rule-based variants."
    if not cands:
        cands = rule_based_variants(title, content_type, original_result, n=n)
    for c in cands:
        r = scorer.score_title(c.title, content_type, subscriber_count)
        c.score = r.score
        c.delta = float(round(r.score - original_result.score))
        c.drivers = r.drivers
    cands.sort(key=lambda c: -(c.score or 0))
    return {"original": original, "candidates": cands, "source": source, "error": error, "model": model if source == "claude" else None}
