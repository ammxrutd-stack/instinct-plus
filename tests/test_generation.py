import os

import pytest

from src.generation import titles as gen
from src.generation.titles import Candidate, GenerationError, TitleIdea, TitleIdeas, rule_based_variants, improve_title


def test_rule_based_variants_apply_the_advice(scorer):
    r = scorer.score_title("RAMEN", "cooking")
    cands = rule_based_variants("RAMEN", "cooking", r, n=6)
    assert 1 <= len(cands) <= 6
    assert all(c.source == "rules" for c in cands)
    assert len({c.title.lower() for c in cands}) == len(cands)
    assert any(any(ch.isdigit() for ch in c.title) for c in cands)  # a number was added


def test_improve_title_offline(scorer, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_PROFILE", raising=False)
    out = improve_title("RAMEN", "cooking", scorer, 100_000, n=5, use_claude=True)
    assert out["source"] == "rules"
    assert "ANTHROPIC_API_KEY" in out["error"]
    assert out["original"].score == scorer.score_title("RAMEN", "cooking", 100_000).score
    scores = [c.score for c in out["candidates"]]
    assert scores == sorted(scores, reverse=True)
    assert all(c.delta == round(c.score - out["original"].score, 1) for c in out["candidates"])


class _FakeResponse:
    def __init__(self, ideas, stop_reason="end_turn"):
        self.parsed_output = TitleIdeas(ideas=ideas)
        self.stop_reason = stop_reason


class _FakeMessages:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class _FakeClient:
    def __init__(self, response):
        self.messages = _FakeMessages(response)


def test_improve_title_with_claude_stub(scorer, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    ideas = [TitleIdea(title="7 Ramen Mistakes You Keep Making", rationale="adds a number"),
             TitleIdea(title="RAMEN", rationale="duplicate of the draft, should be dropped"),
             TitleIdea(title="Is Instant Ramen Worth It?", rationale="question")]
    client = _FakeClient(_FakeResponse(ideas))
    out = improve_title("RAMEN", "cooking", scorer, n=5, use_claude=True, client=client, model="claude-opus-5-5")
    assert out["source"] == "claude" and out["error"] is None and out["model"] == "claude-opus-5-5"
    assert [c.title for c in sorted(out["candidates"], key=lambda c: c.title)] == ["7 Ramen Mistakes You Keep Making", "Is Instant Ramen Worth It?"]
    call = client.messages.calls[0]
    assert call["model"] == "claude-opus-5-5" and call["output_format"] is TitleIdeas
    assert "RAMEN" in call["messages"][0]["content"] and "cooking" in call["messages"][0]["content"]


def test_refusal_falls_back_to_rules(scorer, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    client = _FakeClient(_FakeResponse([], stop_reason="refusal"))
    out = improve_title("RAMEN", "cooking", scorer, n=4, use_claude=True, client=client)
    assert out["source"] == "rules" and "declined" in out["error"]
    assert out["candidates"]


def test_generate_titles_raises_on_empty(scorer, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    client = _FakeClient(_FakeResponse([TitleIdea(title="RAMEN", rationale="same")]))
    with pytest.raises(GenerationError):
        gen.generate_titles("RAMEN", "cooking", None, n=3, client=client)
