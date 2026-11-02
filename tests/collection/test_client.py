"""Client: quota accounting, retries and the fatal/non-fatal split."""

import json

import pytest
import requests

from src.collection.client import (
    MAX_ATTEMPTS,
    Client,
    CollectionError,
    QuotaExceeded,
    load_api_key,
)


class FakeResponse:
    def __init__(self, status_code, body=None, text=""):
        self.status_code = status_code
        self._body = body
        self.text = text if body is None else json.dumps(body)

    def json(self):
        if self._body is None:
            raise ValueError("not json")
        return self._body


class FakeSession:
    """Replays a scripted list of responses or exceptions, in order."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params, timeout))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


OK = FakeResponse(200, {"kind": "youtube#videoListResponse", "items": []})


def api_error(status, reason, message="boom"):
    return FakeResponse(status, {"error": {
        "code": status, "message": message,
        "errors": [{"message": message, "domain": "youtube.quota", "reason": reason}],
    }})


def make_client(outcomes, budget=9000):
    sleeps = []
    session = FakeSession(outcomes)
    client = Client("test-key", budget=budget, session=session, sleep=sleeps.append)
    return client, session, sleeps


def test_budget_exhaustion_raises_before_the_call_not_after():
    client, session, _ = make_client([OK, OK, OK], budget=2)
    client.get("videos", id="a")
    client.get("videos", id="b")
    with pytest.raises(QuotaExceeded):
        client.get("videos", id="c")
    assert len(session.calls) == 2, "the third request must never leave the client"
    assert client.units == 2


def test_each_request_costs_one_unit():
    client, _, _ = make_client([OK, OK])
    client.get("channels", forHandle="@x")
    client.get("playlistItems", playlistId="UUx")
    assert client.units == 2


def test_key_is_added_and_none_params_dropped():
    client, session, _ = make_client([OK])
    client.get("playlistItems", playlistId="UUx", pageToken=None, maxResults=50)
    _, params, timeout = session.calls[0]
    assert params == {"playlistId": "UUx", "maxResults": 50, "key": "test-key"}
    assert timeout == 30


@pytest.mark.parametrize("reason", ["quotaExceeded", "dailyLimitExceeded"])
def test_403_quota_reasons_raise_quota_exceeded(reason):
    client, _, _ = make_client([api_error(403, reason)])
    with pytest.raises(QuotaExceeded):
        client.get("videos", id="a")


def test_403_other_reason_is_fatal_collection_error():
    client, session, _ = make_client([api_error(403, "accessNotConfigured")])
    with pytest.raises(CollectionError) as excinfo:
        client.get("videos", id="a")
    assert "accessNotConfigured" in str(excinfo.value)
    assert len(session.calls) == 1, "fatal errors are not retried"


@pytest.mark.parametrize("status,reason", [(401, "authError"), (400, "keyInvalid")])
def test_401_and_400_are_fatal(status, reason):
    client, _, _ = make_client([api_error(status, reason)])
    with pytest.raises(CollectionError):
        client.get("videos", id="a")


def test_5xx_backs_off_three_times_then_gives_up():
    client, session, sleeps = make_client([
        FakeResponse(500, text="Internal"), FakeResponse(502, text="Bad gateway"),
        FakeResponse(503, text="Unavailable"),
    ])
    assert client.get("videos", id="a") is None
    assert len(session.calls) == MAX_ATTEMPTS == 3
    assert sleeps == [1, 2], "exponential backoff between attempts"
    assert client.units == 3, "every request made is a unit spent"


def test_timeout_then_success_returns_payload():
    client, session, sleeps = make_client([requests.Timeout("slow"), OK])
    assert client.get("videos", id="a") == OK.json()
    assert len(session.calls) == 2
    assert sleeps == [1]


def test_connection_error_is_retried():
    client, _, _ = make_client([requests.ConnectionError("reset"), requests.ConnectionError("reset"), OK])
    assert client.get("videos", id="a") == OK.json()


def test_retries_respect_the_budget():
    # one unit left: the first attempt spends it, the retry must not be sent
    client, session, _ = make_client([FakeResponse(500), OK], budget=1)
    with pytest.raises(QuotaExceeded):
        client.get("videos", id="a")
    assert len(session.calls) == 1


def test_404_returns_none_without_retrying():
    client, session, sleeps = make_client([api_error(404, "playlistNotFound")])
    assert client.get("playlistItems", playlistId="UUnope") is None
    assert len(session.calls) == 1
    assert sleeps == []


def test_load_api_key_missing_is_a_collection_error(tmp_path, monkeypatch):
    monkeypatch.delenv("YOUTUBE_API_KEY", raising=False)
    with pytest.raises(CollectionError) as excinfo:
        load_api_key(tmp_path / ".env")
    assert ".env.example" in str(excinfo.value)


def test_load_api_key_reads_dotenv(tmp_path, monkeypatch):
    monkeypatch.delenv("YOUTUBE_API_KEY", raising=False)
    (tmp_path / ".env").write_text("YOUTUBE_API_KEY=from-file\n")
    assert load_api_key(tmp_path / ".env") == "from-file"
