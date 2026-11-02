"""Shared fakes and fixtures. No network, no API key.

FakeClient serves the recorded-shaped JSON under tests/fixtures/ through the
same ``get(endpoint, **params)`` / ``units`` surface as the real Client, so
every stage runs unmodified against it.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.collection.client import QuotaExceeded

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"

# the instant every test measures ages against; fixture dates are chosen around it
NOW = datetime(2026, 9, 16, 12, 0, 0, tzinfo=timezone.utc)

ALPHA = "UCalpha0000000000000001"
BETA = "UCbeta00000000000000002"
TINY = "UCtiny00000000000000003"


def load_fixture(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


class FakeClient:
    """Routes channels / playlistItems / videos calls to fixture payloads."""

    def __init__(self, channels=None, playlists=None, videos=None, budget=None):
        self.channels = channels or {}    # handle -> channelListResponse
        self.playlists = playlists or {}  # playlist id -> {page token: playlistItemListResponse}
        self.videos = videos or {}        # video id -> video item
        self.budget = budget
        self.units = 0
        self.calls = []

    def get(self, endpoint, **params):
        if self.budget is not None and self.units + 1 > self.budget:
            raise QuotaExceeded(f"budget of {self.budget} units reached")
        self.units += 1
        self.calls.append((endpoint, params))
        if endpoint == "channels":
            empty = {"kind": "youtube#channelListResponse", "pageInfo": {"totalResults": 0}, "items": []}
            return self.channels.get(params["forHandle"], empty)
        if endpoint == "playlistItems":
            return self.playlists[params["playlistId"]][params.get("pageToken") or ""]
        if endpoint == "videos":
            ids = params["id"].split(",")
            return {
                "kind": "youtube#videoListResponse",
                "items": [self.videos[i] for i in ids if i in self.videos],
            }
        raise AssertionError(f"unexpected endpoint {endpoint}")

    def calls_to(self, endpoint):
        return [params for name, params in self.calls if name == endpoint]


@pytest.fixture
def channels_fixture():
    return load_fixture("channels_list.json")


@pytest.fixture
def playlists_fixture():
    return load_fixture("playlist_items.json")


@pytest.fixture
def videos_fixture():
    return {item["id"]: item for item in load_fixture("videos_list.json")["items"]}


@pytest.fixture
def fake_client(channels_fixture, playlists_fixture, videos_fixture):
    return FakeClient(channels_fixture, playlists_fixture, videos_fixture)


@pytest.fixture
def now():
    return NOW


@pytest.fixture
def raw_dir(tmp_path):
    path = tmp_path / "raw"
    path.mkdir()
    return path


@pytest.fixture
def processed_dir(tmp_path):
    path = tmp_path / "processed"
    path.mkdir()
    return path


@pytest.fixture
def thumbnails_dir(tmp_path):
    return tmp_path / "thumbnails"


def write_jsonl(path, items):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for item in items:
            out.write(json.dumps(item) + "\n")


def channel_record(channels_fixture, handle, content_type="gaming", **overrides):
    """A channels_*.jsonl line: the verbatim item plus the two keys discover adds."""
    item = json.loads(json.dumps(channels_fixture[handle]["items"][0]))
    item.update(_content_type=content_type, _handle=handle)
    item.update(overrides)
    return item


@pytest.fixture
def populated_raw(raw_dir, channels_fixture, videos_fixture):
    """A raw dir as discover + fetch would leave it for @alpha, @beta and @tiny."""
    write_jsonl(raw_dir / "channels_20260916T120000Z.jsonl", [
        channel_record(channels_fixture, "@alpha"),
        channel_record(channels_fixture, "@beta"),
        channel_record(channels_fixture, "@tiny"),
    ])
    write_jsonl(raw_dir / "videos_20260916T120000Z.jsonl", list(videos_fixture.values()))
    return raw_dir
