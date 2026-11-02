"""CLI: smoke seed selection, verify output, exit codes."""

import pytest

from config.channels import CHANNELS
from src.collection import __main__ as cli
from src.collection.client import CollectionError

from .conftest import FakeClient


def test_smoke_seeds_take_two_channels_of_the_first_content_type():
    first = next(iter(CHANNELS))
    assert cli.smoke_seeds() == {first: CHANNELS[first][:2]}
    assert cli.smoke_seeds({"x": ["@a", "@b", "@c"], "y": ["@d"]}) == {"x": ["@a", "@b"]}


def test_verify_flags_unresolved_and_small_channels(fake_client, capsys):
    summary = cli.cmd_verify(fake_client, {"gaming": ["@alpha", "@tiny", "@ghost"]})
    out = capsys.readouterr().out
    assert summary["checked"] == 3
    assert summary["flagged"] == [("@tiny", "BELOW MIN_SUBSCRIBERS"), ("@ghost", "unresolved")]
    assert "Alpha Gaming" in out and "2,500,000" in out
    assert "UNRESOLVED" in out
    assert summary["units"] == 3


def test_collection_error_exits_1(monkeypatch, capsys):
    def boom(**kwargs):
        raise CollectionError("bad key")
    monkeypatch.setattr(cli, "run_build", boom)
    assert cli.main(["build"]) == 1
    assert "bad key" in capsys.readouterr().err


def test_quota_exceeded_in_verify_exits_0_with_reset_time(monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_api_key", lambda: "k")
    monkeypatch.setattr(cli, "Client", lambda key, budget: FakeClient(budget=0))
    assert cli.main(["verify"]) == 0
    assert "midnight Pacific" in capsys.readouterr().out


def test_missing_key_exits_1_without_touching_the_network(monkeypatch, capsys):
    def no_key():
        raise CollectionError("YOUTUBE_API_KEY is not set")
    monkeypatch.setattr(cli, "load_api_key", no_key)
    monkeypatch.setattr(cli, "Client", lambda key, budget: pytest.fail("client must not be built"))
    assert cli.main(["discover"]) == 1
    assert "YOUTUBE_API_KEY" in capsys.readouterr().err


def test_parse_args_defaults():
    args = cli.parse_args(["all", "--smoke", "--csv", "--no-thumbs"])
    assert (args.stage, args.budget, args.smoke, args.csv, args.no_thumbs) == ("all", 9000, True, True, True)
    with pytest.raises(SystemExit):
        cli.parse_args(["nonsense"])
