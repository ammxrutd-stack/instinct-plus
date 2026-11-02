"""Checkpoint round trip and the reset-time message."""

from datetime import datetime, timezone

from src.collection.checkpoint import (
    checkpoint_path,
    clear_checkpoint,
    load_checkpoint,
    next_quota_reset,
    quota_reset_message,
    write_checkpoint,
)


def test_missing_checkpoint_is_empty(raw_dir):
    assert load_checkpoint(raw_dir) == {"completed_channel_ids": [], "completed_handles": []}


def test_round_trip_dedupes_and_sorts(raw_dir):
    write_checkpoint(raw_dir, ["UCb", "UCa", "UCa"], ["@b", "@a"])
    assert checkpoint_path(raw_dir).name == "checkpoint.json"
    assert load_checkpoint(raw_dir) == {
        "completed_channel_ids": ["UCa", "UCb"],
        "completed_handles": ["@a", "@b"],
    }


def test_clear_removes_file_and_is_idempotent(raw_dir):
    write_checkpoint(raw_dir, ["UCa"], ["@a"])
    clear_checkpoint(raw_dir)
    assert not checkpoint_path(raw_dir).exists()
    clear_checkpoint(raw_dir)  # nothing to remove, no error


def test_next_quota_reset_is_midnight_pacific_after_now():
    now = datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc)  # 05:00 PDT
    reset = next_quota_reset(now)
    if reset is None:  # no zone database on this machine; message degrades
        return
    assert (reset.hour, reset.minute) == (0, 0)
    assert reset.tzname() in ("PDT", "PST")
    assert reset > now
    assert reset.astimezone(timezone.utc) == datetime(2026, 9, 17, 7, 0, tzinfo=timezone.utc)


def test_quota_reset_message_names_midnight_pacific():
    assert "midnight Pacific" in quota_reset_message()
