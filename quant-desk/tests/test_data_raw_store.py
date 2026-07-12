from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from quantdesk.data.raw_store import RawStore, canonical_json_bytes, sha256_hex


def test_write_creates_expected_path(tmp_path):
    store = RawStore(tmp_path)
    when = datetime(2026, 7, 12, 10, 30, tzinfo=timezone.utc)
    record = store.write("hyperliquid_info", {"a": 1}, ingested_time=when)

    expected_path = tmp_path / "hyperliquid_info" / "2026-07-12.jsonl"
    assert expected_path.exists()
    assert record.raw_payload_hash == sha256_hex(canonical_json_bytes({"a": 1}))


def test_write_is_append_only(tmp_path):
    store = RawStore(tmp_path)
    when = datetime(2026, 7, 12, 10, 30, tzinfo=timezone.utc)
    store.write("src", {"a": 1}, ingested_time=when)
    store.write("src", {"a": 2}, ingested_time=when)

    path = tmp_path / "src" / "2026-07-12.jsonl"
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["payload"] == {"a": 1}
    assert json.loads(lines[1])["payload"] == {"a": 2}


def test_write_rejects_naive_datetime(tmp_path):
    store = RawStore(tmp_path)
    with pytest.raises(ValueError):
        store.write("src", {"a": 1}, ingested_time=datetime(2026, 7, 12))


def test_iter_records_roundtrip(tmp_path):
    store = RawStore(tmp_path)
    when = datetime(2026, 7, 12, 10, 30, tzinfo=timezone.utc)
    store.write("src", {"a": 1}, request={"q": "x"}, ingested_time=when)

    records = list(store.iter_records("src", "2026-07-12"))
    assert len(records) == 1
    assert records[0].payload == {"a": 1}
    assert records[0].request == {"q": "x"}
    assert records[0].source_id == "src"


def test_iter_records_missing_file_yields_nothing(tmp_path):
    store = RawStore(tmp_path)
    assert list(store.iter_records("nope", "2026-07-12")) == []


def test_verify_file_detects_no_tampering(tmp_path):
    store = RawStore(tmp_path)
    when = datetime(2026, 7, 12, 10, 30, tzinfo=timezone.utc)
    store.write("src", {"a": 1}, ingested_time=when)
    store.write("src", {"b": 2}, ingested_time=when)

    results = store.verify_file("src", "2026-07-12")
    assert results == [(0, True), (1, True)]


def test_verify_file_detects_tampering(tmp_path):
    store = RawStore(tmp_path)
    when = datetime(2026, 7, 12, 10, 30, tzinfo=timezone.utc)
    store.write("src", {"a": 1}, ingested_time=when)

    path = tmp_path / "src" / "2026-07-12.jsonl"
    raw = json.loads(path.read_text().strip())
    raw["payload"] = {"a": 999}  # tamper with payload but keep old hash
    path.write_text(json.dumps(raw) + "\n")

    results = store.verify_file("src", "2026-07-12")
    assert results == [(0, False)]


def test_verify_all_covers_multiple_days(tmp_path):
    store = RawStore(tmp_path)
    day1 = datetime(2026, 7, 12, 10, 30, tzinfo=timezone.utc)
    day2 = datetime(2026, 7, 13, 10, 30, tzinfo=timezone.utc)
    store.write("src", {"a": 1}, ingested_time=day1)
    store.write("src", {"a": 2}, ingested_time=day2)

    results = store.verify_all("src")
    assert set(results.keys()) == {"2026-07-12", "2026-07-13"}
    assert results["2026-07-12"] == [(0, True)]
    assert results["2026-07-13"] == [(0, True)]


def test_canonical_json_bytes_is_deterministic():
    a = canonical_json_bytes({"b": 1, "a": 2})
    b = canonical_json_bytes({"a": 2, "b": 1})
    assert a == b
