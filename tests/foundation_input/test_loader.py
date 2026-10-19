from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.foundation_input.loader import iter_normalized_samples
from src.foundation_input.manifest import ManifestError

from .test_manifest import _write_dataset


def test_iter_normalized_samples_reconstructs_profile_and_events(tmp_path: Path) -> None:
    _write_dataset(tmp_path)

    samples = list(iter_normalized_samples(tmp_path, split="train", validate=False))

    assert len(samples) == 1
    sample = samples[0]
    assert sample.dataset_name == "test"
    assert sample.split == "train"
    assert sample.profile == {}
    assert sample.events[0].event_id == "u:0:0"
    assert sample.events[0].context == {}
    assert sample.events[0].action == {}


def test_iter_normalized_samples_rejects_unknown_history_event(tmp_path: Path) -> None:
    _write_dataset(tmp_path)
    path = tmp_path / "snapshots" / "u.parquet"
    table = pq.read_table(path)
    rows = table.to_pylist()
    rows[0]["history_event_ids"] = ["u:missing:0"]
    rows[0]["history_event_ids"] = ["u:missing:0"]
    pq.write_table(pa.Table.from_pylist(rows), path)

    with pytest.raises(ManifestError, match="history_event_ids"):
        list(iter_normalized_samples(tmp_path, split="train", validate=False))


def test_iter_normalized_samples_rejects_malformed_profile_json(tmp_path: Path) -> None:
    _write_dataset(tmp_path)
    path = tmp_path / "snapshots" / "u.parquet"
    table = pq.read_table(path)
    rows = table.to_pylist()
    rows[0]["profile_json"] = "not-json"
    pq.write_table(pa.Table.from_pylist(rows), path)

    with pytest.raises(ManifestError, match="profile_json"):
        list(iter_normalized_samples(tmp_path, split="train", validate=False))


def test_iter_normalized_samples_treats_null_context_and_action_as_empty(tmp_path: Path) -> None:
    _write_dataset(tmp_path)
    path = tmp_path / "events" / "u.parquet"
    table = pq.read_table(path)
    rows = table.to_pylist()
    rows[0]["context_json"] = "null"
    rows[0]["action_json"] = "null"
    pq.write_table(pa.Table.from_pylist(rows), path)

    sample = next(iter_normalized_samples(tmp_path, split="train", validate=False))
    assert sample.events[0].context == {}
    assert sample.events[0].action == {}


def test_iter_normalized_samples_sorts_history_by_eligibility_coordinate(tmp_path: Path) -> None:
    _write_dataset(tmp_path)
    event_path = tmp_path / "events" / "u.parquet"
    event_table = pq.read_table(event_path)
    first = event_table.to_pylist()[0]
    second = dict(first)
    second.update(
        {
            "event_id": "u:1:0",
            "source_record_id": "u:1",
            "source_record_index": 1,
            "eligibility_time": datetime(2025, 9, 1, 12, tzinfo=timezone.utc),
        }
    )
    pq.write_table(pa.Table.from_pylist([first, second]), event_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = __import__("json").loads(manifest_path.read_text(encoding="utf-8"))
    manifest["shards"][0]["event_rows"] = 2
    manifest_path.write_text(__import__("json").dumps(manifest), encoding="utf-8")
    snapshot_path = tmp_path / "snapshots" / "u.parquet"
    snapshot_rows = pq.read_table(snapshot_path).to_pylist()
    snapshot_rows[0]["history_event_ids"] = ["u:0:0", "u:1:0"]
    snapshot_rows[0]["history_stop"] = 2
    pq.write_table(pa.Table.from_pylist(snapshot_rows), snapshot_path)

    sample = next(iter_normalized_samples(tmp_path, split="train", validate=True))
    assert [event.event_id for event in sample.events] == ["u:1:0", "u:0:0"]


def test_iter_normalized_samples_rejects_history_slice_mismatch(tmp_path: Path) -> None:
    _write_dataset(tmp_path)
    snapshot_path = tmp_path / "snapshots" / "u.parquet"
    rows = pq.read_table(snapshot_path).to_pylist()
    rows[0]["history_stop"] = 2
    pq.write_table(pa.Table.from_pylist(rows), snapshot_path)

    with pytest.raises(ManifestError, match="history bounds"):
        list(iter_normalized_samples(tmp_path, split="train", validate=False))


def test_iter_normalized_samples_rejects_snapshot_in_wrong_split(tmp_path: Path) -> None:
    _write_dataset(tmp_path)
    snapshot_path = tmp_path / "snapshots" / "u.parquet"
    rows = pq.read_table(snapshot_path).to_pylist()
    rows[0]["cutoff_time"] = datetime(2025, 11, 20, tzinfo=timezone.utc)
    pq.write_table(pa.Table.from_pylist(rows), snapshot_path)

    with pytest.raises(ManifestError, match="split boundary"):
        list(iter_normalized_samples(tmp_path, split="train", validate=False))
