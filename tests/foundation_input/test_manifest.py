from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.foundation_input.manifest import ManifestError, load_manifest, validate_parquet_schema


def _write_dataset(root: Path, *, omit_event_column: str | None = None) -> None:
    (root / "events").mkdir(parents=True)
    (root / "snapshots").mkdir()
    event_row = {
        "event_id": "u:0:0",
        "user_id": "u",
        "source_record_id": "u:0",
        "source_record_index": 0,
        "action_index": 0,
        "source_action_count": 1,
        "event_time": datetime(2025, 9, 1, tzinfo=timezone.utc),
        "time_precision": "day",
        "eligibility_time": datetime(2025, 9, 2, tzinfo=timezone.utc),
        "eligibility_inclusive": False,
        "scenario": "E-commerce",
        "event_type": "click",
        "context_json": "{}",
        "action_json": "{}",
    }
    if omit_event_column:
        event_row.pop(omit_event_column)
    pq.write_table(pa.Table.from_pylist([event_row]), root / "events" / "u.parquet")
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "sample_id": "u:2025-09-02:open",
                    "user_id": "u",
                    "cutoff_time": datetime(2025, 9, 2, tzinfo=timezone.utc),
                    "cutoff_inclusive": False,
                    "split": "train",
                    "history_start": 0,
                    "history_stop": 1,
                    "history_event_ids": ["u:0:0"],
                    "history_order_known": True,
                    "history_truncated": False,
                    "history_group_truncated": False,
                    "profile_json": "{}",
                }
            ]
        ),
        root / "snapshots" / "u.parquet",
    )
    config = {
        "schema_version": "omnibehavior_pretraining_v2",
        "context_allowlist": {},
        "max_history_events": 30,
        "splits": {
            "start_inclusive": "2025-09-01T00:00:00",
            "train_end_exclusive": "2025-11-01T00:00:00",
        },
    }
    config_path = root / "preprocessing_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    checksum = hashlib.sha256(config_path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": "omnibehavior_pretraining_v2",
        "status": "complete",
        "source_dataset": "test",
        "config_sha256": checksum,
        "shards": [
            {
                "user_id": "u",
                "events_file": "events/u.parquet",
                "snapshots_file": "snapshots/u.parquet",
                "event_rows": 1,
                "snapshot_rows": 1,
                "snapshots_by_split": {"train": 1},
            }
        ],
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_load_manifest_rejects_missing_schema_version(tmp_path: Path) -> None:
    _write_dataset(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["schema_version"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ManifestError, match="manifest.schema_version"):
        load_manifest(tmp_path)


def test_load_manifest_rejects_config_checksum_mismatch(tmp_path: Path) -> None:
    _write_dataset(tmp_path)
    config_path = tmp_path / "preprocessing_config.json"
    config_path.write_text('{"changed": true}', encoding="utf-8")

    with pytest.raises(ManifestError, match="config checksum"):
        load_manifest(tmp_path)


def test_validate_parquet_schema_reports_missing_event_column(tmp_path: Path) -> None:
    _write_dataset(tmp_path, omit_event_column="action_json")
    with pytest.raises(ManifestError, match="action_json"):
        validate_parquet_schema(tmp_path / "events" / "u.parquet", kind="events")


def test_validate_parquet_schema_can_enforce_arrow_types(tmp_path: Path) -> None:
    _write_dataset(tmp_path)

    validate_parquet_schema(
        tmp_path / "events" / "u.parquet", kind="events", strict_types=True
    )
    validate_parquet_schema(
        tmp_path / "snapshots" / "u.parquet", kind="snapshots", strict_types=True
    )


def test_load_manifest_exposes_stable_dataset_name(tmp_path: Path) -> None:
    _write_dataset(tmp_path)
    loaded = load_manifest(tmp_path)
    assert loaded.dataset_name == "test"
    assert loaded.shards[0]["user_id"] == "u"
