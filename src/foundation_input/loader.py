"""Read versioned pretraining Parquet shards into canonical samples."""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from .manifest import DatasetManifest, ManifestError, load_manifest, validate_parquet_schema
from .types import CanonicalSample, EventRecord


def _as_datetime(value: Any, *, field: str) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as error:
            raise ManifestError(f"Invalid {field} timestamp: {value!r}") from error
    raise ManifestError(f"Invalid {field} value: {value!r}")


def _as_bool(value: Any, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise ManifestError(f"{field} must be a boolean, got {value!r}")
    return value


def _as_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ManifestError(f"{field} must be an integer, got {value!r}")
    return value


def _parse_object(
    value: Any, *, field: str, identity: str, allow_null: bool = False
) -> dict[str, Any]:
    if not isinstance(value, str):
        raise ManifestError(f"{identity}: {field} must be a JSON string")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise ManifestError(f"{identity}: malformed {field}: {error.msg}") from error
    if parsed is None and allow_null:
        return {}
    if not isinstance(parsed, dict):
        raise ManifestError(f"{identity}: {field} must decode to an object")
    return parsed


def _event_from_row(row: dict[str, Any]) -> EventRecord:
    event_id = row.get("event_id")
    user_id = row.get("user_id")
    identity = f"event_id={event_id!r}, user_id={user_id!r}"
    required = (
        "event_id",
        "source_record_id",
        "source_record_index",
        "action_index",
        "source_action_count",
        "event_time",
        "time_precision",
        "eligibility_time",
        "eligibility_inclusive",
        "scenario",
        "event_type",
        "context_json",
        "action_json",
    )
    missing = [field for field in required if field not in row]
    if missing:
        raise ManifestError(f"{identity}: missing event fields {missing}")
    return EventRecord(
        event_id=str(event_id),
        user_id=None if user_id is None else str(user_id),
        source_record_id=str(row["source_record_id"]),
        source_record_index=int(row["source_record_index"]),
        action_index=int(row["action_index"]),
        source_action_count=int(row["source_action_count"]),
        event_time=_as_datetime(row["event_time"], field=f"{identity}.event_time"),
        time_precision=str(row["time_precision"]),
        eligibility_time=_as_datetime(
            row["eligibility_time"], field=f"{identity}.eligibility_time"
        ),
        eligibility_inclusive=_as_bool(
            row["eligibility_inclusive"], field=f"{identity}.eligibility_inclusive"
        ),
        scenario=None if row["scenario"] is None else str(row["scenario"]),
        event_type=None if row["event_type"] is None else str(row["event_type"]),
        context=_parse_object(
            row["context_json"], field="context_json", identity=identity, allow_null=True
        ),
        action=_parse_object(
            row["action_json"], field="action_json", identity=identity, allow_null=True
        ),
    )


def _load_user_events(
    manifest: DatasetManifest,
    shard: dict[str, Any],
    needed_event_ids: set[str],
) -> tuple[dict[str, EventRecord], list[str]]:
    event_path = manifest.root / shard["events_file"]
    validate_parquet_schema(event_path, kind="events", strict_types=True)
    events: dict[str, EventRecord] = {}
    all_event_ids: list[str] = []
    try:
        parquet = pq.ParquetFile(event_path)
        for batch in parquet.iter_batches(batch_size=256):
            for row in batch.to_pylist():
                event = _event_from_row(row)
                if event.event_id in all_event_ids:
                    raise ManifestError(f"Duplicate event_id in {event_path}: {event.event_id!r}")
                if row.get("user_id") != shard["user_id"]:
                    raise ManifestError(
                        f"Event user mismatch in {event_path}: expected {shard['user_id']!r}, "
                        f"got {row.get('user_id')!r}"
                    )
                all_event_ids.append(event.event_id)
                if event.event_id in needed_event_ids:
                    events[event.event_id] = event
    except ManifestError:
        raise
    except Exception as error:
        raise ManifestError(f"Cannot read events shard {event_path}: {error}") from error
    expected_rows = shard.get("event_rows")
    if expected_rows is not None and len(all_event_ids) != expected_rows:
        raise ManifestError(
            f"Event row count mismatch in {event_path}: manifest={expected_rows}, "
            f"actual={len(all_event_ids)}"
        )
    return events, all_event_ids


def _validate_split_boundary(
    row: dict[str, Any], *, split: str, manifest: DatasetManifest, identity: str
) -> None:
    boundaries = manifest.config.get("splits")
    if not isinstance(boundaries, dict):
        return
    cutoff = (
        _as_datetime(row.get("cutoff_time"), field=f"{identity}.cutoff_time").replace(tzinfo=None),
        int(_as_bool(row.get("cutoff_inclusive"), field=f"{identity}.cutoff_inclusive")),
    )

    def bound(name: str, default: tuple[datetime, int] | None = None) -> tuple[datetime, int] | None:
        value = boundaries.get(name)
        if value is None:
            return default
        return (_as_datetime(value, field=f"splits.{name}").replace(tzinfo=None), 1)

    start = bound("start_inclusive")
    train_end = bound("train_end_exclusive")
    validation_end = bound("validation_end_exclusive")
    test_end = bound("test_end_exclusive")
    if start is not None and cutoff < start:
        raise ManifestError(f"{identity}: split boundary starts after cutoff")
    lower = {"train": start, "validation": train_end, "test": validation_end}[split]
    upper = {"train": train_end, "validation": validation_end, "test": test_end}[split]
    if lower is not None and cutoff < lower:
        raise ManifestError(f"{identity}: split boundary does not contain cutoff")
    if upper is not None and cutoff > (upper[0], 0):
        raise ManifestError(f"{identity}: split boundary does not contain cutoff")


def _sample_from_row(
    row: dict[str, Any],
    *,
    manifest: DatasetManifest,
    shard: dict[str, Any],
    events: dict[str, EventRecord],
    all_event_ids: list[str],
    requested_split: str,
) -> CanonicalSample:
    sample_id = row.get("sample_id")
    identity = f"sample_id={sample_id!r}"
    if row.get("user_id") != shard["user_id"]:
        raise ManifestError(
            f"{identity}: snapshot user mismatch; expected {shard['user_id']!r}, "
            f"got {row.get('user_id')!r}"
        )
    history_ids = row.get("history_event_ids")
    if not isinstance(history_ids, list):
        raise ManifestError(f"{identity}: history_event_ids must be a list")
    start = _as_int(row.get("history_start"), field=f"{identity}.history_start")
    stop = _as_int(row.get("history_stop"), field=f"{identity}.history_stop")
    if start < 0 or start > stop or stop > len(all_event_ids):
        raise ManifestError(f"{identity}: history bounds are outside the event shard")
    if all_event_ids[start:stop] != [str(event_id) for event_id in history_ids]:
        raise ManifestError(f"{identity}: history_event_ids slice does not match event shard order")
    max_history = int(manifest.config.get("max_history_events", 30))
    expected_truncated = start > 0 and stop - start >= max_history
    if _as_bool(row.get("history_truncated"), field=f"{identity}.history_truncated") != expected_truncated:
        raise ManifestError(f"{identity}: history_truncated is inconsistent with history bounds")
    history: list[EventRecord] = []
    for event_id in history_ids:
        if event_id not in events:
            raise ManifestError(
                f"{identity}: history_event_ids references missing event {event_id!r}"
            )
        history.append(events[event_id])
    history.sort(
        key=lambda event: (
            event.eligibility_time.replace(tzinfo=None),
            int(event.eligibility_inclusive),
            event.source_record_index,
            event.action_index,
        )
    )
    profile = _parse_object(row.get("profile_json"), field="profile_json", identity=identity)
    split = row.get("split")
    if split not in {"train", "validation", "test"}:
        raise ManifestError(f"{identity}: unsupported split {split!r}")
    return CanonicalSample(
        sample_id=str(sample_id),
        dataset_name=manifest.dataset_name,
        split=str(split),
        user_id=str(row["user_id"]),
        cutoff_time=_as_datetime(row["cutoff_time"], field=f"{identity}.cutoff_time"),
        cutoff_inclusive=_as_bool(row["cutoff_inclusive"], field=f"{identity}.cutoff_inclusive"),
        history_order_known=_as_bool(row["history_order_known"], field=f"{identity}.history_order_known"),
        history_truncated=_as_bool(row["history_truncated"], field=f"{identity}.history_truncated"),
        history_group_truncated=_as_bool(
            row["history_group_truncated"], field=f"{identity}.history_group_truncated"
        ),
        profile=profile,
        events=tuple(history),
        history_start=start,
        history_stop=stop,
    )


def iter_normalized_samples(
    dataset_root: str | Path,
    split: str,
    *,
    validate: bool = True,
) -> Iterator[CanonicalSample]:
    """Yield canonical samples from one logical split without modifying source data."""

    if split not in {"train", "validation", "test"}:
        raise ValueError(f"Invalid split: {split!r}")
    manifest = load_manifest(dataset_root, validate_shard_files=False)
    for shard in manifest.shards:
        if not shard["snapshots_by_split"].get(split, 0):
            continue
        snapshot_path = manifest.root / shard["snapshots_file"]
        validate_parquet_schema(snapshot_path, kind="snapshots", strict_types=True)
        snapshot_rows: list[dict[str, Any]] = []
        snapshot_count = 0
        try:
            parquet = pq.ParquetFile(snapshot_path)
            for batch in parquet.iter_batches(batch_size=256):
                for row in batch.to_pylist():
                    snapshot_count += 1
                    if row.get("split") == split:
                        snapshot_rows.append(row)
        except Exception as error:
            raise ManifestError(f"Cannot read snapshots shard {snapshot_path}: {error}") from error
        expected_snapshot_count = shard.get("snapshot_rows")
        if expected_snapshot_count is not None and snapshot_count != expected_snapshot_count:
            raise ManifestError(
                f"Snapshot row count mismatch in {snapshot_path}: manifest={expected_snapshot_count}, "
                f"actual={snapshot_count}"
            )
        expected_split_count = shard.get("snapshots_by_split", {}).get(split, 0)
        if len(snapshot_rows) != expected_split_count:
            raise ManifestError(
                f"Snapshot split row count mismatch in {snapshot_path}: manifest={expected_split_count}, "
                f"actual={len(snapshot_rows)}"
            )
        needed_event_ids = {
            str(event_id)
            for row in snapshot_rows
            for event_id in (row.get("history_event_ids") or [])
        }
        events, all_event_ids = _load_user_events(manifest, shard, needed_event_ids)
        for row in snapshot_rows:
            sample = _sample_from_row(
                row,
                manifest=manifest,
                shard=shard,
                events=events,
                all_event_ids=all_event_ids,
                requested_split=split,
            )
            if sample.split != split:
                raise ManifestError(f"{sample.sample_id}: requested split does not match row split")
            _validate_split_boundary(row, split=split, manifest=manifest, identity=f"sample_id={sample.sample_id!r}")
            if validate:
                from .validation import validate_sample

                validate_sample(sample)
            yield sample
