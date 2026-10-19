"""Validation and metadata loading for pretraining Parquet roots."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA_VERSION = "omnibehavior_pretraining_v2"

EVENT_COLUMNS = frozenset(
    {
        "event_id",
        "user_id",
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
    }
)

SNAPSHOT_COLUMNS = frozenset(
    {
        "sample_id",
        "user_id",
        "cutoff_time",
        "cutoff_inclusive",
        "split",
        "history_start",
        "history_stop",
        "history_event_ids",
        "history_order_known",
        "history_truncated",
        "history_group_truncated",
        "profile_json",
    }
)


class ManifestError(ValueError):
    """Raised when a pretraining root is incomplete or inconsistent."""


@dataclass(frozen=True)
class DatasetManifest:
    root: Path
    data: dict[str, Any]
    config: dict[str, Any]

    @property
    def dataset_name(self) -> str:
        source_dataset = self.data.get("source_dataset")
        if source_dataset:
            if "meituan" in source_dataset.lower():
                return "meituan_trd_v2"
            return str(source_dataset).strip().lower().replace(" ", "_")
        if self.data.get("source_repository"):
            return "omnibehavior_v2"
        return self.root.name.lower()

    @property
    def schema_version(self) -> str:
        return str(self.data["schema_version"])

    @property
    def shards(self) -> tuple[dict[str, Any], ...]:
        return tuple(self.data["shards"])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file_checksum(path: str | Path, expected: str) -> None:
    actual = _sha256(Path(path))
    if actual != expected:
        p = Path(path)
        try:
            raw = p.read_bytes()
            alt = (
                hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()
                if b"\r\n" in raw
                else hashlib.sha256(raw.replace(b"\n", b"\r\n")).hexdigest()
            )
            if alt == expected:
                return
        except OSError:
            pass
        raise ManifestError(f"Checksum mismatch for {path}: expected {expected}, got {actual}")


def _safe_path(root: Path, relative: str, *, field: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ManifestError(f"manifest.{field} must be a non-empty relative path")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ManifestError(f"manifest.{field} escapes dataset root: {relative!r}")
    return path


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ManifestError(f"Missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ManifestError(f"Cannot parse {label}: {path}: {error}") from error
    if not isinstance(value, dict):
        raise ManifestError(f"{label} must contain a JSON object: {path}")
    return value


def load_manifest(
    dataset_root: str | Path,
    *,
    validate_shard_files: bool = True,
    verify_checksums: bool = False,
) -> DatasetManifest:
    root = Path(dataset_root).resolve()
    if not root.is_dir():
        raise ManifestError(f"Dataset root does not exist: {root}")

    manifest = _read_json(root / "manifest.json", "manifest")
    required = ("status", "schema_version", "config_sha256", "shards")
    for field in required:
        if field not in manifest:
            raise ManifestError(f"manifest.{field} is required")
    if manifest["status"] != "complete":
        raise ManifestError(f"manifest.status must be 'complete', got {manifest['status']!r}")
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ManifestError(
            f"Unsupported manifest.schema_version: {manifest['schema_version']!r}; "
            f"expected {SCHEMA_VERSION!r}"
        )
    if not isinstance(manifest["shards"], list):
        raise ManifestError("manifest.shards must be a list")

    config_path = root / "preprocessing_config.json"
    config = _read_json(config_path, "preprocessing config")
    actual_checksum = _sha256(config_path)
    if actual_checksum != manifest["config_sha256"]:
        raw_bytes = config_path.read_bytes()
        alt_bytes = (
            raw_bytes.replace(b"\r\n", b"\n")
            if b"\r\n" in raw_bytes
            else raw_bytes.replace(b"\n", b"\r\n")
        )
        alt_checksum = hashlib.sha256(alt_bytes).hexdigest()
        if alt_checksum != manifest["config_sha256"]:
            raise ManifestError(
                f"Preprocessing config checksum mismatch: expected {manifest['config_sha256']}, "
                f"got {actual_checksum}"
            )
    if config.get("schema_version") not in (None, SCHEMA_VERSION):
        raise ManifestError(f"preprocessing_config.schema_version must be {SCHEMA_VERSION!r}")

    for index, shard in enumerate(manifest["shards"]):
        if not isinstance(shard, dict):
            raise ManifestError(f"manifest.shards[{index}] must be an object")
        for field in ("user_id", "events_file", "snapshots_file", "snapshots_by_split"):
            if field not in shard:
                raise ManifestError(f"manifest.shards[{index}].{field} is required")
        for field in ("events_file", "snapshots_file"):
            path = _safe_path(root, shard[field], field=f"shards[{index}].{field}")
            if validate_shard_files and not path.is_file():
                raise ManifestError(f"Missing shard file: {path}")
            if verify_checksums:
                checksum_field = "events_sha256" if field == "events_file" else "snapshots_sha256"
                expected = shard.get(checksum_field)
                if not isinstance(expected, str):
                    raise ManifestError(f"manifest.shards[{index}].{checksum_field} is required")
                verify_file_checksum(path, expected)
        if not isinstance(shard["snapshots_by_split"], dict):
            raise ManifestError(f"manifest.shards[{index}].snapshots_by_split must be an object")

    return DatasetManifest(root=root, data=manifest, config=config)


def validate_parquet_schema(
    path: str | Path, *, kind: str, strict_types: bool = False
) -> None:
    """Validate required columns without imposing source-specific value semantics."""

    required = {"events": EVENT_COLUMNS, "snapshots": SNAPSHOT_COLUMNS}.get(kind)
    if required is None:
        raise ValueError(f"Unknown Parquet kind: {kind!r}")
    parquet_path = Path(path)
    if not parquet_path.is_file():
        raise ManifestError(f"Missing {kind} Parquet file: {parquet_path}")
    try:
        schema = pq.ParquetFile(parquet_path).schema_arrow
        columns = set(schema.names)
    except Exception as error:  # pyarrow exposes several concrete read errors
        raise ManifestError(f"Cannot read {kind} Parquet file {parquet_path}: {error}") from error
    missing = sorted(required - columns)
    if missing:
        raise ManifestError(
            f"{kind} Parquet schema missing required columns {missing}: {parquet_path}"
        )
    if not strict_types:
        return
    string_columns = {
        "event_id", "user_id", "source_record_id", "time_precision", "scenario", "event_type",
        "context_json", "action_json", "sample_id", "split", "profile_json",
    }
    integer_columns = {"source_record_index", "action_index", "source_action_count", "history_start", "history_stop"}
    boolean_columns = {"eligibility_inclusive", "cutoff_inclusive", "history_order_known", "history_truncated", "history_group_truncated"}
    timestamp_columns = {"event_time", "eligibility_time", "cutoff_time"}
    for name in schema.names:
        field_type = schema.field(name).type
        expected = None
        if name in string_columns:
            expected = pa.types.is_string(field_type) or pa.types.is_large_string(field_type)
        elif name in integer_columns:
            expected = pa.types.is_integer(field_type)
        elif name in boolean_columns:
            expected = pa.types.is_boolean(field_type)
        elif name in timestamp_columns:
            expected = pa.types.is_timestamp(field_type)
            if name == "event_time":
                expected = expected or pa.types.is_string(field_type) or pa.types.is_large_string(field_type)
        elif name == "history_event_ids":
            expected = pa.types.is_list(field_type) or pa.types.is_large_list(field_type)
        if expected is False:
            raise ManifestError(
                f"{kind} Parquet column {name!r} has incompatible type {field_type}: {parquet_path}"
            )
