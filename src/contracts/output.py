"""Output Contract v1: reject invalid predictions and publish complete runs atomically."""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch

PREDICTION_SCHEMA = pa.schema([
    ("schema_version", pa.int32()), ("task", pa.string()), ("split", pa.string()),
    ("data_version", pa.string()), ("run_id", pa.string()), ("request_id", pa.string()),
    ("user_id", pa.string()), ("cutoff_timestamp", pa.int64()),
    ("candidate_id", pa.string()), ("score", pa.float64()),
])
MANIFEST_KEYS = frozenset({
    "contract_version", "run_id", "task", "split", "data_version", "data_manifest_sha256",
    "preprocess_fingerprint", "config_sha256", "code_revision", "seed", "created_at_utc",
    "foundation", "candidate_or_exposure_policy", "row_count", "request_count", "prediction_sha256",
})


class OutputContractError(ValueError):
    code = 5


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(",", ":")).encode()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _error(manifest, rule):
    raise OutputContractError(
        f"run_id={manifest.get('run_id')} task={manifest.get('task')} "
        f"split={manifest.get('split')}: {rule}"
    )


def _hash(value):
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def _utc_timestamp(value):
    if not isinstance(value, str) or not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|\+00:00)", value
    ):
        return False
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).utcoffset().total_seconds() == 0
    except ValueError:
        return False


def validate_run_manifest(manifest):
    missing = sorted(MANIFEST_KEYS - set(manifest))
    if missing:
        _error(manifest, f"manifest missing keys: {missing}")
    if manifest["contract_version"] != 1:
        _error(manifest, "contract_version must be 1")
    if manifest["task"] not in {"food_recommendation", "ad_optimization"}:
        _error(manifest, "task is not a supported downstream task")
    if manifest["split"] not in {"validation", "test"}:
        _error(manifest, "split must be validation or test")
    run_id = manifest["run_id"]
    if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", run_id):
        _error(manifest, "run_id must be a safe unique directory identifier")
    if run_id == "latest":
        _error(manifest, "run_id cannot be latest")
    for name in ("data_version", "code_revision", "candidate_or_exposure_policy"):
        if not isinstance(manifest[name], str) or not manifest[name].strip():
            _error(manifest, f"{name} must be non-empty")
    for name in ("data_manifest_sha256", "preprocess_fingerprint", "config_sha256", "prediction_sha256"):
        if not _hash(manifest[name]):
            _error(manifest, f"{name} must be SHA-256")
    for name in ("row_count", "request_count", "seed"):
        if type(manifest[name]) is not int or (name != "seed" and manifest[name] < 1):
            _error(manifest, f"{name} must be a valid integer")
    if not _utc_timestamp(manifest["created_at_utc"]):
        _error(manifest, "created_at_utc must be RFC-3339 UTC")
    foundation = manifest["foundation"]
    if not isinstance(foundation, dict) or not {
        "checkpoint_path", "checkpoint_sha256", "format_version", "freeze_foundation"
    } <= set(foundation):
        _error(manifest, "foundation checkpoint reference incomplete")
    if foundation["format_version"] != 1 or type(foundation["freeze_foundation"]) is not bool:
        _error(manifest, "foundation format_version/freeze_foundation invalid")
    if not _hash(foundation["checkpoint_sha256"]) or not foundation["checkpoint_path"]:
        _error(manifest, "checkpoint path/hash missing")
    try:
        if sha256_file(foundation["checkpoint_path"]) != foundation["checkpoint_sha256"]:
            _error(manifest, "checkpoint SHA-256 mismatch")
    except OSError:
        _error(manifest, "foundation checkpoint path does not exist")
    try:
        json.dumps(manifest, allow_nan=False)
    except (ValueError, TypeError):
        _error(manifest, "manifest must be finite JSON")


def validate_predictions(table, *, expected_candidates, run_manifest):
    """Check schema, exact candidate membership, identity, ordering and score semantics."""
    manifest = run_manifest
    validate_run_manifest(manifest)
    allowed = set(PREDICTION_SCHEMA.names) | {"raw_logit"}
    if not isinstance(table, pa.Table) or not set(PREDICTION_SCHEMA.names) <= set(table.column_names):
        _error(manifest, "prediction table missing required schema columns")
    if set(table.column_names) - allowed or len(set(table.column_names)) != len(table.column_names):
        _error(manifest, "unexpected/duplicate prediction columns; labels/features forbidden")
    for field in PREDICTION_SCHEMA:
        if table.schema.field(field.name).type != field.type or table[field.name].null_count:
            _error(manifest, f"{field.name}: expected non-null {field.type}")
    if "raw_logit" in table.column_names and (
        table.schema.field("raw_logit").type != pa.float64() or table["raw_logit"].null_count
    ):
        _error(manifest, "raw_logit: expected non-null float64")
    expected = {}
    for row in expected_candidates:
        try:
            key = (row["request_id"], row["candidate_id"])
            identity = (row["user_id"], row["cutoff_timestamp"])
        except KeyError as error:
            _error(manifest, f"expected candidate handoff missing {error}")
        if key in expected:
            _error(manifest, "duplicate candidate in expected Data handoff")
        if row.get("split", manifest["split"]) != manifest["split"]:
            _error(manifest, "expected handoff split mismatch")
        expected[key] = identity
    rows = table.to_pylist()
    seen, requests = set(), {}
    for row in rows:
        for field in ("task", "split", "run_id", "data_version"):
            if row[field] != manifest[field]:
                _error(manifest, f"{field} does not match run manifest")
        if row["schema_version"] != 1:
            _error(manifest, "schema_version must be 1")
        for field in ("request_id", "candidate_id", "user_id"):
            if not row[field].strip():
                _error(manifest, f"{field} must be non-empty")
        score = row["score"]
        if not math.isfinite(score) or ("raw_logit" in row and not math.isfinite(row["raw_logit"])):
            _error(manifest, "NaN/Inf prediction")
        if manifest["task"] == "ad_optimization" and not 0 <= score <= 1:
            _error(manifest, "Ads score outside [0,1]")
        key = (row["request_id"], row["candidate_id"])
        if key in seen:
            _error(manifest, f"duplicate candidate; request_id={row['request_id']}")
        seen.add(key)
        identity = (row["user_id"], row["cutoff_timestamp"])
        if expected.get(key) != identity:
            _error(manifest, f"extra candidate/user/timestamp mismatch; request_id={row['request_id']}")
        if requests.setdefault(row["request_id"], identity) != identity:
            _error(manifest, "request user/cutoff alignment mismatch")
    if seen != set(expected):
        _error(manifest, "missing/extra candidate set")
    if rows != sorted(rows, key=lambda row: (row["request_id"], -row["score"], row["candidate_id"])):
        _error(manifest, "prediction ordering must be request ASC, score DESC, candidate ASC")
    if len(rows) != manifest["row_count"] or len(requests) != manifest["request_count"]:
        _error(manifest, "manifest row/request count mismatch")
    return {
        "status": "passed", "row_count": len(rows), "request_count": len(requests),
        "duplicate_candidate_count": 0, "missing_candidate_count": 0, "extra_candidate_count": 0,
        "nan_inf_count": 0, "timestamp_mismatch_count": 0,
        "score_range": [min(row["score"] for row in rows), max(row["score"] for row in rows)],
        "validated_at_utc": utc_now(),
    }


def validate_foundation_output(output, batch):
    inputs = getattr(batch, "input_ids", None)
    size = inputs.shape[0] if isinstance(inputs, torch.Tensor) and inputs.ndim else "unknown"

    def reject(name, expected, actual):
        if isinstance(actual, torch.Tensor):
            description = f"shape={tuple(actual.shape)} dtype={actual.dtype} device={actual.device}"
        else:
            description = f"type={type(actual).__name__}"
        raise ValueError(f"batch_size={size} {name}: expected {expected}; actual {description}")

    if not isinstance(inputs, torch.Tensor) or inputs.ndim != 2:
        reject("input_ids", "tensor [B, L]", inputs)
    size, length = inputs.shape
    mask = getattr(batch, "attention_mask", None)
    if not isinstance(mask, torch.Tensor) or (
        mask.shape != inputs.shape or mask.dtype != torch.bool or mask.device != inputs.device
    ):
        reject("attention_mask", f"bool {(size, length)} on batch device", mask)
    hidden = getattr(output, "hidden_states", None)
    if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3 or hidden.shape[-1] < 1:
        reject("hidden_states", f"float32 [{size}, {length}, D] with D > 0", hidden)
    dimension = hidden.shape[-1]
    for name, expected in (("hidden_states", (size, length, dimension)),
                           ("long_term", (size, dimension)), ("short_term", (size, dimension))):
        tensor = getattr(output, name, None)
        if not isinstance(tensor, torch.Tensor) or (
            tensor.shape != expected or tensor.dtype != torch.float32 or tensor.device != inputs.device
            or tensor.layout != torch.strided
        ):
            reject(name, f"dense float32 {expected} on batch device", tensor)
        checked = tensor[mask] if name == "hidden_states" else tensor
        if not torch.isfinite(checked).all():
            raise ValueError(f"batch_size={size} {name}: expected finite, actual NaN/Inf")
    padding = getattr(output, "padding_mask", None)
    if not isinstance(padding, torch.Tensor) or (
        padding.shape != mask.shape or padding.dtype != torch.bool or padding.device != mask.device
        or not torch.equal(padding, mask)
    ):
        reject("padding_mask", f"bool {(size, length)} equal to attention_mask on batch device", padding)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8")


def _validate_train_metrics(records, manifest):
    required = {"epoch", "step", "split", "total_loss", "target_counts", "seed", "timestamp_utc"}
    for record in records:
        if not isinstance(record, Mapping) or not required <= set(record):
            _error(manifest, "train_metrics missing epoch/step/split/loss/counts/seed/UTC timestamp")
        for name in ("epoch", "step"):
            if type(record[name]) is not int or record[name] < 0:
                _error(manifest, f"train_metrics {name} must be a nonnegative integer")
        if type(record["seed"]) is not int or record["seed"] != manifest["seed"]:
            _error(manifest, "train_metrics seed does not match run manifest")
        if record["split"] not in {"train", "validation", "test"}:
            _error(manifest, "train_metrics split invalid")
        if not _utc_timestamp(record["timestamp_utc"]):
            _error(manifest, "train_metrics timestamp_utc must be RFC-3339 UTC")
        counts = record["target_counts"]
        if not isinstance(counts, Mapping) or any(
            not isinstance(key, str) or not key or type(value) is not int or value < 0
            for key, value in counts.items()
        ):
            _error(manifest, "train_metrics target_counts must map names to nonnegative integers")
        losses = [key for key in record if isinstance(key, str) and key.endswith("_loss")]
        if not any(key != "total_loss" for key in losses):
            _error(manifest, "train_metrics must include named loss components")
        if any(type(record[key]) not in {int, float} or not math.isfinite(record[key]) for key in losses):
            _error(manifest, "train_metrics losses must be finite numbers")
        try:
            json.dumps(record, allow_nan=False)
        except (ValueError, TypeError):
            _error(manifest, "train_metrics must be finite JSON")


def validate_and_write_predictions(table, *, expected_candidates, run_manifest,
                                   train_metrics, checkpoint_ref, artifact_dir):
    """Stage Parquet, calculate byte hash, validate, then atomically publish the run."""
    manifest = dict(run_manifest)
    # Reject path traversal before touching disk; full manifest is validated after hashing Parquet.
    identifier = manifest.get("run_id", "")
    if not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", identifier):
        _error(manifest, "unsafe run_id")
    if manifest.get("task") not in {"ad_optimization", "food_recommendation"}:
        _error(manifest, "unsupported task")
    root = Path(artifact_dir).resolve() / manifest["task"]
    root.mkdir(parents=True, exist_ok=True)
    destination = root / identifier
    if destination.exists():
        _error(manifest, "run already exists; refusing overwrite")
    stage = root / (".pending-" + str(uuid.uuid4()))
    stage.mkdir()
    try:
        train_metrics = list(train_metrics)
        expected_candidates = list(expected_candidates)
        split = manifest.get("split")
        if split not in {"validation", "test"}:
            _error(manifest, "invalid split")
        path = stage / f"predictions.{split}.parquet"
        pq.write_table(table, path, compression="zstd")
        manifest["prediction_sha256"] = sha256_file(path)
        # Round-trip validation checks the artifact that will actually be consumed.
        report = validate_predictions(pq.read_table(path), expected_candidates=expected_candidates,
                                      run_manifest=manifest)
        if checkpoint_ref != manifest["foundation"]:
            _error(manifest, "checkpoint_ref differs from foundation manifest")
        _validate_train_metrics(train_metrics, manifest)
        write_json(stage / "run_manifest.json", manifest)
        write_json(stage / "validation_report.json", report)
        write_json(stage / "checkpoint_ref.json", checkpoint_ref)
        (stage / "train_metrics.jsonl").write_text("".join(
            json.dumps(record, sort_keys=True, allow_nan=False) + "\n" for record in train_metrics
        ), encoding="utf-8")
        if destination.exists():
            _error(manifest, "run appeared during write; refusing overwrite")
        stage.rename(destination)
        return destination / path.name
    finally:
        if stage.parent == root and stage.name.startswith(".pending-") and stage.exists():
            shutil.rmtree(stage)
