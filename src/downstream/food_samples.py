"""Read and validate the Food ``samples.jsonl`` handoff (Meituan SPU ranking or simulator).

The simulator package (``data.format: sim_v1``, docs/simulation_data_contract.md "Food") extends
the Meituan record format; its history goes through ``src.downstream.sim_events`` so it encodes
exactly like the pretrain package.
"""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.downstream.sim_events import (
    SIM_DATASET_NAME,
    food_history_events,
    sim_canonical_sample,
)
from src.foundation_input.types import CanonicalSample, EventRecord

SPLITS = ("train", "validation", "test")
MAX_REPORTED_ERRORS = 20
MAPPING_VERSION = "meituan-food-samples-to-trd-v2-v1"
SIM_MAPPING_VERSION = "sim-v1-food-samples-via-sim-events-v1"
_SIM_FLAGS = ("history_truncated", "history_group_truncated")
_UNKNOWN_TAGS = {"", "NULL", "未知"}
_ZERO_PROFILE_COUNTS = (
    "advertisement_event_count_30d",
    "customer_service_event_count_30d",
    "live_streaming_event_count_30d",
    "search_behavior_event_count_30d",
    "video_browsing_event_count_30d",
)
_REQUIRED_FIELDS = (
    "sample_id",
    "input.user_id", "input.cutoff_time", "input.profile", "input.history",
    "input.context", "input.candidates",
    "labels",
    "label_metadata.request_id", "label_metadata.split", "label_metadata.label_time",
    "label_metadata.candidate_source",
)


def parse_utc(value: str, *, field: str, request_id: str) -> datetime:
    """Parse an ISO-8601 timestamp with explicit timezone into whole-second UTC."""
    prefix = f"request_id={request_id}: {field}"
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError, AttributeError):
        raise ValueError(f"{prefix} is not an ISO-8601 timestamp: {value!r}") from None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{prefix} has no explicit timezone (Z or offset): {value!r}")
    if parsed.microsecond:
        raise ValueError(f"{prefix} must be whole seconds: {value!r}")
    return parsed.astimezone(timezone.utc)


def read_food_samples(
    path: str | Path, member: str, expected_sha256: str
) -> tuple[list[dict], str]:
    """Read samples.jsonl from a zip member (or a plain file) and check its SHA-256."""
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"Food samples file {source} not found")
    if source.suffix == ".zip" or zipfile.is_zipfile(source):
        try:
            with zipfile.ZipFile(source) as archive:
                if member not in archive.namelist():
                    raise ValueError(f"Food samples member {member} not found in {source}")
                payload = archive.read(member)
        except zipfile.BadZipFile as exc:
            raise ValueError(f"Food samples zip {source} is corrupt (member {member}): {exc}") from None
    else:
        payload = source.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != expected_sha256:
        raise ValueError(
            f"Food samples SHA-256 mismatch for {member}: expected {expected_sha256}, got {digest}"
        )
    records = []
    for number, line in enumerate(payload.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Food samples {member} line {number} is not valid JSON: {exc}") from None
    return records, digest


def _lookup(record: dict, dotted: str):
    node = record
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return False, None
        node = node[part]
    return True, node


def _validate_record(record, candidate_policy, seen_ids):
    """Check rules 2-8 and 11 for one record; return (cutoff, split) for the split check."""
    sample_id = record.get("sample_id") if isinstance(record, dict) else None
    rid = str(sample_id) if isinstance(sample_id, (str, int)) else "<unknown>"
    if not isinstance(record, dict):
        raise ValueError(f"request_id={rid}: record is not a JSON object")  # noqa: TRY004
    for dotted in _REQUIRED_FIELDS:
        if not _lookup(record, dotted)[0]:
            raise ValueError(f"request_id={rid}: missing field {dotted}")
    meta, inp = record["label_metadata"], record["input"]
    if not isinstance(sample_id, (str, int)):
        raise ValueError(f"request_id={rid}: sample_id must be a string or integer")  # noqa: TRY004
    if sample_id in seen_ids:
        raise ValueError(f"request_id={rid}: duplicate sample_id")
    seen_ids.add(sample_id)
    if not isinstance(meta, dict) or not isinstance(inp, dict) or not isinstance(inp["profile"], dict):
        raise ValueError(f"request_id={rid}: input, profile and label_metadata must be objects")  # noqa: TRY004
    for name, value in (("history", inp["history"]), ("candidates", inp["candidates"]),
                        ("labels", record["labels"])):
        if not isinstance(value, list):
            raise ValueError(f"request_id={rid}: {name} must be a list")  # noqa: TRY004
    for name, items in (("history event", inp["history"]), ("candidate", inp["candidates"]),
                        ("label", record["labels"])):
        if not all(isinstance(item, dict) for item in items):
            raise ValueError(f"request_id={rid}: every {name} must be an object")
    if meta["request_id"] != sample_id:
        raise ValueError(f"request_id={rid}: request_id differs from sample_id")
    split = meta["split"]
    if split not in SPLITS:
        raise ValueError(f"request_id={rid}: invalid split {split!r}")
    if meta["candidate_source"] != candidate_policy:
        raise ValueError(
            f"request_id={rid}: candidate policy mismatch "
            f"(got {meta['candidate_source']!r}, expected {candidate_policy!r})"
        )
    cutoff = parse_utc(inp["cutoff_time"], field="cutoff_time", request_id=rid)
    label_time = parse_utc(meta["label_time"], field="label_time", request_id=rid)
    if inp["profile"].get("as_of_time") is not None:
        parse_utc(inp["profile"]["as_of_time"], field="profile.as_of_time", request_id=rid)
    event_times = [
        parse_utc(event.get("event_time"), field="event_time", request_id=rid)
        for event in inp["history"]
    ]
    candidate_ids = [c.get("item_id") for c in inp["candidates"]]
    label_ids = [x.get("item_id") for x in record["labels"]]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError(f"request_id={rid}: duplicate candidate item_id")
    if candidate_ids != label_ids:
        raise ValueError(f"request_id={rid}: candidates and labels differ")
    relevances = [x.get("relevance") for x in record["labels"]]
    if any(rel not in (0, 1) for rel in relevances):
        raise ValueError(f"request_id={rid}: relevance must be 0 or 1")
    if 1 not in relevances:
        raise ValueError(f"request_id={rid}: no positive label")
    if any(moment >= cutoff for moment in event_times):
        raise ValueError(f"request_id={rid}: history event at or after cutoff")
    if label_time <= cutoff:
        raise ValueError(f"request_id={rid}: label_time must be after cutoff")
    return cutoff, split


def validate_food_samples(records: list[dict], candidate_policy: str) -> None:
    """Apply spec section 4 rules 2-9 and 11; raise ValueError listing offending request_ids."""
    errors: list[str] = []
    seen_ids: set = set()
    cutoffs: dict[str, list[datetime]] = {split: [] for split in SPLITS}
    for record in records:
        try:
            cutoff, split = _validate_record(record, candidate_policy, seen_ids)
        except ValueError as exc:
            errors.append(str(exc))
            _collect_cutoff(record, cutoffs)
            continue
        cutoffs[split].append(cutoff)
    for earlier, later in (("train", "validation"), ("validation", "test")):
        if cutoffs[earlier] and cutoffs[later] and max(cutoffs[earlier]) >= min(cutoffs[later]):
            errors.append(f"temporal split overlap: {earlier} cutoffs reach {later} cutoffs")
    if errors:
        shown = errors[:MAX_REPORTED_ERRORS]
        extra = len(errors) - len(shown)
        suffix = f"; ... and {extra} more" if extra else ""
        raise ValueError("Invalid Food samples: " + "; ".join(shown) + suffix)


def _collect_cutoff(record, cutoffs) -> None:
    """Best-effort cutoff for the temporal check when another rule already failed."""
    try:
        split = record["label_metadata"]["split"]
        cutoff = parse_utc(record["input"]["cutoff_time"], field="cutoff_time", request_id="?")
    except (KeyError, TypeError, ValueError):
        return
    if split in cutoffs:
        cutoffs[split].append(cutoff)


def _clean_tag(value: str | None) -> str:
    return "unknown" if value is None or value.strip() in _UNKNOWN_TAGS else value.strip()


def load_poi_catalog(
    path: str | Path, expected_sha256: str
) -> tuple[dict[str, dict[str, str]], str]:
    """Load pois.txt keyed by wm_poi_id (columns read by header name) and check its SHA-256."""
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"POI catalog {source} not found")
    payload = source.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != expected_sha256:
        raise ValueError(
            f"POI catalog SHA-256 mismatch for {source}: expected {expected_sha256}, got {digest}"
        )
    lines = payload.decode("utf-8").splitlines()
    if not lines:
        raise ValueError(f"POI catalog {source} is empty")
    header = lines[0].lstrip("﻿").split("	")
    wanted = (
        "wm_poi_id", "wm_poi_name", "primary_first_tag_name",
        "primary_second_tag_name", "primary_third_tag_name",
    )
    missing = [column for column in wanted if column not in header]
    if missing:
        raise ValueError(f"POI catalog {source} is missing columns {missing}")
    index = {column: header.index(column) for column in wanted}
    catalog: dict[str, dict[str, str]] = {}
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.split("	")
        if len(parts) < len(header):
            parts += [""] * (len(header) - len(parts))
        catalog[parts[index["wm_poi_id"]].strip()] = {
            "name": parts[index["wm_poi_name"]],
            "tag1": _clean_tag(parts[index["primary_first_tag_name"]]),
            "tag2": _clean_tag(parts[index["primary_second_tag_name"]]),
            "tag3": _clean_tag(parts[index["primary_third_tag_name"]]),
        }
    return catalog, digest


def _bare_id(value: str) -> str:
    return str(value).rsplit(":", 1)[-1]


def _poi(restaurant_id, catalog: dict, request_id: str) -> dict[str, str]:
    poi = catalog.get(_bare_id(restaurant_id)) if restaurant_id is not None else None
    if poi is None:
        raise ValueError(f"request_id={request_id}: unknown restaurant {restaurant_id!r}")
    return poi


def to_canonical_sample(record: dict, catalog: dict, namespace: str) -> CanonicalSample:
    """Map one samples.jsonl record to the Person 3 checkpoint schema (spec section 5)."""
    rid = str(record["sample_id"])
    inp = record["input"]
    cutoff = parse_utc(inp["cutoff_time"], field="cutoff_time", request_id=rid)
    events = []
    for position, event in enumerate(inp["history"]):
        poi = _poi(event.get("restaurant_id"), catalog, rid)
        moment = parse_utc(event.get("event_time"), field="event_time", request_id=rid)
        events.append(EventRecord(
            event_id=event["event_id"],
            source_record_id=event["event_id"],
            source_record_index=position,
            action_index=0,
            source_action_count=1,
            event_time=moment,
            time_precision="second",
            eligibility_time=moment,
            eligibility_inclusive=True,
            scenario="E-commerce",
            event_type="purchase",
            context={
                "product_source": "meituan_takeout",
                "product_id": _bare_id(event["restaurant_id"]),
                "item_title": poi["name"],
                "item_desc": "",
                "external_item_title": "",
                "item_price": None,
                "category_level1_name": poi["tag1"],
                "category_level2_name": poi["tag2"],
                "category_level3_name": poi["tag3"],
                "industry_level1_name": "Food Delivery",
                "industry_level2_name": poi["tag1"],
                "industry_level3_name": poi["tag2"],
            },
            action={"type": "purchase", "is_pay": True, "item_cnt": 1, "order_amt": None},
            user_id=_bare_id(inp["user_id"]),
        ))
    source = inp["profile"]
    as_of = source.get("as_of_time")
    if as_of is not None:  # normalize to UTC: validate_sample strips tzinfo without converting
        as_of = parse_utc(as_of, field="profile.as_of_time", request_id=rid).isoformat()
    profile = {
        "as_of_time": as_of,
        "active_days_30d": source.get("active_days_30d"),
        "event_count_30d": source.get("order_count_30d"),
        "e_commerce_event_count_30d": source.get("order_count_30d"),
    }
    profile.update({key: 0 for key in _ZERO_PROFILE_COUNTS})
    return CanonicalSample(
        sample_id=rid,
        dataset_name=namespace,
        split=record["label_metadata"]["split"],
        user_id=_bare_id(inp["user_id"]),
        cutoff_time=cutoff,
        cutoff_inclusive=False,
        history_order_known=True,
        history_truncated=False,
        history_group_truncated=False,
        profile=profile,
        events=tuple(events),
    )


def _request_rows(record: dict, check_candidate) -> list[dict]:
    """One prediction-table row per candidate; ``check_candidate(candidate, request_id)``."""
    rid = str(record["sample_id"])
    inp, meta = record["input"], record["label_metadata"]
    cutoff = int(parse_utc(inp["cutoff_time"], field="cutoff_time", request_id=rid).timestamp())
    label_at = int(parse_utc(meta["label_time"], field="label_time", request_id=rid).timestamp())
    relevance = {label["item_id"]: int(label["relevance"]) for label in record["labels"]}
    rows = []
    for candidate in inp["candidates"]:
        check_candidate(candidate, rid)
        rows.append({
            "request_id": rid,
            "user_id": inp["user_id"],
            "cutoff_timestamp": cutoff,
            "candidate_id": candidate["item_id"],
            "is_positive": relevance[candidate["item_id"]],
            "split": meta["split"],
            "label_available_at_timestamp": label_at,
        })
    return rows


def candidate_rows(record: dict, catalog: dict) -> list[dict]:
    """One row per candidate for the prediction table; checks candidate restaurants (rule 10)."""
    return _request_rows(
        record, lambda candidate, rid: _poi(candidate.get("restaurant_id"), catalog, rid)
    )


# ---- simulator package (data.format: sim_v1) -------------------------------------------------


def _sim_manifest_problem(manifest: dict, expected_version: str) -> str | None:
    if manifest.get("source_dataset") != SIM_DATASET_NAME:
        return (f"source_dataset must be {SIM_DATASET_NAME!r}, "
                f"got {manifest.get('source_dataset')!r}")
    if manifest.get("dataset_version") != expected_version:
        return (f"dataset_version {manifest.get('dataset_version')!r} differs from "
                f"data.expected_version {expected_version!r}")
    sim_config = manifest.get("sim_config_sha256")
    if not isinstance(sim_config, str) or not re.fullmatch(r"[0-9a-f]{64}", sim_config):
        return "sim_config_sha256 must be a SHA-256"
    if not isinstance(manifest.get("candidate_policy"), str) or not manifest["candidate_policy"]:
        return "candidate_policy must be a non-empty string"
    files = manifest.get("files")
    for name in ("samples", "merchants"):
        entry = files.get(name) if isinstance(files, dict) else None
        if not isinstance(entry, dict) or not all(
            isinstance(entry.get(key), str) and entry[key] for key in ("path", "sha256")
        ):
            return f"files.{name} needs path and sha256"
    return None


def read_sim_food_manifest(root: str | Path, expected_version: str) -> tuple[dict, str]:
    """Read and check ``food/manifest.json`` of a simulator package; return it and its SHA-256."""
    path = Path(root) / "manifest.json"
    if not path.is_file():
        raise ValueError(f"Food sim manifest {path} not found")
    payload = path.read_bytes()
    try:
        manifest = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Food sim manifest {path} is not valid JSON: {exc}") from None
    problem = ("must be a JSON object" if not isinstance(manifest, dict)
               else _sim_manifest_problem(manifest, expected_version))
    if problem:
        raise ValueError(f"Food sim manifest {path}: {problem}")
    return manifest, hashlib.sha256(payload).hexdigest()


def load_merchant_catalog(path: str | Path, expected_sha256: str) -> dict[str, dict]:
    """Load ``merchants.parquet`` keyed by ``merchant_id`` and check its SHA-256."""
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"Merchant catalog {source} not found")
    payload = source.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != expected_sha256:
        raise ValueError(
            f"Merchant catalog SHA-256 mismatch for {source}: "
            f"expected {expected_sha256}, got {digest}"
        )
    try:
        rows = pq.read_table(pa.BufferReader(payload)).to_pylist()
    except (pa.ArrowException, OSError) as exc:
        raise ValueError(f"Merchant catalog {source} is not readable Parquet: {exc}") from None
    catalog: dict[str, dict] = {}
    for row in rows:
        merchant = row.get("merchant_id")
        if not isinstance(merchant, str) or not merchant:
            raise ValueError(f"Merchant catalog {source} has a row without merchant_id")
        if merchant in catalog:
            raise ValueError(f"Merchant catalog {source} has duplicate merchant_id {merchant!r}")
        catalog[merchant] = row
    return catalog


def sim_to_canonical_sample(record: dict, dataset_name: str) -> CanonicalSample:
    """Map one sim Food record like the pretrain loader: full ids, profile and flags as given."""
    rid = str(record["sample_id"])
    inp = record["input"]
    for flag in _SIM_FLAGS:
        if type(inp.get(flag)) is not bool:
            raise ValueError(f"request_id={rid}: input.{flag} must be a boolean")
    try:
        events = food_history_events(record)
    except (KeyError, TypeError) as exc:
        raise ValueError(f"request_id={rid}: malformed history event ({exc!r})") from exc
    return sim_canonical_sample(
        sample_id=rid,
        dataset_name=dataset_name,
        split=record["label_metadata"]["split"],
        user_id=str(inp["user_id"]),
        cutoff_time=parse_utc(inp["cutoff_time"], field="cutoff_time", request_id=rid),
        profile=inp["profile"],
        events=events,
        history_truncated=inp["history_truncated"],
        history_group_truncated=inp["history_group_truncated"],
    )


def sim_candidate_rows(record: dict, catalog: dict) -> list[dict]:
    """Rows shaped like ``candidate_rows``; every candidate must be a catalog merchant."""

    def check(candidate: dict, rid: str) -> None:
        item = candidate.get("item_id")
        if not isinstance(item, str) or item not in catalog:
            raise ValueError(f"request_id={rid}: unknown merchant {item!r}")

    return _request_rows(record, check)
