"""Post-export checks (spec §8) and ``data_report.json`` helpers.

``run_checks(out_dir, config)`` re-reads the EXPORTED files under ``out_dir`` (the
``<out>/<dataset_version>`` directory), never the in-memory simulation, and enforces the spec §8
"no leak" and "format" rules across the five packages. Every check is reported with a status
(``passed`` / ``failed`` / ``skipped`` / ``not_run``), a description, the number of violations and
at most ``MAX_IDS`` offending ids. When any check fails, ``CheckError`` (a ``ValueError``) is
raised; its message names every failed check with its ids and ``error.report`` holds the full
result. A phase that crashes on an unreadable file fails ``<package>_readable`` and marks its
unfinished checks ``not_run``.

Phases, in order (later phases reuse what earlier ones read):

1. Hashes: SHA-256 and size of every file except ``data_report.json`` (also returned for the
   report); manifests are compared against these, nothing is hashed twice.
2. ``pretrain/`` (``skipped`` when absent and not expected): manifest + preprocessing-config SHA
   (``src.foundation_input.manifest.load_manifest``), every shard SHA/size against the manifest,
   no stray files; no event on/after local day ``7 * (weeks - 2) + 1`` (all shards, columnar);
   every snapshot: cutoff before that day, split inside its bounds, profile = exactly
   ``as_of_time`` + spec §4.6 fields with ``as_of_time`` = cutoff, ``history_group_truncated``
   false (ruling R11: whole sessions, so next-action targets stay on), every history id an
   event of the same user at or before the cutoff (one Arrow join over all shards). Loader spot
   check
   (the loader costs ~3.5 ms per sample): every sample of ``pretrain_loader_users`` evenly spaced
   users (copied with a manifest listing only them) goes through ``iter_normalized_samples`` +
   ``validate_sample``; ``None`` loads every user. The columnar checks cover every row.
3. ``ads/``: manifest, exact public columns and ``attributes_json`` keys (public context + numeric
   action fields only: no click flag, session size or action count), event table order, each
   window strictly before the exposure and excluding the request's own impression, window policy
   (``history_start = max(0, stop - max)``, ``stop`` = every event before the exposure), the
   impression row and its label (``clicked`` = the event log's outcome: the user's next row is the
   ``ad_click`` of the same ad 2-26 s later), split by exposure week.
4. ``food/``: manifest (incl. ``popularity_end_exclusive``, used by the difficulty report), row
   format, 10 distinct catalog candidates with labels in candidate order and one positive, inputs
   with exactly the public keys (no current location, distance or hidden state; distances inside
   history attributes are past-event fields), history strictly before ``cutoff_time`` and at
   most ``max_history_events``, own session absent (located in
   ``ads/events.parquet``: the user's Food events in ``(cutoff, label_time]`` must start with the
   ``search`` at ``cutoff + 1 s`` and end with the ``order`` of the positive at ``label_time``, and
   none of their ids may be in the history), ``label_time > cutoff_time``, profile as of the
   cutoff, split by cutoff week.
5. ``oracle/``: Food probabilities sum to 1 per request over exactly the request's candidates;
   Ads ids match the opportunities and probabilities lie in (0, 1).
6. ``baseline/``: each file is streamed once (never held as one string): sections, metadata,
   queries/candidates/labels equal to the Food/Ads packages, label targets and availability (the
   outcome time: Food order, Ads click or exposure + 27 s), every label of a split available
   before the next split's first cutoff (so a B1 fit cutoff exists), candidate attributes with
   the B1 names (ruling R12: Food ``category``/``price``, Ads ``brand``); history strictly before
   the user's latest request cutoff with public attributes only and unique ids, ad impressions
   as B1 ``impression`` rows whose ``clicked`` and ``available_at`` match the event log.
   ``src.baselines.runner.load_dataset`` then loads the whole file when it is at most
   ``baseline_full_load_max_bytes``; a larger file (``full``: ~1.5 GB each) is loaded as a subset
   file of ``BASELINE_SUBSET_QUERIES`` evenly spaced requests with their candidates, labels and
   their users' history, written with the exporter's own writer. A rejection by ``load_dataset``
   fails ``baseline_<name>_load_dataset`` (with the error); the other file is still checked.

``only=`` restricts a run to some packages (targeted tests); the CLI always checks everything.
"""

from __future__ import annotations

import codecs
import hashlib
import json
import tempfile
from array import array
from collections import Counter
from collections.abc import Callable, Collection, Iterator
from dataclasses import dataclass
from datetime import datetime
from itertools import pairwise
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from ..baselines.runner import load_dataset
from ..foundation_input.loader import iter_normalized_samples
from ..foundation_input.manifest import load_manifest
from .behavior import AD_CLICK_DELAY_SECONDS, INTENT_KINDS
from .config import PERSONA_SHARES, SimConfig, config_sha256
from .export_ads import CANDIDATE_POLICY as ADS_CANDIDATE_POLICY
from .export_ads import CATALOG_SCHEMA, OPPORTUNITY_SCHEMA, PROFILE_SCHEMA
from .export_ads import EVENT_SCHEMA as ADS_EVENT_SCHEMA
from .export_baseline import (
    AD_OUTCOME_MATURITY_SECONDS,
    AVAILABILITY_POLICY,
    DATASET_NAME,
    IMPRESSION_EVENT_TYPE,
    _write_document,
)
from .export_food import CANDIDATE_POLICY as FOOD_CANDIDATE_POLICY
from .export_food import MERCHANT_SCHEMA, N_CANDIDATES, POSITIVE_ACTION, PREDICTION_ANCHOR
from .export_pretrain import (
    CONTEXT_ALLOWLIST,
    SCENARIOS,
    SOURCE_DATASET,
    pretrain_end_day,
    split_bounds,
)
from .export_pretrain import SCHEMA_VERSION as PRETRAIN_SCHEMA_VERSION
from .history import PROFILE_FIELDS
from .timeutil import day_start_ts, local_iso, naive_utc

REPORT_NAME = "data_report.json"
PACKAGES = ("pretrain", "ads", "food", "oracle", "baseline")  # in check order
MAX_IDS = 20
BASELINE_FULL_LOAD_MAX_BYTES = 256 * 1024 * 1024
BASELINE_SUBSET_QUERIES = 200
PRETRAIN_LOADER_USERS = 20
PROB_TOLERANCE = 1e-9

SPLITS = ("train", "validation", "test")
# Numeric action fields exported as attributes, per service (spec §6.1 action_json).
ACTION_FIELDS: dict[str, frozenset[str]] = {
    "ride": frozenset({"fare", "distance_km", "wait_minutes"}),
    "food": frozenset({"order_amount", "delivery_distance_km"}),
    "ads": frozenset(),
}
PUBLIC_ATTRIBUTES: dict[str, frozenset[str]] = {
    service: frozenset(CONTEXT_ALLOWLIST[scenario]) | ACTION_FIELDS[service]
    for service, scenario in SCENARIOS.items()
}
PROFILE_KEYS = frozenset({"as_of_time", *PROFILE_FIELDS})
FOOD_ROW_KEYS = frozenset({"sample_id", "input", "labels", "label_metadata"})
FOOD_INPUT_KEYS = frozenset({"user_id", "cutoff_time", "profile", "history", "history_truncated",
                             "history_group_truncated", "context", "candidates"})
FOOD_META_KEYS = frozenset({"request_id", "split", "label_time", "candidate_source",
                            "observation_complete", "prediction_anchor", "positive_action"})
FOOD_CANDIDATE_KEYS = frozenset({"item_id", "cuisine", "price_level", "geohash"})
FOOD_HISTORY_KEYS = frozenset({"event_id", "event_time", "service", "event_type", "entity_id",
                               "attributes"})
CONTEXT_KEYS = frozenset({"hour", "weekday"})
BASELINE_SECTIONS = frozenset({"metadata", "queries", "candidates", "labels", "history"})
BASELINE_QUERY_KEYS = frozenset({"request_id", "user_id", "cutoff_time", "split", "context"})
BASELINE_CANDIDATE_KEYS = frozenset({"request_id", "candidate_id", "attributes"})
BASELINE_LABEL_KEYS = frozenset({"request_id", "candidate_id", "target", "available_at"})
BASELINE_HISTORY_KEYS = frozenset({"event_id", "user_id", "entity_id", "event_type", "event_time",
                                   "available_at", "attributes"})
# B1 conventions (docs/b1_features.md, ruling R12): extra attributes of an event type, and the
# public attribute each B1 name repeats. Ad impressions also carry ``clicked`` (their outcome).
BASELINE_HISTORY_ALIASES: dict[str, dict[str, str]] = {
    IMPRESSION_EVENT_TYPE: {"category": "industry_main", "brand": "industry_sub"},
    "order": {"category": "cuisine", "price": "price_level"},
}
BASELINE_HISTORY_EXTRAS: dict[str, frozenset[str]] = {
    IMPRESSION_EVENT_TYPE: frozenset({"clicked", "category", "brand"}),
    "order": frozenset({"category", "price"}),
}
_BASELINE_EXTRA_KEYS = frozenset().union(*BASELINE_HISTORY_EXTRAS.values())
_EXTRAS_OF_SERVICE = {"ads": BASELINE_HISTORY_EXTRAS[IMPRESSION_EVENT_TYPE],
                      "food": BASELINE_HISTORY_EXTRAS["order"]}
BASELINE_HISTORY_ATTRIBUTES = {
    service: keys | {"service"} | _EXTRAS_OF_SERVICE.get(service, frozenset())
    for service, keys in PUBLIC_ATTRIBUTES.items()
}
BASELINE_ATTRIBUTES = {
    "food": frozenset({"cuisine", "price_level", "geohash", "category", "price"}),
    "ads": frozenset({"category", "advertiser", "title", "brand"}),
}
BASELINE_CANDIDATE_ALIASES = {"food": {"category": "cuisine", "price": "price_level"},
                              "ads": {"brand": "advertiser"}}
BASELINE_TASKS = {"food": ("food_recommendation", "merchant", FOOD_CANDIDATE_POLICY),
                  "ads": ("ad_optimization", "ad", ADS_CANDIDATE_POLICY)}


class CheckError(ValueError):
    """At least one post-export check failed; ``report`` is the full ``run_checks`` result."""

    def __init__(self, message: str, report: dict) -> None:
        super().__init__(message)
        self.report = report


# ---- result bookkeeping ---------------------------------------------------------------------


class _Results:
    def __init__(self) -> None:
        self.checks: dict[str, dict] = {}

    def declare(self, name: str, description: str) -> None:
        self.checks[name] = {"status": "passed", "description": description, "violations": 0}

    def fail(self, name: str, ident) -> None:
        entry = self.checks[name]
        entry["status"] = "failed"
        entry["violations"] += 1
        ids = entry.setdefault("ids", [])
        if len(ids) < MAX_IDS:
            ids.append(str(ident))

    def require(self, name: str, ok: bool, ident) -> bool:
        if not ok:
            self.fail(name, ident)
        return ok

    def note(self, name: str, **detail) -> None:
        self.checks[name].update(detail)

    def set_status(self, name: str, status: str, reason: str) -> None:
        self.checks[name].update(status=status, reason=reason)

    def not_run(self, name: str, reason: str) -> None:
        """Mark a check that could not run; a check that already failed stays failed."""
        if self.checks[name]["status"] == "passed":
            self.set_status(name, "not_run", reason)

    def phase(self, package: str, checks: dict[str, str], fn: Callable, *args):
        """Declare ``<package>_readable`` + ``checks`` and run ``fn``; a crash fails
        ``<package>_readable`` and marks the checks that had not failed ``not_run``."""
        readable = f"{package}_readable"
        self.declare(readable, f"{package} files exist and parse")
        for name, description in checks.items():
            self.declare(name, description)
        try:
            return fn(*args)
        except Exception as error:  # noqa: BLE001 - a broken file is a failed check
            self.checks[readable].update(status="failed", violations=1,
                                         error=f"{type(error).__name__}: {error}")
            for name in checks:
                self.not_run(name, f"{readable} failed")
            return None

    def skip(self, package: str, checks: dict[str, str], reason: str) -> None:
        self.declare(f"{package}_readable", f"{package} files exist and parse")
        for name, description in checks.items():
            self.declare(name, description)
        for name in (f"{package}_readable", *checks):
            self.set_status(name, "skipped", reason)

    def failed(self) -> list[str]:
        return [name for name, c in self.checks.items() if c["status"] == "failed"]


# ---- files ----------------------------------------------------------------------------------


def file_hashes(root: Path) -> dict[str, dict]:
    """``{relative posix path: {sha256, bytes}}`` of every file except ``data_report.json``."""
    out: dict[str, dict] = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        if rel == REPORT_NAME:
            continue
        digest = hashlib.sha256()
        with path.open("rb") as fh:
            for block in iter(lambda fh=fh: fh.read(1 << 20), b""):
                digest.update(block)
        out[rel] = {"sha256": digest.hexdigest(), "bytes": path.stat().st_size}
    return out


def tree_sha256(files: dict[str, dict]) -> str:
    """One digest over the sorted ``path<TAB>sha256`` lines of ``file_hashes``."""
    lines = "".join(f"{path}\t{files[path]['sha256']}\n" for path in sorted(files))
    return hashlib.sha256(lines.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _check_provenance(results: _Results, name: str, manifest: dict, config: SimConfig) -> None:
    for key, expected in (("dataset_version", config.dataset_version), ("seed", config.seed),
                          ("sim_config_sha256", config_sha256(config))):
        results.require(name, manifest.get(key) == expected, f"manifest.{key}")


def _check_listed_file(results: _Results, name: str, files: dict, rel: str, sha: str | None,
                       size: int | None = None) -> None:
    actual = files.get(rel)
    if not results.require(name, actual is not None, f"{rel}: missing"):
        return
    results.require(name, actual["sha256"] == sha, f"{rel}: sha256")
    if size is not None:
        results.require(name, actual["bytes"] == size, f"{rel}: bytes")


# ---- streaming JSON (Baseline documents) ----------------------------------------------------

_DECODER = json.JSONDecoder()
_WHITESPACE = " \t\r\n"


class _JsonStream:
    """Incremental reader over one UTF-8 JSON text; values are decoded with ``raw_decode``."""

    def __init__(self, fh, chunk_size: int = 1 << 22) -> None:
        self._fh = fh
        self._chunk = chunk_size
        self._decoder = codecs.getincrementaldecoder("utf-8")()
        self.buf = ""
        self.pos = 0
        self.eof = False

    def _fill(self) -> bool:
        if self.eof:
            return False
        data = self._fh.read(self._chunk)
        if data:
            text = self._decoder.decode(data)
        else:
            self.eof = True
            text = self._decoder.decode(b"", final=True)
            if not text:
                return False
        self.buf = self.buf[self.pos:] + text
        self.pos = 0
        return True

    def peek(self) -> str:
        """Next non-whitespace character (not consumed); ``""`` at end of file."""
        while True:
            buf, pos = self.buf, self.pos
            while pos < len(buf) and buf[pos] in _WHITESPACE:
                pos += 1
            self.pos = pos
            if pos < len(buf):
                return buf[pos]
            if not self._fill():
                return ""

    def take(self, expected: str) -> None:
        char = self.peek()
        if char != expected:
            raise ValueError(f"expected {expected!r} at offset {self.pos}, found {char!r}")
        self.pos += 1

    def value(self):
        if self.pos >= len(self.buf) or self.buf[self.pos] in _WHITESPACE:
            self.peek()
        while True:
            try:
                value, end = _DECODER.raw_decode(self.buf, self.pos)
            except json.JSONDecodeError:
                if not self._fill():
                    raise
                continue
            if end == len(self.buf) and self._fill():
                continue  # a number may go on in the next chunk
            self.pos = end
            return value


def iter_json_document(path: Path) -> Iterator[tuple[str, str, object]]:
    """Stream a top-level JSON object without loading it whole.

    Yields ``(key, "list", None)`` when an array value starts, ``(key, "item", element)`` for
    each of its elements and ``(key, "value", value)`` for any other value.
    """
    with Path(path).open("rb") as fh:
        stream = _JsonStream(fh)
        stream.take("{")
        if stream.peek() == "}":
            stream.pos += 1
        else:
            while True:
                key = stream.value()
                if not isinstance(key, str):
                    raise TypeError(f"object key expected, got {key!r}")
                stream.take(":")
                if stream.peek() == "[":
                    stream.pos += 1
                    yield key, "list", None
                    if stream.peek() == "]":
                        stream.pos += 1
                    else:
                        while True:
                            yield key, "item", stream.value()
                            buf, pos = stream.buf, stream.pos
                            if pos < len(buf) and buf[pos] == ",":  # fast path
                                stream.pos = pos + 1
                                continue
                            sep = stream.peek()
                            stream.pos += 1
                            if sep == "]":
                                break
                            if sep != ",":
                                raise ValueError(f"expected ',' or ']' in {key!r}, found {sep!r}")
                else:
                    yield key, "value", stream.value()
                sep = stream.peek()
                stream.pos += 1
                if sep == "}":
                    break
                if sep != ",":
                    raise ValueError(f"expected ',' or '}}' after {key!r}, found {sep!r}")
        if stream.peek() != "":
            raise ValueError("trailing data after the top-level object")


# ---- shared state between phases -------------------------------------------------------------


@dataclass(slots=True)
class _Request:
    """What the Baseline/oracle phases compare against (read from the Food/Ads packages)."""

    user_id: str
    split: str
    cutoff_iso: str
    candidates: tuple[str, ...]
    targets: tuple[int, ...]
    label_iso: str | None  # when the outcome is observable (Baseline label ``available_at``)


class _EventIndex:
    """``ads/events.parquet`` (every event of every service) as arrays with per-user ranges."""

    def __init__(self, table: pa.Table) -> None:
        self.n = table.num_rows
        self.ts = np.asarray(table["timestamp"].to_numpy(), dtype=np.int64)
        self.event_id = table["event_id"].combine_chunks()
        self.entity = table["entity_id"].combine_chunks()
        self.user_codes, self.user_names = self._encode(table["user_id"])
        self.type_codes, self.type_names = self._encode(table["event_type"])
        self.service_codes, self.service_names = self._encode(table["service"])
        event_index = np.asarray(table["event_index"].to_numpy(), dtype=np.int64)
        self.starts = np.flatnonzero(event_index == 0)
        self.stops = np.append(self.starts[1:], self.n).astype(np.int64)
        self.segment: dict[str, tuple[int, int]] = {}
        self.duplicate_users: list[str] = []
        for start, stop in zip(self.starts.tolist(), self.stops.tolist()):
            user = self.user_names[self.user_codes[start]]
            if user in self.segment:
                self.duplicate_users.append(user)
            self.segment[user] = (start, stop)

    @staticmethod
    def _encode(column) -> tuple[np.ndarray, list[str]]:
        encoded = pc.dictionary_encode(column.combine_chunks())
        return (np.asarray(encoded.indices.to_numpy(zero_copy_only=False), dtype=np.int64),
                encoded.dictionary.to_pylist())

    def type_code(self, name: str) -> int:
        return self.type_names.index(name) if name in self.type_names else -1

    def service_code(self, name: str) -> int:
        return self.service_names.index(name) if name in self.service_names else -1


@dataclass
class _State:
    root: Path
    config: SimConfig
    files: dict[str, dict]
    events: _EventIndex | None = None
    ad_maturity: np.ndarray | None = None  # per events row: impression outcome time, else -1
    food: dict[str, _Request] | None = None
    ads: dict[str, _Request] | None = None


def _split_of(ts: int, config: SimConfig) -> str | None:
    """Split of an epoch-seconds timestamp (weeks 1..weeks-2 / weeks-1 / weeks), None outside."""
    weeks = config.weeks
    if not day_start_ts(1, config) <= ts < day_start_ts(7 * weeks + 1, config):
        return None
    if ts < day_start_ts(7 * (weeks - 2) + 1, config):
        return "train"
    if ts < day_start_ts(7 * (weeks - 1) + 1, config):
        return "validation"
    return "test"


def _epoch(iso: str) -> int:
    value = datetime.fromisoformat(iso)
    if value.tzinfo is None:
        raise ValueError(f"timestamp without offset: {iso!r}")
    return int(value.timestamp())


def _iso_before(a: str, b: str) -> bool:
    """``a < b`` for ISO-8601 times with offsets. Plain string order when both use the
    exporters' layout ``YYYY-MM-DDTHH:MM:SS+HH:MM`` with the same offset, else parsed."""
    if len(a) == 25 == len(b) and a[19:] == b[19:] and a[10] == "T" == b[10]:
        return a < b
    return datetime.fromisoformat(a) < datetime.fromisoformat(b)


# ---- pretrain -------------------------------------------------------------------------------

PRETRAIN_CHECKS = {
    "pretrain_manifest": "manifest complete, provenance matches the config, preprocessing config "
                         "and every shard SHA-256/size match the files, no stray files",
    "pretrain_events_before_week7": "no event on/after the first local day of week weeks-1 "
                                    "(day 43), every shard",
    "pretrain_snapshots": "every snapshot: cutoff before day 43, split within its bounds, profile "
                          "= as_of_time (= cutoff) + spec 4.6 fields only, "
                          "history_group_truncated false (ruling R11: whole sessions, next-action "
                          "targets on)",
    "pretrain_snapshot_history": "every snapshot history id is an event of the same user at or "
                                 "before the cutoff, 1..max_history_events ids",
    "pretrain_loader": "spot check: every sample of evenly spaced users loads with "
                       "iter_normalized_samples and passes validate_sample",
}


def _pretrain_phase(results: _Results, state: _State, loader_users: int | None) -> None:
    root, config, files = state.root / "pretrain", state.config, state.files
    name = "pretrain_manifest"
    manifest = load_manifest(root, validate_shard_files=True).data  # checks the config SHA too
    results.require(name, manifest.get("status") == "complete", "manifest.status")
    results.require(name, manifest.get("schema_version") == PRETRAIN_SCHEMA_VERSION,
                    "manifest.schema_version")
    results.require(name, manifest.get("source_dataset") == SOURCE_DATASET,
                    "manifest.source_dataset")
    _check_provenance(results, name, manifest, config)
    _check_listed_file(results, name, files, "pretrain/preview.json",
                       manifest.get("preview_sha256"))
    listed = {"pretrain/manifest.json", "pretrain/preview.json",
              "pretrain/preprocessing_config.json"}
    event_paths: list[str] = []
    snapshot_paths: list[str] = []
    total_events = 0
    by_split = dict.fromkeys(SPLITS, 0)
    for shard in manifest["shards"]:
        for kind, paths in (("events", event_paths), ("snapshots", snapshot_paths)):
            rel = f"pretrain/{shard[f'{kind}_file']}"
            listed.add(rel)
            paths.append(str(state.root / rel))
            _check_listed_file(results, name, files, rel, shard.get(f"{kind}_sha256"),
                               shard.get(f"{kind}_bytes"))
        total_events += int(shard["event_rows"])
        for split, count in shard["snapshots_by_split"].items():
            by_split[split] = by_split.get(split, 0) + int(count)
    for rel in sorted(r for r in files if r.startswith("pretrain/") and r not in listed):
        results.fail(name, f"{rel}: not in manifest")
    results.require(name, manifest.get("total_events") == total_events, "manifest.total_events")
    results.require(name, len(manifest["shards"]) == config.n_users, "manifest.shards: users")
    results.note(name, shards=len(manifest["shards"]), files_checked=len(listed))

    limit = naive_utc(day_start_ts(pretrain_end_day(config), config))
    limit_scalar = pa.scalar(limit, pa.timestamp("us"))
    name = "pretrain_events_before_week7"
    events = ds.dataset(event_paths, format="parquet").to_table(
        columns=["event_id", "user_id", "time_lower_bound", "time_upper_bound",
                 "eligibility_time"])
    late = pc.or_(pc.greater_equal(events["time_upper_bound"], limit_scalar),
                  pc.greater_equal(events["eligibility_time"], limit_scalar))
    late = pc.or_(late, pc.greater_equal(events["time_lower_bound"], limit_scalar))
    for event_id in events.filter(late)["event_id"].to_pylist():
        results.fail(name, event_id)
    results.require(name, events.num_rows == total_events, "event rows != manifest total_events")
    results.note(name, events=events.num_rows, limit_naive_utc=limit.isoformat())

    name = "pretrain_snapshots"
    bounds = {k: datetime.fromisoformat(v) for k, v in split_bounds(config).items()}
    window = {"train": (bounds["start_inclusive"], bounds["train_end_exclusive"]),
              "validation": (bounds["train_end_exclusive"], bounds["validation_end_exclusive"])}
    snaps = ds.dataset(snapshot_paths, format="parquet").to_table(
        columns=["sample_id", "user_id", "cutoff_time", "split", "profile_json",
                 "history_event_ids", "history_group_truncated"])
    sample_ids = snaps["sample_id"].to_pylist()
    counted = dict.fromkeys(SPLITS, 0)
    for sid, cutoff, split, profile_json, group_truncated in zip(
            sample_ids, snaps["cutoff_time"].to_pylist(), snaps["split"].to_pylist(),
            snaps["profile_json"].to_pylist(), snaps["history_group_truncated"].to_pylist()):
        counted[split] = counted.get(split, 0) + 1
        lo_hi = window.get(split)
        ok = lo_hi is not None and lo_hi[0] <= cutoff < lo_hi[1] and cutoff < limit
        ok = ok and group_truncated is False
        profile = json.loads(profile_json)
        ok = ok and set(profile) == PROFILE_KEYS and profile["as_of_time"] == cutoff.isoformat()
        results.require(name, ok, sid)
    results.require(name, counted == by_split, "snapshot rows by split != manifest")
    results.note(name, snapshots=snaps.num_rows, by_split=counted)

    name = "pretrain_snapshot_history"
    ids = snaps["history_event_ids"].combine_chunks()
    lengths = np.asarray(pc.list_value_length(ids).fill_null(0).to_numpy(), dtype=np.int64)
    _flag(results, name, (lengths < 1) | (lengths > config.max_history_events), sample_ids)
    flat = pa.table({"event_id": pc.list_flatten(ids), "parent": pc.list_parent_indices(ids)})
    joined = flat.join(events.select(["event_id", "user_id", "time_upper_bound"]),
                       keys="event_id", join_type="left outer")
    parent = joined["parent"].combine_chunks()
    bad = pc.or_kleene(pc.is_null(joined["time_upper_bound"]),
                       pc.greater(joined["time_upper_bound"], snaps["cutoff_time"].take(parent)))
    bad = pc.or_kleene(bad, pc.not_equal(joined["user_id"], snaps["user_id"].take(parent)))
    bad = np.asarray(bad.fill_null(True).to_numpy(zero_copy_only=False), dtype=bool)
    parents = np.asarray(parent.to_numpy(), dtype=np.int64)
    _flag(results, name, np.isin(np.arange(len(sample_ids)), parents[bad]), sample_ids)
    results.require(name, joined.num_rows == len(flat), "duplicate event ids in events shards")
    results.note(name, history_ids=len(flat))

    name = "pretrain_loader"
    shards = manifest["shards"]
    step = 1 if loader_users is None else max(1, -(-len(shards) // max(loader_users, 1)))
    chosen = shards[::step]
    loaded_n = dict.fromkeys(("train", "validation"), 0)
    with tempfile.TemporaryDirectory() as tmp:
        # A copy of the chosen shards with a manifest listing only them (same bytes, same SHAs).
        sub = Path(tmp)
        config_file = "preprocessing_config.json"
        (sub / config_file).write_bytes((root / config_file).read_bytes())
        for shard in chosen:
            for kind in ("events", "snapshots"):
                target = sub / shard[f"{kind}_file"]
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((root / shard[f"{kind}_file"]).read_bytes())
        (sub / "manifest.json").write_text(json.dumps({**manifest, "shards": chosen}),
                                           encoding="utf-8")
        for split in loaded_n:
            try:
                for sample in iter_normalized_samples(sub, split):
                    loaded_n[split] += 1
                    results.require(name, sample.dataset_name == SOURCE_DATASET,
                                    sample.sample_id)
            except Exception as error:  # noqa: BLE001 - the loader stops at an invalid sample
                results.fail(name, f"{split}: {type(error).__name__}: {error}")
    expected = {s: sum(int(sh["snapshots_by_split"].get(s, 0)) for sh in chosen)
                for s in loaded_n}
    results.require(name, loaded_n == expected, "loaded sample count != manifest")
    results.note(name, users=len(chosen), samples=loaded_n)


# ---- ads ------------------------------------------------------------------------------------

ADS_CHECKS = {
    "ads_manifest": "manifest provenance, policies, file SHA-256/rows/bytes and request counts",
    "ads_columns_public": "exact public columns in every table; attributes_json keys = public "
                          "context + numeric action fields (no click flag / session size)",
    "ads_events_order": "events grouped per user, event_index 0..n-1, source_row_index = row, "
                        "timestamps non-decreasing per user, unique event ids",
    "ads_history_before_exposure": "every history window holds only events strictly before the "
                                   "exposure (never the request's own impression or click)",
    "ads_history_window_policy": "history_stop = events before exposure, history_start = "
                                 "max(0, stop - max_events), history_end_timestamp, cutoff = "
                                 "exposure",
    "ads_impression_rows": "source_row_index is the user's ad_impression of the candidate at the "
                           "exposure timestamp; clicked = the event log's outcome (the user's next "
                           "row is the ad_click of the same ad 2-26 s later)",
    "ads_split": "split = week of the exposure (train weeks 1..weeks-2, validation, test); "
                 "manifest split bounds; unique request ids",
}


def _check_columns(results: _Results, name: str, label: str, names, schema) -> None:
    for extra in sorted(set(names) - set(schema.names)):
        results.fail(name, f"{label}: unexpected column {extra}")
    for missing in sorted(set(schema.names) - set(names)):
        results.fail(name, f"{label}: missing column {missing}")


def _ads_phase(results: _Results, state: _State) -> None:
    root, config, files = state.root / "ads", state.config, state.files
    manifest = _read_json(root / "manifest.json")
    events = pq.read_table(root / "events.parquet")
    opp = pq.read_table(root / "ad_opportunities.parquet")

    name = "ads_columns_public"
    _check_columns(results, name, "events", events.schema.names, ADS_EVENT_SCHEMA)
    _check_columns(results, name, "ad_opportunities", opp.schema.names, OPPORTUNITY_SCHEMA)
    _check_columns(results, name, "profiles", pq.read_schema(root / "profiles.parquet").names,
                   PROFILE_SCHEMA)
    _check_columns(results, name, "ad_catalog", pq.read_schema(root / "ad_catalog.parquet").names,
                   CATALOG_SCHEMA)
    index = _EventIndex(events)
    attributes = events["attributes_json"].combine_chunks()
    for code, service in enumerate(index.service_names):
        allowed = PUBLIC_ATTRIBUTES.get(service)
        if not results.require(name, allowed is not None, f"events: unknown service {service}"):
            continue
        mask = pa.array(index.service_codes == code)
        for text in pc.unique(attributes.filter(mask)).to_pylist():
            extra = set(json.loads(text)) - allowed
            results.require(name, not extra, f"events[{service}]: keys {sorted(extra)}")

    name = "ads_manifest"
    _check_provenance(results, name, manifest, config)
    for key, table_name in (("events", "events"), ("ad_opportunities", "ad_opportunities"),
                            ("profiles", "profiles"), ("ad_catalog", "ad_catalog")):
        entry = manifest["files"][key]
        _check_listed_file(results, name, files, f"ads/{entry['path']}", entry["sha256"],
                           entry["bytes"])
        rows = pq.read_metadata(root / entry["path"]).num_rows
        results.require(name, entry["row_count"] == rows, f"{table_name}: row_count")
    policy = manifest.get("history_policy", {})
    results.require(name, policy == {"timestamp_rule": "strictly < cutoff",
                                     "target_click_in_history": False,
                                     "equal_timestamp_events_excluded": True,
                                     "max_events": config.max_history_events},
                    "manifest.history_policy")
    for key, expected in (("candidate_policy", ADS_CANDIDATE_POLICY),
                          ("negative_sampling", "none"), ("missing_click_policy", "reject"),
                          ("timestamp_unit", "seconds"), ("row_count", opp.num_rows)):
        results.require(name, manifest.get(key) == expected, f"manifest.{key}")

    name = "ads_events_order"
    for user in index.duplicate_users:
        results.fail(name, f"{user}: events not contiguous")
    seg = np.repeat(np.arange(len(index.starts)), index.stops - index.starts)
    event_index = np.asarray(events["event_index"].to_numpy(), dtype=np.int64)
    row_index = np.asarray(events["source_row_index"].to_numpy(), dtype=np.int64)
    bad = np.zeros(index.n, dtype=bool)
    if index.n:
        bad |= index.user_codes != index.user_codes[index.starts][seg]
        bad |= event_index != np.arange(index.n) - index.starts[seg]
        bad |= row_index != np.arange(index.n)
        same_user = seg[1:] == seg[:-1]
        bad[1:] |= same_user & (np.diff(index.ts) < 0)
    for i in np.flatnonzero(bad)[:MAX_IDS].tolist():
        results.fail(name, index.event_id[i].as_py())
    if bad.sum() > MAX_IDS:
        results.checks[name]["violations"] += int(bad.sum()) - MAX_IDS
    results.require(name, pc.count_distinct(index.event_id).as_py() == index.n,
                    "duplicate event_id")

    _ads_window_checks(results, state, index, opp, manifest)
    state.events = index


def _impression_outcomes(index: _EventIndex) -> tuple[np.ndarray, np.ndarray]:
    """Per row of ``ads/events.parquet``: whether it is an ``ad_impression`` whose next row (same
    user) is the ``ad_click`` of the same ad ``AD_CLICK_DELAY_SECONDS`` later, and when the
    impression's outcome is observable (click time, else exposure +
    ``AD_OUTCOME_MATURITY_SECONDS``; -1 on other rows)."""
    n = index.n
    is_impression = index.type_codes == index.type_code("ad_impression")
    clicked = np.zeros(n, dtype=bool)
    if n > 1:
        low, high = AD_CLICK_DELAY_SECONDS
        delay = np.diff(index.ts)
        same_ad = pc.equal(index.entity.slice(0, n - 1), index.entity.slice(1)).fill_null(False)
        clicked[:-1] = (is_impression[:-1]
                        & (index.type_codes[1:] == index.type_code("ad_click"))
                        & (index.user_codes[1:] == index.user_codes[:-1])
                        & np.asarray(same_ad.to_numpy(zero_copy_only=False), dtype=bool)
                        & (delay >= low) & (delay <= high))
    next_ts = np.append(index.ts[1:], 0).astype(np.int64)
    maturity = np.where(clicked, next_ts, index.ts + AD_OUTCOME_MATURITY_SECONDS)
    maturity[~is_impression] = -1
    return clicked, maturity


def _flag(results: _Results, name: str, mask: np.ndarray, ids: list[str]) -> None:
    hits = np.flatnonzero(mask)
    for i in hits[:MAX_IDS].tolist():
        results.fail(name, ids[i])
    if len(hits) > MAX_IDS:
        results.checks[name]["violations"] += len(hits) - MAX_IDS


def _ads_window_checks(results: _Results, state: _State, index: _EventIndex, opp: pa.Table,
                       manifest: dict) -> None:
    config = state.config
    max_events = config.max_history_events
    rids = opp["request_id"].to_pylist()
    users = opp["user_id"].to_pylist()
    col = {c: np.asarray(opp[c].fill_null(-1).to_numpy(), dtype=np.int64)
           for c in ("history_start", "history_stop", "exposure_timestamp", "cutoff_timestamp",
                     "source_row_index", "history_end_timestamp")}
    hs, hst, exposure = col["history_start"], col["history_stop"], col["exposure_timestamp"]
    seg = np.array([index.segment.get(u, (-1, -1)) for u in users], dtype=np.int64).reshape(-1, 2)
    known = seg[:, 0] >= 0
    seg_len = seg[:, 1] - seg[:, 0]
    valid = known & (hs >= 0) & (hs <= hst) & (hst <= seg_len)
    stop = np.where(valid, seg[:, 0] + hst, 0)
    non_empty = valid & (hst > hs)
    last_ts = np.where(non_empty, index.ts[np.clip(stop - 1, 0, max(index.n - 1, 0))], -1)
    has_next = valid & (stop < seg[:, 1])
    next_ts = np.where(has_next, index.ts[np.clip(stop, 0, max(index.n - 1, 0))], 0)

    name = "ads_history_before_exposure"
    _flag(results, name, ~valid | (non_empty & (last_ts >= exposure)), rids)

    name = "ads_history_window_policy"
    end_ts = col["history_end_timestamp"]
    bad = ~valid | (hs != np.maximum(0, hst - max_events)) | (hst - hs > max_events)
    bad |= has_next & (next_ts < exposure)
    bad |= np.where(non_empty, end_ts != last_ts, end_ts != -1)
    bad |= col["cutoff_timestamp"] != exposure
    _flag(results, name, bad, rids)

    name = "ads_impression_rows"
    src = col["source_row_index"]
    in_range = (src >= 0) & (src < index.n)
    at = np.clip(src, 0, max(index.n - 1, 0))
    user_code = {u: i for i, u in enumerate(index.user_names)}
    expected_user = np.array([user_code.get(u, -1) for u in users], dtype=np.int64)
    entity_at = index.entity.take(pa.array(at)).to_pylist() if index.n else [None] * len(rids)
    candidates = opp["candidate_id"].to_pylist()
    same_entity = np.array([a == b for a, b in zip(entity_at, candidates)], dtype=bool)
    ok = in_range & (index.user_codes[at] == expected_user)
    ok &= index.type_codes[at] == index.type_code("ad_impression")
    ok &= (index.ts[at] == exposure) & same_entity & (~valid | (src >= stop))
    clicked = opp["clicked"].to_pylist()
    clicked_in_log, maturity = _impression_outcomes(index)
    ok &= clicked_in_log[at] == np.array([bool(y) for y in clicked], dtype=bool)
    _flag(results, name, ~ok, rids)
    results.require(name, opp["clicked"].null_count == 0, "clicked has nulls")
    state.ad_maturity = maturity

    name = "ads_split"
    splits = opp["split"].to_pylist()
    wrong = np.array([_split_of(int(t), config) != s for t, s in zip(exposure, splits)],
                     dtype=bool)
    _flag(results, name, wrong, rids)
    results.require(name, len(set(rids)) == len(rids), "duplicate request_id")
    expected_split = {"train_end": day_start_ts(7 * (config.weeks - 2) + 1, config),
                      "valid_end": day_start_ts(7 * (config.weeks - 1) + 1, config)}
    for key, value in expected_split.items():
        results.require(name, manifest["split"].get(key) == value, f"manifest.split.{key}")
    counts = {s: splits.count(s) for s in SPLITS}
    results.require(name, manifest.get("request_counts") == counts, "manifest.request_counts")
    results.note(name, requests=counts)

    mature = maturity[at].tolist()
    state.ads = {
        rid: _Request(user, split, local_iso(int(t), config), (cand,), (int(bool(y)),),
                      local_iso(m, config) if good and m >= 0 else None)
        for rid, user, split, t, cand, y, m, good in zip(
            rids, users, splits, exposure.tolist(), candidates, clicked, mature, ok.tolist())
    }


# ---- food -----------------------------------------------------------------------------------

FOOD_CHECKS = {
    "food_manifest": "manifest provenance, policy, file SHA-256/rows/bytes and request counts",
    "food_format": "row/label_metadata keys and constant fields, unique ids in request order",
    "food_candidates": "10 distinct catalog merchants with catalog attributes; labels in "
                       "candidate order with exactly one positive",
    "food_inputs_public": "inputs carry exactly the public keys: no current location, distance "
                          "or hidden state; history attributes from the public allowlist; "
                          "public merchant catalog columns",
    "food_history_before_cutoff": "every history event strictly before cutoff_time, at most "
                                  "max_history_events",
    "food_own_session_absent": "the request's own session (from ads/events.parquet) starts at "
                               "cutoff + 1 s, ends with the positive's order at label_time and "
                               "is not in the history",
    "food_label_after_cutoff": "label_time strictly after cutoff_time",
    "food_profile_as_of_cutoff": "profile = as_of_time (= cutoff_time) + spec 4.6 fields only",
    "food_split": "split = week of cutoff_time; manifest split bounds",
}


def _food_phase(results: _Results, state: _State) -> None:
    root, config, files = state.root / "food", state.config, state.files
    manifest = _read_json(root / "manifest.json")
    merchants = pq.read_table(root / "merchants.parquet")
    _check_columns(results, "food_inputs_public", "merchants", merchants.schema.names,
                   MERCHANT_SCHEMA)
    catalog = {row["merchant_id"]: (row["cuisine"], row["price_level"], row["geohash"])
               for row in merchants.to_pylist()}

    name = "food_manifest"
    _check_provenance(results, name, manifest, config)
    for key in ("samples", "merchants"):
        entry = manifest["files"][key]
        _check_listed_file(results, name, files, f"food/{entry['path']}", entry["sha256"],
                           entry["bytes"])
    results.require(name, manifest["files"]["merchants"]["row_count"] == merchants.num_rows,
                    "merchants: row_count")
    for key, expected in (("candidate_policy", FOOD_CANDIDATE_POLICY),
                          ("candidates_per_request", N_CANDIDATES),
                          ("max_history_events", config.max_history_events),
                          ("prediction_anchor", PREDICTION_ANCHOR),
                          ("positive_action", POSITIVE_ACTION)):
        results.require(name, manifest.get(key) == expected, f"manifest.{key}")
    train_end = day_start_ts(pretrain_end_day(config), config)
    valid_end = day_start_ts(pretrain_end_day(config) + 7, config)
    results.require(name, manifest["split"].get("train_end") == local_iso(train_end, config)
                    and manifest["split"].get("valid_end") == local_iso(valid_end, config),
                    "manifest.split")
    results.require(name, manifest.get("popularity_end_exclusive") == local_iso(train_end, config),
                    "manifest.popularity_end_exclusive")

    index = state.events
    if index is None:
        results.not_run("food_own_session_absent", "ads/events.parquet not checked")
    else:
        food_code, search_code = index.service_code("food"), index.type_code("search")
        order_code = index.type_code("order")
    requests: dict[str, _Request] = {}
    counts = dict.fromkeys(SPLITS, 0)
    with (root / "samples.jsonl").open("rb") as fh:
        for n, line in enumerate(fh):
            row = json.loads(line)
            rid = str(row.get("sample_id"))
            meta, inp = row["label_metadata"], row["input"]
            cutoff_iso = inp["cutoff_time"]
            cutoff = _epoch(cutoff_iso)
            label = _epoch(meta["label_time"])

            name = "food_format"
            results.require(name, set(row) == FOOD_ROW_KEYS and set(meta) == FOOD_META_KEYS
                            and meta["request_id"] == rid == f"sim:food_req:{n}"
                            and rid not in requests
                            and meta["candidate_source"] == FOOD_CANDIDATE_POLICY
                            and meta["observation_complete"] is True
                            and meta["prediction_anchor"] == PREDICTION_ANCHOR
                            and meta["positive_action"] == POSITIVE_ACTION, rid)

            name = "food_candidates"
            cands = inp["candidates"]
            ids = tuple(c.get("item_id") for c in cands)
            labels = row["labels"]
            ok = len(cands) == N_CANDIDATES and len(set(ids)) == N_CANDIDATES
            ok = ok and all(catalog.get(c.get("item_id")) == (c.get("cuisine"),
                            c.get("price_level"), c.get("geohash")) for c in cands)
            ok = ok and [lb.get("item_id") for lb in labels] == list(ids)
            targets = tuple(lb.get("relevance") for lb in labels)
            ok = ok and set(targets) <= {0, 1} and sum(targets) == 1
            results.require(name, ok, rid)

            name = "food_inputs_public"
            ok = (set(inp) == FOOD_INPUT_KEYS and set(inp["context"]) == CONTEXT_KEYS
                  and set(inp["profile"]) == PROFILE_KEYS
                  and all(set(c) == FOOD_CANDIDATE_KEYS for c in cands))
            for item in inp["history"]:
                allowed = PUBLIC_ATTRIBUTES.get(item.get("service"))
                ok = ok and item.keys() == FOOD_HISTORY_KEYS and allowed is not None
                ok = ok and item["attributes"].keys() <= allowed
            results.require(name, ok, rid)

            name = "food_history_before_cutoff"
            history = inp["history"]
            results.require(name, len(history) <= config.max_history_events
                            and all(_iso_before(e["event_time"], cutoff_iso)
                                    for e in history), rid)

            results.require("food_label_after_cutoff", label > cutoff, rid)
            results.require("food_profile_as_of_cutoff",
                            set(inp["profile"]) == PROFILE_KEYS
                            and inp["profile"]["as_of_time"] == cutoff_iso, rid)
            split = meta["split"]
            results.require("food_split", _split_of(cutoff, config) == split, rid)
            counts[split] = counts.get(split, 0) + 1

            if index is not None:
                positive = ids[targets.index(1)] if targets.count(1) == 1 else None
                ok = _own_session_ok(index, inp["user_id"], cutoff, label, positive,
                                     {e.get("event_id") for e in history},
                                     food_code, search_code, order_code)
                results.require("food_own_session_absent", ok, rid)
            requests[rid] = _Request(inp["user_id"], split, cutoff_iso, ids, targets,
                                     meta["label_time"])

    results.require("food_manifest", manifest["files"]["samples"]["row_count"] == len(requests)
                    and manifest.get("request_counts") == counts, "request counts")
    results.note("food_split", requests=counts)
    state.food = requests


def _own_session_ok(index: _EventIndex, user: str, cutoff: int, label: int,
                    positive: str | None, history_ids: set, food_code: int, search_code: int,
                    order_code: int) -> bool:
    segment = index.segment.get(user)
    if segment is None or positive is None:
        return False
    start, stop = segment
    times = index.ts[start:stop]
    lo = start + int(np.searchsorted(times, cutoff, side="right"))
    hi = start + int(np.searchsorted(times, label, side="right"))
    rows = [r for r in range(lo, hi) if index.service_codes[r] == food_code]
    if not rows:
        return False
    first, last = rows[0], rows[-1]
    if not (index.ts[first] == cutoff + 1 and index.type_codes[first] == search_code
            and index.ts[last] == label and index.type_codes[last] == order_code
            and index.entity[last].as_py() == positive):
        return False
    own = {index.event_id[r].as_py() for r in rows}
    return not (own & history_ids)


# ---- oracle ---------------------------------------------------------------------------------

ORACLE_CHECKS = {
    "oracle_food_sums_to_one": "true probabilities are finite, in [0, 1] and sum to 1 per "
                               "request",
    "oracle_food_matches_candidates": "one row per (request, candidate) of the Food package",
    "oracle_ads": "one probability in (0, 1) per Ads request, in request order",
}


def _oracle_phase(results: _Results, state: _State) -> None:
    root = state.root / "oracle"
    food = pq.read_table(root / "food_oracle.parquet")
    ads = pq.read_table(root / "ads_oracle.parquet")
    name = "oracle_food_sums_to_one"
    rids = food["request_id"].to_pylist()
    probs = np.asarray(food["true_probability"].to_numpy(), dtype=float)
    encoded = pc.dictionary_encode(food["request_id"].combine_chunks())
    codes = np.asarray(encoded.indices.to_numpy(zero_copy_only=False), dtype=np.int64)
    names = encoded.dictionary.to_pylist()
    finite = np.isfinite(probs) & (probs >= 0) & (probs <= 1)
    _flag(results, name, ~finite, rids)
    sums = np.bincount(codes, weights=np.where(finite, probs, 0.0), minlength=len(names))
    _flag(results, name, np.abs(sums - 1.0) > PROB_TOLERANCE, names)
    if list(food.schema.names) != ["request_id", "candidate_id", "true_probability"]:
        results.fail(name, f"columns {food.schema.names}")

    name = "oracle_food_matches_candidates"
    if state.food is None:
        results.not_run(name, "food package not checked")
    else:
        sets: dict[str, list[str]] = {}
        for rid, cid in zip(rids, food["candidate_id"].to_pylist()):
            sets.setdefault(rid, []).append(cid)
        for rid, req in state.food.items():
            got = sets.pop(rid, [])
            results.require(name, len(got) == len(set(got)) and set(got) == set(req.candidates),
                            rid)
        for rid in sets:
            results.fail(name, f"{rid}: unknown request")

    name = "oracle_ads"
    p = np.asarray(ads["true_click_probability"].to_numpy(), dtype=float)
    ad_rids = ads["request_id"].to_pylist()
    _flag(results, name, ~(np.isfinite(p) & (p > 0) & (p < 1)), ad_rids)
    if state.ads is None:
        results.not_run(name, "ads package not checked")
    else:
        results.require(name, ad_rids == list(state.ads), "request ids != ad_opportunities")


# ---- baseline -------------------------------------------------------------------------------


def _baseline_checks() -> dict[str, str]:
    out = {}
    for name in ("food", "ads"):
        out[f"baseline_{name}_structure"] = (
            f"baseline/{name}.json sections and metadata; queries, candidates and labels equal "
            f"to the {name} package (one label per candidate, label available_at = outcome time "
            f">= cutoff: Food order, Ads click or exposure + 27 s); candidate attributes with the "
            f"B1 names (category/price, brand); every label of a split available before the "
            f"next split's first cutoff (a B1 fit cutoff exists)")
        out[f"baseline_{name}_history_before_cutoff"] = (
            "history: users with requests only, event_time strictly before the user's latest "
            "cutoff, public attributes + service + B1 names only, unique event ids; ad "
            "impressions as event_type impression with clicked 0/1 and available_at = click "
            "or exposure + 27 s (matching ads/events.parquet), every other row available_at = "
            "event_time")
        out[f"baseline_{name}_load_dataset"] = (
            "src.baselines.runner.load_dataset (BaselineDataset.validate) accepts the file, or "
            "for a large file a subset of evenly spaced requests")
    return out


def _subset_ids(requests: dict[str, _Request], k: int) -> set[str]:
    ids = list(requests)
    step = max(1, len(ids) // k)
    return set(ids[::step][:k])


def _baseline_file(results: _Results, state: _State, name: str, requests: dict[str, _Request],
                   max_bytes: int) -> None:
    path = state.root / "baseline" / f"{name}.json"
    structure = f"baseline_{name}_structure"
    history_check = f"baseline_{name}_history_before_cutoff"
    load_check = f"baseline_{name}_load_dataset"
    task, item_kind, policy = BASELINE_TASKS[name]
    attribute_keys = BASELINE_ATTRIBUTES[name]
    aliases = BASELINE_CANDIDATE_ALIASES[name]
    outcomes = (None if state.events is None or state.ad_maturity is None
                else (state.events, state.ad_maturity))
    latest: dict[str, str] = {}  # user -> latest request cutoff (ISO)
    for req in requests.values():
        if req.user_id not in latest or _iso_before(latest[req.user_id], req.cutoff_iso):
            latest[req.user_id] = req.cutoff_iso
    subset = None if path.stat().st_size <= max_bytes else _subset_ids(
        requests, BASELINE_SUBSET_QUERIES)
    subset_latest: dict[str, str] = {}
    for rid in subset or ():
        req = requests[rid]
        if req.user_id not in subset_latest or _iso_before(subset_latest[req.user_id],
                                                           req.cutoff_iso):
            subset_latest[req.user_id] = req.cutoff_iso
    collected: dict[str, list] = {"queries": [], "candidates": [], "labels": [], "history": []}
    metadata = None
    sections: list[str] = []
    seen_queries: set[str] = set()
    seen_candidates: dict[str, int] = {}
    seen_labels: dict[str, int] = {}
    event_numbers = array("q")  # numeric part of every history event id

    for key, kind, value in iter_json_document(path):
        if kind == "list" or kind == "value":
            sections.append(key)
            if key == "metadata":
                metadata = value
            elif kind == "value":
                results.fail(structure, f"section {key}: not a list")
            continue
        if key == "history":
            _baseline_history_item(results, history_check, value, latest, event_numbers,
                                   outcomes)
            user = value.get("user_id") if isinstance(value, dict) else None
            if user in subset_latest and _iso_before(value["event_time"], subset_latest[user]):
                collected["history"].append(value)
            continue
        rid = value.get("request_id") if isinstance(value, dict) else None
        req = requests.get(rid)
        if req is None:
            results.fail(structure, f"{key}: unknown request {rid}")
            continue
        if key == "queries":
            ok = (value.keys() == BASELINE_QUERY_KEYS and rid not in seen_queries
                  and value["user_id"] == req.user_id and value["split"] == req.split
                  and value["cutoff_time"] == req.cutoff_iso
                  and value["context"].keys() == CONTEXT_KEYS)
            seen_queries.add(rid)
        elif key == "candidates":
            position = _position(req, value.get("candidate_id"))
            bit = 1 << position if position >= 0 else 0
            attributes = value["attributes"]
            ok = (value.keys() == BASELINE_CANDIDATE_KEYS and bit != 0
                  and not seen_candidates.get(rid, 0) & bit
                  and attributes.keys() == attribute_keys
                  and all(attributes[k] == attributes[v] for k, v in aliases.items()))
            seen_candidates[rid] = seen_candidates.get(rid, 0) | bit
        elif key == "labels":
            position = _position(req, value.get("candidate_id"))
            bit = 1 << position if position >= 0 else 0
            ok = (value.keys() == BASELINE_LABEL_KEYS and bit != 0
                  and not seen_labels.get(rid, 0) & bit
                  and value["target"] == req.targets[position]
                  and value["available_at"] == req.label_iso
                  and not _iso_before(value["available_at"], req.cutoff_iso))
            seen_labels[rid] = seen_labels.get(rid, 0) | bit
        else:
            results.fail(structure, f"unexpected section {key}")
            continue
        results.require(structure, ok, f"{key}: {rid}")
        if subset is not None and rid in subset:
            collected[key].append(value)

    results.require(structure, sorted(sections) == sorted(BASELINE_SECTIONS),
                    f"sections {sections}")
    expected_meta = {"dataset_name": DATASET_NAME, "task": task, "item_kind": item_kind,
                     "candidate_policy": policy, "availability_policy": AVAILABILITY_POLICY}
    for field, expected in expected_meta.items():
        results.require(structure, isinstance(metadata, dict) and metadata.get(field) == expected,
                        f"metadata.{field}")
    for rid, req in requests.items():
        full = (1 << len(req.candidates)) - 1
        results.require(structure, rid in seen_queries and seen_candidates.get(rid) == full
                        and seen_labels.get(rid) == full, f"{rid}: missing rows")
    _check_split_label_order(results, structure, requests)
    numbers = np.frombuffer(event_numbers, dtype=np.int64) if event_numbers else np.zeros(0)
    results.require(history_check, len(np.unique(numbers)) == len(numbers),
                    "duplicate history event_id")
    results.note(structure, queries=len(seen_queries))
    results.note(history_check, events=len(event_numbers))

    mode = "full_file" if subset is None else "subset"
    results.note(load_check, mode=mode)
    try:
        if subset is None:
            dataset = load_dataset(path)
        else:
            with tempfile.TemporaryDirectory() as tmp:
                subset_path = Path(tmp) / f"{name}_subset.json"
                _write_document(subset_path, {"metadata": metadata,
                                              **{k: iter(v) for k, v in collected.items()}})
                dataset = load_dataset(subset_path)
    except Exception as error:  # noqa: BLE001 - a rejected file is a failed check
        results.fail(load_check, f"{type(error).__name__}: {error}")
        return
    results.note(load_check, queries=len(dataset.queries), history=len(dataset.history))


def _check_split_label_order(results: _Results, name: str,
                             requests: dict[str, _Request]) -> None:
    """B1 ``fit`` (``src.baselines.lightgbm``) needs a training cutoff after every train cutoff and
    train label and at/before the first validation cutoff (B2-B4 also: validation labels before
    the first test cutoff); one exists iff each split's labels are available before the next
    split's first cutoff. Notes the latest label time and first cutoff per split."""
    first: dict[str, str] = {}
    last: dict[str, str] = {}
    for req in requests.values():
        if req.split not in first or _iso_before(req.cutoff_iso, first[req.split]):
            first[req.split] = req.cutoff_iso
        if req.label_iso is not None and (req.split not in last
                                          or _iso_before(last[req.split], req.label_iso)):
            last[req.split] = req.label_iso
    for earlier, later in pairwise(SPLITS):
        if earlier in last and later in first:
            results.require(name, _iso_before(last[earlier], first[later]),
                            f"{earlier} label available at/after the first {later} cutoff")
    results.note(name, latest_label_available_at=last, first_cutoff=first)


def _position(req: _Request, candidate_id) -> int:
    try:
        return req.candidates.index(candidate_id)
    except ValueError:
        return -1


def _baseline_history_item(results: _Results, name: str, item, latest: dict[str, str],
                           event_numbers: array, outcomes) -> None:
    ident = item.get("event_id") if isinstance(item, dict) else repr(item)[:60]
    try:
        attributes = item["attributes"]
        service = attributes.get("service")
        allowed = BASELINE_HISTORY_ATTRIBUTES.get(service)
        user_latest = latest.get(item["user_id"])
        event_type = item["event_type"]
        number = int(str(item["event_id"]).rsplit(":", 1)[1])
        event_numbers.append(number)
        aliases = BASELINE_HISTORY_ALIASES.get(event_type, {})
        ok = (item.keys() == BASELINE_HISTORY_KEYS and allowed is not None
              and attributes.keys() <= allowed and user_latest is not None
              and event_type != "ad_impression"  # B1 reads ``impression`` rows only
              and attributes.keys() & _BASELINE_EXTRA_KEYS
              == BASELINE_HISTORY_EXTRAS.get(event_type, frozenset())
              and all(attributes[k] == attributes.get(v) for k, v in aliases.items())
              and _iso_before(item["event_time"], user_latest))
        if event_type == IMPRESSION_EVENT_TYPE:
            ok = ok and service == "ads" and _impression_ok(item, number, outcomes)
        else:
            ok = ok and item["available_at"] == item["event_time"]
    except (KeyError, TypeError, ValueError, AttributeError, IndexError):
        ok = False
    results.require(name, ok, ident)


def _impression_ok(item: dict, number: int, outcomes) -> bool:
    """``clicked`` 0/1 and ``available_at`` = click time (2-26 s after the exposure) or exposure
    + 27 s; with ``outcomes`` = (events index, maturity per row) also equal to the event log."""
    clicked = item["attributes"]["clicked"]
    if type(clicked) is not int or clicked not in (0, 1):
        return False
    exposure, mature = _epoch(item["event_time"]), _epoch(item["available_at"])
    low, high = AD_CLICK_DELAY_SECONDS
    delay = mature - exposure
    if not (low <= delay <= high if clicked else delay == AD_OUTCOME_MATURITY_SECONDS):
        return False
    if outcomes is None:
        return True
    index, maturity = outcomes
    return (0 <= number < index.n and index.event_id[number].as_py() == item["event_id"]
            and int(index.ts[number]) == exposure and int(maturity[number]) == mature)


def _baseline_phase(results: _Results, state: _State, max_bytes: int) -> None:
    for name, requests in (("food", state.food), ("ads", state.ads)):
        if requests is None:
            for suffix in ("structure", "history_before_cutoff", "load_dataset"):
                results.not_run(f"baseline_{name}_{suffix}", f"{name} package not checked")
            continue
        _baseline_file(results, state, name, requests, max_bytes)


# ---- entry point ----------------------------------------------------------------------------


def run_checks(out_dir: Path, config: SimConfig, *, expect_pretrain: bool = True,
               baseline_full_load_max_bytes: int = BASELINE_FULL_LOAD_MAX_BYTES,
               pretrain_loader_users: int | None = PRETRAIN_LOADER_USERS,
               only: Collection[str] | None = None) -> dict:
    """Check the exported files under ``out_dir`` (see module docstring).

    Returns ``{status, results, files, tree_sha256}``; ``results`` maps check name ->
    ``{status, description, violations, ids?, ...}``. Raises ``CheckError`` when any check fails.
    ``expect_pretrain=False`` (``--skip-pretrain``) marks the pretrain checks ``skipped`` when
    ``pretrain/`` is absent. ``baseline_full_load_max_bytes`` and ``pretrain_loader_users`` set
    the Baseline loader mode and the size of the pretrain loader spot check. ``only`` restricts
    the run to some of ``PACKAGES`` (the others are ``skipped``; checks that need an unselected
    package are ``not_run``) -- for targeted tests; the CLI always checks everything.
    """
    selected = set(PACKAGES if only is None else only)
    unknown = selected - set(PACKAGES)
    if unknown:
        raise ValueError(f"unknown packages {sorted(unknown)}; expected some of {PACKAGES}")
    root = Path(out_dir)
    results = _Results()
    files = file_hashes(root)
    state = _State(root, config, files)
    phases = (
        ("pretrain", PRETRAIN_CHECKS, _pretrain_phase, (pretrain_loader_users,)),
        ("ads", ADS_CHECKS, _ads_phase, ()),
        ("food", FOOD_CHECKS, _food_phase, ()),
        ("oracle", ORACLE_CHECKS, _oracle_phase, ()),
        ("baseline", _baseline_checks(), _baseline_phase, (baseline_full_load_max_bytes,)),
    )
    for package, checks, fn, extra in phases:
        if package not in selected:
            results.skip(package, checks, "package not selected")
        elif package == "pretrain" and not expect_pretrain and not (root / "pretrain").exists():
            results.skip(package, checks, "pretrain export skipped (--skip-pretrain)")
        else:
            results.phase(package, checks, fn, results, state, *extra)

    failed = results.failed()
    report = {"status": "failed" if failed else "passed", "results": results.checks,
              "files": files, "tree_sha256": tree_sha256(files)}
    if failed:
        parts = []
        for name in failed:
            entry = results.checks[name]
            detail = entry.get("error") or ", ".join(entry.get("ids", []))
            parts.append(f"{name} ({entry['violations']} violation(s): {detail})")
        raise CheckError("simulator checks failed: " + "; ".join(parts), report)
    return report


# ---- data_report.json -----------------------------------------------------------------------


def write_data_report(root: Path, report: dict) -> Path:
    """Write ``root/data_report.json`` (UTF-8, indented, strict JSON) and return its path."""
    path = Path(root) / REPORT_NAME
    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.write_bytes(text.encode("utf-8"))
    return path



def simulation_summary(sim, food_requests, ad_requests) -> dict:
    """Run-level counts for ``data_report.json`` (spec §6.6) from the in-memory simulation:
    users/events/requests per split, CTR, persona and intent distributions."""
    world = sim.world
    config = world.config
    limit = day_start_ts(pretrain_end_day(config), config)
    personas = Counter(u.persona for u in world.users)
    intents = {}
    for kind in INTENT_KINDS:
        episodes = [i for i in sim.intents if i.kind == kind]
        intents[kind] = {"episodes": len(episodes), "users": len({i.user_id for i in episodes})}
    clicks = sum(bool(i.clicked) for i in sim.ad_impressions)
    ctr_by_split: dict[str, float | None] = {}
    for split in SPLITS:
        labels = [bool(r.impression.clicked) for r in ad_requests if r.split == split]
        ctr_by_split[split] = sum(labels) / len(labels) if labels else None
    return {
        "counts": {
            "users": len(world.users),
            "merchants": len(world.merchants),
            "ads": len(world.ads),
            "drivers": len(world.drivers),
            "events": len(sim.events),
            "events_before_week7": sum(1 for e in sim.events if e.ts < limit),
            "food_sessions_with_order": len(sim.food_sessions),
            "ad_impressions": len(sim.ad_impressions),
            "food_requests_by_split": {s: sum(r.split == s for r in food_requests)
                                       for s in SPLITS},
            "ads_requests_by_split": {s: sum(r.split == s for r in ad_requests) for s in SPLITS},
        },
        "ctr": {
            "all_impressions": clicks / len(sim.ad_impressions) if sim.ad_impressions else None,
            "requests_by_split": ctr_by_split,
        },
        "personas": {p: personas.get(p, 0) for p in PERSONA_SHARES},
        "intents": intents,
    }
