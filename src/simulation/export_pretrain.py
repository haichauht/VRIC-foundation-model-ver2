"""Pretraining package ``pretrain/`` in the ``omnibehavior_pretraining_v2`` schema (spec §6.1).

Layout (readable by ``src.foundation_input`` without changes)::

    pretrain/manifest.json               written last; shard list with row counts and SHA-256
    pretrain/preprocessing_config.json   max_history_events, split bounds, allowlists
    pretrain/preview.json                a few reconstructed samples
    pretrain/events/<user>.parquet       one row per funnel step, user_id with ':' -> '_'
    pretrain/snapshots/<user>.parquet    one row per (user, cutoff) with a non-empty history

Only events before local day ``7 * (weeks - 2) + 1`` (day 43 = start of week 7) are exported.
Each session is one source record; each funnel step is one action. All times are naive UTC.
Users are written one at a time; files are byte-deterministic for a given simulation.
"""

from __future__ import annotations

import hashlib
import json
from bisect import bisect_left
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .behavior import Event, Simulation
from .config import GENERATOR_VERSION, config_sha256
from .history import (
    PROFILE_FIELDS,
    PROFILE_WINDOW_DAYS,
    event_order_key,
    last_history,
    profile_at,
)
from .timeutil import SECONDS_PER_DAY, day_start_ts, local_iso, naive_utc

SCHEMA_VERSION = "omnibehavior_pretraining_v2"
SOURCE_DATASET = "sim_v1"
SPLITS: tuple[str, ...] = ("train", "validation", "test")
SCENARIOS: dict[str, str] = {"ride": "Ride", "food": "Food", "ads": "Advertisement"}
CONTEXT_ALLOWLIST: dict[str, list[str]] = {
    "Ride": ["pickup_geohash", "dropoff_geohash", "pickup_zone_type", "dropoff_zone_type",
             "vehicle_type", "driver_id"],
    "Food": ["merchant_id", "merchant_name", "cuisine", "price_level", "search_query",
             "dish_id", "dish_name"],
    "Advertisement": ["product", "industry_main", "industry_sub", "caption"],
}
COMPRESSION = "zstd"
COMPRESSION_LEVEL = 3
PREVIEW_SAMPLES = 3

_METADATA = {"schema_version": SCHEMA_VERSION}
EVENT_SCHEMA = pa.schema(
    [
        ("row_index", pa.int32()),
        ("event_id", pa.string()),
        ("user_id", pa.string()),
        ("source_record_id", pa.string()),
        ("source_record_index", pa.int32()),
        ("action_index", pa.int32()),
        ("source_action_count", pa.int32()),
        ("source_timestamp", pa.string()),
        ("event_time", pa.string()),
        ("time_precision", pa.string()),
        ("time_lower_bound", pa.timestamp("us")),
        ("time_upper_bound", pa.timestamp("us")),
        ("upper_bound_exclusive", pa.bool_()),
        ("eligibility_time", pa.timestamp("us")),
        ("eligibility_inclusive", pa.bool_()),
        ("scenario", pa.string()),
        ("event_type", pa.string()),
        ("context_json", pa.large_string()),
        ("action_json", pa.large_string()),
    ],
    metadata=_METADATA,
)
SNAPSHOT_SCHEMA = pa.schema(
    [
        ("sample_id", pa.string()),
        ("user_id", pa.string()),
        ("cutoff_time", pa.timestamp("us")),
        ("cutoff_inclusive", pa.bool_()),
        ("split", pa.string()),
        ("history_start", pa.int32()),
        ("history_stop", pa.int32()),
        ("history_event_ids", pa.list_(pa.field("element", pa.string()))),
        ("history_order_known", pa.bool_()),
        ("history_truncated", pa.bool_()),
        ("history_group_truncated", pa.bool_()),
        ("profile_json", pa.string()),
    ],
    metadata=_METADATA,
)


# ---- calendar ---------------------------------------------------------------------------------


def pretrain_end_day(config) -> int:
    """First local day not exported (day 43 = start of week 7 for 8 weeks)."""
    return 7 * (config.weeks - 2) + 1


def train_end_day(config) -> int:
    """First local day of the pretrain validation period: the last 3 snapshot periods (34)."""
    return pretrain_end_day(config) - 3 * config.snapshot_every_days


def snapshot_days(config) -> dict[str, tuple[int, ...]]:
    """Local days whose 23:59:59 is a snapshot cutoff: train 6, 9, ..., 33; validation 37, 40."""
    every = config.snapshot_every_days
    split_day = train_end_day(config)
    return {
        "train": tuple(range(2 * every, split_day, every)),
        "validation": tuple(range(split_day + every, pretrain_end_day(config), every)),
    }


def _iso(ts: int) -> str:
    return naive_utc(ts).isoformat()


def split_bounds(config) -> dict[str, str]:
    """``preprocessing_config.splits`` (naive UTC); the test split is empty."""
    end = _iso(day_start_ts(pretrain_end_day(config), config))
    return {
        "start_inclusive": _iso(day_start_ts(1, config)),
        "train_end_exclusive": _iso(day_start_ts(train_end_day(config), config)),
        "validation_end_exclusive": end,
        "test_end_exclusive": end,
    }


def shard_name(user_id: str) -> str:
    """Parquet file name of a user's shard: ``sim:user:0`` -> ``sim_user_0.parquet``."""
    return user_id.replace(":", "_") + ".parquet"


# ---- writing ----------------------------------------------------------------------------------


def _dumps(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _write_parquet(table: pa.Table, path: Path) -> tuple[int, str]:
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink, compression=COMPRESSION, compression_level=COMPRESSION_LEVEL)
    data = sink.getvalue().to_pybytes()
    path.write_bytes(data)
    return len(data), hashlib.sha256(data).hexdigest()


def _write_json(value, path: Path) -> str:
    data = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    path.write_bytes(data)  # bytes: no newline translation, so the hash matches on Windows
    return hashlib.sha256(data).hexdigest()


def _events_table(user_id: str, events: list[Event], config) -> pa.Table:
    micros = [e.ts * 1_000_000 for e in events]
    times = pa.array(micros, type=pa.timestamp("us"))
    n = len(events)
    return pa.Table.from_arrays(
        [
            pa.array(range(n), type=pa.int32()),
            pa.array([e.event_id for e in events], type=pa.string()),
            pa.array([user_id] * n, type=pa.string()),
            pa.array([e.session_id for e in events], type=pa.string()),
            pa.array([e.session_index for e in events], type=pa.int32()),
            pa.array([e.action_index for e in events], type=pa.int32()),
            pa.array([e.session_size for e in events], type=pa.int32()),
            pa.array([local_iso(e.ts, config) for e in events], type=pa.string()),
            pa.array([_iso(e.ts) for e in events], type=pa.string()),
            pa.array(["second"] * n, type=pa.string()),
            times,
            times,
            pa.array([False] * n, type=pa.bool_()),
            times,
            pa.array([True] * n, type=pa.bool_()),
            pa.array([SCENARIOS[e.service] for e in events], type=pa.string()),
            pa.array([e.event_type for e in events], type=pa.string()),
            pa.array([_dumps(e.context) for e in events], type=pa.large_string()),
            pa.array([_dumps(e.action) for e in events], type=pa.large_string()),
        ],
        schema=EVENT_SCHEMA,
    )


def _snapshot_rows(user, events: list[Event], sim: Simulation) -> list[dict]:
    """One row per cutoff with a non-empty history (whole sessions, at most max events)."""
    config = sim.world.config
    max_events = config.max_history_events
    times = [e.ts for e in events]
    row_of = {e.event_id: i for i, e in enumerate(events)}
    rows: list[dict] = []
    for split, days in snapshot_days(config).items():
        for day in days:
            end = day_start_ts(day + 1, config)  # events strictly before -> up to 23:59:59
            n = bisect_left(times, end)
            history, _ = last_history(events[:n], end, max_events)
            if not history:
                continue
            start = row_of[history[0].event_id]
            stop = start + len(history)
            ids = [e.event_id for e in history]
            if [e.event_id for e in events[start:stop]] != ids:
                raise RuntimeError(f"{user.user_id}: history at day {day} is not contiguous")
            lo = bisect_left(times, end - PROFILE_WINDOW_DAYS * SECONDS_PER_DAY)
            cutoff_iso = _iso(end - 1)
            profile = {"as_of_time": cutoff_iso}
            profile.update(profile_at(events[lo:n], user, end, config, zones=sim.world.zones))
            rows.append({
                "sample_id": f"{user.user_id}:{cutoff_iso}:closed",
                "user_id": user.user_id,
                "cutoff_time": (end - 1) * 1_000_000,
                "cutoff_inclusive": True,
                "split": split,
                "history_start": start,
                "history_stop": stop,
                "history_event_ids": ids,
                "history_order_known": True,
                # the loader's rule: a full window with older events left out
                "history_truncated": start > 0 and len(history) >= max_events,
                # Ruling R11: sessions (event groups) are never split -- whole sessions only --
                # so no group is truncated. True would also turn off foundation_input's
                # next-action target (dataset.encode_sample) for the snapshot.
                "history_group_truncated": False,
                "profile_json": _dumps(profile),
            })
    return rows


def _snapshots_table(rows: list[dict]) -> pa.Table:
    return pa.Table.from_arrays(
        [pa.array([r[f.name] for r in rows], type=f.type) for f in SNAPSHOT_SCHEMA],
        schema=SNAPSHOT_SCHEMA,
    )


def _preview(row: dict, events: list[Event]) -> dict:
    history = events[row["history_start"]:row["history_stop"]]
    return {
        "sample_id": row["sample_id"],
        "input": {
            "user_id": row["user_id"],
            "split": row["split"],
            "cutoff_time": _iso(row["cutoff_time"] // 1_000_000),
            "cutoff_inclusive": row["cutoff_inclusive"],
            "timezone": None,
            "history_order_known": row["history_order_known"],
            "history_truncated": row["history_truncated"],
            "history_group_truncated": row["history_group_truncated"],
            "profile": json.loads(row["profile_json"]),
            "history": [
                {
                    "event_id": e.event_id,
                    "source_record_id": e.session_id,
                    "action_index": e.action_index,
                    "event_time": _iso(e.ts),
                    "time_precision": "second",
                    "scenario": SCENARIOS[e.service],
                    "event_type": e.event_type,
                    "context": e.context,
                    "action": e.action,
                }
                for e in history
            ],
        },
    }


def preprocessing_config(config) -> dict:
    days = snapshot_days(config)
    return {
        "schema_version": SCHEMA_VERSION,
        "language": "vi",
        "timezone": None,
        "time_basis": "naive UTC; source_timestamp keeps the local time with its UTC offset",
        "source_utc_offset_hours": config.utc_offset_hours,
        "max_history_events": config.max_history_events,
        "event_unit": "one funnel step (action) of a ride, food or ads session",
        "record_unit": "one session: source_record_id = session id",
        "profile_text_used": False,
        "context_allowlist": CONTEXT_ALLOWLIST,
        "profile_fields": list(PROFILE_FIELDS),
        "profile_window": "(cutoff_time - 30 days, cutoff_time]: the 30 local days ending on "
                          "the cutoff day, clipped at simulation start",
        "active_day_basis": f"local calendar day (UTC+{config.utc_offset_hours:02d}:00)",
        "history_selection": "most recent completed sessions before the cutoff, at most "
                             "max_history_events events; sessions are never split",
        "snapshot_cadence": "local 23:59:59 of every snapshot day; snapshots with an empty "
                            "history are skipped",
        "snapshot_days": {split: list(d) for split, d in days.items()},
        "events_end_exclusive": _iso(day_start_ts(pretrain_end_day(config), config)),
        "splits": split_bounds(config),
        "compression": {"codec": COMPRESSION, "level": COMPRESSION_LEVEL},
        "training_targets_stored": False,
        "mask_policy": None,
        "vocabulary_fit": False,
    }


def export_pretrain(sim: Simulation, out_dir: Path) -> dict:
    """Write ``out_dir/pretrain/`` (spec §6.1); returns ``{users, events, snapshots_by_split}``.

    Raises ``FileExistsError`` when ``out_dir/pretrain`` already has content.
    """
    world = sim.world
    config = world.config
    root = Path(out_dir) / "pretrain"
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {root}")
    (root / "events").mkdir(parents=True, exist_ok=True)
    (root / "snapshots").mkdir(parents=True, exist_ok=True)

    limit = day_start_ts(pretrain_end_day(config), config)
    by_user: dict[str, list[Event]] = {u.user_id: [] for u in world.users}
    for event in sim.events:
        if event.ts >= limit:
            continue
        bucket = by_user.get(event.user_id)
        if bucket is None:
            raise ValueError(f"event {event.event_id} has unknown user {event.user_id!r}")
        bucket.append(event)

    shards: list[dict] = []
    previews: list[dict] = []
    totals = dict.fromkeys(SPLITS, 0)
    total_events = 0
    for user in world.users:
        events = sorted(by_user.pop(user.user_id), key=event_order_key)
        rows = _snapshot_rows(user, events, sim)
        name = shard_name(user.user_id)
        events_bytes, events_sha = _write_parquet(
            _events_table(user.user_id, events, config), root / "events" / name)
        snaps_bytes, snaps_sha = _write_parquet(
            _snapshots_table(rows), root / "snapshots" / name)
        by_split = dict.fromkeys(SPLITS, 0)
        for row in rows:
            by_split[row["split"]] += 1
        for split, count in by_split.items():
            totals[split] += count
        total_events += len(events)
        if rows and len(previews) < PREVIEW_SAMPLES:
            train = [r for r in rows if r["split"] == "train"]
            previews.append(_preview(train[-1] if train else rows[0], events))
        shards.append({
            "user_id": user.user_id,
            "events_file": f"events/{name}",
            "snapshots_file": f"snapshots/{name}",
            "event_rows": len(events),
            "snapshot_rows": len(rows),
            "snapshots_by_split": by_split,
            "events_bytes": events_bytes,
            "snapshots_bytes": snaps_bytes,
            "events_sha256": events_sha,
            "snapshots_sha256": snaps_sha,
        })

    config_sha = _write_json(preprocessing_config(config), root / "preprocessing_config.json")
    preview_sha = _write_json(previews, root / "preview.json")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "source_dataset": SOURCE_DATASET,
        "dataset_version": config.dataset_version,
        "generator_version": GENERATOR_VERSION,
        "seed": config.seed,
        "sim_config_sha256": config_sha256(config),
        "pyarrow_version": pa.__version__,
        "config_sha256": config_sha,
        "preview_sha256": preview_sha,
        "total_users": len(shards),
        "total_events": total_events,
        "total_snapshots": sum(totals.values()),
        "snapshots_by_split": totals,
        "shards": shards,
    }
    _write_json(manifest, root / "manifest.json")
    return {"users": len(shards), "events": total_events, "snapshots_by_split": dict(totals)}
