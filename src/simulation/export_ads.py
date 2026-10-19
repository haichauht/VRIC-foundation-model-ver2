"""Ads package ``ads/`` (spec §5 Ads, §6.3).

Layout::

    ads/profiles.parquet          static profile fields, one row per user
    ads/ad_catalog.parquet        ad catalog (category, advertiser, title)
    ads/events.parquet            events of ALL services, per user in chronological order
    ads/ad_opportunities.parquet  one row per request = one served ``ad_impression``
    ads/manifest.json             written last; provenance, file SHA-256, history policy

Requests: every simulated ``ad_impression`` (``Simulation.ad_impressions``) is one request with a
single candidate (the ad that was served) and the label ``clicked`` drawn from the true
probability when it was simulated; there is no negative resampling. Split by the local week of the
exposure timestamp. ``config.ads_requests`` = None takes every impression, otherwise exactly that
many per split are drawn without replacement from ``rng`` (all of them when fewer exist).

History pointers: ``events`` rows of a user are ordered ``(timestamp, session_index,
action_index)`` and numbered by ``event_index`` (0..n-1 per user) and ``source_row_index``
(global row number). A request's window is ``events[user][history_start:history_stop]`` with
``history_stop = number of the user's events with timestamp < exposure`` (strict, so events in the
same second are excluded and the request's own impression and click never fall inside) and
``history_start = max(0, history_stop - max_history_events)``.

No label leak: ``attributes_json`` holds only the public context fields of an event (and the
numeric action fields); never session size, action counts or a click flag. In the simulation an
``ad_impression`` session has two events exactly when it was clicked, so none of that may be
exported.
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa

from .behavior import AdImpression, Event, Simulation
from .config import GENERATOR_VERSION, config_sha256
from .export_food import TIMEZONE, _attributes
from .export_pretrain import (
    SOURCE_DATASET,
    SPLITS,
    _dumps,
    _write_json,
    _write_parquet,
)
from .history import event_order_key
from .timeutil import day_start_ts, split_of

SCHEMA_VERSION = 1
CANDIDATE_POLICY = "sim-served-impression-single-candidate-v1"
HISTORY_POLICY_RULE = "strictly < cutoff"

_META = {"schema_version": str(SCHEMA_VERSION)}
PROFILE_SCHEMA = pa.schema(
    [
        ("user_id", pa.string()),
        ("as_of_timestamp", pa.int64()),
        ("age_group", pa.string()),
        ("saved_home_geohash", pa.string()),
        ("saved_work_geohash", pa.string()),
    ],
    metadata=_META,
)
CATALOG_SCHEMA = pa.schema(
    [
        ("candidate_id", pa.string()),
        ("first_observed_timestamp", pa.int64()),
        ("category", pa.string()),
        ("advertiser", pa.string()),
        ("title", pa.string()),
    ],
    metadata=_META,
)
EVENT_SCHEMA = pa.schema(
    [
        ("event_id", pa.string()),
        ("source_row_index", pa.int64()),
        ("user_id", pa.string()),
        ("timestamp", pa.int64()),
        ("event_type", pa.string()),
        ("service", pa.string()),
        ("entity_id", pa.string()),
        ("event_index", pa.int64()),
        ("attributes_json", pa.large_string()),
    ],
    metadata=_META,
)
OPPORTUNITY_SCHEMA = pa.schema(
    [
        ("request_id", pa.string()),
        ("source_row_index", pa.int64()),
        ("user_id", pa.string()),
        ("cutoff_timestamp", pa.int64()),
        ("candidate_id", pa.string()),
        ("clicked", pa.bool_()),
        ("split", pa.string()),
        ("exposure_timestamp", pa.int64()),
        ("candidate_timestamp", pa.int64()),
        ("profile_as_of_timestamp", pa.int64()),
        ("history_end_timestamp", pa.int64()),
        ("history_start", pa.int64()),
        ("history_stop", pa.int64()),
    ],
    metadata=_META,
)


@dataclass(frozen=True)
class AdRequest:
    """One Ads request. ``source_row_index`` = row of the impression in ``events.parquet``."""

    request_id: str
    impression: AdImpression
    split: str
    source_row_index: int


def _limits(config) -> dict[str, int] | None:
    limits = config.ads_requests
    if limits is None:
        return None
    if set(limits) != set(SPLITS):
        raise ValueError(f"ads_requests keys {sorted(limits)} must be exactly {list(SPLITS)}")
    if any(int(v) != v or v < 0 for v in limits.values()):
        raise ValueError(f"ads_requests counts must be non-negative integers: {limits}")
    return {s: int(limits[s]) for s in SPLITS}


def _ordered_events(sim: Simulation) -> list[list[Event]]:
    """Events per user (world user order), each list in chronological order."""
    grouped: dict[str, list[Event]] = {u.user_id: [] for u in sim.world.users}
    for event in sim.events:
        bucket = grouped.get(event.user_id)
        if bucket is None:
            raise ValueError(f"event {event.event_id} has unknown user {event.user_id!r}")
        bucket.append(event)
    return [sorted(grouped[u.user_id], key=event_order_key) for u in sim.world.users]


def _row_indices(per_user: Sequence[Sequence[Event]]) -> dict[str, int]:
    """Global ``events.parquet`` row number of every event id."""
    rows: dict[str, int] = {}
    for events in per_user:
        for event in events:
            rows[event.event_id] = len(rows)
    return rows


def build_ad_requests(sim: Simulation, rng: np.random.Generator) -> list[AdRequest]:
    """Requests grouped by split (train, validation, test), chronological inside each split.

    ``rng`` is used only to draw the per-split samples, when ``config.ads_requests`` is set.
    Ids are ``sim:ad_req:<n>`` in that order.
    """
    config = sim.world.config
    limits = _limits(config)
    rows = _row_indices(_ordered_events(sim))
    by_split: dict[str, list[AdImpression]] = {s: [] for s in SPLITS}
    for imp in sorted(sim.ad_impressions, key=lambda i: i.ts):  # stable: simulation order
        by_split[split_of(imp.ts, config)].append(imp)
    chosen: list[tuple[str, AdImpression]] = []
    for split in SPLITS:
        impressions = by_split[split]
        limit = None if limits is None else limits[split]
        if limit is not None and limit < len(impressions):
            picks = np.sort(rng.choice(len(impressions), size=limit, replace=False))
            impressions = [impressions[int(i)] for i in picks]
        chosen.extend((split, imp) for imp in impressions)
    return [
        AdRequest(f"sim:ad_req:{n}", imp, split, rows[imp.event_id])
        for n, (split, imp) in enumerate(chosen)
    ]


# ---- export ---------------------------------------------------------------------------------


def _table(schema: pa.Schema, columns: dict[str, list]) -> pa.Table:
    return pa.Table.from_arrays(
        [pa.array(columns[f.name], type=f.type) for f in schema], schema=schema)


def _profiles_table(sim: Simulation, as_of: int) -> pa.Table:
    zones = sim.world.zones
    users = sim.world.users
    work = [None if u.work_zone is None else zones[u.work_zone].geohash for u in users]
    return _table(PROFILE_SCHEMA, {
        "user_id": [u.user_id for u in users],
        "as_of_timestamp": [as_of] * len(users),
        "age_group": [u.age_group for u in users],
        "saved_home_geohash": [zones[u.home_zone].geohash for u in users],
        "saved_work_geohash": work,
    })


def _catalog_table(sim: Simulation, first_observed: int) -> pa.Table:
    ads = sim.world.ads
    return _table(CATALOG_SCHEMA, {
        "candidate_id": [a.ad_id for a in ads],
        "first_observed_timestamp": [first_observed] * len(ads),
        "category": [a.category for a in ads],
        "advertiser": [a.advertiser for a in ads],
        "title": [a.title for a in ads],
    })


def export_ads(sim: Simulation, requests: Sequence[AdRequest], out_dir) -> dict:
    """Write ``out_dir/ads/`` (spec §6.3); returns ``{requests, requests_by_split, events}``.

    Raises ``FileExistsError`` when ``out_dir/ads`` already has content.
    """
    world = sim.world
    config = world.config
    root = Path(out_dir) / "ads"
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {root}")
    root.mkdir(parents=True, exist_ok=True)
    day1 = day_start_ts(1, config)
    max_events = config.max_history_events

    per_user = _ordered_events(sim)
    cols: dict[str, list] = {f.name: [] for f in EVENT_SCHEMA}
    times: dict[str, list[int]] = {}
    row_of: dict[str, int] = {}
    for user, events in zip(world.users, per_user):
        times[user.user_id] = [e.ts for e in events]
        for k, e in enumerate(events):
            row_of[e.event_id] = len(row_of)
            cols["event_id"].append(e.event_id)
            cols["source_row_index"].append(row_of[e.event_id])
            cols["user_id"].append(e.user_id)
            cols["timestamp"].append(e.ts)
            cols["event_type"].append(e.event_type)
            cols["service"].append(e.service)
            cols["entity_id"].append(e.entity_id)
            cols["event_index"].append(k)
            cols["attributes_json"].append(_dumps(_attributes(e)))

    opp: dict[str, list] = {f.name: [] for f in OPPORTUNITY_SCHEMA}
    by_split = dict.fromkeys(SPLITS, 0)
    for r in requests:
        imp = r.impression
        if row_of.get(imp.event_id) != r.source_row_index:
            raise ValueError(f"{r.request_id}: source_row_index does not match events table")
        if split_of(imp.ts, config) != r.split:
            raise ValueError(f"{r.request_id}: split does not match the exposure timestamp")
        ts = times[imp.user_id]
        stop = bisect_left(ts, imp.ts)  # strict: equal-second events excluded
        start = max(0, stop - max_events)
        opp["request_id"].append(r.request_id)
        opp["source_row_index"].append(r.source_row_index)
        opp["user_id"].append(imp.user_id)
        opp["cutoff_timestamp"].append(imp.ts)
        opp["candidate_id"].append(imp.ad_id)
        opp["clicked"].append(bool(imp.clicked))
        opp["split"].append(r.split)
        opp["exposure_timestamp"].append(imp.ts)
        opp["candidate_timestamp"].append(day1)
        opp["profile_as_of_timestamp"].append(day1)
        opp["history_end_timestamp"].append(ts[stop - 1] if stop else None)
        opp["history_start"].append(start)
        opp["history_stop"].append(stop)
        by_split[r.split] += 1

    tables = {
        "profiles": _profiles_table(sim, day1),
        "ad_catalog": _catalog_table(sim, day1),
        "events": _table(EVENT_SCHEMA, cols),
        "ad_opportunities": _table(OPPORTUNITY_SCHEMA, opp),
    }
    files = {}
    for name, table in tables.items():
        size, sha = _write_parquet(table, root / f"{name}.parquet")
        files[name] = {"path": f"{name}.parquet", "format": "parquet", "sha256": sha,
                       "row_count": table.num_rows, "bytes": size}
    manifest = {
        "dataset_version": config.dataset_version,
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "seed": config.seed,
        "sim_config_sha256": config_sha256(config),
        "pyarrow_version": pa.__version__,
        "timezone": TIMEZONE,
        "utc_offset": f"+{config.utc_offset_hours:02d}:00",
        "timestamp_unit": "seconds",
        "files": files,
        "split": {
            "kind": "temporal",
            "basis": "exposure_timestamp",
            "train_end": day_start_ts((config.weeks - 2) * 7 + 1, config),
            "valid_end": day_start_ts((config.weeks - 1) * 7 + 1, config),
        },
        "request_counts": dict(by_split),
        "source_dataset": SOURCE_DATASET,
        "history_policy": {
            "timestamp_rule": HISTORY_POLICY_RULE,
            "target_click_in_history": False,
            "equal_timestamp_events_excluded": True,
            "max_events": max_events,
        },
        "candidate_policy": CANDIDATE_POLICY,
        "missing_click_policy": "reject",
        "negative_sampling": "none",
        "row_count": len(requests),
    }
    _write_json(manifest, root / "manifest.json")
    return {"requests": len(requests), "requests_by_split": dict(by_split),
            "events": tables["events"].num_rows}
