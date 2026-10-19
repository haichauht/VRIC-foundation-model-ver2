"""Food recommendation package ``food/`` (spec §5, §6.2).

Layout::

    food/samples.jsonl       one request per line (Meituan ``samples.jsonl`` format, extended)
    food/merchants.parquet   merchant catalog, public attributes only
    food/manifest.json       written last; provenance, file SHA-256, split, candidate policy

Requests: one per Food session that ended in an ``order`` (``Simulation.food_sessions``).
Prediction anchor = session start: ``cutoff_ts = start_ts - 1``. History and profile use only
events with ``ts < cutoff_ts`` (whole completed sessions, ``history.last_history``), so no event
of the request's own session is ever an input. Split by the local week of ``cutoff_ts``
(``timeutil.split_of``). ``config.food_requests`` = None takes every request, otherwise exactly
that many per split are drawn without replacement from ``rng`` (all of them when fewer exist).

Candidates (policy ``sim-popularity-matched-home-work-v1``): the ordered merchant + 9 others.

1. Popularity unit (fixed partition shared by every merchant): merchants are ranked by their
   order count in weeks ``1..weeks-2`` (ordinal rank, ties by catalog order), decile =
   ``rank * 10 // n``, unit = decile pair {0,1}, {2,3}, {4,5}, {6,7}, {8,9}. The pool is the
   positive's unit minus the positive. Last resort, only for catalogs where a pair holds fewer
   than 10 merchants (n < 50): neighbouring pairs are merged bottom-up into units of at least
   10 merchants, a short top remainder joining the unit below (still one shared partition).
2. Zone clusters with exchangeable sizes (controller ruling R8): the distinct anchors among the
   positive merchant's zone, the user's home zone and work zone (the saved work place,
   ``UserTraits.work_zone``; users without one: the mall zone nearest home) each get one cluster
   of the 10 candidates, and the positive counts toward its own zone's cluster. 3 distinct
   anchors: sizes (4, 3, 3) assigned by a uniform random permutation drawn from ``rng``, so the
   positive's cluster has 3 or 2 negatives; 2 distinct anchors: sizes (5, 5), i.e. 4 negatives
   in the positive's cluster and 5 in the other; 1 anchor: all 9 negatives in the positive's
   cluster (``cluster_sizes``, ``zone_groups``).
3. Stochastic distance decay (ruling R10, ``tau = config.food_candidate_tau_km``, default 2.0 km,
   calibrated in Task 8; previously 1.5 km):
   each cluster has a centre zone -- the home / work (mall) zone itself for those clusters, and
   for the positive's cluster a *latent* zone drawn over all zones with P(z) proportional to
   ``exp(-d(z, positive's zone) / tau)``. Each cluster's negatives are drawn from the pool
   without replacement (excluding the positive and merchants already chosen), one after
   another with P(m) proportional to ``exp(-d(zone(m), centre) / tau)`` (``decay_sample``).
   So the positive is not at a fill centre and no negative is a deterministic spill.
4. The 10 candidates are shuffled with ``rng``. ``rng`` order per request: size permutation
   (3 distinct anchors only), latent centre, each cluster's negatives (anchor order), shuffle.

Inputs never contain the current location or a distance from it; it is only inferable from the
history (ride drop-offs). Event ``context``/``action`` dicts are shared and never mutated.
"""

from __future__ import annotations

import hashlib
from bisect import bisect_left
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa

from .behavior import Event, FoodSession, Simulation
from .config import GENERATOR_VERSION, config_sha256
from .export_pretrain import (  # shared deterministic writers (private helpers, same package)
    CONTEXT_ALLOWLIST,
    SCENARIOS,
    SOURCE_DATASET,
    SPLITS,
    _dumps,
    _write_json,
    _write_parquet,
    pretrain_end_day,
)
from .history import (
    PROFILE_WINDOW_DAYS,
    completed_sessions_before,
    event_order_key,
    last_history,
    profile_at,
)
from .timeutil import (
    SECONDS_PER_DAY,
    day_start_ts,
    local_hour,
    local_iso,
    local_weekday,
    split_of,
)
from .world import UserTraits, World, haversine_km

CANDIDATE_POLICY = "sim-popularity-matched-home-work-v1"
SCHEMA_VERSION = 1
TIMEZONE = "Asia/Ho_Chi_Minh"
PREDICTION_ANCHOR = "session_start"
POSITIVE_ACTION = "order"
N_CANDIDATES = 10
# Cluster sizes (positive included) by number of distinct anchors (controller ruling R8).
CLUSTER_SIZES: dict[int, tuple[int, ...]] = {1: (10,), 2: (5, 5), 3: (4, 3, 3)}
# Radius of the "near an anchor" notion; used for reporting (candidate_stats) only.
ANCHOR_REPORT_KM = 3.0
N_DECILES = 10
DECILES_PER_UNIT = 2  # popularity unit = decile pair (controller ruling R6)
# Numeric action fields that become history attributes (``type`` is the event_type itself).
ACTION_ATTRIBUTES: tuple[str, ...] = (
    "fare", "distance_km", "wait_minutes", "order_amount", "delivery_distance_km",
)
MERCHANT_SCHEMA = pa.schema(
    [
        ("merchant_id", pa.string()),
        ("merchant_name", pa.string()),
        ("cuisine", pa.string()),
        ("price_level", pa.int32()),
        ("geohash", pa.string()),
        ("zone_type", pa.string()),
    ],
    metadata={"schema_version": str(SCHEMA_VERSION)},
)


@dataclass(frozen=True)
class FoodRequest:
    """One Food ranking request; ``candidate_ids`` in presentation (shuffled) order."""

    request_id: str
    session: FoodSession
    cutoff_ts: int
    split: str
    candidate_ids: tuple[str, ...]
    positive_id: str


# ---- popularity deciles -------------------------------------------------------------------


def _ranks(order_counts: Sequence[int]) -> list[int]:
    """Ordinal rank of each merchant by (order count, catalog position), 0 = least ordered."""
    order = sorted(range(len(order_counts)), key=lambda i: (order_counts[i], i))
    ranks = [0] * len(order_counts)
    for rank, i in enumerate(order):
        ranks[i] = rank
    return ranks


def popularity_deciles(order_counts: Sequence[int]) -> list[int]:
    """Decile (0..9) of each merchant: ``ordinal rank * 10 // n`` (near-equal sizes)."""
    n = len(order_counts)
    return [rank * N_DECILES // n for rank in _ranks(order_counts)]


def popularity_units(order_counts: Sequence[int]) -> list[int]:
    """Popularity unit of each merchant: a fixed partition shared by all merchants.

    Unit = decile pair {0,1}, {2,3}, {4,5}, {6,7}, {8,9} (ids 0..4). Last resort, only when some
    pair holds fewer than ``N_CANDIDATES`` merchants (catalogs under 50): neighbouring pairs are
    merged bottom-up into units of at least ``N_CANDIDATES`` merchants and a short top remainder
    joins the unit below. Fewer than ``N_CANDIDATES`` merchants in all raises ``ValueError``.
    """
    n = len(order_counts)
    if n < N_CANDIDATES:
        raise ValueError(f"need at least {N_CANDIDATES} merchants for a candidate set, got {n}")
    pairs = [d // DECILES_PER_UNIT for d in popularity_deciles(order_counts)]
    sizes = Counter(pairs)
    unit_of_pair: dict[int, int] = {}
    unit = filled = 0
    for pair in range(N_DECILES // DECILES_PER_UNIT):
        unit_of_pair[pair] = unit
        filled += sizes[pair]
        if filled >= N_CANDIDATES:
            unit += 1
            filled = 0
    if filled:  # short top remainder joins the unit below (one exists since n >= 10)
        for pair, u in unit_of_pair.items():
            if u == unit:
                unit_of_pair[pair] = unit - 1
    return [unit_of_pair[p] for p in pairs]


def train_order_counts(sim: Simulation) -> list[int]:
    """Orders per merchant (catalog order) placed before week ``weeks - 1`` (weeks 1-6)."""
    config = sim.world.config
    limit = day_start_ts(pretrain_end_day(config), config)
    position = {m.merchant_id: i for i, m in enumerate(sim.world.merchants)}
    counts = [0] * len(position)
    for session in sim.food_sessions:
        if session.order_ts < limit:
            counts[position[session.merchant_id]] += 1
    return counts


# ---- zone groups --------------------------------------------------------------------------


def anchor_zones(positive_zone: int, home_zone: int, third_zone: int) -> list[int]:
    """Distinct anchor zones in fill order: positive's zone, home, work (or mall)."""
    return list(dict.fromkeys((positive_zone, home_zone, third_zone)))


def cluster_sizes(n_anchors: int, rng: np.random.Generator) -> tuple[int, ...]:
    """Cluster sizes (positive included) per distinct anchor, in anchor order (ruling R8).

    3 anchors: (4, 3, 3) under a uniform random permutation drawn from ``rng``; 2 anchors:
    (5, 5); 1 anchor: (10,). Only the 3-anchor case consumes ``rng``.
    """
    sizes = CLUSTER_SIZES[n_anchors]
    if n_anchors == 3:
        return tuple(sizes[int(i)] for i in rng.permutation(3))
    return sizes


def zone_groups(positive_zone: int, home_zone: int, third_zone: int,
                sizes: Sequence[int]) -> list[tuple[int, int]]:
    """``[(anchor zone, number of negatives)]`` in fill order; negatives always sum to 9.

    ``sizes`` are the cluster sizes of the distinct anchors (``cluster_sizes``); the positive
    counts toward its own zone's cluster, which therefore gets one negative fewer.
    """
    anchors = anchor_zones(positive_zone, home_zone, third_zone)
    if len(sizes) != len(anchors) or sum(sizes) != N_CANDIDATES:
        raise ValueError(f"cluster sizes {tuple(sizes)} do not fit anchors {anchors}")
    return [(a, s - (a == positive_zone)) for a, s in zip(anchors, sizes)]


def nearest_mall_zone(world: World, zone_id: int) -> int:
    """Mall zone nearest to ``zone_id`` (zone-centre haversine, ties by zone id)."""
    origin = world.zones[zone_id]
    malls = [z for z in world.zones if z.zone_type == "mall"]
    if not malls:
        raise ValueError("the world has no mall zone")
    return min(malls, key=lambda z: (haversine_km(origin, z), z.zone_id)).zone_id


def third_anchor_zone(world: World, user: UserTraits) -> int:
    """Work zone (saved work place) or, for users without one, the mall zone nearest home."""
    return user.work_zone if user.work_zone is not None else nearest_mall_zone(
        world, user.home_zone)


def decay_sample(rng: np.random.Generator, distances_km, k: int, tau_km: float) -> np.ndarray:
    """``k`` distinct indices drawn one after another with P(i) proportional to
    ``exp(-d_i / tau)`` among the indices still left (ruling R10).

    Implemented as an exponential race: the ``k`` smallest keys ``log E_i + d_i / tau`` with
    ``E_i ~ Exp(1)`` (one ``rng`` vector per call), which is exactly successive sampling
    without replacement.
    """
    if tau_km <= 0:
        raise ValueError(f"tau_km must be > 0, got {tau_km}")
    distances = np.asarray(distances_km, dtype=float)
    if not 0 <= k <= len(distances):
        raise ValueError(f"cannot draw {k} of {len(distances)} items")
    keys = np.log(rng.standard_exponential(len(distances))) + distances / tau_km
    return np.argsort(keys, kind="stable")[:k]


class _CandidateSampler:
    """Precomputed units, per-unit merchant arrays and zone distances."""

    def __init__(self, sim: Simulation) -> None:
        world = sim.world
        merchants = world.merchants
        self.tau = float(world.config.food_candidate_tau_km)
        if self.tau <= 0:
            raise ValueError(f"food_candidate_tau_km must be > 0, got {self.tau}")
        self.ids = [m.merchant_id for m in merchants]
        self.position = {mid: i for i, mid in enumerate(self.ids)}
        self.m_zone = [m.zone_id for m in merchants]
        counts = train_order_counts(sim)
        self.units = popularity_units(counts)
        pairs = [d // DECILES_PER_UNIT for d in popularity_deciles(counts)]
        pairs_per_unit: dict[int, set[int]] = {}
        for unit, pair in zip(self.units, pairs):
            pairs_per_unit.setdefault(unit, set()).add(pair)
        self.last_resort_units = {u for u, ps in pairs_per_unit.items() if len(ps) > 1}
        # unit -> merchant positions (catalog order) and their zones
        self.members = {
            unit: np.array([i for i, u in enumerate(self.units) if u == unit], dtype=np.int64)
            for unit in sorted(set(self.units))
        }
        self.member_zones = {
            unit: np.array([self.m_zone[i] for i in members], dtype=np.int64)
            for unit, members in self.members.items()
        }
        zones = world.zones
        self.zone_dist = np.array([[haversine_km(a, b) for b in zones] for a in zones])
        self.users = {u.user_id: u for u in world.users}
        self.third = {u.user_id: third_anchor_zone(world, u) for u in world.users}

    def anchors(self, session: FoodSession) -> tuple[int, int, int]:
        """(positive's zone, home zone, work zone or mall nearest home)."""
        user = self.users[session.user_id]
        return (self.m_zone[self.position[session.merchant_id]], user.home_zone,
                self.third[user.user_id])

    def sample(self, session: FoodSession, rng: np.random.Generator) -> tuple[str, ...]:
        """One candidate set; ``rng`` order: cluster sizes (3 distinct anchors only), latent
        centre of the positive's cluster, each cluster's negatives in anchor order, shuffle."""
        pos = self.position[session.merchant_id]
        anchors = self.anchors(session)
        sizes = cluster_sizes(len(set(anchors)), rng)
        groups = zone_groups(*anchors, sizes)
        positive_zone = anchors[0]
        latent = int(decay_sample(rng, self.zone_dist[positive_zone], 1, self.tau)[0])
        members = self.members[self.units[pos]]
        member_zones = self.member_zones[self.units[pos]]
        available = members != pos
        picks: list[int] = []
        for anchor, quota in groups:
            centre = latent if anchor == positive_zone else anchor
            left = np.flatnonzero(available)
            if len(left) < quota:
                raise RuntimeError(f"{session.session_id}: popularity unit too small")
            chosen = left[decay_sample(rng, self.zone_dist[centre, member_zones[left]], quota,
                                       self.tau)]
            available[chosen] = False
            picks.extend(int(m) for m in members[chosen])
        picks.append(pos)
        if len(set(picks)) != N_CANDIDATES:
            raise RuntimeError(f"{session.session_id}: candidate set is not {N_CANDIDATES} unique")
        return tuple(self.ids[picks[int(i)]] for i in rng.permutation(N_CANDIDATES))


# ---- requests -----------------------------------------------------------------------------


def _limits(config) -> dict[str, int] | None:
    limits = config.food_requests
    if limits is None:
        return None
    if set(limits) != set(SPLITS):
        raise ValueError(f"food_requests keys {sorted(limits)} must be exactly {list(SPLITS)}")
    if any(int(v) != v or v < 0 for v in limits.values()):
        raise ValueError(f"food_requests counts must be non-negative integers: {limits}")
    return {s: int(limits[s]) for s in SPLITS}


def build_food_requests(sim: Simulation, rng: np.random.Generator) -> list[FoodRequest]:
    """Requests in chronological order (``sim:food_req:0..``), candidates per spec §5.

    ``rng`` draws, in order: the per-split samples (only when ``config.food_requests`` is
    set), then for each request the cluster-size permutation (3 distinct anchors only), the
    latent centre of the positive's cluster, each cluster's negatives (anchor order) and the
    candidate shuffle.
    """
    config = sim.world.config
    limits = _limits(config)
    by_split: dict[str, list[FoodSession]] = {s: [] for s in SPLITS}
    for session in sorted(sim.food_sessions, key=lambda s: s.start_ts):  # stable: user order
        by_split[split_of(session.start_ts - 1, config)].append(session)
    chosen: list[tuple[str, FoodSession]] = []
    for split in SPLITS:
        sessions = by_split[split]
        limit = None if limits is None else limits[split]
        if limit is not None and limit < len(sessions):
            picks = np.sort(rng.choice(len(sessions), size=limit, replace=False))
            sessions = [sessions[int(i)] for i in picks]
        chosen.extend((split, s) for s in sessions)
    sampler = _CandidateSampler(sim)
    return [
        FoodRequest(f"sim:food_req:{n}", session, session.start_ts - 1, split,
                    sampler.sample(session, rng), session.merchant_id)
        for n, (split, session) in enumerate(chosen)
    ]


def candidate_stats(sim: Simulation, requests: Sequence[FoodRequest]) -> dict:
    """Fallback rates and, for reporting only, how close negatives sit to the anchor zones.

    "Anchor" = the positive's zone, home zone and work zone (or mall nearest home). Negatives
    are drawn stochastically around cluster centres (R10), so these are averages, not rules.
    """
    sampler = _CandidateSampler(sim)
    last_resort = in_anchor_zone = near_anchor = negatives = 0
    n_anchors: Counter = Counter()
    for r in requests:
        pos = sampler.position[r.positive_id]
        last_resort += sampler.units[pos] in sampler.last_resort_units
        anchors = sorted(set(sampler.anchors(r.session)))
        n_anchors[len(anchors)] += 1
        for c in r.candidate_ids:
            if c == r.positive_id:
                continue
            zone = sampler.m_zone[sampler.position[c]]
            negatives += 1
            in_anchor_zone += zone in anchors
            near_anchor += int(min(float(sampler.zone_dist[a, zone]) for a in anchors)
                               <= ANCHOR_REPORT_KM)
    return {
        "popularity_unit": "decile_pair",
        "tau_km": sampler.tau,
        "last_resort_requests": last_resort,
        "requests_by_anchor_count": {str(k): n_anchors[k] for k in sorted(n_anchors)},
        "merged_anchor_requests": sum(v for k, v in n_anchors.items() if k < 3),
        "negatives": negatives,
        "negatives_in_an_anchor_zone": in_anchor_zone,
        f"negatives_within_{ANCHOR_REPORT_KM:g}km_of_an_anchor": near_anchor,
    }


# ---- export -------------------------------------------------------------------------------


def _attributes(event: Event) -> dict:
    """Public context + numeric action fields of an event (new dict; inputs untouched)."""
    allowed = CONTEXT_ALLOWLIST[SCENARIOS[event.service]]
    out = {k: v for k, v in event.context.items() if k in allowed}
    out.update((k, v) for k, v in event.action.items() if k in ACTION_ATTRIBUTES)
    return out


def _history_item(event: Event, config) -> dict:
    return {
        "event_id": event.event_id,
        "event_time": local_iso(event.ts, config),
        "service": event.service,
        "event_type": event.event_type,
        "entity_id": event.entity_id,
        "attributes": _attributes(event),
    }


class _UserEvents:
    """A user's events in chronological order plus their timestamps (for bisection)."""

    __slots__ = ("events", "times")

    def __init__(self, events: list[Event]) -> None:
        self.events = sorted(events, key=event_order_key)
        self.times = [e.ts for e in self.events]


def _sample_row(r: FoodRequest, user_events: _UserEvents, traits: UserTraits, world: World,
                candidate_payload: dict[str, dict]) -> dict:
    config = world.config
    if len(r.candidate_ids) != N_CANDIDATES or len(set(r.candidate_ids)) != N_CANDIDATES:
        raise ValueError(f"{r.request_id}: needs {N_CANDIDATES} distinct candidates")
    if r.positive_id not in r.candidate_ids or r.positive_id != r.session.merchant_id:
        raise ValueError(f"{r.request_id}: positive must be the ordered merchant, a candidate")
    cutoff = r.cutoff_ts
    events, times = user_events.events, user_events.times
    n = bisect_left(times, cutoff)
    prefix = events[:n]
    history, group_truncated = last_history(prefix, cutoff, config.max_history_events)
    truncated = len(completed_sessions_before(prefix, cutoff)) > len(history)
    lo = bisect_left(times, cutoff - PROFILE_WINDOW_DAYS * SECONDS_PER_DAY)
    cutoff_iso = local_iso(cutoff, config)
    profile = {"as_of_time": cutoff_iso}
    profile.update(profile_at(events[lo:n], traits, cutoff, config, zones=world.zones))
    return {
        "sample_id": r.request_id,
        "input": {
            "user_id": r.session.user_id,
            "cutoff_time": cutoff_iso,
            "profile": profile,
            "history": [_history_item(e, config) for e in history],
            "history_truncated": truncated,
            "history_group_truncated": group_truncated,
            "context": {"hour": local_hour(cutoff, config),
                        "weekday": local_weekday(cutoff, config)},
            "candidates": [candidate_payload[c] for c in r.candidate_ids],
        },
        "labels": [{"item_id": c, "relevance": int(c == r.positive_id)}
                   for c in r.candidate_ids],
        "label_metadata": {
            "request_id": r.request_id,
            "split": r.split,
            "label_time": local_iso(r.session.order_ts, config),
            "candidate_source": CANDIDATE_POLICY,
            "observation_complete": True,
            "prediction_anchor": PREDICTION_ANCHOR,
            "positive_action": POSITIVE_ACTION,
        },
    }


def _merchants_table(world: World) -> pa.Table:
    merchants, zones = world.merchants, world.zones
    columns = {
        "merchant_id": [m.merchant_id for m in merchants],
        "merchant_name": [m.name for m in merchants],
        "cuisine": [m.cuisine for m in merchants],
        "price_level": [m.price_level for m in merchants],
        "geohash": [zones[m.zone_id].geohash for m in merchants],
        "zone_type": [zones[m.zone_id].zone_type for m in merchants],
    }
    return pa.Table.from_arrays(
        [pa.array(columns[f.name], type=f.type) for f in MERCHANT_SCHEMA], schema=MERCHANT_SCHEMA)


def export_food(sim: Simulation, requests: Sequence[FoodRequest], out_dir) -> dict:
    """Write ``out_dir/food/`` (spec §6.2); returns ``{requests, requests_by_split, merchants}``.

    Raises ``FileExistsError`` when ``out_dir/food`` already has content.
    """
    world = sim.world
    config = world.config
    root = Path(out_dir) / "food"
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {root}")
    root.mkdir(parents=True, exist_ok=True)

    grouped: dict[str, list[Event]] = {u.user_id: [] for u in world.users}
    for event in sim.events:
        bucket = grouped.get(event.user_id)
        if bucket is None:
            raise ValueError(f"event {event.event_id} has unknown user {event.user_id!r}")
        bucket.append(event)
    user_events: dict[str, _UserEvents] = {}
    traits = {u.user_id: u for u in world.users}
    candidate_payload = {
        m.merchant_id: {"item_id": m.merchant_id, "cuisine": m.cuisine,
                        "price_level": m.price_level, "geohash": world.zones[m.zone_id].geohash}
        for m in world.merchants
    }

    by_split = dict.fromkeys(SPLITS, 0)
    digest = hashlib.sha256()
    size = 0
    with (root / "samples.jsonl").open("wb") as fh:
        for r in requests:
            uid = r.session.user_id
            if uid not in user_events:
                user_events[uid] = _UserEvents(grouped[uid])
            row = _sample_row(r, user_events[uid], traits[uid], world, candidate_payload)
            line = (_dumps(row) + "\n").encode("utf-8")
            fh.write(line)
            digest.update(line)
            size += len(line)
            by_split[r.split] += 1
    merchants_bytes, merchants_sha = _write_parquet(_merchants_table(world),
                                                    root / "merchants.parquet")

    manifest = {
        "dataset_version": config.dataset_version,
        "schema_version": SCHEMA_VERSION,
        "source_dataset": SOURCE_DATASET,
        "generator_version": GENERATOR_VERSION,
        "seed": config.seed,
        "sim_config_sha256": config_sha256(config),
        "pyarrow_version": pa.__version__,
        "timezone": TIMEZONE,
        "utc_offset": f"+{config.utc_offset_hours:02d}:00",
        "item_kind": "merchant",
        "files": {
            "samples": {"path": "samples.jsonl", "sha256": digest.hexdigest(),
                        "row_count": len(requests), "bytes": size},
            "merchants": {"path": "merchants.parquet", "sha256": merchants_sha,
                          "row_count": len(world.merchants), "bytes": merchants_bytes},
        },
        "split": {
            "kind": "temporal",
            "basis": "cutoff_time",
            "train_end": local_iso(day_start_ts(pretrain_end_day(config), config), config),
            "valid_end": local_iso(day_start_ts(pretrain_end_day(config) + 7, config), config),
        },
        "request_counts": by_split,
        "candidate_policy": CANDIDATE_POLICY,
        "candidates_per_request": N_CANDIDATES,
        "candidate_stats": candidate_stats(sim, requests),
        "popularity_end_exclusive": local_iso(
            day_start_ts(pretrain_end_day(config), config), config),
        "max_history_events": config.max_history_events,
        "prediction_anchor": PREDICTION_ANCHOR,
        "positive_action": POSITIVE_ACTION,
    }
    _write_json(manifest, root / "manifest.json")
    return {"requests": len(requests), "requests_by_split": dict(by_split),
            "merchants": len(world.merchants)}
