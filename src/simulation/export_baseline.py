"""Baseline packages ``baseline/food.json`` and ``baseline/ads.json`` (spec §6.4).

Format of ``src.baselines.runner.load_dataset``: ``metadata``, ``queries``, ``candidates``,
``labels`` (with ``available_at``) and ``history``. Written deterministically (sorted keys, UTF-8,
no NaN) and streamed item by item, so no whole document is ever held as one string.

B1 conventions (``docs/b1_features.md``, ``src.baselines.features.FeatureBuilder``; ruling R12):

- History = every event strictly before the latest request cutoff of each user that has a
  request in that file; the runner's ``eligible_history`` cuts it per query (``event_time`` and
  ``available_at`` both before the cutoff). One row per simulated event, same rows in both files.
- An ad impression is written as ``event_type="impression"`` with ``attributes.clicked`` 0/1
  (plus ``category`` / ``brand`` = the ad's category / advertiser) and ``available_at`` = the
  moment its outcome is observable: the click time when clicked, else exposure +
  ``AD_OUTCOME_MATURITY_SECONDS`` (27 s, one second after the latest possible click). The
  ``ad_click`` row stays an ordinary event. Every other row has ``available_at = event_time``;
  ``order`` rows add ``category`` / ``price`` (= cuisine / price_level).
- A user's ad exposures are >= ``SESSION_GAP_SECONDS`` (60 s) apart (asserted on export), so
  every earlier impression of a user is mature at the cutoff of their next Ads request.
- Candidates: Food ``cuisine, price_level, geohash`` + ``category``, ``price``; Ads ``category,
  advertiser, title`` + ``brand`` (no price).
- Labels: Food ``available_at`` = order time (``label_time``); Ads = the impression's outcome
  maturity (as in the history).

Hidden state and true probabilities are never written.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Sequence
from pathlib import Path

from .behavior import AD_CLICK_DELAY_SECONDS, SESSION_GAP_SECONDS, AdImpression, Event, Simulation
from .config import GENERATOR_VERSION, config_sha256
from .export_ads import CANDIDATE_POLICY as ADS_CANDIDATE_POLICY
from .export_ads import AdRequest
from .export_food import CANDIDATE_POLICY as FOOD_CANDIDATE_POLICY
from .export_food import TIMEZONE, FoodRequest, _attributes
from .timeutil import local_hour, local_iso, local_weekday

DATASET_NAME = "sim_v1"
AVAILABILITY_POLICY = "handoff_timestamp"
# B1 (docs/b1_features.md): an Ads history row is ``event_type="impression"``.
IMPRESSION_EVENT_TYPE = "impression"
# An impression without a click is observable as such one second after the latest click time.
AD_OUTCOME_MATURITY_SECONDS = AD_CLICK_DELAY_SECONDS[1] + 1


def _dump(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False)


def _write_document(path: Path, sections: dict) -> tuple[int, str]:
    """Write ``{key: value-or-iterator-of-items}`` with sorted keys; returns (bytes, sha256)."""
    digest = hashlib.sha256()
    size = 0

    def put(fh, text: str) -> None:
        nonlocal size
        data = text.encode("utf-8")
        digest.update(data)
        size += len(data)
        fh.write(data)

    with path.open("wb") as fh:  # binary: no newline translation, hash matches on Windows
        put(fh, "{")
        for i, key in enumerate(sorted(sections)):
            put(fh, ("," if i else "") + _dump(key) + ":")
            value = sections[key]
            if isinstance(value, Iterator):
                put(fh, "[")
                for j, item in enumerate(value):
                    put(fh, ("," if j else "") + _dump(item))
                put(fh, "]")
            else:
                put(fh, _dump(value))
        put(fh, "}\n")
    return size, digest.hexdigest()


def _metadata(config, task: str, item_kind: str, candidate_policy: str) -> dict:
    return {
        "dataset_name": DATASET_NAME,
        "data_version": f"{GENERATOR_VERSION}+{config_sha256(config)[:12]}",
        "task": task,
        "item_kind": item_kind,
        "timezone": TIMEZONE,
        "candidate_policy": candidate_policy,
        "availability_policy": AVAILABILITY_POLICY,
    }


def _click_times(sim: Simulation) -> dict[str, int]:
    """Session id -> click time, for every clicked ad session."""
    return {e.session_id: e.ts for e in sim.events if e.event_type == "ad_click"}


def _outcome(impression: Event, clicks: dict[str, int]) -> tuple[int, int]:
    """(clicked 0/1, epoch seconds at which the impression's outcome becomes observable)."""
    click = clicks.get(impression.session_id)
    if click is None:
        return 0, impression.ts + AD_OUTCOME_MATURITY_SECONDS
    return 1, click


def _impression_event(sim: Simulation, impression: AdImpression) -> Event:
    """The ``ad_impression`` event of an impression (``sim:event:<n>`` is ``sim.events[n]``)."""
    event = sim.events[int(impression.event_id.rsplit(":", 1)[1])]
    if event.event_id != impression.event_id or event.event_type != "ad_impression":
        raise RuntimeError(f"impression {impression.event_id} is not sim.events[n]")
    return event


def _check_exposure_gaps(sim: Simulation) -> None:
    """Raise ``ValueError`` unless each user's consecutive ad exposures are at least
    ``SESSION_GAP_SECONDS`` apart; outcomes mature within ``AD_OUTCOME_MATURITY_SECONDS`` (<
    the gap), so no earlier impression of a user is pending at their next exposure."""
    previous: dict[str, int] = {}
    for imp in sim.ad_impressions:
        last = previous.get(imp.user_id)
        if last is not None and imp.ts - last < SESSION_GAP_SECONDS:
            raise ValueError(f"{imp.user_id}: ad exposure {imp.event_id} is {imp.ts - last} s "
                             f"after the previous one (< {SESSION_GAP_SECONDS} s)")
        previous[imp.user_id] = imp.ts


def _history(sim: Simulation, latest_cutoff: dict[str, int],
             clicks: dict[str, int]) -> Iterator[dict]:
    config = sim.world.config
    for event in sim.events:
        cutoff = latest_cutoff.get(event.user_id)
        if cutoff is None or event.ts >= cutoff:
            continue
        when = available = local_iso(event.ts, config)
        event_type = event.event_type
        attributes = _attributes(event)
        attributes["service"] = event.service
        if event_type == "ad_impression":
            clicked, mature = _outcome(event, clicks)
            event_type, available = IMPRESSION_EVENT_TYPE, local_iso(mature, config)
            attributes.update(clicked=clicked, category=attributes["industry_main"],
                              brand=attributes["industry_sub"])
        elif event_type == "order":
            attributes.update(category=attributes["cuisine"], price=attributes["price_level"])
        yield {
            "event_id": event.event_id,
            "user_id": event.user_id,
            "entity_id": event.entity_id,
            "event_type": event_type,
            "event_time": when,
            "available_at": available,
            "attributes": attributes,
        }


def _context(ts: int, config) -> dict:
    return {"hour": local_hour(ts, config), "weekday": local_weekday(ts, config)}


def _latest(pairs) -> dict[str, int]:
    latest: dict[str, int] = {}
    for user_id, ts in pairs:
        latest[user_id] = max(ts, latest.get(user_id, ts))
    return latest


def _food_document(sim: Simulation, requests: Sequence[FoodRequest],
                   clicks: dict[str, int]) -> dict:
    world = sim.world
    config = world.config
    merchants = {  # category / price: the B1 candidate attribute names
        m.merchant_id: {"cuisine": m.cuisine, "price_level": m.price_level,
                        "geohash": world.zones[m.zone_id].geohash,
                        "category": m.cuisine, "price": m.price_level}
        for m in world.merchants
    }

    def queries():
        for r in requests:
            yield {"request_id": r.request_id, "user_id": r.session.user_id,
                   "cutoff_time": local_iso(r.cutoff_ts, config), "split": r.split,
                   "context": _context(r.cutoff_ts, config)}

    def candidates():
        for r in requests:
            for c in r.candidate_ids:
                yield {"request_id": r.request_id, "candidate_id": c, "attributes": merchants[c]}

    def labels():
        for r in requests:
            for c in r.candidate_ids:
                yield {"request_id": r.request_id, "candidate_id": c,
                       "target": int(c == r.positive_id),
                       "available_at": local_iso(r.session.order_ts, config)}

    latest = _latest((r.session.user_id, r.cutoff_ts) for r in requests)
    return {
        "metadata": _metadata(config, "food_recommendation", "merchant", FOOD_CANDIDATE_POLICY),
        "queries": queries(), "candidates": candidates(), "labels": labels(),
        "history": _history(sim, latest, clicks),
    }


def _ads_document(sim: Simulation, requests: Sequence[AdRequest],
                  clicks: dict[str, int]) -> dict:
    world = sim.world
    config = world.config
    ads = {a.ad_id: {"category": a.category, "advertiser": a.advertiser, "title": a.title,
                     "brand": a.advertiser}  # brand: the B1 candidate attribute name
           for a in world.ads}

    def queries():
        for r in requests:
            yield {"request_id": r.request_id, "user_id": r.impression.user_id,
                   "cutoff_time": local_iso(r.impression.ts, config), "split": r.split,
                   "context": _context(r.impression.ts, config)}

    def candidates():
        for r in requests:
            yield {"request_id": r.request_id, "candidate_id": r.impression.ad_id,
                   "attributes": ads[r.impression.ad_id]}

    def labels():
        for r in requests:
            clicked, mature = _outcome(_impression_event(sim, r.impression), clicks)
            if clicked != int(r.impression.clicked):
                raise RuntimeError(f"{r.request_id}: click label disagrees with the event log")
            yield {"request_id": r.request_id, "candidate_id": r.impression.ad_id,
                   "target": clicked, "available_at": local_iso(mature, config)}

    latest = _latest((r.impression.user_id, r.impression.ts) for r in requests)
    return {
        "metadata": _metadata(config, "ad_optimization", "ad", ADS_CANDIDATE_POLICY),
        "queries": queries(), "candidates": candidates(), "labels": labels(),
        "history": _history(sim, latest, clicks),
    }


def export_baseline(sim: Simulation, food_requests: Sequence[FoodRequest],
                    ad_requests: Sequence[AdRequest], out_dir) -> dict:
    """Write ``out_dir/baseline/{food,ads}.json``; returns per-file ``{requests, bytes, sha256}``.

    Raises ``FileExistsError`` when ``out_dir/baseline`` already has content and
    ``ValueError`` (before writing anything) when two ad exposures of a user are closer than
    ``SESSION_GAP_SECONDS``.
    """
    root = Path(out_dir) / "baseline"
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {root}")
    _check_exposure_gaps(sim)
    root.mkdir(parents=True, exist_ok=True)
    clicks = _click_times(sim)
    result = {}
    for name, document, count in (
        ("food", _food_document(sim, food_requests, clicks), len(food_requests)),
        ("ads", _ads_document(sim, ad_requests, clicks), len(ad_requests)),
    ):
        size, sha = _write_document(root / f"{name}.json", document)
        result[name] = {"path": f"baseline/{name}.json", "requests": count, "bytes": size,
                        "sha256": sha}
    return result
