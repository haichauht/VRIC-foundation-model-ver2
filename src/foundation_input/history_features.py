"""Long-history features: a fixed-size summary of every event a user had before a cutoff.

The ≤30-event window that the sequence encoder reads misses most of a user's history (on sim-v1,
76% of Ads test requests have no ad click inside it, against 2.66 clicks on average over the whole
history). These features carry that history into the model as the input of the long-term encoder
(``FoundationModelConfig.long_term_source == "history"``), replacing the per-ID user table.

Features (all in [0, 1], computed from the events before the cutoff only):
  log-counts per service and per key event type, order share per cuisine and per price level,
  ad click and impression share per ad category, smoothed click rate per category, and days since
  the last order / ride / ad click.
Targets (pretraining only): the user's order-cuisine and ad-click-category distributions over the
``target_days`` after the cutoff, each with a 0/1 mask when that window has no such event.

The category lists are the ``context.cuisine`` / ``context.industry_main`` values seen in the
pretraining train split; they are frozen in the checkpoint and reused downstream.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np

HISTORY_FEATURE_VERSION = 1
SERVICES = ("ride", "food", "ads")
SCENARIO_SERVICE = {"Ride": "ride", "Food": "food", "Advertisement": "ads",
                    "ride": "ride", "food": "food", "ads": "ads"}
COUNTED_TYPES = ("complete_ride", "order", "view_merchant", "ad_impression", "ad_click")
RECENCY_TYPES = ("order", "complete_ride", "ad_click")
PRICE_LEVELS = 3
CUISINE_KEY, CATEGORY_KEY, PRICE_KEY = "cuisine", "industry_main", "price_level"
LOG_COUNT_SCALE = 8.0  # log1p(count) / 8 stays below 1 up to ~3,000 events
RECENCY_CAP_DAYS = 60.0
CTR_PRIOR_CLICKS, CTR_PRIOR_IMPRESSIONS = 0.06, 1.0
SECONDS_PER_DAY = 86_400


@dataclass(frozen=True)
class HistoryEvent:
    """What the summary needs from one event: time, service, type and its context values."""

    timestamp: int  # epoch seconds
    service: str | None
    event_type: str | None
    context: Mapping[str, Any]


@dataclass(frozen=True)
class HistorySchema:
    cuisines: tuple[str, ...]
    categories: tuple[str, ...]
    target_days: int = 7
    version: int = HISTORY_FEATURE_VERSION

    @property
    def names(self) -> tuple[str, ...]:
        return (tuple(f"log_count.service.{s}" for s in SERVICES)
                + tuple(f"log_count.type.{t}" for t in COUNTED_TYPES)
                + tuple(f"order_share.cuisine.{c}" for c in self.cuisines)
                + tuple(f"order_share.price.{p}" for p in range(PRICE_LEVELS))
                + tuple(f"click_share.category.{c}" for c in self.categories)
                + tuple(f"impression_share.category.{c}" for c in self.categories)
                + tuple(f"click_rate.category.{c}" for c in self.categories)
                + tuple(f"recency.{t}" for t in RECENCY_TYPES))

    @property
    def size(self) -> int:
        return len(self.names)

    @property
    def target_sizes(self) -> tuple[int, int]:
        return (len(self.cuisines), len(self.categories))

    def to_dict(self) -> dict:
        return {"version": self.version, "cuisines": list(self.cuisines),
                "categories": list(self.categories), "target_days": self.target_days,
                "size": self.size}

    @classmethod
    def from_dict(cls, payload: Mapping) -> HistorySchema:
        if payload.get("version") != HISTORY_FEATURE_VERSION:
            raise ValueError(f"unsupported history feature version {payload.get('version')!r}")
        schema = cls(tuple(payload["cuisines"]), tuple(payload["categories"]),
                     int(payload["target_days"]))
        if payload.get("size") != schema.size:
            raise ValueError("history feature schema size mismatch")
        return schema


class HistorySchemaCollector:
    """Collects the cuisine and ad-category values of the events of the samples streamed through
    ``watch`` (the train split), so the schema is fitted in the same pass as the vocabulary.
    These context fields are hashed text tokens, so the vocabulary does not list their values."""

    def __init__(self) -> None:
        self.cuisines: set[str] = set()
        self.categories: set[str] = set()

    def watch(self, samples: Iterable) -> Iterable:
        for sample in samples:
            for event in sample.events:
                if event.context.get(CUISINE_KEY) is not None:
                    self.cuisines.add(str(event.context[CUISINE_KEY]))
                if event.context.get(CATEGORY_KEY) is not None:
                    self.categories.add(str(event.context[CATEGORY_KEY]))
            yield sample

    def schema(self, *, target_days: int = 7) -> HistorySchema:
        if type(target_days) is not int or target_days < 1:
            raise ValueError("history target_days must be a positive integer")
        return HistorySchema(tuple(sorted(self.cuisines)), tuple(sorted(self.categories)), target_days)


def history_schema_from_samples(samples: Iterable, *, target_days: int = 7) -> HistorySchema:
    collector = HistorySchemaCollector()
    for _ in collector.watch(samples):
        pass
    return collector.schema(target_days=target_days)


def epoch_seconds(moment: datetime) -> int:
    """Epoch seconds; a naive datetime is read as UTC so the result never depends on the machine's
    time zone (pretraining packages store naive local times; only differences matter there)."""
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return int(moment.timestamp())


class UserHistory:
    """One user's events in order, as compact arrays; summaries over any prefix or time window."""

    def __init__(self, events: Iterable[HistoryEvent], schema: HistorySchema) -> None:
        events = list(events)
        cuisine = {c: i for i, c in enumerate(schema.cuisines)}
        category = {c: i for i, c in enumerate(schema.categories)}
        self.schema = schema
        self.timestamps = np.array([e.timestamp for e in events], dtype=np.int64)
        if len(self.timestamps) > 1 and (np.diff(self.timestamps) < 0).any():
            raise ValueError("history events must be in time order")
        self.service = np.array([SERVICES.index(SCENARIO_SERVICE[e.service])
                                 if e.service in SCENARIO_SERVICE else -1 for e in events], dtype=np.int8)
        self.kind = np.array([COUNTED_TYPES.index(e.event_type) if e.event_type in COUNTED_TYPES else -1
                              for e in events], dtype=np.int8)
        self.cuisine = np.array([cuisine.get(str(e.context.get(CUISINE_KEY)), -1) for e in events],
                                dtype=np.int16)
        self.category = np.array([category.get(str(e.context.get(CATEGORY_KEY)), -1) for e in events],
                                 dtype=np.int16)
        self.price = np.array([_price(e.context.get(PRICE_KEY)) for e in events], dtype=np.int8)

    def stop_before(self, cutoff: int, *, inclusive: bool = False) -> int:
        """Prefix length holding the events at or before (``inclusive``) / strictly before ``cutoff``."""
        return int(np.searchsorted(self.timestamps, cutoff, side="right" if inclusive else "left"))

    def features(self, stop: int, as_of: int) -> tuple[float, ...]:
        """Summary of the first ``stop`` events, with recency measured at ``as_of`` (epoch seconds)."""
        if not 0 <= stop <= len(self.timestamps):
            raise ValueError("history prefix outside the user's events")
        s = self.schema
        service = np.bincount(self.service[:stop][self.service[:stop] >= 0], minlength=len(SERVICES))
        kind_prefix = self.kind[:stop]
        kind = np.bincount(kind_prefix[kind_prefix >= 0], minlength=len(COUNTED_TYPES))
        orders = kind_prefix == COUNTED_TYPES.index("order")
        clicks = kind_prefix == COUNTED_TYPES.index("ad_click")
        impressions = kind_prefix == COUNTED_TYPES.index("ad_impression")
        by_cuisine = _counts(self.cuisine[:stop][orders], len(s.cuisines))
        by_price = _counts(self.price[:stop][orders], PRICE_LEVELS)
        click_cat = _counts(self.category[:stop][clicks], len(s.categories))
        impression_cat = _counts(self.category[:stop][impressions], len(s.categories))
        recency = []
        for name in RECENCY_TYPES:
            hits = np.flatnonzero(kind_prefix == COUNTED_TYPES.index(name))
            if len(hits) == 0:
                recency.append(1.0)
            else:
                days = max(0.0, (as_of - int(self.timestamps[hits[-1]])) / SECONDS_PER_DAY)
                recency.append(min(RECENCY_CAP_DAYS, days) / RECENCY_CAP_DAYS)
        values = np.concatenate([
            np.log1p(service) / LOG_COUNT_SCALE, np.log1p(kind) / LOG_COUNT_SCALE,
            _share(by_cuisine), _share(by_price), _share(click_cat), _share(impression_cat),
            (click_cat + CTR_PRIOR_CLICKS) / (impression_cat + CTR_PRIOR_IMPRESSIONS),
            np.array(recency)])
        return tuple(float(min(1.0, value)) for value in values)

    def targets(self, start: int, until: int) -> tuple[tuple[float, ...], tuple[float, float]]:
        """Order-cuisine and click-category distributions of the events from index ``start`` with
        timestamp < ``until``; mask = (any order, any click)."""
        stop = int(np.searchsorted(self.timestamps, until, side="left"))
        stop = max(start, stop)
        kind = self.kind[start:stop]
        orders = _counts(self.cuisine[start:stop][kind == COUNTED_TYPES.index("order")],
                         len(self.schema.cuisines))
        clicks = _counts(self.category[start:stop][kind == COUNTED_TYPES.index("ad_click")],
                         len(self.schema.categories))
        mask = (float(orders.sum() > 0), float(clicks.sum() > 0))
        return tuple(float(v) for v in np.concatenate([_share(orders), _share(clicks)])), mask


def _price(value) -> int:
    try:
        level = int(value)
    except (TypeError, ValueError):
        return -1
    return min(max(level, 0), PRICE_LEVELS - 1)


def _counts(indices: np.ndarray, size: int) -> np.ndarray:
    indices = indices[indices >= 0]
    return np.bincount(indices, minlength=size)[:size].astype(float) if size else np.zeros(0)


def _share(counts: np.ndarray) -> np.ndarray:
    total = counts.sum()
    return counts / total if total > 0 else np.zeros_like(counts, dtype=float)


def history_event_from_record(event) -> HistoryEvent:
    """From a loader ``EventRecord`` (pretraining package)."""
    moment = event.event_time or event.eligibility_time
    return HistoryEvent(epoch_seconds(moment), event.scenario, event.event_type, event.context)


def history_event_from_row(row: Mapping[str, Any]) -> HistoryEvent:
    """From an ``events.parquet`` row of the simulator Ads package (all services)."""
    attributes = row.get("attributes_json")
    context = json.loads(attributes) if isinstance(attributes, str) and attributes else {}
    return HistoryEvent(int(row["timestamp"]), row.get("service"), row.get("event_type"), context)


def check_features(values, size: int) -> tuple[float, ...]:
    values = tuple(float(v) for v in values)
    if len(values) != size or not all(math.isfinite(v) for v in values):
        raise ValueError(f"history features must be {size} finite values")
    return values
