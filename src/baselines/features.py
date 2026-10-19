"""Point-in-time aggregate features with explicit history coverage."""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np

from .contracts import BaselineDataset, ContractError, Interaction, eligible_history

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def _micros(value: datetime) -> int:
    return (value - _EPOCH) // timedelta(microseconds=1)


def _valid_click(event) -> bool:
    click = event.attributes.get("clicked")
    return type(click) is int and click in {0, 1}


@dataclass(frozen=True)
class FeatureRow:
    request_id: str
    candidate_id: str
    split: str
    values: dict[str, object]


@dataclass(frozen=True)
class _ItemEvents:
    """One item's typed events as arrays (any order); only window counts are read."""

    available_at: np.ndarray
    event_time: np.ndarray
    clicked: np.ndarray


class FeatureBuilder:
    """No labels are read; Ads history contains mature labeled impressions.

    Set frozen_cutoff to reuse the same historical state throughout a split.
    history_start is a caller assertion of complete log coverage, not min(event).

    An event is visible at ``cutoff`` when ``event_time < cutoff`` and ``available_at < cutoff``;
    ``BaselineDataset.validate`` guarantees ``available_at >= event_time``, so that is
    ``available_at < cutoff``. The history is indexed once per call (by user, and by item as
    arrays), so the cost is linear in candidates + history rather than their product.
    """

    def build(self, dataset: BaselineDataset, split: str, *,
              frozen_cutoff: datetime | None = None,
              history_start: datetime | None = None) -> tuple[FeatureRow, ...]:
        dataset.validate()
        if split not in {"train", "validation", "test"}:
            raise ContractError("invalid feature split")
        if split == "train" and frozen_cutoff is not None:
            raise ContractError("training features require per-query prefix history")
        if frozen_cutoff is not None:
            eligible_history((), frozen_cutoff)
        if history_start is not None:
            eligible_history((), history_start)
        ads = dataset.metadata.task == "ad_optimization"
        kind = "impression" if ads else "order"
        typed = [e for e in dataset.history if e.event_type == kind]
        invalid = [e for e in typed if ads and not _valid_click(e)]
        by_user: dict[str, list[tuple[int, Interaction]]] = {}
        item_lists: dict[str, tuple[list[int], list[int], list[int]]] = {}
        for event in typed:
            available = _micros(event.available_at)
            by_user.setdefault(event.user_id, []).append((available, event))
            columns = item_lists.setdefault(event.entity_id, ([], [], []))
            columns[0].append(available)
            columns[1].append(_micros(event.event_time))
            # An invalid click raises once its event becomes visible (below), as before.
            columns[2].append(event.attributes["clicked"] if ads and _valid_click(event) else 0)
        by_item = {item: _ItemEvents(*(np.asarray(column, dtype=np.int64) for column in columns))
                   for item, columns in item_lists.items()}
        empty = _ItemEvents(*(np.zeros(0, dtype=np.int64) for _ in range(3)))
        queries = {q.request_id: q for q in dataset.queries if q.split == split}
        visible_key, visible_user = None, ()
        rows = []
        for candidate in dataset.candidates:
            query = queries.get(candidate.request_id)
            if query is None:
                continue
            cutoff = frozen_cutoff if frozen_cutoff is not None else query.cutoff_time
            if cutoff > query.cutoff_time:
                raise ContractError(f"request={query.request_id}: frozen cutoff is in future")
            if history_start is not None and history_start > cutoff:
                raise ContractError("history coverage starts after cutoff")
            for event in invalid:
                if event.available_at < cutoff and event.event_time < cutoff:
                    raise ContractError(f"event={event.event_id}: mature impression needs clicked 0/1")
            cutoff_us = _micros(cutoff)
            if visible_key != (query.user_id, cutoff_us):
                visible_key = (query.user_id, cutoff_us)
                visible_user = tuple(e for available, e in by_user.get(query.user_id, ())
                                     if available < cutoff_us)
            user = visible_user
            item = by_item.get(candidate.candidate_id, empty)
            item_visible = item.available_at < cutoff_us
            pair = tuple(e for e in user if e.entity_id == candidate.candidate_id)
            values: dict[str, object] = {}
            for days in (7, 30):
                start = cutoff - timedelta(days=days)
                values[f"history_complete_{days}d"] = int(
                    history_start is not None and history_start <= start)
                item_window = item_visible & (item.event_time >= _micros(start))
                for name, group in (("user", user), ("item", None), ("user_item", pair)):
                    if group is None:
                        count = int(np.count_nonzero(item_window))
                        clicks = int(item.clicked[item_window].sum()) if ads else 0
                    else:
                        window = tuple(e for e in group if e.event_time >= start)
                        count = len(window)
                        clicks = sum(e.attributes["clicked"] for e in window) if ads else 0
                    values[f"{name}_count_{days}d"] = count
                    if ads:
                        values[f"{name}_clicks_{days}d"] = clicks
                        values[f"{name}_ctr_{days}d"] = clicks / count if count else None
            for name, group in (("user", user), ("user_item", pair)):
                values[f"{name}_recency_seconds"] = (
                    (query.cutoff_time - max(e.event_time for e in group)).total_seconds()
                    if group else None)
            values["user_repeat_ratio"] = (
                1 - len({e.entity_id for e in user}) / len(user) if user else None)
            local = query.cutoff_time.astimezone(ZoneInfo(dataset.metadata.timezone))
            values.update(hour=local.hour, weekday=local.weekday())
            for key in ("meal_period", "placement", "zone"):
                values[f"context_{key}"] = query.context.get(key)
            # Only declared business attributes; labels, IDs, audit and arbitrary payload excluded.
            for key in ("category", "brand", "price"):
                values[f"item_{key}"] = candidate.attributes.get(key)
            category = candidate.attributes.get("category")
            values["user_category_affinity"] = (
                sum(e.attributes.get("category") == category for e in user) / len(user)
                if user and category is not None else None)
            rows.append(FeatureRow(query.request_id, candidate.candidate_id, split, values))
        if not rows:
            raise ContractError("feature split has no candidates")
        return tuple(rows)
