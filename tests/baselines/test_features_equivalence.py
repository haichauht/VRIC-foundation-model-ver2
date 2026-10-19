"""The fast FeatureBuilder returns exactly the rows of the original (quadratic) implementation.

The original algorithm is kept here verbatim as the oracle: for every candidate it rescanned the
whole history and cached one filtered copy per cutoff, which ran out of 30 GB on the full sim-v1
Food Baseline and needed > 11 h for Ads.
"""

from dataclasses import replace
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from src.baselines.contracts import BaselineDataset, ContractError, eligible_history
from src.baselines.features import FeatureBuilder, FeatureRow
from src.baselines.runner import load_dataset
from src.simulation.behavior import simulate
from src.simulation.config import PRESETS
from src.simulation.export_ads import build_ad_requests
from src.simulation.export_baseline import export_baseline
from src.simulation.export_food import build_food_requests
from src.simulation.world import build_world

VALIDATION_START = datetime.fromisoformat("2026-02-16T00:00:00+07:00")
DAY_ONE = datetime.fromisoformat("2026-01-05T00:00:00+07:00")


def _reference_build(dataset: BaselineDataset, split: str, *, frozen_cutoff=None,
                     history_start=None) -> tuple[FeatureRow, ...]:
    """The pre-optimisation FeatureBuilder.build, unchanged."""
    dataset.validate()
    if split not in {"train", "validation", "test"}:
        raise ContractError("invalid feature split")
    if split == "train" and frozen_cutoff is not None:
        raise ContractError("training features require per-query prefix history")
    if frozen_cutoff is not None:
        eligible_history((), frozen_cutoff)
    if history_start is not None:
        eligible_history((), history_start)
    queries = {q.request_id: q for q in dataset.queries if q.split == split}
    cache = {}
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
        if cutoff not in cache:
            cache[cutoff] = eligible_history(dataset.history, cutoff)
        events = cache[cutoff]
        if dataset.metadata.task == "food_recommendation":
            events = tuple(e for e in events if e.event_type == "order")
        else:
            events = tuple(e for e in events if e.event_type == "impression")
            for event in events:
                click = event.attributes.get("clicked")
                if type(click) is not int or click not in {0, 1}:
                    raise ContractError(f"event={event.event_id}: mature impression needs clicked 0/1")
        user = tuple(e for e in events if e.user_id == query.user_id)
        item = tuple(e for e in events if e.entity_id == candidate.candidate_id)
        pair = tuple(e for e in user if e.entity_id == candidate.candidate_id)
        values: dict[str, object] = {}
        for days in (7, 30):
            start = cutoff - timedelta(days=days)
            values[f"history_complete_{days}d"] = int(
                history_start is not None and history_start <= start)
            for name, group in (("user", user), ("item", item), ("user_item", pair)):
                window = tuple(e for e in group if e.event_time >= start)
                values[f"{name}_count_{days}d"] = len(window)
                if dataset.metadata.task == "ad_optimization":
                    clicks = sum(e.attributes["clicked"] for e in window)
                    values[f"{name}_clicks_{days}d"] = clicks
                    values[f"{name}_ctr_{days}d"] = clicks / len(window) if window else None
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


def _subset(dataset: BaselineDataset, per_split: int) -> BaselineDataset:
    """Evenly spaced requests per split (all candidates, labels and the full history)."""
    keep = set()
    for split in ("train", "validation", "test"):
        ids = [q.request_id for q in dataset.queries if q.split == split]
        keep.update(ids[:: max(1, len(ids) // per_split)][:per_split])
    return replace(dataset,
                   queries=tuple(q for q in dataset.queries if q.request_id in keep),
                   candidates=tuple(c for c in dataset.candidates if c.request_id in keep),
                   labels=tuple(label for label in dataset.labels if label.request_id in keep))


@pytest.fixture(scope="module")
def datasets(tmp_path_factory):
    config = replace(PRESETS["smoke"], n_users=60, n_merchants=24, n_ads=40, n_drivers=40)
    rng = np.random.default_rng(config.seed)
    sim = simulate(build_world(config, rng), rng)
    food, ads = build_food_requests(sim, rng), build_ad_requests(sim, rng)
    root = tmp_path_factory.mktemp("baseline")
    export_baseline(sim, food, ads, root)
    return {name: _subset(load_dataset(root / "baseline" / f"{name}.json"), 40)
            for name in ("food", "ads")}


@pytest.mark.parametrize("task", ["food", "ads"])
@pytest.mark.parametrize("split,frozen", [("train", None), ("validation", VALIDATION_START),
                                          ("test", VALIDATION_START)])
@pytest.mark.parametrize("history_start", [None, DAY_ONE])
def test_fast_builder_matches_reference_exactly(datasets, task, split, frozen, history_start):
    dataset = datasets[task]
    expected = _reference_build(dataset, split, frozen_cutoff=frozen, history_start=history_start)
    actual = FeatureBuilder().build(dataset, split, frozen_cutoff=frozen,
                                    history_start=history_start)
    assert len(actual) == len(expected) > 0
    for got, want in zip(actual, expected):
        assert (got.request_id, got.candidate_id, got.split) == (
            want.request_id, want.candidate_id, want.split)
        assert list(got.values) == list(want.values)  # same keys, same order
        assert got.values == want.values


@pytest.mark.parametrize("bad", ["1", None, 2, 1.0])
def test_invalid_click_raises_contract_error_like_the_reference(datasets, bad):
    dataset = datasets["ads"]
    first = next(i for i, e in enumerate(dataset.history) if e.event_type == "impression")
    broken = replace(dataset.history[first],
                     attributes={**dataset.history[first].attributes, "clicked": bad})
    dataset = replace(dataset, history=(*dataset.history[:first], broken,
                                        *dataset.history[first + 1:]))
    for build in (_reference_build, FeatureBuilder().build):
        with pytest.raises(ContractError, match="mature impression"):
            build(dataset, "validation", frozen_cutoff=VALIDATION_START)


def test_ads_reference_covers_delayed_label_availability(datasets):
    """The Ads history has impressions whose outcome matures after the impression."""
    history = datasets["ads"].history
    assert any(e.event_type == "impression" and e.available_at > e.event_time for e in history)
