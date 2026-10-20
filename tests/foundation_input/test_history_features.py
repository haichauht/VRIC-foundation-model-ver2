"""Long-history summary: values, cutoff discipline, targets, schema."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from src.foundation_input.history_features import (
    HistoryEvent,
    HistorySchema,
    HistorySchemaCollector,
    UserHistory,
    history_schema_from_samples,
)

DAY = 86_400
SCHEMA = HistorySchema(cuisines=("bún", "phở"), categories=("du lịch", "tài chính"), target_days=7)


def event(day, service, kind, **context):
    return HistoryEvent(day * DAY, service, kind, context)


EVENTS = [
    event(1, "ride", "complete_ride"),
    event(2, "food", "order", cuisine="phở", price_level=1),
    event(3, "ads", "ad_impression", industry_main="du lịch"),
    event(3, "ads", "ad_click", industry_main="du lịch"),
    event(4, "food", "order", cuisine="bún", price_level=2),
    event(5, "food", "order", cuisine="phở", price_level=1),
    event(9, "ads", "ad_click", industry_main="tài chính"),
    event(10, "food", "order", cuisine="bún", price_level=0),
]


def values(history, stop, as_of):
    return dict(zip(SCHEMA.names, history.features(stop, as_of), strict=True))


def test_features_summarise_only_the_prefix():
    history = UserHistory(EVENTS, SCHEMA)
    stop = history.stop_before(6 * DAY)
    assert stop == 6
    got = values(history, stop, 6 * DAY)
    assert got["log_count.type.order"] == pytest.approx(math.log1p(3) / 8)
    assert got["log_count.service.ads"] == pytest.approx(math.log1p(2) / 8)
    assert got["order_share.cuisine.phở"] == pytest.approx(2 / 3)
    assert got["order_share.price.1"] == pytest.approx(2 / 3)
    assert got["click_share.category.du lịch"] == 1.0
    assert got["click_share.category.tài chính"] == 0.0  # the day-9 click is after the cutoff
    assert got["click_rate.category.du lịch"] == pytest.approx((1 + 0.06) / (1 + 1))
    assert got["recency.order"] == pytest.approx(1 / 60)
    assert got["recency.complete_ride"] == pytest.approx(5 / 60)
    assert all(0.0 <= v <= 1.0 for v in got.values())


def test_cutoff_is_strict_unless_inclusive():
    history = UserHistory(EVENTS, SCHEMA)
    assert history.stop_before(3 * DAY) == 2
    assert history.stop_before(3 * DAY, inclusive=True) == 4


def test_empty_history_has_zero_counts_and_unknown_recency():
    history = UserHistory([], SCHEMA)
    got = values(history, 0, 10 * DAY)
    assert got["log_count.type.order"] == 0.0 and got["order_share.cuisine.bún"] == 0.0
    assert got["recency.ad_click"] == 1.0


def test_targets_cover_the_window_after_the_cutoff():
    history = UserHistory(EVENTS, SCHEMA)
    stop = history.stop_before(6 * DAY)
    targets, mask = history.targets(stop, 6 * DAY + 7 * DAY)
    assert targets[:2] == (1.0, 0.0)  # the day-10 order is bún
    assert targets[2:] == (0.0, 1.0) and mask == (1.0, 1.0)
    _, empty = history.targets(len(EVENTS), 30 * DAY)
    assert empty == (0.0, 0.0)


def test_events_must_be_time_ordered():
    with pytest.raises(ValueError, match="time order"):
        UserHistory(list(reversed(EVENTS)), SCHEMA)


def test_schema_is_collected_from_the_streamed_samples_and_round_trips():
    sample = SimpleNamespace(events=[SimpleNamespace(context={"cuisine": "phở"}),
                                     SimpleNamespace(context={"cuisine": "bún", "industry_main": "tài chính"}),
                                     SimpleNamespace(context={"pickup_geohash": "w3gv"})])
    collector = HistorySchemaCollector()
    assert list(collector.watch([sample])) == [sample]  # samples pass through unchanged
    schema = collector.schema(target_days=5)
    assert schema.cuisines == ("bún", "phở") and schema.categories == ("tài chính",)
    assert history_schema_from_samples([sample], target_days=5) == schema
    assert HistorySchema.from_dict(schema.to_dict()) == schema
    assert schema.size == len(schema.names) == 3 + 5 + 2 + 3 + 3 * 1 + 3
    with pytest.raises(ValueError, match="size"):
        HistorySchema.from_dict({**schema.to_dict(), "size": 1})
    with pytest.raises(ValueError, match="version"):
        HistorySchema.from_dict({**schema.to_dict(), "version": 99})


def test_unknown_values_are_ignored_not_failed():
    history = UserHistory([event(1, "food", "order", cuisine="pizza", price_level="cao"),
                           event(2, "unknown", "noise")], SCHEMA)
    got = np.array(history.features(2, 3 * DAY))
    assert np.isfinite(got).all()
    assert values(history, 2, 3 * DAY)["log_count.type.order"] == pytest.approx(math.log1p(1) / 8)
