from dataclasses import replace
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from src.baselines.contracts import (
    BaselineDataset,
    Candidate,
    ContractError,
    DatasetMetadata,
    Interaction,
    Label,
    Query,
)
from src.baselines.feature_schema import FeatureEncoder
from src.baselines.features import FeatureBuilder, FeatureRow

T = datetime(2026, 9, 1, tzinfo=timezone.utc)


@pytest.fixture(params=["food_recommendation", "ad_optimization"])
def dataset(request):
    task = request.param
    event_type = "order" if task.startswith("food") else "impression"
    return BaselineDataset(
        DatasetMetadata("fixture", "v1", task, "food" if task.startswith("food") else "ad",
                        "Asia/Shanghai", "logged_impressions", "explicit-v1"),
        (Query("q", "u", T, "train", {"meal_period": "lunch"}),),
        (Candidate("q", "a", {"category": "rice", "price": 0}), Candidate("q", "new")),
        (Label("q", "a", 1, T), Label("q", "new", 0, T)),
        tuple(Interaction(str(i), "u", "a", event_type, T - timedelta(days=days),
                          T - timedelta(days=days), {"category": "rice", "clicked": i % 2})
              for i, days in enumerate([1, 2, 10])),
    )


def test_counts_recency_affinity_and_coverage(dataset):
    rows = FeatureBuilder().build(dataset, "train", history_start=T - timedelta(days=21))
    values = rows[0].values
    assert values["user_count_7d"] == values["item_count_7d"] == 2
    assert values["user_count_30d"] == 3
    assert values["history_complete_7d"] == 1
    assert values["history_complete_30d"] == 0
    assert values["user_recency_seconds"] == 86400
    assert values["user_category_affinity"] == 1
    assert values["hour"] == 8
    assert rows[1].values["user_item_recency_seconds"] is None
    if dataset.metadata.task == "ad_optimization":
        assert values["user_clicks_7d"] == 1
        assert values["user_ctr_7d"] == 0.5
        assert rows[1].values["item_ctr_7d"] is None


def test_future_ties_delayed_events_labels_and_audit_do_not_change_features(dataset):
    builder = FeatureBuilder()
    original = builder.build(dataset, "train")
    event = dataset.history[0]
    delayed = replace(event, event_id="delayed", available_at=T)
    future = replace(event, event_id="future", event_time=T, available_at=T)
    changed = replace(dataset, history=dataset.history + (delayed, future),
                      labels=tuple(replace(label, target=1-label.target) for label in dataset.labels),
                      candidates=tuple(replace(c, attributes={**c.attributes, "clicked": 1,
                                                               "target": 1, "request_id": "bad"})
                                       for c in dataset.candidates))
    assert builder.build(changed, "train") == original


def test_frozen_cutoff_and_encoder_roundtrip(dataset, tmp_path):
    builder = FeatureBuilder()
    train = builder.build(dataset, "train")
    validation = replace(dataset, queries=(replace(dataset.queries[0], split="validation",
                                                  cutoff_time=T + timedelta(days=1)),),
                         labels=tuple(replace(label, available_at=T + timedelta(days=1))
                                      for label in dataset.labels))
    rows = builder.build(validation, "validation", frozen_cutoff=T)
    assert rows[0].values["user_count_7d"] == 2
    assert rows[0].values["user_recency_seconds"] == 172800
    encoder = FeatureEncoder().fit(train)
    path = tmp_path / "schema.json"
    encoder.save(path)
    assert np.array_equal(encoder.transform(rows), FeatureEncoder.load(path).transform(rows))
    with pytest.raises(ContractError, match="train rows"):
        FeatureEncoder().fit(rows)
    with pytest.raises(ContractError, match="prefix"):
        builder.build(dataset, "train", frozen_cutoff=T)


def test_missing_zero_unknown_and_schema_mismatch():
    row = FeatureRow("q", "a", "train", {"item_price": 0, "item_category": "rice"})
    encoder = FeatureEncoder().fit((row,))
    unknown = replace(row, split="test", values={"item_price": None, "item_category": "new"})
    matrix = encoder.transform((row, unknown))
    assert matrix.tolist() == [[2, 0, 0], [1, 0, 1]]
    with pytest.raises(ContractError, match="schema mismatch"):
        encoder.transform((replace(row, values={"item_price": 1}),))


def test_invalid_ctr_history(dataset):
    if dataset.metadata.task != "ad_optimization":
        return
    malformed = replace(dataset, history=(replace(dataset.history[0], attributes={}),))
    with pytest.raises(ContractError, match="mature impression"):
        FeatureBuilder().build(malformed, "train")
