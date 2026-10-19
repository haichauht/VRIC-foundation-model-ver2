from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from src.baselines.baseline import B0Scorer
from src.baselines.contracts import (
    BaselineDataset,
    Candidate,
    ContractError,
    DatasetMetadata,
    Interaction,
    Label,
    Query,
)
from src.evaluation.evaluator import DatasetRegistry, evaluate_predictions
from src.evaluation.predictions import write_predictions

T = datetime(2026, 9, 1, tzinfo=timezone.utc)


def food():
    return BaselineDataset(
        DatasetMetadata("food", "v1", "food_recommendation", "merchant", "UTC",
                        "sampled_catalog", "explicit-v1"),
        (Query("q", "u", T, "test", {"meal_period": "lunch"}),),
        (Candidate("q", "a"), Candidate("q", "b"), Candidate("q", "new")),
        (Label("q", "a", 1, T), Label("q", "b", 0, T), Label("q", "new", 0, T)),
        tuple(Interaction(str(i), "u", item, "order", T - timedelta(days=days),
                          T - timedelta(days=days), {"meal_period": meal})
              for i, (item, days, meal) in enumerate([
                  ("a", 1, "lunch"), ("b", 2, "dinner"), ("b", 10, "dinner")
              ])),
    )


@pytest.mark.parametrize("variant,expected", [
    ("popularity", [1, 2, 0]), ("recent", [1, 1, 0]), ("meal_popularity", [1, 2, 0]),
])
def test_food_counts_and_unseen(variant, expected):
    dataset = food()
    model = B0Scorer(variant).fit(dataset, T)
    assert [model.score(dataset.queries[0], c) for c in dataset.candidates] == expected


@pytest.mark.parametrize("variant", sorted(B0Scorer.FOOD))
def test_food_roundtrip_future_invariance_and_evaluator(variant, tmp_path):
    dataset = food()
    model = B0Scorer(variant).fit(dataset, T)
    future = replace(dataset.history[0], event_id="future", event_time=T + timedelta(days=1),
                     available_at=T + timedelta(days=1))
    delayed = replace(dataset.history[0], event_id="delayed", available_at=T)
    other = B0Scorer(variant).fit(replace(dataset, history=dataset.history + (future, delayed)), T)
    assert model.predict(dataset, "test", "run").equals(other.predict(dataset, "test", "run"))
    path = tmp_path / "model.json"
    model.save(path)
    restored = B0Scorer.load(path)
    predictions = restored.predict(dataset, "test", "run")
    assert predictions.equals(model.predict(dataset, "test", "run"))
    registry = DatasetRegistry()
    registry.register(dataset)
    output = tmp_path / "predictions.parquet"
    write_predictions(output, predictions, dataset, "test")
    assert "ndcg_at_10" in evaluate_predictions(str(output), "food_recommendation", "v1",
                                              "test", registry=registry)


def test_random_independent_of_candidate_order_and_batch():
    dataset = food()
    model = B0Scorer("random").fit(dataset, T)
    assert model.predict(dataset, "test", "run").equals(
        model.predict(replace(dataset, candidates=tuple(reversed(dataset.candidates))), "test", "run"))
    assert all(0 <= model.score(dataset.queries[0], c) < 1 for c in dataset.candidates)


def ads():
    queries, candidates, labels = [], [], []
    for i, (item, target) in enumerate([("a", 1), ("a", 0), ("b", 0)]):
        request = str(i)
        queries.append(Query(request, "u", T - timedelta(days=2), "train"))
        candidates.append(Candidate(request, item))
        labels.append(Label(request, item, target, T - timedelta(days=1)))
    queries.append(Query("test", "u", T, "test"))
    candidates.append(Candidate("test", "new"))
    labels.append(Label("test", "new", 1, T))
    return BaselineDataset(
        DatasetMetadata("ads", "v1", "ad_optimization", "ad", "UTC",
                        "logged_impressions", "explicit-v1"),
        tuple(queries), tuple(candidates), tuple(labels),
    )


@pytest.mark.parametrize("variant", sorted(B0Scorer.ADS))
def test_ctr_ignores_eval_labels_and_delayed_train_labels(variant, tmp_path):
    dataset = ads()
    model = B0Scorer(variant, alpha=2).fit(dataset, T)
    assert model.score(dataset.queries[-1], dataset.candidates[-1]) == pytest.approx(1 / 3)
    changed = replace(dataset, labels=dataset.labels[:-1] + (replace(dataset.labels[-1], target=0),))
    other = B0Scorer(variant, alpha=2).fit(changed, T)
    assert model.state == other.state
    delayed = replace(dataset, labels=tuple(replace(label, available_at=T) for label in dataset.labels))
    assert B0Scorer(variant).fit(delayed, T).state["prior"] == 0.5
    path = tmp_path / "ctr.json"
    model.save(path)
    assert B0Scorer.load(path).predict(dataset, "test", "run").equals(
        model.predict(dataset, "test", "run"))
    registry = DatasetRegistry()
    registry.register(dataset)
    output = tmp_path / "ad.parquet"
    write_predictions(output, model.predict(dataset, "test", "run"), dataset, "test")
    assert evaluate_predictions(str(output), "ad_optimization", "v1", "test",
                                registry=registry)["discrimination_defined"] == 0


def test_ctr_smoothing_and_category_fallback():
    dataset = ads()
    dataset = replace(dataset, candidates=tuple(replace(c, attributes={"category": "x"})
                                               for c in dataset.candidates))
    model = B0Scorer("ad_ctr", alpha=2).fit(dataset, T)
    query = dataset.queries[-1]
    assert model.score(query, Candidate("test", "a", {"category": "x"})) == pytest.approx(5 / 12)
    assert model.score(query, Candidate("test", "unknown")) == pytest.approx(1 / 3)


def test_unfitted_backward_and_namespace_fail():
    dataset = food()
    with pytest.raises(ContractError):
        B0Scorer("popularity").score(dataset.queries[0], dataset.candidates[0])
    model = B0Scorer("popularity").fit(dataset, T)
    with pytest.raises(ContractError, match="precedes"):
        model.score(replace(dataset.queries[0], cutoff_time=T - timedelta(days=1)),
                    dataset.candidates[0])
    with pytest.raises(ContractError, match="namespace"):
        model.predict(replace(dataset, metadata=replace(dataset.metadata, dataset_name="other")),
                      "test", "run")
