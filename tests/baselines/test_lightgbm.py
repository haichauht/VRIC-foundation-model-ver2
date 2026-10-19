from dataclasses import replace
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from src.baselines.contracts import (
    BaselineDataset,
    Candidate,
    ContractError,
    DatasetMetadata,
    Label,
    Query,
)
from src.baselines.lightgbm import B1Classifier
from src.evaluation.evaluator import DatasetRegistry, evaluate_predictions
from src.evaluation.predictions import write_predictions

T = datetime(2026, 9, 1, tzinfo=timezone.utc)


@pytest.fixture(params=["food_recommendation", "ad_optimization"])
def dataset(request):
    queries, candidates, labels = [], [], []
    for split, day in [("train", -2), ("validation", 0), ("test", 2)]:
        for i in range(32):
            q = f"{split}-{i}"
            cutoff = T + timedelta(days=day)
            queries.append(Query(q, "u", cutoff, split))
            candidates.append(Candidate(q, str(i), {"price": i % 2, "category": "x"}))
            labels.append(Label(q, str(i), i % 2, cutoff + timedelta(hours=1)))
    task = request.param
    return BaselineDataset(
        DatasetMetadata("fixture", "v1", task, "food" if task.startswith("food") else "ad",
                        "UTC", "logged_impressions", "explicit-v1"),
        tuple(queries), tuple(candidates), tuple(labels),
    )


def model():
    return B1Classifier(n_estimators=30, learning_rate=0.2, num_leaves=3,
                        min_child_samples=2, stopping_rounds=3)


def test_train_predict_reload_and_evaluate(dataset, tmp_path):
    fitted = model().fit(dataset, training_cutoff=T)
    table = fitted.predict(dataset, "test", "run", frozen_cutoff=T + timedelta(days=2))
    scores = np.asarray(table["score"])
    assert np.isfinite(scores).all()
    path = tmp_path / "model"
    fitted.save(path)
    restored = B1Classifier.load(path)
    assert table.equals(restored.predict(dataset, "test", "run",
                                        frozen_cutoff=T + timedelta(days=2)))
    registry = DatasetRegistry()
    registry.register(dataset)
    output = tmp_path / "predictions.parquet"
    write_predictions(output, table, dataset, "test")
    result = evaluate_predictions(str(output), dataset.metadata.task, "v1", "test",
                                  registry=registry)
    assert result["brier_score"] < 0.25 if dataset.metadata.task == "ad_optimization" else (
        result["request_count"] == 32)
    with pytest.raises(FileExistsError):
        fitted.save(path)


def test_test_labels_and_categories_do_not_affect_training(dataset):
    changed = replace(dataset,
                      labels=tuple(replace(label, target=1-label.target)
                                   if label.request_id.startswith("test") else label
                                   for label in dataset.labels),
                      candidates=tuple(replace(c, attributes={"price": 999, "category": "test-only"})
                                       if c.request_id.startswith("test") else c
                                       for c in dataset.candidates))
    original = model().fit(dataset, training_cutoff=T)
    other = model().fit(changed, training_cutoff=T)
    assert original.booster.model_to_string() == other.booster.model_to_string()
    assert original.encoder.vocabularies == other.encoder.vocabularies


def test_delayed_train_label_and_single_class_rejected(dataset):
    delayed = replace(dataset, labels=(replace(dataset.labels[0], available_at=T),)
                      + dataset.labels[1:])
    with pytest.raises(ContractError, match="not available"):
        model().fit(delayed, training_cutoff=T)
    single = replace(dataset, labels=tuple(replace(label, target=0) for label in dataset.labels))
    with pytest.raises(ContractError, match="both target classes"):
        model().fit(single, training_cutoff=T)


def test_nonchronological_split_and_unfitted_prediction(dataset):
    with pytest.raises(ContractError, match="chronological"):
        model().fit(dataset, training_cutoff=T + timedelta(days=1))
    with pytest.raises(ContractError, match="fitted"):
        model().predict(dataset, "test", "run", frozen_cutoff=T)
