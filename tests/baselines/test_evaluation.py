from dataclasses import replace
from datetime import datetime, timezone
from math import isnan, log, log2

import pyarrow as pa
import pytest

from src.baselines.adapters import adapt_records
from src.baselines.contracts import ContractError, DatasetMetadata
from src.evaluation.evaluator import DatasetRegistry, evaluate_predictions
from src.evaluation.metrics import advertising_metrics, ranking_metrics
from src.evaluation.predictions import PREDICTION_SCHEMA, validate_predictions, write_predictions


@pytest.fixture(params=["food_recommendation", "ad_optimization"])
def handoff(request):
    task = request.param
    now = datetime(2026, 9, 1, 12, tzinfo=timezone.utc)
    meta = DatasetMetadata("fixture", "v1", task, "food" if task.startswith("food") else "ad",
                           "UTC", "logged_impressions", "explicit-v1")
    if task.startswith("food"):
        records = [{"input": {"user_id": "u", "cutoff_time": now.isoformat(),
                              "candidates": [{"item_id": "a"}, {"item_id": "b"}]},
                    "label_metadata": {"request_id": "q", "observation_complete": True,
                                       "candidate_source": "logged_impressions"},
                    "labels": [{"item_id": "a", "relevance": 1},
                               {"item_id": "b", "relevance": 0}]}]
        availability = {("q", "a"): now, ("q", "b"): now}
    else:
        records = [{"impression_id": q, "prediction_time": now.isoformat(),
                    "inputs": {"user": {"user_id": "u"}, "ad": {"ad_id": item}},
                    "labels": {"clicked": target, "converted": None}}
                   for q, item, target in [("q", "a", 1), ("r", "b", 0)]]
        availability = {("q", "a"): now, ("r", "b"): now}
    dataset = adapt_records(records, meta, "test", availability)
    queries = {q.request_id: q for q in dataset.queries}
    table = pa.Table.from_pylist([
        {"schema_version": 1, "task": task, "split": "test", "data_version": "v1",
         "run_id": "run1", "request_id": c.request_id, "candidate_id": c.candidate_id,
         "user_id": "u", "cutoff_timestamp": int(queries[c.request_id].cutoff_time.timestamp()),
         "score": 0.9 if c.candidate_id == "a" else 0.1}
        for c in dataset.candidates
    ], schema=PREDICTION_SCHEMA)
    return dataset, table, records, availability


def test_end_to_end_public_evaluator(handoff, tmp_path):
    dataset, table, _, _ = handoff
    path = tmp_path / "predictions.test.parquet"
    write_predictions(path, table, dataset, "test")
    registry = DatasetRegistry()
    registry.register(dataset)
    result = evaluate_predictions(str(path), dataset.metadata.task, "v1", "test", registry=registry)
    metric = "ndcg_at_10" if dataset.metadata.task.startswith("food") else "roc_auc"
    assert result[metric] == 1.0


@pytest.mark.parametrize("field,value", [
    ("data_version", "wrong"), ("user_id", "wrong"), ("cutoff_timestamp", 0),
    ("score", float("nan")), ("run_id", ""), ("candidate_id", "absent"),
])
def test_reject_invalid_prediction(handoff, field, value):
    dataset, table, _, _ = handoff
    rows = table.to_pylist()
    rows[0][field] = value
    with pytest.raises(ContractError):
        validate_predictions(pa.Table.from_pylist(rows, schema=PREDICTION_SCHEMA), dataset, "test")


def test_missing_duplicate_extra_dtype_and_atomic_failure(handoff, tmp_path):
    dataset, table, _, _ = handoff
    path = tmp_path / "bad.parquet"
    for bad in (table.slice(0, 1), pa.concat_tables([table, table.slice(0, 1)]),
                table.append_column("clicked", pa.array([0] * len(table))),
                table.set_column(0, "schema_version", pa.array([1] * len(table), pa.int64()))):
        with pytest.raises(ContractError):
            write_predictions(path, bad, dataset, "test")
        assert not path.exists()


def test_missing_availability_and_unknown_timezone(handoff):
    dataset, _, records, _ = handoff
    with pytest.raises(ContractError, match="availability"):
        adapt_records(records, dataset.metadata, "test", {})
    with pytest.raises(ContractError, match="timezone"):
        replace(dataset, metadata=replace(dataset.metadata, timezone="unknown/zone")).validate()


def test_ties_multiple_positives_and_large_k():
    result = ranking_metrics([("c", 1, 0.5), ("a", 0, 0.5), ("b", 1, 0.5)], (1, 2, 20))
    assert result["mrr"] == 0.5
    assert result["recall_at_2"] == 0.5
    assert result["ndcg_at_2"] == pytest.approx((1 / log2(3)) / (1 + 1 / log2(3)))
    assert result["recall_at_20"] == 1


def test_zero_positive_and_invalid_k():
    assert ranking_metrics([("a", 0, 1.0)])["mrr"] == 0
    with pytest.raises(ValueError):
        ranking_metrics([], (0,))


def test_ad_metrics_by_hand_and_single_class():
    result = advertising_metrics([1, 0], [0.8, 0.2])
    assert result["roc_auc"] == result["pr_auc"] == 1
    assert result["log_loss"] == pytest.approx(-log(0.8))
    assert result["brier_score"] == pytest.approx(0.04)
    single = advertising_metrics([0, 0], [0.2, 0.2])
    assert isnan(single["roc_auc"])
    assert single["discrimination_defined"] == 0
