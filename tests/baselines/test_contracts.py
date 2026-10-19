from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from src.baselines.contracts import (
    BaselineDataset,
    Candidate,
    ContractError,
    DatasetMetadata,
    Interaction,
    Label,
    Query,
    eligible_history,
    validate_score,
)


@pytest.fixture(params=[("food_recommendation", "merchant"), ("ad_optimization", "ad")])
def dataset(request):
    task, kind = request.param
    cutoff = datetime(2021, 3, 25, tzinfo=timezone.utc)
    return BaselineDataset(
        DatasetMetadata("fixture", "fixture-v1", task, kind, "UTC", "sampled_catalog",
                        "explicit available_at"),
        (Query("q1", "u1", cutoff, "train"),),
        (Candidate("q1", "i1"), Candidate("q1", "i2")),
        (Label("q1", "i1", 1, cutoff + timedelta(minutes=5)),
         Label("q1", "i2", 0, cutoff + timedelta(minutes=30))),
        (Interaction("e1", "u1", "i1", "order", cutoff - timedelta(days=1),
                     cutoff - timedelta(hours=23)),),
    )


def test_valid_handoff(dataset):
    dataset.validate()


@pytest.mark.parametrize("mutation,rule", [
    (lambda d: replace(d, candidates=d.candidates + d.candidates[:1]), "duplicate candidate"),
    (lambda d: replace(d, labels=d.labels[:1]), "exactly one label"),
    (lambda d: replace(d, queries=d.queries + d.queries), "duplicate request"),
    (lambda d: replace(d, candidates=(replace(d.candidates[0], request_id="absent"),)),
     "unknown request"),
    (lambda d: replace(d, labels=(replace(d.labels[0], target=True),) + d.labels[1:]),
     "binary integer"),
    (lambda d: replace(d, queries=(replace(d.queries[0], cutoff_time=datetime(2021, 3, 25)),)),
     "timezone-aware"),
    (lambda d: replace(d, history=d.history + d.history), "duplicate interaction"),
    (lambda d: replace(d, metadata=replace(d.metadata, schema_version=2)), "schema_version"),
])
def test_invalid_handoff(dataset, mutation, rule):
    with pytest.raises(ContractError, match=rule):
        mutation(dataset).validate()


def test_history_excludes_delayed_future_and_tied_events(dataset):
    cutoff = dataset.queries[0].cutoff_time
    event = dataset.history[0]
    delayed = replace(event, event_id="delayed", available_at=cutoff + timedelta(hours=1))
    tied = replace(event, event_id="tied", event_time=cutoff, available_at=cutoff)
    future = replace(tied, event_id="future", event_time=cutoff + timedelta(hours=1),
                     available_at=cutoff + timedelta(hours=1))
    assert eligible_history(dataset.history + (delayed, tied, future), cutoff) == dataset.history


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -0.1, 1.1, True])
def test_invalid_ad_probability(score):
    with pytest.raises(ContractError):
        validate_score(score, "ad_optimization")


def test_food_score_can_be_negative():
    validate_score(-5.0, "food_recommendation")
