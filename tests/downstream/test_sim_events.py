from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone

import pyarrow.parquet as pq
import pytest

from src.downstream.sim_events import (
    SimEvent,
    ads_window_events,
    food_history_events,
    sim_canonical_sample,
    sim_event_records,
)
from src.foundation_input.dataset import DatasetConfig, encode_sample
from src.foundation_input.loader import _event_from_row, iter_normalized_samples
from src.foundation_input.manifest import load_manifest
from src.foundation_input.tokenizer import TokenizerConfig
from src.foundation_input.types import EventRecord
from src.foundation_input.validation import validate_sample
from src.foundation_input.vocab import (
    VocabularyArtifacts,
    VocabularyConfig,
    fit_vocabulary_from_samples,
)
from src.training.pretrain import tokenizer_config_for_manifest

# Large enough that encode_sample never truncates a sim history (<= 30 events).
NO_TRUNCATION = DatasetConfig(max_tokens=2048, max_event_count=64)
T0 = datetime(2026, 1, 5, 8, 0, 0, tzinfo=timezone(timedelta(hours=7)))


def _event(event_id, event_type, service, attributes=None, *, minutes=0):
    return SimEvent(
        event_id=event_id,
        user_id="sim:user:7",
        event_time=T0 + timedelta(minutes=minutes),
        service=service,
        event_type=event_type,
        entity_id=None,
        attributes=dict(attributes or {}),
    )


def test_scenario_context_action_split():
    ride_context = {
        "pickup_geohash": "w3gvdm",
        "dropoff_geohash": "w3gvsg",
        "pickup_zone_type": "residential",
        "dropoff_zone_type": "office",
        "vehicle_type": "bike",
        "driver_id": "sim:driver:3",
    }
    ride_numbers = {"distance_km": 17.0, "fare": 86000.0, "wait_minutes": 6.8}
    ride, click = sim_event_records([
        _event("sim:event:1", "book_ride", "ride", {**ride_context, **ride_numbers}),
        _event(
            "sim:event:2",
            "ad_click",
            "ads",
            {"product": "sim:ad:4", "industry_main": "food", "industry_sub": "x", "caption": "c"},
            minutes=5,
        ),
    ])
    assert ride.scenario == "Ride"
    assert ride.action == {"type": "book_ride", **ride_numbers}
    assert ride.context == ride_context
    assert ride.user_id == "sim:user:7"
    assert ride.event_time == datetime(2026, 1, 5, 1, 0, 0)
    assert ride.eligibility_time == ride.event_time
    assert (ride.time_precision, ride.eligibility_inclusive) == ("second", True)
    assert click.scenario == "Advertisement"
    assert click.action == {"type": "click"}


def test_groups_start_at_session_start_types():
    kinds = [
        ("view_merchant", "food"),
        ("order", "food"),
        ("search", "food"),
        ("view_merchant", "food"),
        ("order", "food"),
        ("ad_impression", "ads"),
        ("ad_click", "ads"),
        ("search_destination", "ride"),
    ]
    events = [
        _event(f"sim:event:{index}", event_type, service, minutes=index)
        for index, (event_type, service) in enumerate(kinds)
    ]
    records = sim_event_records(events)
    assert [record.source_action_count for record in records] == [2, 2, 3, 3, 3, 2, 2, 1]
    assert [record.source_record_index for record in records] == [0, 0, 1, 1, 1, 2, 2, 3]
    assert [record.action_index for record in records] == [0, 1, 0, 1, 2, 0, 1, 0]
    assert [record.source_record_id for record in records] == [
        "sim:event:0", "sim:event:0",
        "sim:event:2", "sim:event:2", "sim:event:2",
        "sim:event:5", "sim:event:5",
        "sim:event:7",
    ]
    assert records[5].action == {"type": "impression"}


def test_canonical_sample_is_naive_utc_and_copies_profile():
    profile = {"as_of_time": "2026-01-05T15:05:54+07:00", "age_group": "25-34"}
    sample = sim_canonical_sample(
        sample_id="sim:food_req:73",
        dataset_name="sim_v1",
        split="train",
        user_id="sim:user:7",
        cutoff_time=datetime.fromisoformat("2026-01-05T15:05:54+07:00"),
        profile=profile,
        events=[_event("sim:event:1", "search", "food", {"search_query": "phở"})],
        history_truncated=False,
        history_group_truncated=True,
    )
    assert sample.cutoff_time == datetime(2026, 1, 5, 8, 5, 54)
    assert sample.profile == {"as_of_time": "2026-01-05T08:05:54", "age_group": "25-34"}
    assert profile["as_of_time"] == "2026-01-05T15:05:54+07:00"
    assert (sample.cutoff_inclusive, sample.history_order_known) == (False, True)
    assert (sample.history_truncated, sample.history_group_truncated) == (False, True)
    assert [event.event_id for event in sample.events] == ["sim:event:1"]


def test_naive_times_are_rejected():
    with pytest.raises(ValueError, match="sim:event:1"):
        replace(_event("sim:event:1", "search", "food"), event_time=datetime(2026, 1, 5))
    with pytest.raises(ValueError, match="cutoff_time"):
        sim_canonical_sample(
            sample_id="s",
            dataset_name="sim_v1",
            split="train",
            user_id="sim:user:7",
            cutoff_time=datetime(2026, 1, 5),
            profile={},
            events=[],
            history_truncated=False,
            history_group_truncated=False,
        )


# ---- parity with the pretrain package --------------------------------------------------------


@dataclass(frozen=True)
class Pretrain:
    dataset_name: str
    events_end: datetime  # naive UTC start of day 43; pretrain has no event at or after it
    tokenizer_config: TokenizerConfig
    vocab: VocabularyArtifacts
    events: dict[str, EventRecord]


@pytest.fixture(scope="module")
def pretrain(sim_tiny_packages) -> Pretrain:
    manifest = load_manifest(sim_tiny_packages.pretrain, validate_shard_files=False)
    tokenizer_config = tokenizer_config_for_manifest(manifest)
    vocab = fit_vocabulary_from_samples(
        iter_normalized_samples(sim_tiny_packages.pretrain, split="train", validate=False),
        VocabularyConfig(dataset_name=manifest.dataset_name),
        tokenizer_config,
    )
    events: dict[str, EventRecord] = {}
    for shard in manifest.shards:
        for row in pq.read_table(manifest.root / shard["events_file"]).to_pylist():
            event = _event_from_row(row)
            events[event.event_id] = event
    return Pretrain(
        dataset_name=manifest.dataset_name,
        events_end=datetime.fromisoformat(manifest.config["events_end_exclusive"]),
        tokenizer_config=tokenizer_config,
        vocab=vocab,
        events=events,
    )


def _token_tuples(sample, pretrain: Pretrain) -> list[tuple]:
    encoded = encode_sample(sample, pretrain.vocab, pretrain.tokenizer_config, NO_TRUNCATION)
    return [
        (
            record.value_id,
            record.key_id,
            record.modality_id,
            record.source_id,
            record.position_id,
            record.time_gap_id,
            record.numeric_value,
            record.numeric_missing,
        )
        for record in encoded.records
    ]


def _assert_parity(sample, event_ids: list[str], pretrain: Pretrain) -> None:
    """The sim sample tokenizes exactly like the same events read by the pretrain loader."""
    reference = replace(sample, events=tuple(pretrain.events[eid] for eid in event_ids))
    validate_sample(sample)
    validate_sample(reference)
    assert [event.event_id for event in sample.events] == event_ids
    assert _token_tuples(sample, pretrain) == _token_tuples(reference, pretrain), sample.sample_id


def _food_records(food_root) -> list[dict]:
    lines = (food_root / "samples.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def test_empty_history_encodes(sim_tiny_packages, pretrain):
    record = next(
        record
        for record in _food_records(sim_tiny_packages.food)
        if record["label_metadata"]["split"] == "train" and not record["input"]["history"]
    )
    inp = record["input"]
    sample = sim_canonical_sample(
        sample_id=record["sample_id"],
        dataset_name=pretrain.dataset_name,
        split="train",
        user_id=inp["user_id"],
        cutoff_time=datetime.fromisoformat(inp["cutoff_time"]),
        profile=inp["profile"],
        events=food_history_events(record),
        history_truncated=inp["history_truncated"],
        history_group_truncated=inp["history_group_truncated"],
    )
    validate_sample(sample)
    encoded = encode_sample(sample, pretrain.vocab, pretrain.tokenizer_config, NO_TRUNCATION)
    assert sample.events == ()
    assert encoded.records
    assert {record.token.source for record in encoded.records} == {"tabular"}
    assert encoded.records[0].token.key == "user_id"


def test_food_history_matches_pretrain_tokens(sim_tiny_packages, pretrain):
    compared = with_history = 0
    for record in _food_records(sim_tiny_packages.food):
        if record["label_metadata"]["split"] != "train":
            continue
        inp = record["input"]
        sample = sim_canonical_sample(
            sample_id=record["sample_id"],
            dataset_name=pretrain.dataset_name,
            split="train",
            user_id=inp["user_id"],
            cutoff_time=datetime.fromisoformat(inp["cutoff_time"]),
            profile={},
            events=food_history_events(record),
            history_truncated=inp["history_truncated"],
            history_group_truncated=inp["history_group_truncated"],
        )
        _assert_parity(sample, [item["event_id"] for item in inp["history"]], pretrain)
        compared += 1
        with_history += bool(inp["history"])
    assert with_history >= 50, (compared, with_history)


def test_ads_window_starting_mid_session_matches_pretrain(sim_tiny_packages, pretrain):
    events_by_user: dict[str, list[dict]] = defaultdict(list)
    for row in pq.read_table(sim_tiny_packages.ads / "events.parquet").to_pylist():
        events_by_user[row["user_id"]].append(row)
    for rows in events_by_user.values():
        rows.sort(key=lambda row: row["event_index"])
    events_end = pretrain.events_end.replace(tzinfo=timezone.utc)
    opportunities = pq.read_table(sim_tiny_packages.ads / "ad_opportunities.parquet").to_pylist()
    compared = mid_session = 0
    for opportunity in opportunities:
        if opportunity["split"] != "train":
            continue
        window = events_by_user[opportunity["user_id"]][
            opportunity["history_start"] : opportunity["history_stop"]
        ]
        events = ads_window_events(window)
        if any(event.event_time >= events_end for event in events):
            continue
        sample = sim_canonical_sample(
            sample_id=opportunity["request_id"],
            dataset_name=pretrain.dataset_name,
            split="train",
            user_id=opportunity["user_id"],
            cutoff_time=datetime.fromtimestamp(opportunity["cutoff_timestamp"], timezone.utc),
            profile={},
            events=events,
            history_truncated=opportunity["history_start"] > 0,
            history_group_truncated=False,
        )
        event_ids = [row["event_id"] for row in window]
        _assert_parity(sample, event_ids, pretrain)
        compared += 1
        mid_session += bool(event_ids) and pretrain.events[event_ids[0]].action_index > 0
    assert compared >= 50, compared
    assert mid_session >= 1, (compared, mid_session)
