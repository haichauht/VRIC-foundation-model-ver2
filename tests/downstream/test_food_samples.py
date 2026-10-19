"""Tests for reading and validating the Food samples.jsonl handoff."""

from __future__ import annotations

import copy
import hashlib
import json
import zipfile
from datetime import datetime, timedelta, timezone

import pytest

from src.downstream.food_samples import (
    candidate_rows,
    load_poi_catalog,
    parse_utc,
    read_food_samples,
    to_canonical_sample,
    validate_food_samples,
)


def _z(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def make_record(
    request_id: str,
    split: str,
    cutoff: str,
    *,
    user: str = "meituan:user:1",
    history: list[dict] | None = None,
    items: tuple[str, ...] = ("meituan:spu:1", "meituan:spu:2"),
    positives: tuple[int, ...] = (1, 0),
    restaurant: str = "meituan:poi:10",
) -> dict:
    """Build one record shaped like a real samples.jsonl line."""
    moment = parse_utc(cutoff, field="cutoff_time", request_id=request_id)
    if history is None:
        history = [{
            "event_id": f"meituan:order:{request_id}-h0",
            "event_time": _z(moment - timedelta(seconds=60)),
            "event_type": "order",
            "service": "Food",
            "item_ids": ["meituan:spu:1"],
            "restaurant_id": restaurant,
        }]
    return {
        "sample_id": request_id,
        "input": {
            "user_id": user,
            "cutoff_time": cutoff,
            "profile": {
                "as_of_time": cutoff,
                "order_count_30d": 1,
                "active_days_30d": 1,
                "distinct_food_items_30d": 1,
                "avg_pay_amt": "[29,36)",
                "avg_pay_amt_weekdays": "[29,36)",
                "avg_pay_amt_weekends": "[29,36)",
            },
            "history": history,
            "context": {"hour": 1, "weekday": 0},
            "candidates": [
                {"item_id": item, "restaurant_id": restaurant, "price": 3.0,
                 "category": [1, 2, 3], "ingredients": None, "taste": None}
                for item in items
            ],
        },
        "labels": [{"item_id": item, "relevance": rel} for item, rel in zip(items, positives)],
        "label_metadata": {
            "request_id": request_id,
            "request_source": "meituan_order",
            "split": split,
            "label_time": _z(moment + timedelta(seconds=1)),
            "prediction_anchor": "order_time",
            "candidate_source": "sampled_catalog",
            "positive_action": "order",
            "label_protocol": "order_items",
            "negative_label_semantics": "unobserved",
            "observation_complete": None,
            "catalog_feature_time_assumption": "static",
            "user_profile_feature_time_assumption": "as_of_time",
        },
    }


def write_samples_zip(tmp_path, records):
    """Write records to samples.zip; return (path, sha256 of the member bytes)."""
    payload = "".join(json.dumps(r) + "\n" for r in records).encode("utf-8")
    path = tmp_path / "samples.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("samples.jsonl", payload)
    return path, hashlib.sha256(payload).hexdigest()


def valid_records() -> list[dict]:
    return [
        make_record("train-1", "train", "2021-03-01T01:00:00Z"),
        make_record("train-2", "train", "2021-03-01T02:00:00Z"),
        make_record("validation-1", "validation", "2021-03-15T01:00:00Z"),
        make_record("test-1", "test", "2021-03-18T01:00:00Z"),
    ]


def test_reads_member_from_zip_and_returns_member_sha(tmp_path):
    path, sha = write_samples_zip(tmp_path, valid_records())
    records, digest = read_food_samples(path, "samples.jsonl", sha)
    assert digest == sha
    assert [r["sample_id"] for r in records] == [r["sample_id"] for r in valid_records()]


def test_reads_plain_jsonl_file(tmp_path):
    payload = "".join(json.dumps(r) + "\n" for r in valid_records()).encode("utf-8")
    path = tmp_path / "samples.jsonl"
    path.write_bytes(payload)
    sha = hashlib.sha256(payload).hexdigest()
    records, digest = read_food_samples(path, "samples.jsonl", sha)
    assert len(records) == 4 and digest == sha


def test_rejects_sha_mismatch(tmp_path):
    path, _ = write_samples_zip(tmp_path, valid_records())
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        read_food_samples(path, "samples.jsonl", "0" * 64)


def test_missing_path_or_member_is_clear(tmp_path):
    path, sha = write_samples_zip(tmp_path, valid_records())
    with pytest.raises(ValueError, match="nope.zip.*not found"):
        read_food_samples(tmp_path / "nope.zip", "samples.jsonl", sha)
    with pytest.raises(ValueError, match="missing.jsonl.*not found"):
        read_food_samples(path, "missing.jsonl", sha)


def test_valid_records_pass():
    validate_food_samples(valid_records(), "sampled_catalog")


def _drop_context(r):
    del r[0]["input"]["context"]


def _dup_sample_id(r):
    r[1]["sample_id"] = r[0]["sample_id"]


def _request_id(r):
    r[0]["label_metadata"]["request_id"] = "x"


def _split(r):
    r[0]["label_metadata"]["split"] = "dev"


def _swap_labels(r):
    r[0]["labels"].reverse()


def _dup_candidate(r):
    r[0]["input"]["candidates"][1]["item_id"] = r[0]["input"]["candidates"][0]["item_id"]


def _bad_relevance(r):
    r[0]["labels"][0]["relevance"] = 2


def _no_positive(r):
    for label in r[0]["labels"]:
        label["relevance"] = 0


def _late_event(r):
    r[0]["input"]["history"][0]["event_time"] = r[0]["input"]["cutoff_time"]


def _early_label(r):
    r[0]["label_metadata"]["label_time"] = r[0]["input"]["cutoff_time"]


def _policy(r):
    r[0]["label_metadata"]["candidate_source"] = "other"


def _naive(r):
    r[0]["input"]["cutoff_time"] = "2021-03-01T01:00:00"


def _fractional(r):
    r[0]["input"]["cutoff_time"] = "2021-03-01T01:00:00.5Z"


@pytest.mark.parametrize(
    ("mutate", "pattern"),
    [
        (_drop_context, "missing field input.context"),
        (_dup_sample_id, "duplicate sample_id"),
        (_request_id, "request_id differs from sample_id"),
        (_split, "invalid split"),
        (_swap_labels, "candidates and labels differ"),
        (_dup_candidate, "duplicate candidate"),
        (_bad_relevance, "relevance must be 0 or 1"),
        (_no_positive, "no positive"),
        (_late_event, "history event at or after cutoff"),
        (_early_label, "label_time must be after cutoff"),
        (_policy, "candidate policy mismatch"),
        (_naive, "timezone"),
        (_fractional, "whole seconds"),
    ],
)
def test_rejects_invalid_record(mutate, pattern):
    records = valid_records()
    mutate(records)
    with pytest.raises(ValueError, match=pattern):
        validate_food_samples(records, "sampled_catalog")


def test_error_message_names_request_id():
    records = valid_records()
    _split(records)
    with pytest.raises(ValueError, match=r"request_id=train-1: invalid split"):
        validate_food_samples(records, "sampled_catalog")


def test_rejects_overlapping_temporal_splits():
    records = valid_records()
    records[2]["input"]["cutoff_time"] = "2021-03-01T01:30:00Z"  # validation before last train
    with pytest.raises(ValueError, match="temporal split overlap"):
        validate_food_samples(records, "sampled_catalog")


def test_accepts_offset_timestamps_and_equal_event_times():
    records = valid_records()
    records[0]["input"]["cutoff_time"] = "2021-03-01T09:00:00+08:00"
    first = records[0]["input"]["history"][0]  # two orders share one event_time
    records[0]["input"]["history"] = [first, {**first, "event_id": "meituan:order:dup"}]
    validate_food_samples(records, "sampled_catalog")
    assert parse_utc("2021-03-01T09:00:00+08:00", field="cutoff_time", request_id="r") == \
        datetime(2021, 3, 1, 1, tzinfo=timezone.utc)


def test_validation_does_not_mutate_records():
    records = valid_records()
    before = copy.deepcopy(records)
    validate_food_samples(records, "sampled_catalog")
    assert records == before


CATALOG_TSV = ("wm_poi_id\twm_poi_name\tprimary_second_tag_name\tprimary_third_tag_name\t"
               "primary_first_tag_name\tpoi_brand_id\taor_id\tpoi_score\t"
               "delivery_comment_avg_score\tfood_comment_avg_score\r\n"
               "10\tname-10\tsecond\t未知\tfirst\t0\t1\t4.5\t4.6\t4.7\r\n"
               "11\tname-11\tNULL\t\tfirst-11\t0\t1\t4.5\t4.6\t4.7\r\n")


def load_catalog_fixture(tmp_path):
    payload = CATALOG_TSV.encode("utf-8")
    path = tmp_path / "pois.txt"
    path.write_bytes(payload)
    return load_poi_catalog(path, hashlib.sha256(payload).hexdigest())


def test_catalog_cleans_unknown_tags_and_crlf(tmp_path):
    catalog, _ = load_catalog_fixture(tmp_path)
    assert catalog["10"] == {"name": "name-10", "tag1": "first", "tag2": "second", "tag3": "unknown"}
    assert catalog["11"] == {"name": "name-11", "tag1": "first-11", "tag2": "unknown", "tag3": "unknown"}


def test_maps_order_to_purchase_with_builder_context(tmp_path):
    catalog, _ = load_catalog_fixture(tmp_path)
    sample = to_canonical_sample(valid_records()[0], catalog, "meituan_trd_v2")
    event = sample.events[0]
    assert (sample.dataset_name, sample.user_id, sample.cutoff_inclusive) == (
        "meituan_trd_v2", "1", False)
    assert (event.scenario, event.event_type, event.source_action_count) == (
        "E-commerce", "purchase", 1)
    assert event.action == {"type": "purchase", "is_pay": True, "item_cnt": 1, "order_amt": None}
    assert event.context == {
        "product_source": "meituan_takeout", "product_id": "10", "item_title": "name-10",
        "item_desc": "", "external_item_title": "", "item_price": None,
        "category_level1_name": "first", "category_level2_name": "second",
        "category_level3_name": "unknown", "industry_level1_name": "Food Delivery",
        "industry_level2_name": "first", "industry_level3_name": "second"}
    assert event.event_time.tzinfo is not None and event.eligibility_inclusive is True


def test_maps_profile_fields(tmp_path):
    catalog, _ = load_catalog_fixture(tmp_path)
    record = valid_records()[0]
    record["input"]["profile"].update(order_count_30d=4, active_days_30d=2)
    profile = to_canonical_sample(record, catalog, "meituan_trd_v2").profile
    assert {k: v for k, v in profile.items() if k != "as_of_time"} == {
        "active_days_30d": 2, "event_count_30d": 4, "e_commerce_event_count_30d": 4,
        "advertisement_event_count_30d": 0, "customer_service_event_count_30d": 0,
        "live_streaming_event_count_30d": 0, "search_behavior_event_count_30d": 0,
        "video_browsing_event_count_30d": 0}


def test_empty_history_maps_to_profile_only(tmp_path):
    catalog, _ = load_catalog_fixture(tmp_path)
    record = make_record("train-1", "train", "2021-03-01T01:00:00Z", history=[])
    assert to_canonical_sample(record, catalog, "meituan_trd_v2").events == ()


def test_to_canonical_sample_does_not_mutate_record(tmp_path):
    catalog, _ = load_catalog_fixture(tmp_path)
    record = valid_records()[0]
    before = copy.deepcopy(record)
    to_canonical_sample(record, catalog, "meituan_trd_v2")
    assert record == before


def test_catalog_sha_mismatch(tmp_path):
    load_catalog_fixture(tmp_path)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_poi_catalog(tmp_path / "pois.txt", "0" * 64)


def test_unknown_restaurant_rejected_in_history_and_candidates(tmp_path):
    catalog, _ = load_catalog_fixture(tmp_path)
    bad_history = make_record("train-1", "train", "2021-03-01T01:00:00Z")
    bad_history["input"]["history"][0]["restaurant_id"] = "meituan:poi:99"
    with pytest.raises(ValueError, match="request_id=train-1: unknown restaurant"):
        to_canonical_sample(bad_history, catalog, "meituan_trd_v2")
    bad_candidate = make_record("train-1", "train", "2021-03-01T01:00:00Z")
    bad_candidate["input"]["candidates"][1]["restaurant_id"] = "meituan:poi:99"
    with pytest.raises(ValueError, match="request_id=train-1: unknown restaurant"):
        candidate_rows(bad_candidate, catalog)


def test_candidate_rows_keep_original_ids_and_label_time(tmp_path):
    catalog, _ = load_catalog_fixture(tmp_path)
    rows = candidate_rows(valid_records()[0], catalog)
    assert rows[0] == {"request_id": "train-1", "user_id": "meituan:user:1",
                       "cutoff_timestamp": 1614560400, "candidate_id": "meituan:spu:1",
                       "is_positive": 1, "split": "train", "label_available_at_timestamp": 1614560401}


def test_offset_as_of_time_is_normalized_and_passes_validate_sample(tmp_path):
    from src.foundation_input.validation import validate_sample

    catalog, _ = load_catalog_fixture(tmp_path)
    record = valid_records()[0]
    record["input"]["cutoff_time"] = "2021-03-01T09:00:00+08:00"
    record["input"]["profile"]["as_of_time"] = "2021-03-01T09:00:00+08:00"
    record["input"]["history"][0]["event_time"] = "2021-03-01T08:59:00+08:00"
    validate_food_samples([record], "sampled_catalog")
    validate_sample(to_canonical_sample(record, catalog, "meituan_trd_v2"))


def test_offset_as_of_time_after_cutoff_is_rejected(tmp_path):
    from src.foundation_input.validation import TemporalValidationError, validate_sample

    catalog, _ = load_catalog_fixture(tmp_path)
    record = valid_records()[0]
    record["input"]["cutoff_time"] = "2021-03-01T01:00:00-08:00"  # 09:00Z
    record["input"]["profile"]["as_of_time"] = "2021-03-01T02:00:00-08:00"  # 10:00Z
    record["input"]["history"][0]["event_time"] = "2021-03-01T00:59:00-08:00"
    record["label_metadata"]["label_time"] = "2021-03-01T01:00:01-08:00"
    with pytest.raises(TemporalValidationError, match="as_of_time is after cutoff"):
        validate_sample(to_canonical_sample(record, catalog, "meituan_trd_v2"))


def test_as_of_time_requires_timezone():
    records = valid_records()
    records[0]["input"]["profile"]["as_of_time"] = "2021-03-01T01:00:00"
    with pytest.raises(ValueError, match="request_id=train-1: profile.as_of_time"):
        validate_food_samples(records, "sampled_catalog")


def test_malformed_shapes_become_request_id_errors():
    records = valid_records()
    records[0]["input"]["history"] = ["not-a-dict"]
    records[1]["input"]["candidates"] = "not-a-list"
    records[2]["sample_id"] = ["unhashable"]
    with pytest.raises(ValueError) as caught:
        validate_food_samples(records, "sampled_catalog")
    message = str(caught.value)
    assert "request_id=train-1:" in message
    assert "request_id=train-2:" in message
    assert "request_id=<unknown>:" in message or "request_id=['unhashable']:" in message


def test_non_dict_label_and_non_list_labels_rejected():
    records = valid_records()
    records[0]["labels"] = [1, 0]
    records[1]["labels"] = "nope"
    with pytest.raises(ValueError, match=r"request_id=train-1: .*\n?.*request_id=train-2:"):
        validate_food_samples(records, "sampled_catalog")


def test_bad_zip_is_value_error_naming_path(tmp_path):
    path = tmp_path / "broken.zip"
    path.write_bytes(b"this is not a zip archive")
    with pytest.raises(ValueError, match="broken.zip"):
        read_food_samples(path, "samples.jsonl", "0" * 64)


def test_corrupt_zip_member_is_value_error(tmp_path):
    path, _ = write_samples_zip(tmp_path, valid_records())
    data = bytearray(path.read_bytes())
    data[50] ^= 0xFF  # inside the stored member payload -> CRC mismatch
    path.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="samples.zip"):
        read_food_samples(path, "samples.jsonl", "0" * 64)
