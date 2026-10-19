from __future__ import annotations

from datetime import datetime, timezone

from src.foundation_input.types import (
    SPECIAL_TOKEN_IDS,
    CanonicalSample,
    EventRecord,
    KeyValueToken,
)

from .fixtures.samples import sample_mapping


def _build_sample(dataset_name: str) -> CanonicalSample:
    raw = sample_mapping(dataset_name)
    event = EventRecord(**raw["events"][0])
    return CanonicalSample(**{**raw, "events": (event,)})


def test_current_pretrain_sources_share_canonical_sample_interface() -> None:
    meituan = _build_sample("meituan_trd_v2")
    omnibehavior = _build_sample("omnibehavior_v2")

    assert isinstance(meituan, CanonicalSample)
    assert isinstance(omnibehavior, CanonicalSample)
    assert meituan.user_id == omnibehavior.user_id
    assert meituan.dataset_name != omnibehavior.dataset_name
    assert meituan.events[0].source_record_id == "user-001:0"
    assert meituan.events[0].eligibility_time < meituan.cutoff_time


def test_key_value_token_keeps_event_group_provenance() -> None:
    token = KeyValueToken(
        key="event_type",
        value="click",
        modality="categorical",
        source="sequence",
        timestamp=datetime(2025, 9, 18, tzinfo=timezone.utc),
        event_group_id="user-001:0",
    )

    assert token.event_group_id == "user-001:0"
    assert token.source == "sequence"


def test_special_token_ids_are_stable() -> None:
    assert SPECIAL_TOKEN_IDS == {"PAD": 0, "CLS": 1, "SEP": 2, "MASK": 3, "UNK": 4}
