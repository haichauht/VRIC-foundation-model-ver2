from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.foundation_input.types import CanonicalSample, EventRecord
from src.foundation_input.validation import TemporalValidationError, validate_sample

from .fixtures.samples import sample_mapping


def _sample(
    *, events: tuple[EventRecord, ...] | None = None, profile: dict | None = None
) -> CanonicalSample:
    raw = sample_mapping("omnibehavior_v2")
    first = EventRecord(**raw["events"][0])
    return CanonicalSample(
        **{
            **raw,
            "events": events if events is not None else (first,),
            "profile": profile if profile is not None else raw["profile"],
        }
    )


def test_validate_sample_rejects_event_after_cutoff() -> None:
    raw = sample_mapping("omnibehavior_v2")["events"][0]
    event = EventRecord(
        **{
            **raw,
            "eligibility_time": datetime(2025, 9, 20, tzinfo=timezone.utc),
        }
    )

    with pytest.raises(TemporalValidationError, match="after cutoff"):
        validate_sample(_sample(events=(event,)))


def test_validate_sample_rejects_out_of_order_events() -> None:
    raw = sample_mapping("omnibehavior_v2")["events"][0]
    first = EventRecord(
        **{
            **raw,
            "eligibility_time": datetime(2025, 9, 18, tzinfo=timezone.utc),
        }
    )
    later = EventRecord(
        **{
            **raw,
            "event_id": "user-001:1:0",
            "source_record_id": "user-001:1",
            "source_record_index": 1,
            "eligibility_time": datetime(2025, 9, 19, tzinfo=timezone.utc),
        }
    )

    with pytest.raises(TemporalValidationError, match="not ordered"):
        validate_sample(_sample(events=(later, first)))


def test_validate_sample_rejects_profile_after_cutoff() -> None:
    profile = {"as_of_time": "2025-09-20T00:00:00+00:00", "event_count_30d": 1}

    with pytest.raises(TemporalValidationError, match="profile.as_of_time"):
        validate_sample(_sample(profile=profile))


def test_validate_sample_rejects_nonfinite_numeric_values() -> None:
    with pytest.raises(TemporalValidationError, match="finite"):
        validate_sample(_sample(profile={"event_count_30d": float("nan")}))


def test_validate_sample_accepts_event_at_inclusive_cutoff() -> None:
    raw = sample_mapping("omnibehavior_v2")["events"][0]
    event = EventRecord(
        **{
            **raw,
            "eligibility_time": datetime(2025, 9, 19, 2, 37, 25, tzinfo=timezone.utc),
            "eligibility_inclusive": True,
        }
    )
    sample = _sample(events=(event,))
    assert validate_sample(sample) is sample
