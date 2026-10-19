"""Temporal, as-of, and value validation for canonical pretraining samples."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

from .types import CanonicalSample


class TemporalValidationError(ValueError):
    """Raised when a sample violates the pretraining temporal contract."""


def _comparable(value: datetime) -> datetime:
    """Compare timestamps without inventing a timezone for source data."""

    return value.replace(tzinfo=None) if value.tzinfo is not None else value


def _coordinate_at_or_before(
    value: datetime,
    value_inclusive: bool,
    cutoff: datetime,
    cutoff_inclusive: bool,
) -> bool:
    """Compare the v2 eligibility coordinate, not timestamps alone."""

    return (_comparable(value), int(value_inclusive)) <= (
        _comparable(cutoff),
        int(cutoff_inclusive),
    )


def _finite(value: Any, path: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise TemporalValidationError(f"{path} must be finite, got {value!r}")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _finite(item, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _finite(item, f"{path}[{index}]")


def _profile_as_of(profile: dict[str, Any]) -> datetime | None:
    value = profile.get("as_of_time")
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as error:
            raise TemporalValidationError(
                f"profile.as_of_time is not a valid ISO timestamp: {value!r}"
            ) from error
    raise TemporalValidationError(
        f"profile.as_of_time has unsupported type: {type(value).__name__}"
    )


def validate_sample(sample: CanonicalSample) -> CanonicalSample:
    """Validate and return one canonical sample without mutating it."""

    seen_event_ids: set[str] = set()
    previous_coordinate: tuple[datetime, int] | None = None
    cutoff = _comparable(sample.cutoff_time)
    for index, event in enumerate(sample.events):
        if event.event_id in seen_event_ids:
            raise TemporalValidationError(
                f"{sample.sample_id}: duplicate event_id {event.event_id!r}"
            )
        seen_event_ids.add(event.event_id)
        if event.user_id is not None and event.user_id != sample.user_id:
            raise TemporalValidationError(
                f"{sample.sample_id}: event {event.event_id!r} belongs to user "
                f"{event.user_id!r}, expected {sample.user_id!r}"
            )
        event_time = _comparable(event.eligibility_time)
        if not _coordinate_at_or_before(
            event_time,
            event.eligibility_inclusive,
            cutoff,
            sample.cutoff_inclusive,
        ):
            raise TemporalValidationError(
                f"{sample.sample_id}: event {event.event_id!r} is after cutoff"
            )
        if event.time_precision not in {"day", "second"}:
            raise TemporalValidationError(
                f"{sample.sample_id}: unsupported time_precision {event.time_precision!r}"
            )
        coordinate = (event_time, int(event.eligibility_inclusive))
        if previous_coordinate is not None and coordinate < previous_coordinate:
            raise TemporalValidationError(
                f"{sample.sample_id}: events are not ordered at index {index}"
            )
        previous_coordinate = coordinate
        _finite(event.context, f"event[{event.event_id}].context")
        _finite(event.action, f"event[{event.event_id}].action")

    _finite(sample.profile, f"{sample.sample_id}.profile")
    profile_as_of = _profile_as_of(sample.profile)
    if profile_as_of is not None and _comparable(profile_as_of) > cutoff:
        raise TemporalValidationError(f"{sample.sample_id}: profile.as_of_time is after cutoff")
    return sample
