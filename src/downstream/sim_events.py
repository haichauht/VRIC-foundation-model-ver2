"""Public simulator events (Food history, Ads events) as pretrain-identical event records.

The frozen foundation model only transfers if a downstream reader turns an event into exactly
the tokens the pretrain loader produced for it. The pretrain package stores each session as one
source record; the downstream packages carry no session id, so sessions are rebuilt from the
funnel: every session starts with a session-start event type, and events before the first
start (an Ads window cut mid-session) form one leading group. Context and action are split back
out of the flat ``attributes``, and times become naive UTC like the pretrain package.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from src.foundation_input.types import CanonicalSample, EventRecord

SIM_DATASET_NAME = "sim_v1"  # source_dataset of every sim package == pretrain vocabulary name
SCENARIOS: dict[str, str] = {"ride": "Ride", "food": "Food", "ads": "Advertisement"}
NUMERIC_ACTION_FIELDS: tuple[str, ...] = (
    "fare",
    "distance_km",
    "wait_minutes",
    "order_amount",
    "delivery_distance_km",
)
ADS_ACTION_TYPES: dict[str, str] = {"ad_impression": "impression", "ad_click": "click"}
SESSION_START_TYPES: frozenset[str] = frozenset({"search_destination", "search", "ad_impression"})


@dataclass(frozen=True)
class SimEvent:
    """One public sim event; ``event_time`` is timezone-aware."""

    event_id: str
    user_id: str
    event_time: datetime
    service: str
    event_type: str
    entity_id: str | None
    attributes: dict

    def __post_init__(self) -> None:
        if self.event_time.tzinfo is None:
            raise ValueError(f"event_id={self.event_id!r}: event_time must be timezone-aware")


def _aware(value: Any, *, field: str) -> datetime:
    moment = value
    if isinstance(value, str):
        moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if not isinstance(moment, datetime) or moment.tzinfo is None:
        raise ValueError(f"{field} must be a timezone-aware timestamp, got {value!r}")
    return moment


def _naive_utc(moment: datetime) -> datetime:
    return moment.astimezone(timezone.utc).replace(tzinfo=None)


def food_history_events(record: dict) -> list[SimEvent]:
    """Events of a Food ``samples.jsonl`` row's ``input.history`` (``event_time`` ISO +07:00)."""
    inp = record["input"]
    user_id = str(inp["user_id"])
    events = []
    for item in inp["history"]:
        event_id = str(item["event_id"])
        events.append(
            SimEvent(
                event_id=event_id,
                user_id=user_id,
                event_time=_aware(item["event_time"], field=f"event_id={event_id!r}.event_time"),
                service=str(item["service"]),
                event_type=str(item["event_type"]),
                entity_id=None if item["entity_id"] is None else str(item["entity_id"]),
                attributes=dict(item["attributes"]),
            )
        )
    return events


def ads_window_events(rows: Sequence[dict]) -> list[SimEvent]:
    """Events of ``ads/events.parquet`` rows (``timestamp`` epoch seconds, ``attributes_json``)."""
    events = []
    for row in rows:
        attributes = json.loads(row["attributes_json"])
        if not isinstance(attributes, dict):
            raise TypeError(f"event_id={row['event_id']!r}: attributes_json must be an object")
        events.append(
            SimEvent(
                event_id=str(row["event_id"]),
                user_id=str(row["user_id"]),
                event_time=datetime.fromtimestamp(int(row["timestamp"]), timezone.utc),
                service=str(row["service"]),
                event_type=str(row["event_type"]),
                entity_id=None if row["entity_id"] is None else str(row["entity_id"]),
                attributes=attributes,
            )
        )
    return events


def _scenario(event: SimEvent) -> str:
    try:
        return SCENARIOS[event.service]
    except KeyError:
        raise ValueError(
            f"event_id={event.event_id!r}: unknown service {event.service!r}"
        ) from None


def sim_event_records(events: Sequence[SimEvent]) -> tuple[EventRecord, ...]:
    """Event records grouped into sessions the way the pretrain package groups them."""
    groups: list[list[SimEvent]] = []
    for event in events:
        if not groups or event.event_type in SESSION_START_TYPES:
            groups.append([])
        groups[-1].append(event)
    records = []
    for group_index, group in enumerate(groups):
        for action_index, event in enumerate(group):
            moment = _naive_utc(event.event_time)
            action = {"type": ADS_ACTION_TYPES.get(event.event_type, event.event_type)}
            action.update(
                (key, event.attributes[key])
                for key in NUMERIC_ACTION_FIELDS
                if key in event.attributes
            )
            records.append(
                EventRecord(
                    event_id=event.event_id,
                    source_record_id=group[0].event_id,
                    source_record_index=group_index,
                    action_index=action_index,
                    source_action_count=len(group),
                    event_time=moment,
                    time_precision="second",
                    eligibility_time=moment,
                    eligibility_inclusive=True,
                    scenario=_scenario(event),
                    event_type=event.event_type,
                    context={
                        key: value
                        for key, value in event.attributes.items()
                        if key not in NUMERIC_ACTION_FIELDS
                    },
                    action=action,
                    user_id=event.user_id,
                )
            )
    return tuple(records)


def sim_canonical_sample(
    *,
    sample_id: str,
    dataset_name: str,
    split: str,
    user_id: str,
    cutoff_time: datetime,
    profile: dict,
    events: Sequence[SimEvent],
    history_truncated: bool,
    history_group_truncated: bool,
) -> CanonicalSample:
    """A canonical sample whose history is strictly before ``cutoff_time`` (naive UTC)."""
    cutoff = _naive_utc(_aware(cutoff_time, field=f"sample_id={sample_id!r}: cutoff_time"))
    profile = dict(profile)
    if profile.get("as_of_time") is not None:
        as_of = _aware(profile["as_of_time"], field=f"sample_id={sample_id!r}: profile.as_of_time")
        profile["as_of_time"] = _naive_utc(as_of).isoformat()
    return CanonicalSample(
        sample_id=sample_id,
        dataset_name=dataset_name,
        split=split,
        user_id=user_id,
        cutoff_time=cutoff,
        cutoff_inclusive=False,
        history_order_known=True,
        history_truncated=history_truncated,
        history_group_truncated=history_group_truncated,
        profile=profile,
        events=sim_event_records(events),
    )
