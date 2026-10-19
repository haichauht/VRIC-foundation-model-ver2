"""Raw downstream handoff types; labels never belong to model features."""

from dataclasses import dataclass, field
from datetime import datetime
from math import isfinite
from typing import Any
from zoneinfo import ZoneInfo


class ContractError(ValueError):
    """The handoff cannot be consumed without changing its meaning."""


def _require(condition: bool, identity: str, rule: str) -> None:
    if not condition:
        raise ContractError(f"{identity}: {rule}")


def _identifier(value: str, identity: str) -> None:
    _require(isinstance(value, str) and bool(value.strip()), identity, "non-empty string required")


def _time(value: datetime, identity: str) -> None:
    _require(
        isinstance(value, datetime) and value.tzinfo is not None
        and value.utcoffset() is not None,
        identity,
        "timezone-aware datetime required; do not infer source timezone",
    )


@dataclass(frozen=True)
class DatasetMetadata:
    dataset_name: str
    data_version: str
    task: str
    item_kind: str
    timezone: str
    candidate_policy: str
    availability_policy: str
    schema_version: int = 1


@dataclass(frozen=True)
class Query:
    request_id: str
    user_id: str
    cutoff_time: datetime
    split: str
    context: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Candidate:
    request_id: str
    candidate_id: str
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Label:
    request_id: str
    candidate_id: str
    target: int
    available_at: datetime


@dataclass(frozen=True)
class Interaction:
    event_id: str
    user_id: str
    entity_id: str
    event_type: str
    event_time: datetime
    available_at: datetime
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BaselineDataset:
    metadata: DatasetMetadata
    queries: tuple[Query, ...]
    candidates: tuple[Candidate, ...]
    labels: tuple[Label, ...]
    history: tuple[Interaction, ...] = ()

    def validate(self) -> None:
        """Validate complete labeled handoff without repairing or dropping rows."""
        meta = self.metadata
        for name in ("dataset_name", "data_version", "candidate_policy", "availability_policy"):
            _identifier(getattr(meta, name), f"metadata.{name}")
        _require(type(meta.schema_version) is int and meta.schema_version == 1,
                 "metadata", "schema_version must be 1")
        allowed = {"food_recommendation": {"food", "merchant"}, "ad_optimization": {"ad"}}
        _require(meta.task in allowed, "metadata", "unknown task")
        _require(meta.item_kind in allowed[meta.task], "metadata", "task/item_kind mismatch")
        try:
            ZoneInfo(meta.timezone)
        except (KeyError, ValueError, TypeError) as error:
            raise ContractError("metadata.timezone: valid IANA timezone required") from error
        queries: dict[str, Query] = {}
        for query in self.queries:
            identity = f"request={query.request_id}"
            _identifier(query.request_id, identity)
            _identifier(query.user_id, identity)
            _time(query.cutoff_time, identity)
            _require(query.split in {"train", "validation", "test"}, identity, "invalid split")
            _require(query.request_id not in queries, identity, "duplicate request")
            queries[query.request_id] = query
        _require(bool(queries), "dataset", "at least one query required")
        keys: set[tuple[str, str]] = set()
        covered: set[str] = set()
        for candidate in self.candidates:
            identity = f"request={candidate.request_id}, candidate={candidate.candidate_id}"
            _identifier(candidate.candidate_id, identity)
            _require(candidate.request_id in queries, identity, "unknown request")
            key = (candidate.request_id, candidate.candidate_id)
            _require(key not in keys, identity, "duplicate candidate")
            keys.add(key)
            covered.add(candidate.request_id)
        _require(covered == set(queries), "dataset", "every query requires candidates")
        label_keys: set[tuple[str, str]] = set()
        for label in self.labels:
            key = (label.request_id, label.candidate_id)
            identity = f"request={label.request_id}, candidate={label.candidate_id}"
            _require(key in keys, identity, "label has no candidate")
            _require(key not in label_keys, identity, "duplicate label")
            _require(type(label.target) is int and label.target in {0, 1}, identity,
                     "binary integer target required")
            _time(label.available_at, identity)
            _require(label.available_at >= queries[label.request_id].cutoff_time,
                     identity, "label availability precedes prediction")
            label_keys.add(key)
        _require(label_keys == keys, "dataset", "exactly one label per candidate required")
        event_ids: set[str] = set()
        for event in self.history:
            identity = f"event={event.event_id}"
            for value in (event.event_id, event.user_id, event.entity_id, event.event_type):
                _identifier(value, identity)
            _require(event.event_id not in event_ids, identity, "duplicate interaction")
            _time(event.event_time, identity)
            _time(event.available_at, identity)
            _require(event.available_at >= event.event_time, identity,
                     "availability precedes event")
            event_ids.add(event.event_id)


def eligible_history(history: tuple[Interaction, ...], cutoff: datetime) -> tuple[Interaction, ...]:
    """Exclude future and not-yet-observed interactions, including timestamp ties."""
    _time(cutoff, "history cutoff")
    return tuple(event for event in history
                 if event.event_time < cutoff and event.available_at < cutoff)


def validate_score(score: float, task: str) -> None:
    _require(task in {"food_recommendation", "ad_optimization"}, "score", "unknown task")
    _require(isinstance(score, (int, float)) and not isinstance(score, bool)
             and isfinite(score), "score", "finite numeric score required")
    if task == "ad_optimization":
        _require(0 <= score <= 1, "score", "CTR probability must be in [0, 1]")
