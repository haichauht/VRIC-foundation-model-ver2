"""Shared types for pretrain samples, field tokens, and model batches."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

SPECIAL_TOKEN_IDS: dict[str, int] = {
    "PAD": 0,
    "CLS": 1,
    "SEP": 2,
    "MASK": 3,
    "UNK": 4,
}


@dataclass(frozen=True)
class KeyValueToken:
    """A semantic field before vocabulary encoding."""

    key: str
    value: Any
    modality: str
    source: str
    timestamp: datetime | None
    event_group_id: str | None = None


@dataclass(frozen=True)
class TokenRecord:
    """A semantic token after vocabulary encoding, before sequence padding."""

    token: KeyValueToken
    value_id: int
    key_id: int
    modality_id: int
    source_id: int
    position_id: int
    time_gap_id: int
    numeric_value: float = 0.0
    numeric_missing: bool = False


@dataclass(frozen=True)
class EncodedSample:
    """A sample ready for collation, still carrying field provenance."""

    sample_id: str
    dataset_name: str
    user_id: str
    records: tuple[TokenRecord, ...]
    user_id_value_id: int | None = None
    next_action_records: tuple[TokenRecord, ...] = ()
    # Long-history summary (history_features.py); targets/mask only for pretraining samples.
    history_features: tuple[float, ...] | None = None
    history_targets: tuple[float, ...] | None = None
    history_target_mask: tuple[float, ...] | None = None


@dataclass(frozen=True)
class EventRecord:
    """One action observation reconstructed from the Parquet event table."""

    event_id: str
    source_record_id: str
    source_record_index: int
    action_index: int
    source_action_count: int
    event_time: datetime | None
    time_precision: str
    eligibility_time: datetime
    eligibility_inclusive: bool
    scenario: str | None
    event_type: str | None
    context: dict[str, Any] = field(default_factory=dict)
    action: dict[str, Any] = field(default_factory=dict)
    user_id: str | None = None


@dataclass(frozen=True)
class CanonicalSample:
    """Dataset-independent representation of one pretraining snapshot."""

    sample_id: str
    dataset_name: str
    split: str
    user_id: str
    cutoff_time: datetime
    cutoff_inclusive: bool
    history_order_known: bool
    history_truncated: bool
    history_group_truncated: bool
    profile: dict[str, Any]
    events: tuple[EventRecord, ...]
    history_start: int = 0
    history_stop: int = 0
    history_features: tuple[float, ...] | None = None
    history_targets: tuple[float, ...] | None = None
    history_target_mask: tuple[float, ...] | None = None


@dataclass
class FoundationBatch:
    """Tensor contract consumed by the shared foundation encoder."""

    input_ids: torch.LongTensor
    key_ids: torch.LongTensor
    modality_ids: torch.LongTensor
    source_ids: torch.LongTensor
    position_ids: torch.LongTensor
    time_gap_ids: torch.LongTensor
    attention_mask: torch.BoolTensor
    numeric_values: torch.FloatTensor
    numeric_missing_mask: torch.BoolTensor
    user_ids: torch.LongTensor
    targets: dict[str, torch.Tensor]
    history_features: torch.FloatTensor | None = None  # [B, F] with long_term_source "history"
