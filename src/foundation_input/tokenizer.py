"""Semantic field tokenization for profile and ordered event observations."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .types import CanonicalSample, KeyValueToken, TokenRecord
from .vocab import VocabularyArtifacts

DEFAULT_CONTEXT_ALLOWLIST: dict[str, tuple[str, ...]] = {
    "Advertisement": (
        "product",
        "industry_main",
        "industry_sub",
        "caption",
        "ocr_text",
        "asr_text",
    ),
    "Live Streaming": ("live_category", "live_title", "is_shop_live", "live_cover_content"),
    "Video Browsing": ("caption", "photo_type", "duration", "ocr", "asr"),
    "E-commerce": (
        "product_source",
        "product_id",
        "item_title",
        "item_desc",
        "item_price",
        "external_item_title",
        "category_level1_name",
        "category_level2_name",
        "category_level3_name",
        "industry_level1_name",
        "industry_level2_name",
        "industry_level3_name",
    ),
    "Customer Service": ("ticket_category", "product_name", "item_price", "product_category_info"),
    "Search Behavior": (),
}


@dataclass(frozen=True)
class TokenizerConfig:
    context_allowlist: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: dict(DEFAULT_CONTEXT_ALLOWLIST)
    )
    max_text_chars: int = 256
    include_action_content: bool = False
    numeric_fields: frozenset[tuple[str, str]] = field(
        default_factory=lambda: DEFAULT_NUMERIC_FIELDS
    )
    numeric_transform_policy: dict[tuple[str, str], str] = field(default_factory=dict)
    time_gap_boundaries_seconds: tuple[float, ...] = (
        60.0,
        300.0,
        3600.0,
        21_600.0,
        86_400.0,
        604_800.0,
        2_592_000.0,
    )


_TEXT_FIELD_MARKERS = (
    "caption",
    "title",
    "desc",
    "description",
    "ocr",
    "asr",
    "query",
    "comment",
    "text",
    "content",
)


DEFAULT_NUMERIC_FIELDS: frozenset[tuple[str, str]] = frozenset(
    {
        ("tabular", "profile.active_days_30d"),
        ("tabular", "profile.event_count_30d"),
        ("tabular", "profile.video_browsing_event_count_30d"),
        ("tabular", "profile.live_streaming_event_count_30d"),
        ("tabular", "profile.e_commerce_event_count_30d"),
        ("tabular", "profile.advertisement_event_count_30d"),
        ("tabular", "profile.search_behavior_event_count_30d"),
        ("tabular", "profile.customer_service_event_count_30d"),
        ("sequence", "context.duration"),
        ("sequence", "context.item_price"),
        ("sequence", "action.comment_cnt"),
        ("sequence", "action.conversion_cnt"),
        ("sequence", "action.play_count"),
        ("sequence", "action.play_duration"),
        ("sequence", "action.played_loop_cnt"),
        ("sequence", "action.show_cnt"),
        ("sequence", "action.click_cnt"),
        ("sequence", "action.item_cnt"),
        ("sequence", "action.order_amt"),
    }
)


def _modality(
    key: str,
    value: Any,
    *,
    source: str = "sequence",
    numeric_fields: frozenset[tuple[str, str]] = frozenset(),
) -> str:
    if value is None:
        return "numeric" if (source, key) in numeric_fields else "missing"
    if isinstance(value, bool):
        return "categorical"
    lowered = key.lower()
    if "interval" in lowered:
        return "categorical"
    if isinstance(value, (int, float)):
        return "numeric"
    if any(marker in lowered for marker in _TEXT_FIELD_MARKERS):
        return "text"
    return "categorical"


def _flatten_text_value(value: Any) -> str:
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                role = str(item.get("role", "unknown"))
                content = str(item.get("content", ""))
                parts.append(f"{role}: {content}")
            else:
                parts.append(str(item))
        return "\n".join(parts)
    if isinstance(value, dict):
        return "\n".join(f"{key}: {value[key]}" for key in sorted(value))
    return str(value)


def _normalize_value(value: Any, modality: str, max_text_chars: int) -> Any:
    if modality == "text" and value is not None:
        text = _flatten_text_value(value)
        if len(text) <= max_text_chars:
            return text
        head = max_text_chars // 2
        tail = max_text_chars - head - 5
        return f"{text[:head]} ... {text[-tail:]}"
    return value


def _append_token(
    output: list[KeyValueToken],
    *,
    key: str,
    value: Any,
    source: str,
    timestamp: datetime | None,
    event_group_id: str | None,
    max_text_chars: int,
    numeric_fields: frozenset[tuple[str, str]],
) -> None:
    modality = _modality(key, value, source=source, numeric_fields=numeric_fields)
    output.append(
        KeyValueToken(
            key=key,
            value=_normalize_value(value, modality, max_text_chars),
            modality=modality,
            source=source,
            timestamp=timestamp,
            event_group_id=event_group_id,
        )
    )


def semantic_tokens(sample: CanonicalSample, config: TokenizerConfig) -> list[KeyValueToken]:
    """Convert a canonical sample to deterministic, unencoded field tokens."""

    tokens: list[KeyValueToken] = []
    _append_token(
        tokens,
        key="user_id",
        value=sample.user_id,
        source="tabular",
        timestamp=None,
        event_group_id=None,
        max_text_chars=config.max_text_chars,
        numeric_fields=config.numeric_fields,
    )
    for key in sorted(sample.profile):
        if key == "as_of_time":
            continue
        _append_token(
            tokens,
            key=f"profile.{key}",
            value=sample.profile[key],
            source="tabular",
            timestamp=None,
            event_group_id=None,
            max_text_chars=config.max_text_chars,
            numeric_fields=config.numeric_fields,
        )

    for event in sample.events:
        _append_token(
            tokens,
            key="event.scenario",
            value=event.scenario,
            source="sequence",
            timestamp=event.event_time,
            event_group_id=event.source_record_id,
            max_text_chars=config.max_text_chars,
            numeric_fields=config.numeric_fields,
        )
        _append_token(
            tokens,
            key="event.event_type",
            value=event.event_type,
            source="sequence",
            timestamp=event.event_time,
            event_group_id=event.source_record_id,
            max_text_chars=config.max_text_chars,
            numeric_fields=config.numeric_fields,
        )
        allowlist = config.context_allowlist.get(event.scenario or "", ())
        for key in sorted(set(event.context).intersection(allowlist)):
            _append_token(
                tokens,
                key=f"context.{key}",
                value=event.context[key],
                source="sequence",
                timestamp=event.event_time,
                event_group_id=event.source_record_id,
                max_text_chars=config.max_text_chars,
                numeric_fields=config.numeric_fields,
            )
        for key in sorted(event.action):
            if key == "content" and not config.include_action_content:
                continue
            _append_token(
                tokens,
                key=f"action.{key}",
                value=event.action[key],
                source="sequence",
                timestamp=event.event_time,
                event_group_id=event.source_record_id,
                max_text_chars=config.max_text_chars,
                numeric_fields=config.numeric_fields,
            )
    return tokens


def _naive(value: datetime) -> datetime:
    return value.replace(tzinfo=None) if value.tzinfo is not None else value


def _time_gap_id(
    previous: datetime | None, current: datetime, boundaries: tuple[float, ...]
) -> int:
    if previous is None:
        return 0
    seconds = max(0.0, (_naive(current) - _naive(previous)).total_seconds())
    return bisect_right(boundaries, seconds)


def tokenize_sample(
    sample: CanonicalSample,
    artifacts: VocabularyArtifacts,
    config: TokenizerConfig,
) -> list[TokenRecord]:
    if artifacts.dataset_name != sample.dataset_name:
        raise ValueError(
            f"Vocabulary dataset {artifacts.dataset_name!r} does not match "
            f"sample dataset {sample.dataset_name!r}"
        )
    artifacts.validate_tokenizer_config(config)
    if artifacts.include_action_content != config.include_action_content:
        raise ValueError(
            "Tokenizer action.content policy does not match vocabulary metadata: "
            f"vocabulary={artifacts.include_action_content}, "
            f"tokenizer={config.include_action_content}"
        )
    records: list[TokenRecord] = []
    event_positions: dict[str, int] = {}
    previous_event_time: datetime | None = None
    last_group: str | None = None
    for token in semantic_tokens(sample, config):
        if token.source == "tabular":
            position_id = 0
            time_gap_id = 0
        else:
            group = token.event_group_id or "<missing-event-group>"
            if group not in event_positions:
                event_positions[group] = len(event_positions) + 1
                if sample.history_order_known:
                    time_gap_id = _time_gap_id(
                        previous_event_time,
                        token.timestamp or sample.cutoff_time,
                        config.time_gap_boundaries_seconds,
                    )
                    previous_event_time = token.timestamp
                else:
                    time_gap_id = 0
                last_group = group
            else:
                time_gap_id = records[-1].time_gap_id if records else 0
            position_id = event_positions[group] if sample.history_order_known else 0
            if group != last_group:
                last_group = group
        records.append(
            TokenRecord(
                token=token,
                value_id=artifacts.encode_value(token),
                key_id=artifacts.encode_key(token),
                modality_id=artifacts.encode_modality(token),
                source_id=artifacts.encode_source(token),
                position_id=position_id,
                time_gap_id=time_gap_id,
                numeric_value=(
                    artifacts.normalize_numeric(token)[0]
                    if token.modality == "numeric"
                    else 0.0
                ),
                numeric_missing=(
                    artifacts.normalize_numeric(token)[1]
                    if token.modality == "numeric"
                    else False
                ),
            )
        )
    return records
