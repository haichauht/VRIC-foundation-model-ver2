"""Streaming dataset and padded batch collation for foundation pretraining."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

from .loader import iter_normalized_samples
from .tokenizer import TokenizerConfig, tokenize_sample
from .types import SPECIAL_TOKEN_IDS, EncodedSample, FoundationBatch, TokenRecord
from .vocab import VocabularyArtifacts


@dataclass(frozen=True)
class DatasetConfig:
    max_event_count: int = 30
    max_tokens: int = 256
    next_action_prediction: bool = False


@dataclass(frozen=True)
class TruncationResult:
    records: tuple[TokenRecord, ...]
    rejected_recent_group: bool


def _truncate_records(records: list[TokenRecord], config: DatasetConfig) -> TruncationResult:
    tabular = [record for record in records if record.token.source == "tabular"]
    sequence = [record for record in records if record.token.source == "sequence"]
    groups: list[list[TokenRecord]] = []
    for record in sequence:
        if not groups or groups[-1][0].token.event_group_id != record.token.event_group_id:
            groups.append([])
        groups[-1].append(record)
    kept_groups = groups[-config.max_event_count :] if config.max_event_count > 0 else []
    record_budget = max(0, config.max_tokens - 2)
    if record_budget == 0:
        return TruncationResult((), rejected_recent_group=bool(kept_groups))

    # Complete recent event groups have priority over tabular context. Reserving a
    # tabular slot first can incorrectly reject a group that fits the full budget.
    remaining_event_budget = record_budget
    selected_groups: list[list[TokenRecord]] = []
    for group in reversed(kept_groups):
        if len(group) <= remaining_event_budget:
            selected_groups.append(group)
            remaining_event_budget -= len(group)
        elif not selected_groups:
            return TruncationResult(
                records=tuple(tabular[:record_budget]),
                rejected_recent_group=True,
            )
        else:
            break
    selected_groups.reverse()
    event_records = [record for group in selected_groups for record in group]
    tabular_budget = record_budget - len(event_records)
    selected = tabular[:tabular_budget]
    selected.extend(event_records)
    return TruncationResult(tuple(selected), rejected_recent_group=False)


def encode_sample(
    sample,
    artifacts: VocabularyArtifacts,
    tokenizer_config: TokenizerConfig,
    dataset_config: DatasetConfig,
) -> EncodedSample:
    records = tokenize_sample(sample, artifacts, tokenizer_config)
    user_id_record = next(
        (record for record in records if record.token.key == "user_id"), None
    )
    truncation = _truncate_records(records, dataset_config)
    if truncation.rejected_recent_group:
        raise ValueError(
            f"sample_id={sample.sample_id}: most recent complete event group "
            "does not fit within max_tokens; refusing to split the group"
        )
    selected = truncation.records
    next_action_records: tuple[TokenRecord, ...] = ()
    if (
        dataset_config.next_action_prediction
        and sample.history_order_known
        and not sample.history_group_truncated
    ):
        sequence_groups: list[list[TokenRecord]] = []
        for record in selected:
            if record.token.source != "sequence":
                continue
            if not sequence_groups or sequence_groups[-1][0].token.event_group_id != record.token.event_group_id:
                sequence_groups.append([])
            sequence_groups[-1].append(record)
        if len(sequence_groups) >= 2:
            target_group = sequence_groups[-1]
            target_ids = {id(record) for record in target_group}
            selected = tuple(record for record in selected if id(record) not in target_ids)
            next_action_records = tuple(target_group)
    safe_user_id_value_id = (
        None
        if user_id_record is None
        or user_id_record.value_id == SPECIAL_TOKEN_IDS["UNK"]
        else user_id_record.value_id
    )
    return EncodedSample(
        sample_id=sample.sample_id,
        dataset_name=sample.dataset_name,
        user_id=sample.user_id,
        records=selected,
        user_id_value_id=safe_user_id_value_id,
        next_action_records=next_action_records,
    )


class FoundationPretrainDataset:
    """Lazy Parquet-backed dataset; samples are encoded only when iterated."""

    def __init__(
        self,
        dataset_root: str,
        split: str,
        artifacts: VocabularyArtifacts,
        tokenizer_config: TokenizerConfig | None = None,
        dataset_config: DatasetConfig | None = None,
        sample_filter: Callable[[Any], bool] | None = None,
        validate: bool = False,
    ) -> None:
        self.dataset_root = dataset_root
        self.split = split
        self.artifacts = artifacts
        self.sample_filter = sample_filter
        self.validate = validate
        if tokenizer_config is None:
            contract = artifacts.tokenizer_contract
            if not contract:
                raise ValueError("Vocabulary artifact lacks tokenizer contract")
            tokenizer_config = TokenizerConfig(
                context_allowlist={
                    str(key): tuple(value)
                    for key, value in contract["context_allowlist"].items()
                },
                max_text_chars=int(contract["max_text_chars"]),
                include_action_content=bool(contract["include_action_content"]),
                numeric_fields=frozenset(
                    (str(source), str(key)) for source, key in contract["numeric_fields"]
                ),
                time_gap_boundaries_seconds=tuple(
                    float(value) for value in contract["time_gap_boundaries_seconds"]
                ),
            )
        self.tokenizer_config = tokenizer_config
        self.dataset_config = dataset_config or DatasetConfig()
    def __iter__(self) -> Iterator[EncodedSample]:
        for sample in iter_normalized_samples(self.dataset_root, self.split, validate=self.validate):
            if self.sample_filter is not None and not self.sample_filter(sample):
                continue
            try:
                encoded = encode_sample(
                    sample, self.artifacts, self.tokenizer_config, self.dataset_config
                )
            except ValueError:
                continue
            yield encoded

    def __getitem__(self, index: int) -> EncodedSample:
        if index < 0:
            raise IndexError("Negative indices are not supported by the streaming dataset")
        for current, encoded in enumerate(self):
            if current == index:
                return encoded
        raise IndexError(index)


def _default_tensor_backend() -> Any:
    try:
        import torch
    except Exception as error:  # torch may be absent or have unavailable native DLLs
        raise RuntimeError(
            "FoundationBatch collation requires a working PyTorch installation"
        ) from error
    return torch


def _stable_user_id(dataset_name: str, user_id: str) -> int:
    digest = hashlib.sha256(f"{dataset_name}\0{user_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def foundation_collate(
    samples: list[EncodedSample], *, tensor_backend: Any | None = None
) -> FoundationBatch:
    if not samples:
        raise ValueError("Cannot collate an empty sample list")
    dataset_names = {sample.dataset_name for sample in samples}
    if len(dataset_names) != 1:
        raise ValueError(
            "Mixed-source batches are disabled; collate one dataset namespace at a time"
        )
    backend = tensor_backend or _default_tensor_backend()

    rows: list[dict[str, list[int | bool]]] = []
    numeric_rows: list[list[float]] = []
    numeric_missing_rows: list[list[bool]] = []
    next_action_type_ids: list[int] = []
    next_action_group_ids: list[int] = []
    next_action_valid_mask: list[bool] = []
    for sample in samples:
        next_action = sample.next_action_records
        group_to_id: dict[str, int] = {}
        for record in (*sample.records, *next_action):
            if record.token.source != "sequence":
                continue
            group = record.token.event_group_id or "<missing-event-group>"
            group_to_id.setdefault(group, len(group_to_id) + 1)
        target_record_ids = {id(record) for record in next_action}
        tabular = [
            record
            for record in sample.records
            if record.token.source == "tabular" and id(record) not in target_record_ids
        ]
        sequence = [
            record
            for record in sample.records
            if record.token.source == "sequence" and id(record) not in target_record_ids
        ]
        input_ids = [1]
        key_ids = [0]
        modality_ids = [0]
        source_ids = [0]
        position_ids = [0]
        time_gap_ids = [0]
        group_ids = [0]
        numeric_values = [0.0]
        numeric_missing = [False]
        input_ids.extend(record.value_id for record in tabular)
        key_ids.extend(record.key_id for record in tabular)
        modality_ids.extend(record.modality_id for record in tabular)
        source_ids.extend(record.source_id for record in tabular)
        position_ids.extend(record.position_id for record in tabular)
        time_gap_ids.extend(record.time_gap_id for record in tabular)
        group_ids.extend(0 for _ in tabular)
        numeric_values.extend(record.numeric_value for record in tabular)
        numeric_missing.extend(record.numeric_missing for record in tabular)
        input_ids.append(2)
        key_ids.append(0)
        modality_ids.append(0)
        source_ids.append(0)
        position_ids.append(0)
        time_gap_ids.append(0)
        group_ids.append(0)
        numeric_values.append(0.0)
        numeric_missing.append(False)
        for record in sequence:
            input_ids.append(record.value_id)
            key_ids.append(record.key_id)
            modality_ids.append(record.modality_id)
            source_ids.append(record.source_id)
            position_ids.append(record.position_id)
            time_gap_ids.append(record.time_gap_id)
            group_ids.append(
                group_to_id[record.token.event_group_id or "<missing-event-group>"]
            )
            numeric_values.append(record.numeric_value)
            numeric_missing.append(record.numeric_missing)
        rows.append(
            {
                "input_ids": input_ids,
                "key_ids": key_ids,
                "modality_ids": modality_ids,
                "source_ids": source_ids,
                "position_ids": position_ids,
                "time_gap_ids": time_gap_ids,
                "group_ids": group_ids,
            }
        )
        numeric_rows.append(numeric_values)
        numeric_missing_rows.append(numeric_missing)
        target_type = next(
            (
                record.value_id
                for record in next_action
                if record.token.key in {"event.event_type", "event_type"}
            ),
            0,
        )
        next_action_type_ids.append(target_type)
        next_action_group_ids.append(
            group_to_id[next_action[0].token.event_group_id or "<missing-event-group>"]
            if next_action
            else 0
        )
        next_action_valid_mask.append(bool(next_action))

    max_length = max(len(row["input_ids"]) for row in rows)
    padded: dict[str, list[list[int | bool]]] = {}
    for field in (
        "input_ids",
        "key_ids",
        "modality_ids",
        "source_ids",
        "position_ids",
        "time_gap_ids",
        "group_ids",
    ):
        padded[field] = [row[field] + [0] * (max_length - len(row[field])) for row in rows]
    padded["attention_mask"] = [
        [True] * len(row["input_ids"]) + [False] * (max_length - len(row["input_ids"]))
        for row in rows
    ]
    padded_numeric = [
        row + [0.0] * (max_length - len(row)) for row in numeric_rows
    ]
    padded_numeric_missing = [
        row + [False] * (max_length - len(row)) for row in numeric_missing_rows
    ]
    tensor = backend.tensor
    return FoundationBatch(
        input_ids=tensor(padded["input_ids"], dtype=backend.long),
        key_ids=tensor(padded["key_ids"], dtype=backend.long),
        modality_ids=tensor(padded["modality_ids"], dtype=backend.long),
        source_ids=tensor(padded["source_ids"], dtype=backend.long),
        position_ids=tensor(padded["position_ids"], dtype=backend.long),
        time_gap_ids=tensor(padded["time_gap_ids"], dtype=backend.long),
        attention_mask=tensor(padded["attention_mask"], dtype=backend.bool),
        numeric_values=tensor(padded_numeric, dtype=backend.float32),
        numeric_missing_mask=tensor(padded_numeric_missing, dtype=backend.bool),
        user_ids=tensor(
            [
                [
                    samples[index].user_id_value_id
                    if samples[index].user_id_value_id is not None
                    else _stable_user_id(samples[index].dataset_name, samples[index].user_id)
                ]
                for index in range(len(samples))
            ],
            dtype=backend.long,
        ).reshape(len(samples)),
        targets={
            "event_group_ids": tensor(padded["group_ids"], dtype=backend.long),
            "next_action_type_ids": tensor(next_action_type_ids, dtype=backend.long),
            "next_token": tensor(next_action_type_ids, dtype=backend.long),
            "next_action_group_ids": tensor(next_action_group_ids, dtype=backend.long),
            "next_action_valid_mask": tensor(next_action_valid_mask, dtype=backend.bool),
            "next_token_valid_mask": tensor(next_action_valid_mask, dtype=backend.bool),
        },
    )
