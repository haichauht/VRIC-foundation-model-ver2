"""Transformer encoder and public embedding API for foundation batches."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from src.foundation_input.types import FoundationBatch

from .adapters import AdapterRegistry, UserIdAdapter
from .pooling import SequencePooler


@dataclass(frozen=True)
class FoundationModelConfig:
    """Static dimensions and mask policy used to construct a foundation model."""

    d_model: int = 128
    num_heads: int = 4
    num_layers: int = 4
    dropout: float = 0.1
    max_length: int = 256
    value_vocab_size: int | None = None
    key_vocab_size: int | None = None
    modality_vocab_size: int | None = None
    source_vocab_size: int | None = None
    time_gap_vocab_size: int | None = None
    num_user_buckets: int = 50_000
    categorical_pair_buckets: int = 100_003
    sequence_source_id: int | None = None
    numeric_modality_id: int | None = None
    default_attention_mode: str = "bidirectional"
    long_term_injection: str = "none"
    # None keeps nn.Embedding's N(0, 1) init (old checkpoints, defaults); a float draws N(0, std).
    user_embedding_init_std: float | None = None
    # "user_id": long_term = per-ID table (UserIdAdapter); "history": long_term = MLP over the
    # long-history features of the batch (foundation_input/history_features.py).
    long_term_source: str = "user_id"
    history_feature_dim: int = 0
    # Output sizes of the pretraining history-target heads (order cuisines, ad-click categories).
    history_target_sizes: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.long_term_source not in {"user_id", "history"}:
            raise ValueError(
                f"long_term_source must be user_id or history, got {self.long_term_source!r}")
        if (type(self.history_feature_dim) is not int or self.history_feature_dim < 0
                or (self.long_term_source == "history") != (self.history_feature_dim > 0)):
            raise ValueError("history_feature_dim must be positive exactly when long_term_source is history")
        object.__setattr__(self, "history_target_sizes", tuple(self.history_target_sizes))
        if not all(type(size) is int and size >= 0 for size in self.history_target_sizes):
            raise ValueError("history_target_sizes must be nonnegative integers")
        if self.long_term_injection not in {"none", "cls"}:
            raise ValueError(
                f"long_term_injection must be none or cls, got {self.long_term_injection!r}"
            )
        std = self.user_embedding_init_std
        if std is not None and (
            isinstance(std, bool) or not isinstance(std, (int, float))
            or not math.isfinite(std) or std <= 0
        ):
            raise ValueError(
                f"user_embedding_init_std must be None or a positive finite number, got {std!r}"
            )
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {self.num_heads}")
        if self.d_model <= 0 or self.d_model % self.num_heads != 0:
            raise ValueError(
                f"d_model={self.d_model} must be positive and divisible by "
                f"num_heads={self.num_heads}"
            )
        if self.num_layers <= 0 or self.max_length <= 0:
            raise ValueError("num_layers and max_length must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0.0, 1.0)")
        if self.default_attention_mode not in {"bidirectional", "next_action"}:
            raise ValueError("default_attention_mode must be bidirectional or next_action")
        vocabulary_metadata = (
            "value_vocab_size",
            "key_vocab_size",
            "modality_vocab_size",
            "source_vocab_size",
            "time_gap_vocab_size",
        )
        routing_metadata = (
            "sequence_source_id",
            "numeric_modality_id",
        )
        for name in vocabulary_metadata + routing_metadata:
            if getattr(self, name) is None:
                raise ValueError(f"{name} must be supplied from vocabulary metadata")
        for name in vocabulary_metadata:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 <= self.sequence_source_id < self.source_vocab_size:
            raise ValueError(
                "sequence_source_id must be inside source vocabulary: "
                f"got sequence_source_id={self.sequence_source_id}, "
                f"source_vocab_size={self.source_vocab_size}"
            )
        if not 0 < self.numeric_modality_id < self.modality_vocab_size:
            raise ValueError(
                "numeric_modality_id must be inside modality vocabulary: "
                f"got numeric_modality_id={self.numeric_modality_id}, "
                f"modality_vocab_size={self.modality_vocab_size}"
            )
        for name in (
            "num_user_buckets",
            "categorical_pair_buckets",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass
class FoundationOutput:
    """Encoder output consumed by objectives and the public embedding API."""

    hidden_states: torch.Tensor
    long_term: torch.Tensor
    short_term: torch.Tensor
    padding_mask: torch.BoolTensor


def build_attention_mask(
    batch: FoundationBatch, attention_mode: str, *, sequence_source_id: int
) -> torch.BoolTensor:
    """Build per-example query/key blocking masks using True=blocked semantics."""

    if attention_mode not in {"bidirectional", "next_action"}:
        raise ValueError(
            "attention_mode must be bidirectional or next_action, "
            f"got {attention_mode!r}"
        )
    if batch.input_ids.ndim != 2:
        raise ValueError(f"input_ids must have shape [B, L], got {tuple(batch.input_ids.shape)}")
    expected = batch.input_ids.shape
    for name in ("source_ids", "position_ids"):
        value = getattr(batch, name)
        if value.shape != expected:
            raise ValueError(f"{name} must have shape {tuple(expected)}, got {tuple(value.shape)}")
    if batch.attention_mask.shape != expected or batch.attention_mask.dtype != torch.bool:
        raise ValueError(
            "attention_mask must be bool with shape "
            f"{tuple(expected)}, got shape={tuple(batch.attention_mask.shape)}, "
            f"dtype={batch.attention_mask.dtype}"
        )

    batch_size, sequence_length = expected
    blocked = torch.zeros(
        (batch_size, sequence_length, sequence_length),
        dtype=torch.bool,
        device=batch.input_ids.device,
    )
    if attention_mode == "bidirectional":
        return blocked

    valid = batch.attention_mask
    is_event = valid & batch.source_ids.eq(sequence_source_id)
    future_event = batch.position_ids.unsqueeze(1) > batch.position_ids.unsqueeze(2)
    query_is_event = is_event.unsqueeze(2)
    key_is_event = is_event.unsqueeze(1)
    blocked |= query_is_event & key_is_event & future_event
    blocked |= (~is_event).unsqueeze(2) & key_is_event
    return blocked


def _validate_output(output: FoundationOutput, batch: FoundationBatch) -> None:
    batch_size, sequence_length = batch.input_ids.shape
    expected_hidden = (batch_size, sequence_length, output.hidden_states.shape[-1])
    for name, value, expected in (
        ("hidden_states", output.hidden_states, expected_hidden),
        ("long_term", output.long_term, (batch_size, output.hidden_states.shape[-1])),
        ("short_term", output.short_term, (batch_size, output.hidden_states.shape[-1])),
    ):
        if value.shape != expected or value.dtype != torch.float32:
            raise ValueError(
                f"{name}: expected shape={expected}, dtype=torch.float32; "
                f"got shape={tuple(value.shape)}, dtype={value.dtype}; batch_size={batch_size}"
            )
        if value.device != batch.input_ids.device:
            raise ValueError(
                f"{name}: output device differs from batch device; batch_size={batch_size}"
            )
    if output.padding_mask.dtype != torch.bool or not torch.equal(
        output.padding_mask, batch.attention_mask
    ):
        raise ValueError(f"padding_mask must equal attention_mask; batch_size={batch_size}")
    if not torch.isfinite(output.hidden_states[batch.attention_mask]).all():
        raise ValueError(f"hidden_states contains non-finite valid values; batch_size={batch_size}")
    if not torch.isfinite(output.long_term).all() or not torch.isfinite(output.short_term).all():
        raise ValueError(f"user embedding contains non-finite values; batch_size={batch_size}")


class FoundationModel(nn.Module):
    """Foundation Transformer exposing only long- and short-term user embeddings."""

    def __init__(self, config: FoundationModelConfig) -> None:
        super().__init__()
        self.config = config
        self.adapters = AdapterRegistry(config)
        self.user_adapter = UserIdAdapter(
            num_user_buckets=config.num_user_buckets,
            d_model=config.d_model,
            init_std=config.user_embedding_init_std,
        )
        if config.long_term_source == "history":
            self.history_encoder = nn.Sequential(
                nn.Linear(config.history_feature_dim, config.d_model),
                nn.GELU(),
                nn.Linear(config.d_model, config.d_model),
            )
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=config.num_layers, enable_nested_tensor=False
        )
        self.pooler = SequencePooler(sequence_source_id=config.sequence_source_id)

    def forward(self, batch: FoundationBatch, *, attention_mode: str) -> FoundationOutput:
        if batch.user_ids.ndim != 1 or batch.user_ids.shape[0] != batch.input_ids.shape[0]:
            raise ValueError(
                "user_ids must have shape [B] matching input_ids, "
                f"got {tuple(batch.user_ids.shape)} for batch size {batch.input_ids.shape[0]}"
            )
        if batch.user_ids.dtype != torch.long:
            raise ValueError(f"user_ids must have dtype=torch.long, got {batch.user_ids.dtype}")

        token_features = self.adapters(batch)
        if self.config.long_term_source == "history":
            features = batch.history_features
            if features is None or features.shape != (
                    batch.input_ids.shape[0], self.config.history_feature_dim):
                raise ValueError(
                    "long_term_source history needs batch.history_features "
                    f"[B, {self.config.history_feature_dim}]")
            long_term = self.history_encoder(features.to(token_features.dtype))
        else:
            long_term = self.user_adapter(batch.user_ids)
        if self.config.long_term_injection == "cls":
            token_features = token_features.clone()
            token_features[:, 0, :] = token_features[:, 0, :] + long_term
        edge_mask = build_attention_mask(
            batch, attention_mode, sequence_source_id=self.config.sequence_source_id
        )
        padding_mask = ~batch.attention_mask
        if attention_mode == "bidirectional" or not edge_mask.any():
            hidden_states = self.transformer(
                token_features,
                src_key_padding_mask=padding_mask,
            )
        else:
            encoded_rows = []
            for index in range(token_features.shape[0]):
                encoded_rows.append(
                    self.transformer(
                        token_features[index : index + 1],
                        mask=edge_mask[index],
                        src_key_padding_mask=padding_mask[index : index + 1],
                    )
                )
            hidden_states = torch.cat(encoded_rows, dim=0)
        hidden_states = hidden_states.masked_fill(~batch.attention_mask.unsqueeze(-1), 0.0)
        output = FoundationOutput(
            hidden_states=hidden_states,
            long_term=long_term,
            short_term=self.pooler(hidden_states, batch.attention_mask, batch.source_ids),
            padding_mask=batch.attention_mask,
        )
        _validate_output(output, batch)
        return output

    def encode(self, batch: FoundationBatch) -> FoundationOutput:
        return self.forward(batch, attention_mode=self.config.default_attention_mode)

    def embed_users(self, batch: FoundationBatch) -> dict[str, torch.Tensor]:
        output = self.encode(batch)
        return {"long_term": output.long_term, "short_term": output.short_term}
