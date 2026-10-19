"""Modality-aware adapters consumed by the foundation Transformer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from src.foundation_input.types import FoundationBatch

if TYPE_CHECKING:
    from .encoder import FoundationModelConfig


class UserIdAdapter(nn.Module):
    """Maps bounded vocabulary IDs or stable hashes into long-term embeddings."""

    def __init__(
        self, *, num_user_buckets: int, d_model: int, init_std: float | None = None
    ) -> None:
        super().__init__()
        if num_user_buckets <= 0 or d_model <= 0:
            raise ValueError("num_user_buckets and d_model must be positive")
        self.embedding = nn.Embedding(num_user_buckets, d_model)
        if init_std is not None:  # None keeps nn.Embedding's own N(0, 1) init untouched
            nn.init.normal_(self.embedding.weight, mean=0.0, std=init_std)

    def forward(self, user_ids: torch.LongTensor) -> torch.Tensor:
        if user_ids.ndim != 1:
            raise ValueError(f"user_ids must have shape [B], got {tuple(user_ids.shape)}")
        bucket_ids = torch.remainder(user_ids.to(torch.long), self.embedding.num_embeddings)
        return self.embedding(bucket_ids)


class GeoAdapter(nn.Module):
    """Standalone normalized-coordinate adapter for a future geo batch extension."""

    def __init__(self, *, d_model: int) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        self.projection = nn.Sequential(
            nn.Linear(6, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(
        self, coordinates: torch.Tensor, missing_mask: torch.BoolTensor
    ) -> torch.Tensor:
        if coordinates.ndim != 3 or coordinates.shape[-1] != 2:
            raise ValueError(
                "coordinates must have shape [B, L, 2], "
                f"got {tuple(coordinates.shape)}"
            )
        if missing_mask.shape != coordinates.shape[:2] or missing_mask.dtype != torch.bool:
            raise ValueError(
                "missing_mask must be bool with shape [B, L], "
                f"got shape={tuple(missing_mask.shape)}, dtype={missing_mask.dtype}"
            )
        if not torch.isfinite(coordinates).all():
            raise ValueError("coordinates must be finite")
        coordinates = coordinates.to(torch.float32)
        features = torch.cat(
            (coordinates, torch.sin(coordinates), torch.cos(coordinates)), dim=-1
        )
        output = self.projection(features)
        return output.masked_fill(missing_mask.unsqueeze(-1), 0.0)


class AdapterRegistry(nn.Module):
    """Converts the FoundationBatch v2 contract into token vectors.

    Categorical, text, and missing tokens use bounded ID/pair embeddings. Numeric tokens keep
    their key, modality, and source context but replace value/pair lookup with an MLP over the
    normalized raw value and its missing flag. Geo tensors remain unsupported until a versioned
    geo field is added to ``FoundationBatch``.
    """

    def __init__(self, config: FoundationModelConfig) -> None:
        super().__init__()
        self.config = config
        self.value_embedding = nn.Embedding(
            config.value_vocab_size, config.d_model, padding_idx=0
        )
        self.pair_embedding = nn.Embedding(
            config.categorical_pair_buckets, config.d_model, padding_idx=0
        )
        self.key_embedding = nn.Embedding(config.key_vocab_size, config.d_model, padding_idx=0)
        self.modality_embedding = nn.Embedding(
            config.modality_vocab_size, config.d_model, padding_idx=0
        )
        self.source_embedding = nn.Embedding(
            config.source_vocab_size, config.d_model, padding_idx=0
        )
        self.position_embedding = nn.Embedding(config.max_length, config.d_model, padding_idx=0)
        self.time_gap_embedding = nn.Embedding(
            config.time_gap_vocab_size, config.d_model, padding_idx=0
        )
        self.numeric_mlp = nn.Sequential(
            nn.Linear(2, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.modality_projections = nn.ModuleList(
            nn.Linear(config.d_model, config.d_model, bias=False)
            for _ in range(config.modality_vocab_size)
        )

    @staticmethod
    def _batch_shape(batch: FoundationBatch) -> tuple[int, int]:
        if batch.input_ids.ndim != 2:
            raise ValueError(f"input_ids must have shape [B, L], got {tuple(batch.input_ids.shape)}")
        return tuple(batch.input_ids.shape)

    def _ids(self, batch: FoundationBatch, name: str, size: int) -> torch.LongTensor:
        values = getattr(batch, name)
        expected_shape = self._batch_shape(batch)
        if values.shape != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape}, got {tuple(values.shape)}"
            )
        if values.dtype != torch.long:
            raise ValueError(f"{name} must have dtype=torch.long, got {values.dtype}")
        if torch.any(values < 0).item() or torch.any(values >= size).item():
            raise ValueError(f"{name} contains an ID outside [0, {size - 1}]")
        return values

    def _numeric_tensors(self, batch: FoundationBatch) -> tuple[torch.Tensor, torch.BoolTensor]:
        expected_shape = self._batch_shape(batch)
        values = batch.numeric_values
        missing_mask = batch.numeric_missing_mask
        if values.shape != expected_shape or values.dtype != torch.float32:
            raise ValueError(
                "numeric_values must be torch.float32 with shape "
                f"{expected_shape}, got shape={tuple(values.shape)}, dtype={values.dtype}"
            )
        if missing_mask.shape != expected_shape or missing_mask.dtype != torch.bool:
            raise ValueError(
                "numeric_missing_mask must be torch.bool with shape "
                f"{expected_shape}, got shape={tuple(missing_mask.shape)}, "
                f"dtype={missing_mask.dtype}"
            )
        if not torch.isfinite(values).all():
            raise ValueError("numeric_values must be finite")
        return values, missing_mask

    def _project_by_modality(
        self, features: torch.Tensor, modality_ids: torch.LongTensor
    ) -> torch.Tensor:
        projected = torch.zeros_like(features)
        for modality_id, projection in enumerate(self.modality_projections):
            selected = modality_ids.eq(modality_id).unsqueeze(-1)
            projected = torch.where(selected, projection(features), projected)
        return projected

    def forward(self, batch: FoundationBatch) -> torch.Tensor:
        batch_shape = self._batch_shape(batch)
        if batch.attention_mask.shape != batch_shape or batch.attention_mask.dtype != torch.bool:
            raise ValueError(
                "attention_mask must be bool with shape "
                f"{batch_shape}, got shape={tuple(batch.attention_mask.shape)}, "
                f"dtype={batch.attention_mask.dtype}"
            )

        input_ids = self._ids(batch, "input_ids", self.config.value_vocab_size)
        key_ids = self._ids(batch, "key_ids", self.config.key_vocab_size)
        modality_ids = self._ids(batch, "modality_ids", self.config.modality_vocab_size)
        source_ids = self._ids(batch, "source_ids", self.config.source_vocab_size)
        position_ids = self._ids(batch, "position_ids", self.config.max_length)
        time_gap_ids = self._ids(batch, "time_gap_ids", self.config.time_gap_vocab_size)
        numeric_values, numeric_missing_mask = self._numeric_tensors(batch)

        pair_ids = torch.remainder(
            key_ids * 1_000_003 + input_ids, self.config.categorical_pair_buckets
        )
        id_features = (
            self.value_embedding(input_ids)
            + self.pair_embedding(pair_ids)
        )
        shared_features = (
            self.key_embedding(key_ids)
            + self.modality_embedding(modality_ids)
            + self.source_embedding(source_ids)
        )
        numeric_features = self.numeric_mlp(
            torch.stack((numeric_values, numeric_missing_mask.to(torch.float32)), dim=-1)
        )
        is_numeric = modality_ids.eq(self.config.numeric_modality_id).unsqueeze(-1)
        features = shared_features + torch.where(is_numeric, numeric_features, id_features)

        sequence_mask = batch.attention_mask & source_ids.eq(self.config.sequence_source_id)
        features = self._project_by_modality(features, modality_ids)
        features = features + self.position_embedding(position_ids) * sequence_mask.unsqueeze(-1)
        features = features + self.time_gap_embedding(time_gap_ids) * sequence_mask.unsqueeze(-1)
        return features.masked_fill(~batch.attention_mask.unsqueeze(-1), 0.0)
