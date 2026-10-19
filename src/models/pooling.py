"""Pooling layers for public user embeddings."""

from __future__ import annotations

import torch
from torch import nn


class SequencePooler(nn.Module):
    """Mean-pools valid event tokens and falls back to the CLS representation."""

    def __init__(self, *, sequence_source_id: int) -> None:
        super().__init__()
        self.sequence_source_id = sequence_source_id

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.BoolTensor,
        source_ids: torch.LongTensor,
    ) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise ValueError(
                f"hidden_states must have shape [B, L, D], got {tuple(hidden_states.shape)}"
            )
        expected = hidden_states.shape[:2]
        if attention_mask.shape != expected or attention_mask.dtype != torch.bool:
            raise ValueError(
                "attention_mask must be bool with shape "
                f"{expected}, got shape={tuple(attention_mask.shape)}, "
                f"dtype={attention_mask.dtype}"
            )
        if source_ids.shape != expected:
            raise ValueError(
                f"source_ids must have shape {expected}, got {tuple(source_ids.shape)}"
            )
        if not torch.all(attention_mask[:, 0]).item():
            raise ValueError("CLS at index 0 must be valid for zero-event fallback")

        event_mask = attention_mask & source_ids.eq(self.sequence_source_id)
        weights = event_mask.unsqueeze(-1).to(hidden_states.dtype)
        pooled_events = (hidden_states * weights).sum(dim=1)
        event_count = weights.sum(dim=1)
        pooled = pooled_events / event_count.clamp_min(1.0)
        return torch.where(event_count.gt(0), pooled, hidden_states[:, 0, :])
