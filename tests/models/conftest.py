from __future__ import annotations

import pytest
import torch

from src.foundation_input.types import FoundationBatch


def _batch_factory(*, two_tabular_tokens: bool = False) -> FoundationBatch:
    if two_tabular_tokens:
        input_ids = [[1, 11, 12, 2, 21, 22, 0], [1, 13, 14, 2, 0, 0, 0]]
        key_ids = [[0, 3, 4, 0, 5, 6, 0], [0, 3, 4, 0, 0, 0, 0]]
        modality_ids = [[0, 1, 1, 0, 1, 1, 0], [0, 1, 1, 0, 0, 0, 0]]
        source_ids = [[0, 2, 2, 0, 1, 1, 0], [0, 2, 2, 0, 0, 0, 0]]
        attention_mask = [[True, True, True, True, True, True, False], [True, True, True, True, False, False, False]]
    else:
        input_ids = [[1, 11, 2, 21, 22, 0, 0], [1, 13, 2, 0, 0, 0, 0]]
        key_ids = [[0, 3, 0, 5, 6, 0, 0], [0, 3, 0, 0, 0, 0, 0]]
        modality_ids = [[0, 1, 0, 1, 1, 0, 0], [0, 1, 0, 0, 0, 0, 0]]
        source_ids = [[0, 2, 0, 1, 1, 0, 0], [0, 2, 0, 0, 0, 0, 0]]
        attention_mask = [[True, True, True, True, True, False, False], [True, True, True, False, False, False, False]]

    return FoundationBatch(
        input_ids=torch.tensor(input_ids, dtype=torch.long),
        key_ids=torch.tensor(key_ids, dtype=torch.long),
        modality_ids=torch.tensor(modality_ids, dtype=torch.long),
        source_ids=torch.tensor(source_ids, dtype=torch.long),
        position_ids=torch.tensor([[0, 0, 0, 1, 2, 0, 0], [0, 0, 0, 0, 0, 0, 0]], dtype=torch.long),
        time_gap_ids=torch.tensor([[0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0]], dtype=torch.long),
        attention_mask=torch.tensor(attention_mask, dtype=torch.bool),
        numeric_values=torch.zeros((2, 7), dtype=torch.float32),
        numeric_missing_mask=torch.zeros((2, 7), dtype=torch.bool),
        user_ids=torch.tensor([2**62 + 123, 5], dtype=torch.long),
        targets={},
    )


def _model_config(**overrides):
    from src.models.encoder import FoundationModelConfig

    values = {
        "d_model": 16,
        "num_heads": 4,
        "num_layers": 2,
        "dropout": 0.0,
        "value_vocab_size": 64,
        "key_vocab_size": 32,
        "modality_vocab_size": 8,
        "source_vocab_size": 4,
        "time_gap_vocab_size": 8,
        "num_user_buckets": 17,
        "categorical_pair_buckets": 97,
        "sequence_source_id": 1,
        "numeric_modality_id": 3,
    }
    values.update(overrides)
    return FoundationModelConfig(**values)


@pytest.fixture
def batch_factory():
    return _batch_factory


@pytest.fixture
def model_config():
    return _model_config
