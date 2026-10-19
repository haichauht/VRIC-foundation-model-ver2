from __future__ import annotations

import pytest
import torch

from src.foundation_input.types import FoundationBatch
from src.models.adapters import AdapterRegistry, GeoAdapter, UserIdAdapter


def _three_modality_batch() -> FoundationBatch:
    return FoundationBatch(
        input_ids=torch.tensor([[1, 9, 9, 9, 2]], dtype=torch.long),
        key_ids=torch.tensor([[0, 3, 3, 3, 0]], dtype=torch.long),
        modality_ids=torch.tensor([[0, 1, 2, 3, 0]], dtype=torch.long),
        source_ids=torch.tensor([[0, 2, 2, 2, 0]], dtype=torch.long),
        position_ids=torch.zeros((1, 5), dtype=torch.long),
        time_gap_ids=torch.zeros((1, 5), dtype=torch.long),
        attention_mask=torch.ones((1, 5), dtype=torch.bool),
        numeric_values=torch.zeros((1, 5), dtype=torch.float32),
        numeric_missing_mask=torch.zeros((1, 5), dtype=torch.bool),
        user_ids=torch.tensor([1], dtype=torch.long),
        targets={},
    )


def _numeric_batch(value: float, missing: bool) -> FoundationBatch:
    return FoundationBatch(
        input_ids=torch.tensor([[1, 9, 8, 2]], dtype=torch.long),
        key_ids=torch.tensor([[0, 3, 3, 0]], dtype=torch.long),
        modality_ids=torch.tensor([[0, 3, 1, 0]], dtype=torch.long),
        source_ids=torch.tensor([[0, 2, 2, 0]], dtype=torch.long),
        position_ids=torch.zeros((1, 4), dtype=torch.long),
        time_gap_ids=torch.zeros((1, 4), dtype=torch.long),
        attention_mask=torch.ones((1, 4), dtype=torch.bool),
        numeric_values=torch.tensor([[0.0, value, 777.0, 0.0]], dtype=torch.float32),
        numeric_missing_mask=torch.tensor([[False, missing, False, False]], dtype=torch.bool),
        user_ids=torch.tensor([1], dtype=torch.long),
        targets={},
    )


def test_pair_keying_distinguishes_reused_value_ids(batch_factory, model_config) -> None:
    batch = batch_factory()
    batch.input_ids[0, 1] = 9
    batch.input_ids[0, 3] = 9
    batch.key_ids[0, 1] = 3
    batch.key_ids[0, 3] = 4

    representations = AdapterRegistry(model_config())(batch)

    assert not torch.equal(representations[0, 1], representations[0, 3])


def test_registry_routes_categorical_text_and_numeric_mlp_through_distinct_projections(
    model_config,
) -> None:
    config = model_config(modality_vocab_size=4)
    registry = AdapterRegistry(config).eval()
    with torch.no_grad():
        for embedding in (
            registry.value_embedding,
            registry.pair_embedding,
            registry.key_embedding,
            registry.modality_embedding,
            registry.source_embedding,
            registry.position_embedding,
            registry.time_gap_embedding,
        ):
            embedding.weight.zero_()
        registry.value_embedding.weight[9].fill_(1.0)
        for projection in registry.modality_projections:
            projection.weight.zero_()
        registry.modality_projections[1].weight.copy_(torch.eye(config.d_model))
        registry.modality_projections[2].weight.copy_(2.0 * torch.eye(config.d_model))
        registry.modality_projections[3].weight.copy_(3.0 * torch.eye(config.d_model))
        first_numeric_layer = registry.numeric_mlp[0]
        second_numeric_layer = registry.numeric_mlp[2]
        first_numeric_layer.weight.zero_()
        first_numeric_layer.bias.fill_(1.0)
        second_numeric_layer.weight.copy_(torch.eye(config.d_model))
        second_numeric_layer.bias.zero_()

    output = registry(_three_modality_batch())

    torch.testing.assert_close(output[0, 1], torch.ones(config.d_model))
    torch.testing.assert_close(output[0, 2], torch.full((config.d_model,), 2.0))
    torch.testing.assert_close(
        output[0, 3],
        3.0 * torch.nn.functional.gelu(torch.ones(config.d_model)),
    )


def test_numeric_value_changes_only_the_numeric_token(model_config) -> None:
    registry = AdapterRegistry(model_config()).eval()

    first = registry(_numeric_batch(1.0, False))
    changed = registry(_numeric_batch(99.0, False))

    assert not torch.equal(first[0, 1], changed[0, 1])
    torch.testing.assert_close(first[0, 2], changed[0, 2])


def test_numeric_zero_and_missing_are_distinct(model_config) -> None:
    registry = AdapterRegistry(model_config()).eval()

    zero = registry(_numeric_batch(0.0, False))
    missing = registry(_numeric_batch(0.0, True))

    assert not torch.equal(zero[0, 1], missing[0, 1])


def test_adapter_rejects_malformed_numeric_tensors(model_config) -> None:
    batch = _numeric_batch(1.0, False)
    batch.numeric_values = batch.numeric_values[:, :-1]

    with pytest.raises(ValueError, match="numeric_values"):
        AdapterRegistry(model_config())(batch)


def test_large_hashed_user_id_is_mapped_inside_embedding_range() -> None:
    adapter = UserIdAdapter(num_user_buckets=17, d_model=8)

    output = adapter(torch.tensor([2**62 + 123, 5], dtype=torch.long))

    assert output.shape == (2, 8)
    assert torch.isfinite(output).all()


def test_position_and_time_gaps_apply_only_to_sequence_tokens(batch_factory, model_config) -> None:
    base = batch_factory()
    changed = batch_factory()
    changed.position_ids[0, 1] = 5
    changed.time_gap_ids[0, 1] = 3
    changed.position_ids[0, 3] = 5
    changed.time_gap_ids[0, 3] = 3
    registry = AdapterRegistry(model_config()).eval()

    base_features = registry(base)
    changed_features = registry(changed)

    torch.testing.assert_close(base_features[0, 1], changed_features[0, 1])
    assert not torch.equal(base_features[0, 3], changed_features[0, 3])


def test_padding_rows_are_zero_and_negative_ids_fail(batch_factory, model_config) -> None:
    registry = AdapterRegistry(model_config())
    batch = batch_factory()

    features = registry(batch)
    assert torch.equal(features[0, 5:], torch.zeros_like(features[0, 5:]))

    batch.input_ids[0, 1] = -1
    with pytest.raises(ValueError, match="input_ids"):
        registry(batch)


def test_geo_adapter_returns_zero_for_missing_coordinates() -> None:
    adapter = GeoAdapter(d_model=8)
    coordinates = torch.tensor([[[0.1, 0.2], [0.3, 0.4]]], dtype=torch.float32)
    missing = torch.tensor([[False, True]])

    output = adapter(coordinates, missing)

    assert output.shape == (1, 2, 8)
    assert torch.isfinite(output).all()
    assert torch.equal(output[:, 1], torch.zeros_like(output[:, 1]))
