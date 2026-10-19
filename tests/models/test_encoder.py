from __future__ import annotations

import pytest
import torch

from src.foundation_input.dataset import foundation_collate
from src.foundation_input.types import EncodedSample, KeyValueToken, TokenRecord
from src.foundation_input.vocab import VocabularyConfig, fit_vocabulary
from src.models.encoder import FoundationModel, FoundationModelConfig, build_attention_mask


def test_config_rejects_non_divisible_attention_dimension() -> None:
    with pytest.raises(ValueError, match="d_model=.*num_heads"):
        FoundationModelConfig(d_model=30, num_heads=8)


def test_config_rejects_unknown_default_attention_mode() -> None:
    with pytest.raises(ValueError, match="default_attention_mode"):
        FoundationModelConfig(default_attention_mode="causal")


def test_config_rejects_zero_attention_heads(model_config) -> None:
    with pytest.raises(ValueError, match="num_heads"):
        model_config(num_heads=0)


def test_config_rejects_sequence_source_outside_source_vocab(model_config) -> None:
    with pytest.raises(ValueError, match="sequence_source_id"):
        model_config(source_vocab_size=2, sequence_source_id=2)


def test_config_rejects_numeric_modality_outside_modality_vocab(model_config) -> None:
    with pytest.raises(ValueError, match="numeric_modality_id"):
        model_config(modality_vocab_size=3, numeric_modality_id=3)
    with pytest.raises(ValueError, match="numeric_modality_id"):
        model_config(numeric_modality_id=0)


def test_config_requires_real_input_cardinalities() -> None:
    with pytest.raises(ValueError, match="value_vocab_size"):
        FoundationModelConfig()


def test_next_action_blocks_future_events_but_allows_profile_context(batch_factory) -> None:
    batch = batch_factory()

    blocked = build_attention_mask(batch, "next_action", sequence_source_id=1)

    assert blocked[0, 3, 4].item() is True
    assert blocked[0, 3, 1].item() is False
    assert blocked[0, 1, 3].item() is True


def test_bidirectional_does_not_add_causal_blocks(batch_factory) -> None:
    batch = batch_factory()

    blocked = build_attention_mask(batch, "bidirectional", sequence_source_id=1)

    assert blocked[0, 3, 4].item() is False


def test_forward_returns_contract_shapes_and_finite_values(batch_factory, model_config) -> None:
    model = FoundationModel(model_config())

    output = model(batch_factory(), attention_mode="bidirectional")

    assert output.hidden_states.shape == (2, 7, 16)
    assert output.long_term.shape == (2, 16)
    assert output.short_term.shape == (2, 16)
    assert output.padding_mask.dtype is torch.bool
    assert torch.isfinite(output.hidden_states[output.padding_mask]).all()


def test_embed_users_returns_only_the_two_public_embeddings(batch_factory, model_config) -> None:
    model = FoundationModel(model_config()).eval()

    result = model.embed_users(batch_factory())

    assert set(result) == {"long_term", "short_term"}
    assert result["long_term"].shape == (2, 16)


def test_eval_encoding_is_deterministic(batch_factory, model_config) -> None:
    model = FoundationModel(model_config(dropout=0.2)).eval()
    batch = batch_factory()

    first = model.encode(batch).short_term
    second = model.encode(batch).short_term

    torch.testing.assert_close(first, second, rtol=0.0, atol=1e-6)


def test_changing_padding_ids_does_not_change_valid_outputs(batch_factory, model_config) -> None:
    model = FoundationModel(model_config()).eval()
    original = batch_factory()
    changed = batch_factory()
    changed.input_ids[0, 5:] = torch.tensor([31, 47])
    changed.key_ids[0, 5:] = torch.tensor([9, 10])

    first = model.encode(original).hidden_states[0, :5]
    second = model.encode(changed).hidden_states[0, :5]

    torch.testing.assert_close(first, second, rtol=0.0, atol=1e-6)


def test_permuting_tabular_columns_preserves_short_term_in_eval(batch_factory, model_config) -> None:
    model = FoundationModel(model_config()).eval()
    first = batch_factory(two_tabular_tokens=True)
    second = batch_factory(two_tabular_tokens=True)
    second.input_ids[0, 1:3] = second.input_ids[0, [2, 1]]
    second.key_ids[0, 1:3] = second.key_ids[0, [2, 1]]

    torch.testing.assert_close(
        model.encode(first).short_term, model.encode(second).short_term, rtol=0.0, atol=1e-6
    )


def test_future_event_cannot_change_past_event_hidden_state(batch_factory, model_config) -> None:
    model = FoundationModel(model_config()).eval()
    first = batch_factory()
    changed = batch_factory()
    changed.input_ids[0, 4] = 37

    before = model(first, attention_mode="next_action").hidden_states[0, 3]
    after = model(changed, attention_mode="next_action").hidden_states[0, 3]

    torch.testing.assert_close(before, after, rtol=0.0, atol=1e-6)


def test_malformed_batches_raise_contextual_errors(batch_factory, model_config) -> None:
    model = FoundationModel(model_config())
    negative = batch_factory()
    negative.input_ids[0, 1] = -1
    with pytest.raises(ValueError, match="input_ids"):
        model.encode(negative)

    mismatched = batch_factory()
    mismatched.key_ids = mismatched.key_ids[:, :-1]
    with pytest.raises(ValueError, match="key_ids"):
        model.encode(mismatched)

    with pytest.raises(ValueError, match="attention_mode"):
        model(batch_factory(), attention_mode="causal")

    invalid_cls = batch_factory()
    invalid_cls.attention_mask[1, 0] = False
    with pytest.raises(ValueError, match="CLS"):
        model.encode(invalid_cls)


def test_foundation_input_smoke_uses_public_collator(model_config) -> None:
    records = (
        TokenRecord(
            token=KeyValueToken("profile.city", "Hanoi", "categorical", "tabular", None),
            value_id=7,
            key_id=3,
            modality_id=1,
            source_id=2,
            position_id=0,
            time_gap_id=0,
        ),
        TokenRecord(
            token=KeyValueToken(
                "event.event_type", "view", "categorical", "sequence", None, "event-1"
            ),
            value_id=9,
            key_id=4,
            modality_id=1,
            source_id=1,
            position_id=1,
            time_gap_id=0,
        ),
    )
    batch = foundation_collate(
        [
            EncodedSample("sample-1", "fixture-v1", "user-1", records, user_id_value_id=6),
            EncodedSample("sample-2", "fixture-v1", "user-2", records, user_id_value_id=8),
        ]
    )
    config = model_config()

    output = FoundationModel(config).encode(batch)

    assert output.hidden_states.shape[0] == 2
    assert output.hidden_states.shape[1] == batch.input_ids.shape[1]
    assert output.long_term.shape == output.short_term.shape == (2, config.d_model)
    assert torch.isfinite(output.long_term).all()
    assert torch.isfinite(output.short_term).all()


def test_artifact_metadata_builds_numeric_model_and_handles_zero_and_missing() -> None:
    numeric = KeyValueToken("profile.score", 0.0, "numeric", "tabular", None)
    numeric_for_fit = KeyValueToken("profile.score", 10.0, "numeric", "tabular", None)
    sequence = KeyValueToken("event.type", "view", "categorical", "sequence", None, "event-1")
    artifacts = fit_vocabulary(
        [numeric, numeric_for_fit, sequence], VocabularyConfig(dataset_name="fixture")
    )
    cardinalities = artifacts.model_cardinalities()

    def numeric_record(value: float | None) -> TokenRecord:
        token = KeyValueToken("profile.score", value, "numeric", "tabular", None)
        normalized, missing = artifacts.normalize_numeric(token)
        return TokenRecord(
            token=token,
            value_id=artifacts.encode_value(token),
            key_id=artifacts.encode_key(token),
            modality_id=artifacts.encode_modality(token),
            source_id=artifacts.encode_source(token),
            position_id=0,
            time_gap_id=0,
            numeric_value=normalized,
            numeric_missing=missing,
        )

    batch = foundation_collate(
        [
            EncodedSample("observed", "fixture", "user-1", (numeric_record(0.0),)),
            EncodedSample("missing", "fixture", "user-2", (numeric_record(None),)),
        ]
    )
    config = FoundationModelConfig(
        d_model=8,
        num_heads=2,
        num_layers=1,
        dropout=0.0,
        max_length=16,
        **{key: cardinalities[key] for key in cardinalities if key.endswith("_vocab_size")},
        num_user_buckets=17,
        categorical_pair_buckets=97,
        sequence_source_id=artifacts.source_to_id["sequence"],
        numeric_modality_id=artifacts.modality_to_id["numeric"],
    )

    output = FoundationModel(config).eval().encode(batch)

    assert batch.numeric_values.shape == batch.input_ids.shape
    assert batch.numeric_missing_mask.shape == batch.input_ids.shape
    assert output.hidden_states.shape == (2, batch.input_ids.shape[1], config.d_model)
    assert torch.isfinite(output.hidden_states[output.padding_mask]).all()


@pytest.mark.parametrize("seed", [0, 3])  # seed 3: the plain sum's gradient was exactly zero
def test_cls_injection_gives_user_adapter_gradient(batch_factory, model_config, seed) -> None:
    torch.manual_seed(seed)
    model = FoundationModel(model_config(long_term_injection="cls"))
    batch = batch_factory()
    output = model.encode(batch)
    # Each hidden state leaves a LayerNorm, so its plain feature sum does not depend on the input
    # (gradient = float noise, exactly zero for some seeds); project onto a fixed direction instead.
    direction = torch.randn(output.hidden_states.shape[-1], generator=torch.Generator().manual_seed(0))
    (output.hidden_states[batch.attention_mask] @ direction).sum().backward()
    grad = model.user_adapter.embedding.weight.grad
    assert grad is not None
    assert grad.abs().sum() > 1e-3


def test_cls_injection_is_identity_with_zero_adapter_and_changes_output_otherwise(
    batch_factory, model_config
) -> None:
    batch = batch_factory()
    base = FoundationModel(model_config(long_term_injection="none")).eval()
    injected = FoundationModel(model_config(long_term_injection="cls")).eval()
    injected.load_state_dict(base.state_dict())
    with torch.no_grad():
        expected = base.encode(batch)
        for parameter in injected.user_adapter.parameters():
            parameter.zero_()
        zeroed = injected.encode(batch)
        assert torch.equal(zeroed.hidden_states, expected.hidden_states)
        assert torch.equal(zeroed.short_term, expected.short_term)
        for parameter in injected.user_adapter.parameters():
            parameter.fill_(0.5)
        changed = injected.encode(batch)
    assert not torch.equal(changed.hidden_states, expected.hidden_states)
    assert not torch.equal(changed.short_term, expected.short_term)


def test_unknown_injection_mode_rejected() -> None:
    with pytest.raises(ValueError, match="long_term_injection"):
        FoundationModelConfig(long_term_injection="sum")


def test_default_user_embedding_init_is_unchanged(model_config) -> None:
    """None keeps the plain nn.Embedding N(0, 1) init and consumes no extra RNG."""
    from torch import nn

    from src.models.adapters import UserIdAdapter

    assert FoundationModelConfig.__dataclass_fields__["user_embedding_init_std"].default is None
    torch.manual_seed(3)
    reference = nn.Embedding(17, 16)
    reference_rng = torch.get_rng_state()
    torch.manual_seed(3)
    adapter = UserIdAdapter(num_user_buckets=17, d_model=16)
    assert torch.equal(adapter.embedding.weight, reference.weight)
    assert torch.equal(torch.get_rng_state(), reference_rng)
    torch.manual_seed(5)
    implicit = FoundationModel(model_config()).state_dict()
    torch.manual_seed(5)
    explicit = FoundationModel(model_config(user_embedding_init_std=None)).state_dict()
    assert all(torch.equal(implicit[name], explicit[name]) for name in implicit)


def test_user_embedding_init_std_sets_the_embedding_scale(model_config) -> None:
    from src.models.adapters import UserIdAdapter

    torch.manual_seed(0)
    adapter = UserIdAdapter(num_user_buckets=20_000, d_model=64, init_std=0.02)
    weight = adapter.embedding.weight.detach()
    assert abs(float(weight.std()) - 0.02) < 5e-4
    assert abs(float(weight.mean())) < 5e-4
    model = FoundationModel(model_config(user_embedding_init_std=0.02, num_user_buckets=4_000))
    assert abs(float(model.user_adapter.embedding.weight.detach().std()) - 0.02) < 1e-3


@pytest.mark.parametrize("value", [0.0, -0.02, float("nan"), float("inf")])
def test_user_embedding_init_std_must_be_positive_and_finite(value) -> None:
    with pytest.raises(ValueError, match="user_embedding_init_std"):
        FoundationModelConfig(user_embedding_init_std=value)
