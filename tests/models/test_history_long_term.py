"""long_term_source "history": the long-term embedding comes from the long-history features."""

from __future__ import annotations

import dataclasses

import pytest
import torch

from src.foundation_input.types import FoundationBatch
from src.models.encoder import FoundationModel, FoundationModelConfig
from src.models.objectives import PretrainObjectives, PretrainObjectivesConfig

F = 6


def config(**overrides):
    base = {"d_model": 8, "num_heads": 2, "num_layers": 1, "dropout": 0.0, "max_length": 16,
            "value_vocab_size": 40, "key_vocab_size": 10, "modality_vocab_size": 4,
            "source_vocab_size": 4, "time_gap_vocab_size": 8, "num_user_buckets": 50,
            "categorical_pair_buckets": 11, "sequence_source_id": 1, "numeric_modality_id": 2,
            "long_term_injection": "cls", "long_term_source": "history",
            "history_feature_dim": F, "history_target_sizes": (3, 2)}
    return FoundationModelConfig(**{**base, **overrides})


def batch(features=None, targets=None):
    b, length = 2, 5
    return FoundationBatch(
        input_ids=torch.randint(4, 40, (b, length)), key_ids=torch.randint(1, 10, (b, length)),
        modality_ids=torch.ones(b, length, dtype=torch.long), source_ids=torch.ones(b, length, dtype=torch.long),
        position_ids=torch.arange(length).repeat(b, 1), time_gap_ids=torch.zeros(b, length, dtype=torch.long),
        attention_mask=torch.ones(b, length, dtype=torch.bool), numeric_values=torch.zeros(b, length),
        numeric_missing_mask=torch.ones(b, length, dtype=torch.bool), user_ids=torch.tensor([3, 4]),
        targets=targets or {}, history_features=features)


def test_long_term_follows_the_history_features():
    torch.manual_seed(0)
    model = FoundationModel(config()).eval()
    data = batch(torch.rand(2, F))
    first = model.embed_users(data)["long_term"]
    moved = model.embed_users(dataclasses.replace(data, history_features=data.history_features + 1))
    same_user_other_id = model.embed_users(dataclasses.replace(data, user_ids=torch.tensor([9, 9])))
    assert not torch.allclose(first, moved["long_term"])
    assert torch.allclose(first, same_user_other_id["long_term"])  # IDs no longer matter


@pytest.mark.parametrize("features", [None, torch.rand(2, F + 1)])
def test_history_model_rejects_missing_or_wrong_features(features):
    model = FoundationModel(config())
    with pytest.raises(ValueError, match="history_features"):
        model.embed_users(batch(features))


@pytest.mark.parametrize(("overrides", "match"), [
    ({"long_term_source": "both"}, "long_term_source"),
    ({"history_feature_dim": 0}, "history_feature_dim"),
    ({"long_term_source": "user_id"}, "history_feature_dim"),
])
def test_config_validation(overrides, match):
    with pytest.raises(ValueError, match=match):
        config(**overrides)


def test_user_id_default_is_unchanged():
    plain = FoundationModelConfig(d_model=8, num_heads=2, num_layers=1, max_length=16,
                                  value_vocab_size=40, key_vocab_size=10, modality_vocab_size=4,
                                  source_vocab_size=4, time_gap_vocab_size=8, num_user_buckets=50,
                                  categorical_pair_buckets=11, sequence_source_id=1, numeric_modality_id=2)
    assert plain.long_term_source == "user_id" and plain.history_target_sizes == ()
    assert not hasattr(FoundationModel(plain), "history_encoder")


def test_history_target_loss_is_masked_and_trains_the_history_encoder():
    torch.manual_seed(0)
    cfg = config()
    model = FoundationModel(cfg)
    objectives = PretrainObjectives(cfg, PretrainObjectivesConfig(history_target_weight=1.0))
    targets = {"history_targets": torch.tensor([[1.0, 0, 0, 0.5, 0.5], [0, 1.0, 0, 1.0, 0]]),
               "history_target_mask": torch.tensor([[1.0, 1.0], [0.0, 0.0]])}
    data = batch(torch.rand(2, F), targets)
    masked, mask_targets = objectives.masking_policy.apply_mask(data)
    assert masked.history_features is data.history_features
    losses = objectives(model(masked, attention_mode="bidirectional"), mask_targets, batch_targets=data.targets)
    assert torch.isfinite(losses["history_target_loss"]) and losses["history_target_loss"] > 0
    losses["loss"].backward()
    assert model.history_encoder[0].weight.grad.abs().sum() > 0
    none = {"history_targets": targets["history_targets"], "history_target_mask": torch.zeros(2, 2)}
    zero = objectives(model(masked, attention_mode="bidirectional"), mask_targets, batch_targets=none)
    assert zero["history_target_loss"] == 0
