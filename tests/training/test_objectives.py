from __future__ import annotations

import pytest
import torch

from src.foundation_input.types import SPECIAL_TOKEN_IDS
from src.models.encoder import FoundationModel
from src.models.objectives import MaskingPolicy, PretrainObjectives, PretrainObjectivesConfig


@pytest.fixture
def objectives_config():
    return PretrainObjectivesConfig(
        mask_prob=0.3,
        masked_categorical_weight=1.0,
        masked_numeric_weight=1.0,
        next_action_weight=0.5,
    )


def test_masking_policy_never_masks_pad_cls_sep(batch_factory, objectives_config):
    batch = batch_factory()
    policy = MaskingPolicy(objectives_config)
    masked_batch, targets = policy.apply_mask(batch)

    # CLS is at index 0
    assert not targets["masked_positions"][:, 0].any()
    assert masked_batch.input_ids[:, 0].tolist() == batch.input_ids[:, 0].tolist()

    # PAD tokens (attention_mask == False) are never masked
    pad_mask = ~batch.attention_mask
    assert not targets["masked_positions"][pad_mask].any()

    # SEP is at index 3 or 2
    for b in range(batch.input_ids.shape[0]):
        for pos in range(batch.input_ids.shape[1]):
            if batch.input_ids[b, pos].item() == SPECIAL_TOKEN_IDS["SEP"]:
                assert not targets["masked_positions"][b, pos].item()


def test_pretrain_objectives_forward_computes_finite_loss(
    batch_factory, model_config, objectives_config
):
    cfg = model_config()
    model = FoundationModel(cfg)
    objectives = PretrainObjectives(cfg, objectives_config)

    batch = batch_factory()
    batch.targets["next_action_type_ids"] = torch.tensor([5, 6], dtype=torch.long)
    batch.targets["next_action_valid_mask"] = torch.tensor([True, True], dtype=torch.bool)

    policy = MaskingPolicy(objectives_config)
    masked_batch, targets = policy.apply_mask(batch)

    output = model(masked_batch, attention_mode="bidirectional")
    loss_dict = objectives(output, targets, batch_targets=batch.targets)

    assert "loss" in loss_dict
    assert torch.isfinite(loss_dict["loss"])
    assert loss_dict["loss"].item() > 0.0
    assert "masked_categorical_loss" in loss_dict
    assert "masked_numeric_loss" in loss_dict
    assert "next_action_loss" in loss_dict


def test_pretrain_objectives_backward(batch_factory, model_config, objectives_config):
    cfg = model_config()
    model = FoundationModel(cfg)
    objectives = PretrainObjectives(cfg, objectives_config)

    batch = batch_factory()
    policy = MaskingPolicy(objectives_config)
    masked_batch, targets = policy.apply_mask(batch)

    output = model(masked_batch, attention_mode="bidirectional")
    loss_dict = objectives(output, targets)
    loss = loss_dict["loss"]

    loss.backward()

    # Check gradients exist and are finite
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            assert torch.isfinite(p.grad).all()


def test_unmaskable_keys_are_never_masked(batch_factory):
    batch = batch_factory()
    user_key_id = 9
    valid = batch.attention_mask & batch.input_ids.ne(SPECIAL_TOKEN_IDS["CLS"])
    valid &= batch.input_ids.ne(SPECIAL_TOKEN_IDS["SEP"])
    batch.key_ids[valid] = user_key_id
    batch.key_ids[0, 1] = 3  # the single maskable token
    policy = MaskingPolicy(PretrainObjectivesConfig(mask_prob=1.0, unmaskable_key_ids=(user_key_id,)))
    for seed in range(20):
        generator = torch.Generator().manual_seed(seed)
        _, targets = policy.apply_mask(batch, generator)
        masked = targets["masked_positions"]
        assert not masked[batch.key_ids.eq(user_key_id)].any()
        assert masked[0, 1]
