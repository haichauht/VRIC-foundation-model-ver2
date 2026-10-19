from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.foundation_input.vocab import VocabularyArtifacts
from src.models.encoder import FoundationModel
from src.training.checkpoint import (
    load_checkpoint,
    save_checkpoint,
)


@pytest.fixture
def dummy_vocab():
    return VocabularyArtifacts(
        dataset_name="test_dataset",
        value_to_id={"test_dataset:tabular:categorical:cat_1": {"val1": 5, "val2": 6}},
        numeric_boundaries={"test_dataset:tabular:numeric:num_1": (1.0, 5.0, 10.0)},
        key_to_id={"test_dataset:tabular:categorical:cat_1": 1, "test_dataset:tabular:numeric:num_1": 2},
        modality_to_id={"categorical": 1, "numeric": 2},
        source_to_id={"tabular": 1, "sequence": 2},
        text_hash_buckets=64,
        numeric_normalization={"test_dataset:tabular:numeric:num_1": (5.0, 2.5)},
        numeric_transform_policy={"test_dataset:tabular:numeric:num_1": "zscore"},
        time_gap_boundaries_seconds=(60.0, 3600.0),
    )


def test_save_and_load_checkpoint(tmp_path: Path, model_config, dummy_vocab, batch_factory):
    cfg = model_config()
    model = FoundationModel(cfg)

    ckpt_path = tmp_path / "checkpoint.pt"
    checksum = save_checkpoint(
        ckpt_path,
        model,
        cfg,
        dummy_vocab,
        epoch=1,
        step=50,
        metrics={"loss": 1.23},
    )

    assert ckpt_path.is_file()
    assert isinstance(checksum, str) and len(checksum) == 64

    # Load back
    loaded_model, loaded_cfg, loaded_vocab, raw_payload = load_checkpoint(ckpt_path)

    assert loaded_cfg == cfg
    assert loaded_vocab.dataset_name == dummy_vocab.dataset_name
    assert loaded_vocab.value_to_id == dummy_vocab.value_to_id
    assert raw_payload["epoch"] == 1
    assert raw_payload["step"] == 50
    assert raw_payload["metrics"]["loss"] == 1.23

    # Embeddings must match exactly
    batch = batch_factory()
    model.eval()
    loaded_model.eval()
    with torch.no_grad():
        orig_out = model.embed_users(batch)
        loaded_out = loaded_model.embed_users(batch)

    torch.testing.assert_close(orig_out["long_term"], loaded_out["long_term"])
    torch.testing.assert_close(orig_out["short_term"], loaded_out["short_term"])


def test_load_checkpoint_rejects_corrupted_format(tmp_path: Path):
    bad_path = tmp_path / "bad.pt"
    torch.save({"format_version": 999}, bad_path)

    with pytest.raises(ValueError, match="Unsupported checkpoint format_version"):
        load_checkpoint(bad_path)


def test_old_checkpoint_without_injection_key_loads(
    tmp_path: Path, model_config, dummy_vocab, batch_factory
):
    cfg = model_config()
    model = FoundationModel(cfg)
    ckpt_path = tmp_path / "old.pt"
    save_checkpoint(ckpt_path, model, cfg, dummy_vocab, epoch=1, step=1, metrics={})
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    del payload["model_config"]["long_term_injection"]
    del payload["model_config"]["user_embedding_init_std"]
    torch.save(payload, ckpt_path)

    loaded_model, loaded_cfg, _, _ = load_checkpoint(ckpt_path)

    assert loaded_cfg.long_term_injection == "none"
    assert loaded_cfg.user_embedding_init_std is None
    batch = batch_factory()
    model.eval()
    loaded_model.eval()
    with torch.no_grad():
        original = model.encode(batch)
        loaded = loaded_model.encode(batch)
    assert torch.equal(original.hidden_states, loaded.hidden_states)
    assert torch.equal(original.short_term, loaded.short_term)


def test_user_id_key_id_matches_semantic_token_namespace(dummy_vocab):
    from src.foundation_input.tokenizer import TokenizerConfig
    from src.training.pretrain import user_id_key_id

    dummy_vocab.key_to_id["test_dataset:tabular:categorical:user_id"] = 7
    assert user_id_key_id(dummy_vocab, TokenizerConfig()) == 7
    del dummy_vocab.key_to_id["test_dataset:tabular:categorical:user_id"]
    with pytest.raises(ValueError, match="user_id"):
        user_id_key_id(dummy_vocab, TokenizerConfig())
