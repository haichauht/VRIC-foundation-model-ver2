from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from src.foundation_input.loader import iter_normalized_samples
from src.foundation_input.manifest import DatasetManifest, load_manifest
from src.foundation_input.tokenizer import DEFAULT_CONTEXT_ALLOWLIST, TokenizerConfig
from src.foundation_input.vocab import VocabularyConfig, fit_vocabulary_from_samples
from src.training.pretrain import (
    check_vocab_allowlist,
    pretrain_preprocess_metadata,
    tokenizer_config_for_manifest,
)


def _fit(packages, tokenizer_config):
    manifest = load_manifest(packages.pretrain, validate_shard_files=False)
    samples = iter_normalized_samples(packages.pretrain, split="train", validate=False)
    return manifest, fit_vocabulary_from_samples(
        samples, VocabularyConfig(dataset_name=manifest.dataset_name), tokenizer_config
    )


def test_package_allowlist_reaches_the_vocab(sim_tiny_packages):
    manifest = load_manifest(sim_tiny_packages.pretrain, validate_shard_files=False)
    config = tokenizer_config_for_manifest(manifest)
    assert "pickup_geohash" in config.context_allowlist["Ride"]
    assert "merchant_id" in config.context_allowlist["Food"]
    assert config.context_allowlist["E-commerce"] == DEFAULT_CONTEXT_ALLOWLIST["E-commerce"]
    _, vocab = _fit(sim_tiny_packages, config)
    keys = set(vocab.key_to_id)
    assert any(k.endswith("context.pickup_geohash") for k in keys)
    assert any(k.endswith("context.cuisine") for k in keys)


def test_manifest_without_allowlist_uses_defaults(tmp_path):
    manifest = DatasetManifest(root=tmp_path, data={}, config={})
    assert tokenizer_config_for_manifest(manifest) == TokenizerConfig()


def test_reused_vocab_missing_package_allowlist_is_rejected(sim_tiny_packages):
    manifest, vocab = _fit(sim_tiny_packages, TokenizerConfig())
    config = tokenizer_config_for_manifest(manifest)
    with pytest.raises(ValueError, match="Ride"):
        check_vocab_allowlist(vocab, config)


def test_matching_vocab_allowlist_is_accepted(sim_tiny_packages):
    manifest = load_manifest(sim_tiny_packages.pretrain, validate_shard_files=False)
    config = tokenizer_config_for_manifest(manifest)
    _, vocab = _fit(sim_tiny_packages, config)
    check_vocab_allowlist(vocab, config)


def test_preprocess_metadata_records_sim_provenance(sim_tiny_packages):
    manifest = load_manifest(sim_tiny_packages.pretrain, validate_shard_files=False)
    _, vocab = _fit(sim_tiny_packages, tokenizer_config_for_manifest(manifest))
    md = pretrain_preprocess_metadata(manifest, vocab, max_length=512, max_event_count=30)
    assert md["dataset_provenance"]["sim_config_sha256"] == manifest.data["sim_config_sha256"]
    assert md["sequence"] == {"max_tokens": 512, "max_event_count": 30}
    expected = hashlib.sha256((sim_tiny_packages.pretrain / "manifest.json").read_bytes())
    assert md["source_manifest_sha256"] == expected.hexdigest()
    assert md["dataset_name"] == vocab.dataset_name
    assert md["source_fingerprint"] == vocab.source_fingerprint


@pytest.mark.parametrize("name", ["pretrain_sim", "pretrain_sim_kaggle"])
def test_sim_config_targets_its_data_dir(name):
    import yaml

    from src.training.pretrain import resolve_targets

    config = yaml.safe_load(Path(f"configs/{name}.yaml").read_text(encoding="utf-8"))
    assert resolve_targets(config, {}) == [config["data_dir"]]


def _tiny_model_and_objectives():
    from src.models.encoder import FoundationModel, FoundationModelConfig
    from src.models.objectives import PretrainObjectives

    config = FoundationModelConfig(
        d_model=8, num_heads=2, num_layers=1, value_vocab_size=32, key_vocab_size=16,
        modality_vocab_size=5, source_vocab_size=4, time_gap_vocab_size=8, num_user_buckets=11,
        categorical_pair_buckets=31, sequence_source_id=1, numeric_modality_id=2,
    )
    return FoundationModel(config), PretrainObjectives(config)


@pytest.mark.parametrize("multiplier", [1.0, 10.0])
def test_user_adapter_parameters_get_lr_times_multiplier(multiplier):
    from torch.optim import AdamW

    from src.training.pretrain import optimizer_param_groups

    model, objectives = _tiny_model_and_objectives()
    groups = optimizer_param_groups(
        model, objectives, learning_rate=1e-3, weight_decay=0.01,
        user_adapter_lr_multiplier=multiplier,
    )
    optimizer = AdamW(groups, lr=1e-3)
    user = {id(p) for p in model.user_adapter.parameters()}
    names = {id(p): n for n, p in [*model.named_parameters(), *objectives.named_parameters()]}
    seen = []
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            seen.append(id(parameter))
            expected = 1e-3 * multiplier if id(parameter) in user else 1e-3
            assert group["lr"] == pytest.approx(expected)
            no_decay = any(part in names[id(parameter)] for part in ("bias", "LayerNorm.weight"))
            assert group["weight_decay"] == (0.0 if no_decay else 0.01)
    assert len(seen) == len(set(seen)) and set(seen) == set(names)
    assert user <= set(seen)


@pytest.mark.parametrize("multiplier", [0.0, -1.0, float("nan"), float("inf"), True])
def test_user_adapter_lr_multiplier_must_be_positive_and_finite(multiplier):
    from src.training.pretrain import optimizer_param_groups

    model, objectives = _tiny_model_and_objectives()
    with pytest.raises(ValueError, match="user_adapter_lr_multiplier"):
        optimizer_param_groups(model, objectives, learning_rate=1e-3, weight_decay=0.01,
                               user_adapter_lr_multiplier=multiplier)


def test_pretrain_reads_user_embedding_knobs_from_yaml(sim_tiny_packages, tmp_path):
    import json

    import torch

    from src.training.pretrain import pretrain

    config = {
        "seed": 7, "dataset": None, "data_dir": str(sim_tiny_packages.pretrain),
        "output_dir": str(tmp_path / "out"), "device": "cpu",
        "model": {"d_model": 8, "num_heads": 2, "num_layers": 1, "dropout": 0.0,
                  "num_user_buckets": 50_000, "categorical_pair_buckets": 31,
                  "long_term_injection": "cls", "user_embedding_init_std": 0.02},
        "sequence": {"max_length": 512, "max_event_count": 30},
        "training": {"batch_size": 8, "epochs": 1, "learning_rate": 0.001,
                     "user_adapter_lr_multiplier": 10.0},
    }
    record = pretrain(config, {"smoke": True})
    assert record["model_config"]["user_embedding_init_std"] == 0.02
    assert record["user_adapter_lr_multiplier"] == 10.0
    payload = torch.load(tmp_path / "out" / "checkpoints" / "latest_checkpoint.pt",
                         map_location="cpu", weights_only=False)
    assert payload["model_config"]["user_embedding_init_std"] == 0.02
    manifest = json.loads((sim_tiny_packages.pretrain / "manifest.json").read_text(encoding="utf-8"))
    provenance = payload["preprocess_metadata"]["dataset_provenance"]
    assert provenance["sim_config_sha256"] == manifest["sim_config_sha256"]
    from src.training.checkpoint import load_checkpoint

    vocabularies = load_checkpoint(tmp_path / "out" / "checkpoints" / "latest_checkpoint.pt",
                                   load_model=False)[2]
    allowlist = vocabularies.tokenizer_contract["context_allowlist"]
    assert "Ride" in allowlist and "Food" in allowlist
    lrs = {group["name"]: group["lr"] for group in payload["optimizer_state_dict"]["param_groups"]}
    assert lrs["user_adapter"] == pytest.approx(0.01)
    assert lrs["decay"] == pytest.approx(0.001)
    saved = json.loads((tmp_path / "out" / "experiment_record.json").read_text(encoding="utf-8"))
    assert saved["user_adapter_lr_multiplier"] == 10.0


def test_pretrain_casts_user_embedding_init_std_to_float(sim_tiny_packages, tmp_path):
    from src.training.pretrain import pretrain

    config = {
        "seed": 7, "dataset": None, "data_dir": str(sim_tiny_packages.pretrain),
        "output_dir": str(tmp_path / "out"), "device": "cpu",
        "model": {"d_model": 8, "num_heads": 2, "num_layers": 1, "dropout": 0.0,
                  "num_user_buckets": 50_000, "categorical_pair_buckets": 31,
                  "long_term_injection": "cls", "user_embedding_init_std": "0.02"},
        "sequence": {"max_length": 512, "max_event_count": 30},
        "training": {"batch_size": 8, "epochs": 1, "learning_rate": 0.001},
    }
    record = pretrain(config, {"smoke": True})
    assert record["model_config"]["user_embedding_init_std"] == 0.02
    assert isinstance(record["model_config"]["user_embedding_init_std"], float)


@pytest.mark.parametrize("name", ["pretrain_sim", "pretrain_sim_kaggle"])
def test_sim_configs_set_the_user_embedding_knobs(name):
    import yaml

    config = yaml.safe_load(Path(f"configs/{name}.yaml").read_text(encoding="utf-8"))
    assert config["model"]["long_term_injection"] == "cls"
    assert config["model"]["user_embedding_init_std"] == 0.02
    assert config["training"]["user_adapter_lr_multiplier"] == 10.0
