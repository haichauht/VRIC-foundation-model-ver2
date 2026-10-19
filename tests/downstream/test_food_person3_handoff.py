"""Real Person 3 loader bindings and downstream input provenance, without training."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import pytest
import torch

from src.contracts.output import fingerprint, sha256_file
from src.downstream.food_inputs import (
    food_baseline_evaluator,
    load_food_foundation,
    load_food_inputs,
    resolve_food_checkpoint_metadata,
)
from src.downstream.food_recommendation import FoodServices, run_food_recommendation
from src.foundation_input.types import CanonicalSample, EventRecord
from src.foundation_input.vocab import VocabularyConfig, fit_vocabulary_from_samples
from src.models.encoder import FoundationModel, FoundationModelConfig
from src.training.checkpoint import load_checkpoint, save_checkpoint
from tests.downstream.test_food_samples import (
    load_catalog_fixture,
    make_record,
    valid_records,
    write_samples_zip,
)


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def person3(tmp_path):
    event_time = datetime.fromtimestamp(50, timezone.utc)
    event = EventRecord(
        event_id="event-1", source_record_id="event-1", source_record_index=0,
        action_index=0, source_action_count=1, event_time=event_time, time_precision="second",
        eligibility_time=event_time, eligibility_inclusive=True, scenario="E-commerce",
        event_type="click", context={"product_id": "food-a"}, action={"type": "click"}, user_id="u1",
    )
    sample = CanonicalSample(
        sample_id="pretrain-u1", dataset_name="test-food", split="train", user_id="u1",
        cutoff_time=datetime.fromtimestamp(90, timezone.utc), cutoff_inclusive=False,
        history_order_known=True, history_truncated=False, history_group_truncated=False,
        profile={"active_days_30d": 1}, events=(event,),
    )
    vocab = fit_vocabulary_from_samples(
        [sample], VocabularyConfig(dataset_name="test-food", text_hash_buckets=16),
    )
    sizes = vocab.model_cardinalities()
    sizes.pop("text_hash_buckets")
    model_config = FoundationModelConfig(
        **sizes, d_model=8, num_heads=2, num_layers=1, max_length=64,
        num_user_buckets=16, categorical_pair_buckets=31,
        sequence_source_id=vocab.source_to_id["sequence"], numeric_modality_id=vocab.modality_to_id["numeric"],
    )
    checkpoint = tmp_path / "best_checkpoint.pt"
    save_checkpoint(checkpoint, FoundationModel(model_config), model_config, vocab, epoch=10, step=100)
    vocab_path, record_path = tmp_path / "vocab.json", tmp_path / "experiment_record.json"
    write_json(vocab_path, vocab.to_metadata())
    record = {
        "dataset_name": "test-food", "dataset_version": "v1", "model_config": asdict(model_config),
        "best_checkpoint_sha256": sha256_file(checkpoint), "epochs_completed": 10,
    }
    write_json(record_path, record)
    handoff = {
        "experiment_record_path": str(record_path), "expected_record_sha256": sha256_file(record_path),
        "vocab_path": str(vocab_path), "expected_vocab_sha256": sha256_file(vocab_path),
        "expected_checkpoint_sha256": sha256_file(checkpoint),
    }
    records = valid_records()
    records[0] = make_record("train-1", "train", "2021-03-01T01:00:00Z", user="meituan:user:u1")
    for index, record in enumerate(records[1:], start=1):
        records[index] = make_record(
            record["sample_id"], record["label_metadata"]["split"], record["input"]["cutoff_time"],
            user="meituan:user:9",
            **({"items": ("meituan:spu:1", "meituan:spu:test-only"), "positives": (1, 0)}
               if record["label_metadata"]["split"] == "test" else {}),
        )
    samples_path, samples_sha = write_samples_zip(tmp_path, records)
    _, catalog_sha = load_catalog_fixture(tmp_path)
    return {
        "data": {"path": str(samples_path), "member": "samples.jsonl", "expected_sha256": samples_sha,
                 "expected_version": "food-v1", "namespace": "test-food"},
        "catalog": {"pois_path": str(tmp_path / "pois.txt"), "expected_sha256": catalog_sha},
        "checkpoint": {"path": str(checkpoint), "expected_version": "v1", "handoff": handoff},
        "sequence": {"max_tokens": 64, "max_event_count": 5}, "training": {"batch_size": 2},
        "candidate_policy": "sampled_catalog",
    }


def load_bound(cfg):
    return load_food_foundation(cfg["checkpoint"]["path"], expected_version=cfg["checkpoint"]["expected_version"],
                                handoff=cfg["checkpoint"]["handoff"])


def test_person3_public_loader_encode_and_effective_provenance(person3):
    original_sha = sha256_file(person3["checkpoint"]["path"])
    foundation = load_bound(person3).eval()
    inputs = load_food_inputs(person3, foundation)
    _, config, vocabulary, payload = load_checkpoint(person3["checkpoint"]["path"], load_model=False)
    effective = resolve_food_checkpoint_metadata(person3, payload, config, vocabulary)
    assert "dataset_version" not in payload["preprocess_metadata"]
    assert "sequence" not in payload["preprocess_metadata"]
    assert effective["checkpoint_preprocess_metadata"] == payload["preprocess_metadata"]
    assert effective["downstream_sequence"] == person3["sequence"]
    assert effective["person3_handoff"]["pretrain_code_revision"] == "unknown"
    assert inputs.preprocess_metadata == {**effective, "food_samples": inputs.preprocess_metadata["food_samples"]}
    assert inputs.preprocess_fingerprint == fingerprint(inputs.preprocess_metadata)
    stats = inputs.preprocess_metadata["food_samples"]
    assert stats["requests_by_split"] == {"train": 2, "validation": 1, "test": 1}
    assert stats["requests_with_user_in_vocab_by_split"] == {"train": 1, "validation": 0, "test": 0}
    assert stats["users_in_vocab"] == 1 and stats["users_total"] == 2
    assert len(stats["caveats"]) == 4
    assert inputs.data_manifest_sha256 == person3["data"]["expected_sha256"]
    assert inputs.manifest["candidate_policy"] == "sampled_catalog"
    for split in ("train", "validation", "test"):
        with torch.no_grad():
            encoded = foundation.embed_users(inputs.batches[split][0].foundation)
        assert all(value.shape == (len(inputs.batches[split][0].rows), 8) and torch.isfinite(value).all() for value in encoded.values())
    assert sha256_file(person3["checkpoint"]["path"]) == original_sha


@pytest.mark.parametrize("field", ["expected_record_sha256", "expected_vocab_sha256", "expected_checkpoint_sha256"])
def test_rejects_tampered_binding_hash(person3, field):
    person3["checkpoint"]["handoff"][field] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_bound(person3)


@pytest.mark.parametrize("field,value,match", [
    ("dataset_name", "taobao", "namespace mismatch"),
    ("dataset_version", "other-version", "version mismatch"),
    ("model_config", {}, "model configuration mismatch"),
    ("best_checkpoint_sha256", "0" * 64, "does not identify this best checkpoint"),
])
def test_rejects_semantically_wrong_record_even_with_updated_hash(person3, field, value, match):
    binding = person3["checkpoint"]["handoff"]
    path = Path(binding["experiment_record_path"])
    record = json.loads(path.read_text(encoding="utf-8"))
    record[field] = value
    write_json(path, record)
    binding["expected_record_sha256"] = sha256_file(path)
    with pytest.raises(ValueError, match=match):
        load_bound(person3)


# model_config keys of the frozen Person 3 record (experiments/person3_checkpoint), which predates
# FoundationModelConfig fields added later with backward-compatible defaults.
PERSON3_RECORD_MODEL_KEYS = (
    "categorical_pair_buckets", "d_model", "default_attention_mode", "dropout", "key_vocab_size",
    "max_length", "modality_vocab_size", "num_heads", "num_layers", "num_user_buckets",
    "numeric_modality_id", "sequence_source_id", "source_vocab_size", "time_gap_vocab_size",
    "value_vocab_size",
)


def _rebind_record(person3, update):
    binding = person3["checkpoint"]["handoff"]
    path = Path(binding["experiment_record_path"])
    record = json.loads(path.read_text(encoding="utf-8"))
    update(record)
    write_json(path, record)
    binding["expected_record_sha256"] = sha256_file(path)


def _keep_person3_keys(record):
    record["model_config"] = {key: record["model_config"][key] for key in PERSON3_RECORD_MODEL_KEYS}


def test_record_predating_new_model_config_fields_still_binds(person3):
    _rebind_record(person3, _keep_person3_keys)
    assert isinstance(load_bound(person3), FoundationModel)


def _resave_checkpoint(person3, **changes):
    """Re-save the bound checkpoint with ``changes`` to its model config; return the new config."""
    path = Path(person3["checkpoint"]["path"])
    _, config, vocab, _ = load_checkpoint(path, load_model=False)
    config = replace(config, **changes)
    save_checkpoint(path, FoundationModel(config), config, vocab, epoch=10, step=100)
    person3["checkpoint"]["handoff"]["expected_checkpoint_sha256"] = sha256_file(path)
    return config


def test_record_omitting_a_field_the_checkpoint_changed_is_rejected(person3):
    _resave_checkpoint(person3, long_term_injection="cls")

    def update(record):
        _keep_person3_keys(record)
        record["best_checkpoint_sha256"] = person3["checkpoint"]["handoff"]["expected_checkpoint_sha256"]

    _rebind_record(person3, update)
    with pytest.raises(ValueError, match="model configuration mismatch"):
        load_bound(person3)


@pytest.mark.parametrize("key", ["default_attention_mode", "num_user_buckets", "categorical_pair_buckets"])
def test_record_omitting_a_preexisting_field_is_rejected_even_at_its_default(person3, key):
    # Every listed field holds its dataclass default; only fields added later may be omitted.
    config = _resave_checkpoint(person3, num_user_buckets=50_000, categorical_pair_buckets=100_003)

    def update(record):
        record["model_config"] = {k: v for k, v in asdict(config).items() if k != key}
        record["best_checkpoint_sha256"] = person3["checkpoint"]["handoff"]["expected_checkpoint_sha256"]

    _rebind_record(person3, update)
    with pytest.raises(ValueError, match="model configuration mismatch"):
        load_bound(person3)


def test_rejects_other_vocabulary_even_with_updated_hash(person3):
    binding = person3["checkpoint"]["handoff"]
    path = Path(binding["vocab_path"])
    vocab = json.loads(path.read_text(encoding="utf-8"))
    vocab["text_hash_buckets"] += 1
    write_json(path, vocab)
    binding["expected_vocab_sha256"] = sha256_file(path)
    with pytest.raises(ValueError, match="vocabulary differs"):
        load_bound(person3)


@pytest.mark.parametrize("sequence", [None, {}, {"max_tokens": 65, "max_event_count": 5},
                                     {"max_tokens": 64, "max_event_count": 0},
                                     {"max_tokens": True, "max_event_count": 5}])
def test_requires_explicit_bounded_downstream_sequence(person3, sequence):
    person3["sequence"] = sequence
    with pytest.raises(ValueError, match="explicit valid downstream sequence"):
        load_food_inputs(person3, load_bound(person3))


def test_rejects_data_namespace_mismatch(person3):
    person3["data"]["namespace"] = "taobao"
    with pytest.raises(ValueError, match="namespace incompatible"):
        load_food_inputs(person3, load_bound(person3))


def test_unseen_candidate_rates_use_train_vocabulary_only(person3):
    stats = load_food_inputs(person3, load_bound(person3)).preprocess_metadata["food_samples"]
    assert stats["unseen_candidate_rate_by_split"]["train"] == 0.0
    assert stats["unseen_candidate_rate_by_split"]["test"] == 0.5
    assert stats["unseen_positive_rate_by_split"]["test"] == 0.0


def test_legacy_path_requires_checkpoint_metadata_and_keeps_fingerprint(person3):
    path = person3["checkpoint"]["path"]
    person3["checkpoint"].pop("handoff")
    with pytest.raises(ValueError, match="dataset version mismatch"):
        load_food_foundation(path, expected_version="v1")
    payload = torch.load(path, weights_only=True)
    payload["preprocess_metadata"].update(dataset_version="v1", sequence=person3["sequence"])
    torch.save(payload, path)
    foundation = load_food_foundation(path, expected_version="v1")
    inputs = load_food_inputs(person3, foundation)
    expected = {**payload["preprocess_metadata"], "food_samples": inputs.preprocess_metadata["food_samples"]}
    assert inputs.preprocess_metadata == expected
    assert inputs.preprocess_fingerprint == fingerprint(expected)


def test_end_to_end_samples_run_exports_and_evaluates(person3, tmp_path):
    cfg = {**person3, "contract_version": 1, "seed": 42, "freeze_foundation": True, "split": "test",
           "artifact_dir": str(tmp_path / "artifacts"), "export_model": True, "evaluate": True,
           "evaluation": {"item_kind": "food", "label_availability_policy": "handoff_timestamp",
                          "observation_complete_policy": "assume_complete_diagnostic", "ks": [1, 3, 5, 10]},
           "training": {"epochs": 2, "learning_rate": 0.01, "batch_size": 2,
                        "cache_foundation_embeddings": True, "cpu_threads": 1},
           "model": {"normalization": "l2", "hidden_dim": 8, "candidate_dim": 4}}
    result = run_food_recommendation(cfg, FoodServices(evaluator=food_baseline_evaluator(cfg)))
    assert result["evaluation_status"] == "diagnostic"
    assert {"mrr", "ndcg_at_10", "recall_at_1", "recall_at_5"} <= set(result["metrics"])
    manifest = json.loads(Path(result["predictions_path"]).with_name("run_manifest.json").read_text())
    assert manifest["preprocessing"]["food_samples"]["mapping_version"] == "meituan-food-samples-to-trd-v2-v1"
    assert manifest["evaluation_policy"]["observation_complete_policy"] == "assume_complete_diagnostic"
    stats = manifest["preprocessing"]["food_samples"]
    assert stats["samples_sha256"] == person3["data"]["expected_sha256"]
    assert stats["pois_sha256"] == person3["catalog"]["expected_sha256"]
    for key in ("requests_with_user_in_vocab_by_split", "users_in_vocab", "unseen_candidate_rate_by_split"):
        assert key in stats


def test_runner_data_error_exits_code_three_before_any_embedding(person3, tmp_path):
    from src.downstream.food_recommendation import FoodRunError

    path = Path(person3["data"]["path"])
    import zipfile
    with zipfile.ZipFile(path) as archive:
        lines = archive.read("samples.jsonl").decode("utf-8").splitlines()
    records = [json.loads(line) for line in lines]
    records[0]["input"]["history"][0]["event_time"] = records[0]["input"]["cutoff_time"]
    path.unlink()
    new_path, new_sha = write_samples_zip(tmp_path, records)
    person3["data"].update(path=str(new_path), expected_sha256=new_sha)
    artifacts = tmp_path / "artifacts"
    cfg = {**person3, "contract_version": 1, "seed": 42, "freeze_foundation": True, "split": "test",
           "artifact_dir": str(artifacts), "export_model": False,
           "training": {"epochs": 1, "learning_rate": 0.01, "batch_size": 2},
           "model": {"hidden_dim": 8, "candidate_dim": 4}}
    calls = []
    real_loader = load_bound

    def recording_loader(*args, **kwargs):
        model = real_loader({**person3, "checkpoint": {
            "path": args[0], "expected_version": kwargs["expected_version"],
            "handoff": kwargs["handoff"]}})
        original = model.embed_users
        model.embed_users = lambda batch: calls.append(1) or original(batch)
        return model

    with pytest.raises(FoodRunError) as caught:
        run_food_recommendation(cfg, FoodServices(load_checkpoint=recording_loader))
    assert caught.value.code == 3
    assert "history event at or after cutoff" in str(caught.value)
    assert calls == []
    assert not artifacts.exists()
