"""Food on the simulator package (``data.format: sim_v1``) and the random-init control arm."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
from dataclasses import dataclass, replace
from pathlib import Path

import pyarrow.parquet as pq
import pytest
import torch
import yaml

from src.contracts.output import sha256_file
from src.downstream.food_inputs import (
    food_baseline_evaluator,
    load_food_foundation,
    load_food_inputs,
    tokenizer_from_vocabulary,
)
from src.downstream.food_recommendation import (
    FoodRunError,
    FoodServices,
    run_food_recommendation,
)
from src.downstream.food_samples import (
    load_merchant_catalog,
    sim_candidate_rows,
    sim_to_canonical_sample,
)
from src.downstream.foundation_init import reinitialize_foundation
from src.foundation_input.dataset import DatasetConfig, encode_sample
from src.foundation_input.loader import iter_normalized_samples
from src.foundation_input.manifest import load_manifest
from src.foundation_input.validation import validate_sample
from src.foundation_input.vocab import (
    VocabularyArtifacts,
    VocabularyConfig,
    fit_vocabulary_from_samples,
)
from src.models.encoder import FoundationModel, FoundationModelConfig
from src.training.checkpoint import load_checkpoint, save_checkpoint
from src.training.pretrain import pretrain_preprocess_metadata, tokenizer_config_for_manifest

UNKNOWN_MERCHANT = "sim:merchant:999999"
EVALUATION = {
    "item_kind": "food", "label_availability_policy": "handoff_timestamp",
    "observation_complete_policy": "assume_complete_diagnostic", "ks": [1, 3, 5, 10],
}


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@dataclass(frozen=True)
class SimFood:
    packages: object  # tests/conftest.py SimPackages
    manifest: dict  # food/manifest.json
    checkpoint: Path
    vocab: VocabularyArtifacts


@pytest.fixture(scope="module")
def sim_food(sim_tiny_packages, tmp_path_factory) -> SimFood:
    """A tiny sim checkpoint built the way pretrain builds one, on the same simulated world."""
    pretrain = load_manifest(sim_tiny_packages.pretrain, validate_shard_files=False)
    vocab = fit_vocabulary_from_samples(
        iter_normalized_samples(sim_tiny_packages.pretrain, split="train", validate=False),
        VocabularyConfig(dataset_name=pretrain.dataset_name),
        tokenizer_config_for_manifest(pretrain),
    )
    sizes = vocab.model_cardinalities()
    sizes.pop("text_hash_buckets")
    config = FoundationModelConfig(
        **sizes, d_model=16, num_heads=2, num_layers=1, max_length=512,
        num_user_buckets=64, categorical_pair_buckets=31,
        sequence_source_id=vocab.source_to_id["sequence"],
        numeric_modality_id=vocab.modality_to_id["numeric"], long_term_injection="cls",
    )
    torch.manual_seed(0)
    checkpoint = tmp_path_factory.mktemp("sim_food") / "best_checkpoint.pt"
    save_checkpoint(
        checkpoint, FoundationModel(config), config, vocab,
        preprocess_metadata=pretrain_preprocess_metadata(
            pretrain, vocab, max_length=512, max_event_count=30),
    )
    manifest = json.loads((sim_tiny_packages.food / "manifest.json").read_text(encoding="utf-8"))
    return SimFood(sim_tiny_packages, manifest, checkpoint, vocab)


def sim_config(sim_food: SimFood, tmp_path, *, root=None, checkpoint=None, init="pretrained"):
    return {
        "contract_version": 1, "seed": 42,
        "data": {"format": "sim_v1", "root": str(root or sim_food.packages.food),
                 "expected_version": sim_food.manifest["dataset_version"]},
        "checkpoint": {"path": str(checkpoint or sim_food.checkpoint)},
        "foundation": {"init": init},
        "freeze_foundation": True, "split": "test",
        "artifact_dir": str(tmp_path / "artifacts"), "export_model": False, "evaluate": True,
        "evaluation": copy.deepcopy(EVALUATION),
        "model": {"normalization": "l2", "hidden_dim": 8, "candidate_dim": 4},
        "training": {"epochs": 1, "learning_rate": 0.01, "batch_size": 64,
                     "cache_foundation_embeddings": True, "cpu_threads": 1},
        "sequence": {"max_tokens": 512, "max_event_count": 30},
    }


def load_sim_foundation(sim_food: SimFood, path=None):
    return load_food_foundation(
        path or sim_food.checkpoint, expected_version=sim_food.manifest["dataset_version"],
        expected_sim_config_sha256=sim_food.manifest["sim_config_sha256"],
    )


def food_records(food_root: Path) -> list[dict]:
    lines = (food_root / "samples.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def write_food_package(source: Path, target: Path, records: list[dict]) -> Path:
    """Copy a Food package with replaced samples; the manifest pins the new samples file."""
    target.mkdir(parents=True)
    payload = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n" for record in records
    ).encode("utf-8")
    (target / "samples.jsonl").write_bytes(payload)
    shutil.copyfile(source / "merchants.parquet", target / "merchants.parquet")
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    manifest["files"]["samples"].update(
        sha256=hashlib.sha256(payload).hexdigest(), row_count=len(records), bytes=len(payload))
    (target / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return target


@pytest.fixture(scope="module")
def sim_inputs(sim_food, tmp_path_factory):
    config = sim_config(sim_food, tmp_path_factory.mktemp("unused"))
    return load_food_inputs(config, load_sim_foundation(sim_food))


def test_sim_food_inputs_build(sim_food, sim_inputs):
    counts = sim_food.manifest["request_counts"]
    assert {split: sum(len(batch.rows) for batch in batches)
            for split, batches in sim_inputs.batches.items()} == counts
    assert sim_inputs.preprocess_metadata["food_samples"]["requests_by_split"] == counts
    merchants = pq.read_table(sim_food.packages.food / "merchants.parquet")
    candidate_ids = {row["candidate_id"] for row in sim_inputs.candidates}
    assert candidate_ids <= set(merchants.column("merchant_id").to_pylist())
    per_request = sim_food.manifest["candidates_per_request"]
    assert len(sim_inputs.candidates) == per_request * sum(counts.values())
    assert sim_inputs.manifest["candidate_policy"] == sim_food.manifest["candidate_policy"]
    assert sim_inputs.manifest["dataset_namespace"] == "sim_v1"
    assert sim_inputs.data_manifest_sha256 == sha256_file(sim_food.packages.food / "manifest.json")
    binding = sim_inputs.preprocess_metadata["sim_checkpoint_binding"]
    assert binding["sim_config_sha256"] == sim_food.manifest["sim_config_sha256"]
    assert sim_inputs.preprocess_metadata["downstream_sequence"] == {
        "max_tokens": 512, "max_event_count": 30}
    foundation = load_sim_foundation(sim_food).eval()
    for split, batches in sim_inputs.batches.items():
        with torch.no_grad():
            embedded = foundation.embed_users(batches[0].foundation)
        assert all(value.shape == (len(batches[0].rows), 16) and torch.isfinite(value).all()
                   for value in embedded.values()), split


def test_sim_user_ids_are_not_stripped(sim_food, sim_inputs):
    record = next(record for record in food_records(sim_food.packages.food)
                  if record["label_metadata"]["split"] == "train" and record["input"]["history"])
    sample = validate_sample(sim_to_canonical_sample(record, "sim_v1"))
    assert sample.user_id == record["input"]["user_id"] and sample.user_id.startswith("sim:user:")
    assert {event.user_id for event in sample.events} == {sample.user_id}
    tokenizer = tokenizer_from_vocabulary(sim_food.vocab)
    sequence = DatasetConfig(max_tokens=512, max_event_count=30)
    encoded = encode_sample(sample, sim_food.vocab, tokenizer, sequence)
    assert encoded.records[0].token.key == "user_id"
    assert encoded.records[0].token.value == record["input"]["user_id"]
    assert encoded.user_id_value_id is not None
    # The Meituan reader's bare id would be UNK in a vocabulary fitted on the sim package.
    bare = replace(sample, user_id=sample.user_id.rsplit(":", 1)[-1], events=())
    assert encode_sample(bare, sim_food.vocab, tokenizer, sequence).user_id_value_id is None
    stats = sim_inputs.preprocess_metadata["food_samples"]
    assert stats["requests_with_user_in_vocab_by_split"]["train"] == \
        stats["requests_by_split"]["train"]


def test_sim_sample_reads_truncation_flags_from_record(sim_food):
    seen = set()
    for record in food_records(sim_food.packages.food):
        flags = (record["input"]["history_truncated"], record["input"]["history_group_truncated"])
        if flags in seen:
            continue
        seen.add(flags)
        sample = sim_to_canonical_sample(record, "sim_v1")
        assert (sample.history_truncated, sample.history_group_truncated) == flags
        assert sample.sample_id == record["sample_id"]
        assert sample.split == record["label_metadata"]["split"]
    assert {(True, True), (True, False), (False, False)} <= seen


@pytest.mark.parametrize(("mutation", "match"), [
    ("other_sim_config", "sim_config_sha256"),
    ("no_provenance", "sim_config_sha256"),
    ("other_namespace", "dataset_name"),
])
def test_sim_checkpoint_provenance_mismatch_is_rejected(
    sim_food, tmp_path, monkeypatch, mutation, match
):
    payload = torch.load(sim_food.checkpoint, map_location="cpu", weights_only=False)
    if mutation == "other_sim_config":
        payload["preprocess_metadata"]["dataset_provenance"]["sim_config_sha256"] = "0" * 64
    elif mutation == "no_provenance":
        del payload["preprocess_metadata"]["dataset_provenance"]
    else:
        payload["vocabularies"]["dataset_name"] = "taobao"
    bad = tmp_path / "other_world.pt"
    torch.save(payload, bad)
    with pytest.raises(ValueError, match=match):
        load_sim_foundation(sim_food, bad)
    # The input loader re-checks the binding before any request is encoded or batched.
    from src.downstream import food_inputs

    built = []
    monkeypatch.setattr(food_inputs, "encode_sample", lambda *args: built.append(args))
    monkeypatch.setattr(food_inputs, "foundation_collate", lambda *args: built.append(args))
    config = sim_config(sim_food, tmp_path, checkpoint=bad)
    with pytest.raises(ValueError, match=match):
        load_food_inputs(config, load_sim_foundation(sim_food))
    assert built == []
    with pytest.raises(FoodRunError, match=match) as caught:
        run_food_recommendation(config, FoodServices(evaluator=lambda **kwargs: {}))
    assert caught.value.code == 4
    assert built == [] and not (tmp_path / "artifacts").exists()


def test_sim_candidate_not_in_merchants_is_rejected(sim_food, tmp_path):
    records = food_records(sim_food.packages.food)
    index = next(i for i, record in enumerate(records)
                 if record["label_metadata"]["split"] == "train")
    record = records[index]
    record["input"]["candidates"][0]["item_id"] = UNKNOWN_MERCHANT
    record["labels"][0]["item_id"] = UNKNOWN_MERCHANT
    catalog = load_merchant_catalog(sim_food.packages.food / "merchants.parquet",
                                    sim_food.manifest["files"]["merchants"]["sha256"])
    assert "sim:merchant:0" in catalog
    pattern = f"request_id={record['sample_id']}: .*{UNKNOWN_MERCHANT}"
    with pytest.raises(ValueError, match=pattern):
        sim_candidate_rows(record, catalog)
    package = write_food_package(sim_food.packages.food, tmp_path / "food", records)
    config = sim_config(sim_food, tmp_path, root=package)
    with pytest.raises(FoodRunError, match=pattern) as caught:
        run_food_recommendation(config, FoodServices(evaluator=lambda **kwargs: {}))
    assert caught.value.code == 3 and not (tmp_path / "artifacts").exists()


def test_merchant_catalog_is_hash_pinned(sim_food):
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_merchant_catalog(sim_food.packages.food / "merchants.parquet", "0" * 64)


def test_random_init_changes_weights_not_config(sim_food):
    model = load_checkpoint(sim_food.checkpoint)[0]
    before = {name: value.clone() for name, value in model.state_dict().items()}
    rng = torch.get_rng_state()
    first = reinitialize_foundation(model, seed=7)
    second = reinitialize_foundation(model, seed=7)
    other = reinitialize_foundation(model, seed=8)
    assert torch.equal(torch.get_rng_state(), rng)  # downstream seeding is not disturbed
    assert first is not model and first.config == model.config
    assert not first.training and not second.training
    assert {p.device for p in first.parameters()} == {p.device for p in model.parameters()}
    state, again = first.state_dict(), second.state_dict()
    assert state.keys() == before.keys()
    assert all(torch.equal(state[name], again[name]) for name in state)
    assert any(not torch.equal(state[name], before[name]) for name in state)
    assert any(not torch.equal(state[name], other.state_dict()[name]) for name in state)
    assert all(torch.equal(model.state_dict()[name], before[name]) for name in before)


@pytest.mark.parametrize("arm", ["pretrained", "random"])
def test_sim_run_evaluates_and_records_foundation_arm(sim_food, tmp_path, arm):
    config = sim_config(sim_food, tmp_path, init=arm)
    config["export_model"] = True
    used = []

    def load_inputs(cfg, foundation):
        used.append(foundation)
        return load_food_inputs(cfg, foundation)

    result = run_food_recommendation(config, FoodServices(
        load_inputs=load_inputs, evaluator=food_baseline_evaluator(config)))
    assert {"mrr", "ndcg_at_10", "recall_at_1"} <= set(result["metrics"])
    assert result["evaluation_status"] == "diagnostic"
    manifest = json.loads(Path(result["predictions_path"]).with_name(
        "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["foundation"]["init"] == arm
    assert manifest["candidate_or_exposure_policy"] == sim_food.manifest["candidate_policy"]
    assert manifest["request_count"] == sim_food.manifest["request_counts"]["test"]
    assert torch.load(result["model_path"], weights_only=True)["foundation_init"] == arm
    pretrained = load_checkpoint(sim_food.checkpoint)[0].state_dict()
    state = used[0].state_dict()
    unchanged = all(torch.equal(state[name], pretrained[name]) for name in state)
    assert unchanged == (arm == "pretrained")
    assert used[0].config == load_checkpoint(sim_food.checkpoint, load_model=False)[1]


@pytest.mark.parametrize(("change", "match"), [
    (lambda c: c["data"].update(format="parquet"), "data.format"),
    (lambda c: c.update(foundation={"init": "zeros"}), "foundation.init"),
    (lambda c: c.update(candidate_policy="sim-popularity-matched-home-work-v1"),
     "candidate_policy"),
    (lambda c: c.update(catalog={"pois_path": "pois.txt"}), "catalog"),
    (lambda c: c["checkpoint"].update(handoff={}), "handoff"),
    (lambda c: c["data"].pop("root"), "root"),
])
def test_sim_config_errors_exit_two(sim_food, tmp_path, change, match):
    config = sim_config(sim_food, tmp_path)
    change(config)
    with pytest.raises(FoodRunError, match=match) as caught:
        run_food_recommendation(config, FoodServices(evaluator=lambda **kwargs: {}))
    assert caught.value.code == 2


def test_sim_config_sequence_must_match_the_checkpoint(sim_food, tmp_path):
    config = sim_config(sim_food, tmp_path)
    config["sequence"] = {"max_tokens": 256, "max_event_count": 30}
    with pytest.raises(FoodRunError, match="sequence") as caught:
        run_food_recommendation(config, FoodServices(evaluator=lambda **kwargs: {}))
    assert caught.value.code == 3
    assert not (tmp_path / "artifacts").exists()


def test_checked_in_sim_config():
    sim = yaml.safe_load(Path("configs/food_recommendation_sim.yaml").read_text(encoding="utf-8"))
    assert sim["data"] == {"format": "sim_v1", "root": "experiments/sim/sim-smoke-v1/food",
                           "expected_version": "sim-smoke-v1",
                           "events_root": "experiments/sim/sim-smoke-v1/ads"}
    assert sim["checkpoint"] == {"path": "experiments/pretrain_sim/checkpoints/best_checkpoint.pt"}
    assert sim["foundation"] == {"init": "pretrained"}
    assert sim["sequence"] == {"max_tokens": 512, "max_event_count": 30}
    assert "catalog" not in sim and "candidate_policy" not in sim


def test_sim_run_with_candidate_attributes_exports_a_reloadable_head(sim_food, tmp_path):
    from src.downstream.food_recommendation import load_food_ranker_checkpoint

    config = sim_config(sim_food, tmp_path)
    config["export_model"] = True
    config["model"].update(candidate_attributes=["cuisine", "price_level", "geohash"],
                           attribute_dim=3)
    result = run_food_recommendation(config, FoodServices(evaluator=food_baseline_evaluator(config)))
    assert {"mrr", "ndcg_at_10"} <= set(result["metrics"])
    run_dir = Path(result["predictions_path"]).parent
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["model_hyperparameters"]["candidate_attributes"] == [
        "cuisine", "price_level", "geohash"]
    assert manifest["model_hyperparameters"]["attribute_dim"] == 3
    ranker, _, payload = load_food_ranker_checkpoint(
        result["model_path"], expected_foundation_sha256=sha256_file(sim_food.checkpoint),
        expected_data_version=sim_food.manifest["dataset_version"])
    catalog = load_merchant_catalog(
        sim_food.packages.food / "merchants.parquet",
        sim_food.manifest["files"]["merchants"]["sha256"])
    cuisines = {json.dumps(row["cuisine"], ensure_ascii=False) for row in catalog.values()}
    assert set(payload["attribute_vocabularies"]["cuisine"]) <= cuisines
    assert ranker.attribute_sizes == tuple(
        len(payload["attribute_vocabularies"][name]) + 1 for name in ("cuisine", "price_level", "geohash"))


def test_sim_inputs_carry_catalog_attribute_ids(sim_food, tmp_path):
    config = sim_config(sim_food, tmp_path)
    config["model"]["candidate_attributes"] = ["cuisine", "price_level"]
    inputs = load_food_inputs(config, load_sim_foundation(sim_food))
    catalog = load_merchant_catalog(
        sim_food.packages.food / "merchants.parquet",
        sim_food.manifest["files"]["merchants"]["sha256"])
    vocabularies = inputs.attribute_vocabularies
    batch = inputs.batches["test"][0]
    for index, group in enumerate(batch.rows):
        for column, row in enumerate(group):
            merchant = catalog[row["candidate_id"]]
            assert batch.candidate_attributes[index, column].tolist() == [
                vocabularies["cuisine"].get(json.dumps(merchant["cuisine"], ensure_ascii=False), 0),
                vocabularies["price_level"].get(json.dumps(merchant["price_level"]), 0)]


@pytest.mark.parametrize(("change", "code", "match"), [
    (lambda c: c["model"].update(candidate_attributes=["cuisine", "cuisine"]), 2,
     "candidate_attributes"),
    (lambda c: c["model"].update(candidate_attributes=["cuisine"], attribute_dim=0), 2,
     "attribute_dim"),
    (lambda c: c["training"].update(encoder_device="tpu"), 2, "encoder_device"),
    (lambda c: c["model"].update(candidate_attributes=["opening_hours"]), 3, "catalog column"),
])
def test_sim_attribute_and_device_config_errors(sim_food, tmp_path, change, code, match):
    config = sim_config(sim_food, tmp_path)
    change(config)
    with pytest.raises(FoodRunError, match=match) as caught:
        run_food_recommendation(config, FoodServices(evaluator=lambda **kwargs: {}))
    assert caught.value.code == code


def test_user_embeddings_export_matches_the_downstream_encoding(sim_food, tmp_path):
    import numpy as np

    from src.downstream.user_embeddings import export_user_embeddings, main

    config = sim_config(sim_food, tmp_path)
    paths = export_user_embeddings(config, task="food", out_dir=tmp_path / "emb")
    pretrained, random_arm = (np.load(path) for path in paths)
    requests = {record["sample_id"] for record in food_records(sim_food.packages.food)}
    assert set(pretrained["request_id"]) == requests and len(pretrained["request_id"]) == len(requests)
    assert pretrained["long_term"].shape == (len(requests), 16)
    assert not np.allclose(pretrained["short_term"], random_arm["short_term"])
    meta = json.loads(paths[0].with_suffix(".json").read_text(encoding="utf-8"))
    assert meta["init"] == "pretrained" and meta["checkpoint_sha256"] == sha256_file(sim_food.checkpoint)
    foundation = load_sim_foundation(sim_food)
    batch = load_food_inputs(config, foundation).batches["test"][0]
    with torch.no_grad():
        expected = foundation.embed_users(batch.foundation)["long_term"].numpy()
    rows = [list(pretrained["request_id"]).index(group[0]["request_id"]) for group in batch.rows]
    assert np.allclose(pretrained["long_term"][rows], expected, atol=1e-6)
    config_path = tmp_path / "food.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    assert main(["--task", "food", "--config", str(config_path), "--out", str(tmp_path / "cli"),
                 "--init", "random", "--device", "tpu"]) == 2

