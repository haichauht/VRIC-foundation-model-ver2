"""Pretrain v2 end to end on the tiny simulator world: long_term from the long history."""

from __future__ import annotations

import copy
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch

from src.contracts.output import validate_and_write_predictions
from src.downstream.ad_inputs import ad_baseline_evaluator, load_ad_foundation, load_ad_inputs
from src.downstream.ad_optimization import AdServices, run_ad_optimization
from src.downstream.food_inputs import (
    food_baseline_evaluator,
    load_food_foundation,
    load_food_inputs,
)
from src.downstream.food_recommendation import FoodServices, run_food_recommendation
from src.foundation_input.history_features import HistorySchema, epoch_seconds
from src.foundation_input.loader import iter_normalized_samples
from src.training.checkpoint import load_checkpoint
from src.training.pretrain import pretrain


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def v2(sim_tiny_packages, tmp_path_factory):
    out = tmp_path_factory.mktemp("pretrain_v2")
    config = {
        "seed": 7, "dataset": None, "data_dir": str(sim_tiny_packages.pretrain),
        "output_dir": str(out), "device": "cpu",
        "model": {"d_model": 16, "num_heads": 2, "num_layers": 1, "dropout": 0.0,
                  "num_user_buckets": 64, "categorical_pair_buckets": 31,
                  "long_term_injection": "cls", "long_term_source": "history"},
        "objectives": {"history_target_weight": 0.5},
        "sequence": {"max_length": 512, "max_event_count": 30},
        "training": {"batch_size": 8, "epochs": 1, "learning_rate": 0.001},
    }
    record = pretrain(config, {"smoke": True})
    return record, out / "checkpoints" / "latest_checkpoint.pt"


def manifest(root: Path) -> dict:
    return json.loads((root / "manifest.json").read_text(encoding="utf-8"))


def test_checkpoint_records_the_history_schema(v2):
    record, checkpoint = v2
    _, model_config, _, payload = load_checkpoint(checkpoint, load_model=False)
    schema = HistorySchema.from_dict(payload["preprocess_metadata"]["history_features"])
    assert model_config.long_term_source == "history"
    assert model_config.history_feature_dim == schema.size
    assert model_config.history_target_sizes == schema.target_sizes
    assert schema.cuisines and schema.categories
    assert record["model_config"]["long_term_source"] == "history"


def test_loader_counts_match_the_raw_event_shard(sim_tiny_packages, v2):
    payload = load_checkpoint(v2[1], load_model=False)[3]
    schema = HistorySchema.from_dict(payload["preprocess_metadata"]["history_features"])
    sample = next(iter_normalized_samples(sim_tiny_packages.pretrain, "train", validate=False,
                                          history_schema=schema))
    shard = next(s for s in manifest(sim_tiny_packages.pretrain)["shards"]
                 if s["user_id"] == sample.user_id)
    events = pq.read_table(sim_tiny_packages.pretrain / shard["events_file"]).to_pylist()
    names = dict(zip(schema.names, sample.history_features, strict=True))
    cutoff = epoch_seconds(sample.cutoff_time)

    def when(row):
        value = row["event_time"]
        return epoch_seconds(value if isinstance(value, datetime) else datetime.fromisoformat(value))

    orders = sum(row["event_type"] == "order" and when(row) <= cutoff for row in events)
    assert names["log_count.type.order"] == pytest.approx(np.log1p(orders) / 8)
    assert len(sample.history_targets) == sum(schema.target_sizes)
    assert set(sample.history_target_mask) <= {0.0, 1.0}


def ads_config(packages, checkpoint, tmp_path, init="pretrained"):
    return {
        "contract_version": 1, "seed": 42,
        "data": {"format": "sim_v1", "root": str(packages.ads),
                 "expected_version": manifest(packages.ads)["dataset_version"]},
        "checkpoint": {"path": str(checkpoint)}, "foundation": {"init": init},
        "freeze_foundation": True, "split": "test", "artifact_dir": str(tmp_path / "ads"),
        "exposure_policy": "sim-served-impression-single-candidate-v1",
        "missing_click_policy": "reject", "export_model": True, "evaluate": True,
        "evaluation": {"label_availability_policy": "handoff_timestamp"},
        "model": {"hidden_dim": 8, "candidate_dim": 4, "candidate_attributes": ["category"]},
        "training": {"epochs": 1, "learning_rate": 0.01, "batch_size": 256,
                     "cache_foundation_embeddings": True, "cpu_threads": 1},
        "calibration": {"enabled": False, "split": "validation"},
    }


def food_config(packages, checkpoint, tmp_path, init="pretrained", events_root=True):
    data = {"format": "sim_v1", "root": str(packages.food),
            "expected_version": manifest(packages.food)["dataset_version"]}
    if events_root:
        data["events_root"] = str(packages.ads)
    return {
        "contract_version": 1, "seed": 42, "data": data,
        "checkpoint": {"path": str(checkpoint)}, "foundation": {"init": init},
        "freeze_foundation": True, "split": "test", "artifact_dir": str(tmp_path / "food"),
        "export_model": False, "evaluate": True,
        "evaluation": {"item_kind": "food", "label_availability_policy": "handoff_timestamp",
                       "observation_complete_policy": "assume_complete_diagnostic", "ks": [1, 10]},
        "model": {"normalization": "l2", "hidden_dim": 8, "candidate_dim": 4},
        "training": {"epochs": 1, "learning_rate": 0.01, "batch_size": 64,
                     "cache_foundation_embeddings": True, "cpu_threads": 1},
        "sequence": {"max_tokens": 512, "max_event_count": 30},
    }


@pytest.mark.parametrize("init", ["pretrained", "random"])
def test_ads_runs_on_a_history_checkpoint(sim_tiny_packages, v2, tmp_path, init):
    config = ads_config(sim_tiny_packages, v2[1], tmp_path, init)
    result = run_ad_optimization(config, AdServices(
        load_ad_foundation, load_ad_inputs, validate_and_write_predictions,
        ad_baseline_evaluator(config)))
    assert "roc_auc" in result["metrics"]


def test_ads_batches_carry_history_features(sim_tiny_packages, v2, tmp_path):
    config = ads_config(sim_tiny_packages, v2[1], tmp_path)
    foundation = load_ad_foundation(
        str(v2[1]), expected_version=config["data"]["expected_version"],
        expected_sim_config_sha256=manifest(sim_tiny_packages.ads)["sim_config_sha256"])
    inputs = load_ad_inputs(config, foundation)
    batch = inputs.batches["test"][0]
    assert batch.foundation.history_features.shape == (len(batch.rows), foundation.config.history_feature_dim)
    with torch.no_grad():
        long_term = foundation.embed_users(batch.foundation)["long_term"]
    assert long_term.std(0).sum() > 0  # users differ through their history, not their ID


def test_food_runs_with_events_root_and_needs_it(sim_tiny_packages, v2, tmp_path):
    config = food_config(sim_tiny_packages, v2[1], tmp_path)
    result = run_food_recommendation(config, FoodServices(evaluator=food_baseline_evaluator(config)))
    assert "ndcg_at_10" in result["metrics"]
    missing = food_config(sim_tiny_packages, v2[1], tmp_path, events_root=False)
    foundation = load_food_foundation(
        str(v2[1]), expected_version=missing["data"]["expected_version"],
        expected_sim_config_sha256=manifest(sim_tiny_packages.food)["sim_config_sha256"])
    with pytest.raises(ValueError, match="events_root"):
        load_food_inputs(missing, foundation)


def test_food_rejects_an_event_log_from_another_simulation(sim_tiny_packages, v2, tmp_path):
    other = tmp_path / "ads"
    other.mkdir()
    payload = copy.deepcopy(manifest(sim_tiny_packages.ads))
    payload["sim_config_sha256"] = "0" * 64
    (other / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    config = food_config(sim_tiny_packages, v2[1], tmp_path)
    config["data"]["events_root"] = str(other)
    foundation = load_food_foundation(
        str(v2[1]), expected_version=config["data"]["expected_version"],
        expected_sim_config_sha256=manifest(sim_tiny_packages.food)["sim_config_sha256"])
    with pytest.raises(ValueError, match="different simulation"):
        load_food_inputs(config, foundation)
