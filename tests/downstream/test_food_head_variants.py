"""Head compatibility and cache safety for frozen-foundation model selection."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest
import torch

from src.contracts.output import sha256_file
from src.downstream.food_recommendation import (
    FoodRanker,
    FoodRunError,
    food_loss,
    load_food_ranker_checkpoint,
    prepare_frozen_embeddings,
    run_food_recommendation,
)
from tests.downstream.test_food_recommendation import (
    FakeFoundation,
    config,
    food_inputs,
    services,
)


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("normalization", ["none", "layernorm", "l2"])
@pytest.mark.parametrize("architecture", ["mlp", "mlp_dot"])
def test_head_variants_train_with_padding_and_large_scale_embeddings(normalization, architecture):
    torch.manual_seed(42)
    ranker = FoodRanker(4, 5, hidden_dim=8, candidate_dim=3, dropout=0.1,
                        normalization=normalization, architecture=architecture,
                        interaction_dim=2)
    ids = torch.tensor([[1, 2, 3], [2, 4, 0]])
    mask = ids.ne(0)
    scores = ranker(torch.randn(2, 4) * 100, torch.randn(2, 4) * 0.01, ids)
    assert scores.shape == (2, 3) and torch.isfinite(scores).all()
    loss = food_loss(scores, torch.tensor([[1., 0., 0.], [1., 0., 0.]]), mask)
    loss.backward()
    assert torch.isfinite(loss)
    for parameter in ranker.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    assert ranker.candidates.weight.grad[0].count_nonzero() == 0
    if architecture == "mlp_dot":
        assert ranker.user_projection.weight.grad.abs().sum() > 0
        assert ranker.candidate_projection.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("normalization", ["layernorm", "l2"])
def test_normalization_makes_scores_invariant_to_positive_user_rescaling(normalization):
    torch.manual_seed(14)
    ranker = FoodRanker(4, 5, normalization=normalization, architecture="mlp_dot").eval()
    long_term, short_term = torch.randn(2, 4), torch.randn(2, 4)
    ids = torch.tensor([[1, 2], [3, 4]])
    expected = ranker(long_term, short_term, ids)
    actual = ranker(long_term * 100, short_term * 40, ids)
    assert torch.allclose(actual, expected, atol=1e-4, rtol=1e-4)
    # A missing/zero user vector must also remain finite under L2 or LayerNorm.
    assert torch.isfinite(ranker(torch.zeros_like(long_term), torch.zeros_like(short_term), ids)).all()


@pytest.mark.parametrize("normalization", ["none", "layernorm", "l2"])
@pytest.mark.parametrize("architecture", ["mlp", "mlp_dot"])
def test_exported_variant_restores_exact_predictions(tmp_path, normalization, architecture):
    foundation, inputs = FakeFoundation(), food_inputs()
    inputs.preprocess_metadata = {"history_source": "canonical", "history_cutoff_checked": True}
    cfg = config(tmp_path)
    cfg.update(export_model=True, evaluate=False)
    cfg["model"].update(normalization=normalization, architecture=architecture,
                         interaction_dim=3, candidate_dim=2, dropout=0.1)
    result = run_food_recommendation(cfg, services(foundation, inputs))
    loaded, vocabulary, payload = load_food_ranker_checkpoint(result["model_path"],
        expected_foundation_sha256=sha256_file(cfg["checkpoint"]["path"]),
        expected_data_version=cfg["data"]["expected_version"])
    assert vocabulary == inputs.candidate_vocabulary
    assert (payload["normalization"], payload["architecture"], payload["interaction_dim"]) == (
        normalization, architecture, 3)
    assert payload["preprocess_fingerprint"] == inputs.preprocess_fingerprint
    manifest = json.loads(Path(result["predictions_path"]).with_name("run_manifest.json").read_text())
    assert manifest["preprocessing"] == inputs.preprocess_metadata
    assert manifest["model_hyperparameters"]["normalization_scope"] == "long_term_and_short_term_only"
    exported = {(row["request_id"], row["candidate_id"]): row["score"]
                for row in pq.read_table(result["predictions_path"]).to_pylist()}
    with torch.no_grad():
        for batch in inputs.batches["test"]:
            embeddings = foundation.embed_users(batch.foundation)
            scores = loaded(embeddings["long_term"], embeddings["short_term"], batch.candidate_ids)
            for index, rows in enumerate(batch.rows):
                for column, row in enumerate(rows):
                    assert float(scores[index, column]) == exported[row["request_id"], row["candidate_id"]]


@pytest.mark.parametrize("bad_model", [
    {"normalization": "batchnorm"}, {"architecture": "unknown"}, {"interaction_dim": 0},
])
def test_invalid_head_configuration_fails_before_checkpoint_load(tmp_path, bad_model):
    cfg = config(tmp_path)
    cfg["model"].update(bad_model)
    dependency = services()
    dependency.load_checkpoint = lambda *args, **kwargs: pytest.fail("checkpoint should not be loaded")
    with pytest.raises(FoodRunError) as error:
        run_food_recommendation(cfg, dependency)
    assert error.value.code == 2


def test_frozen_cache_only_encodes_requested_splits_and_matches_live_training(tmp_path):
    torch.manual_seed(62)
    foundation, inputs = FakeFoundation(), food_inputs()
    cfg = config(tmp_path)
    cfg.update(split="validation", export_model=True, evaluate=False)
    cfg["model"].update(normalization="layernorm", architecture="mlp_dot", dropout=0.1)
    cfg["training"].update(shuffle=True)
    cache = prepare_frozen_embeddings(foundation, inputs, cfg["checkpoint"]["path"])
    assert len(foundation.calls) == 2
    assert all(not training and not gradient for training, gradient in foundation.calls)
    assert id(inputs.batches["test"][0]) not in cache.embeddings
    assert all(not tensor.requires_grad for value in cache.embeddings.values() for tensor in value.values())
    dependency = services(foundation, inputs)
    dependency.frozen_embeddings = cache
    cached = run_food_recommendation(cfg, dependency)
    assert len(foundation.calls) == 2  # No repeated foundation computation while tuning.
    fresh = services(foundation, inputs)

    def load_with_rng_consumption(*args, **kwargs):
        # A fresh checkpoint loader constructs its model and consumes RNG.
        torch.randn(97)
        return foundation

    fresh.load_checkpoint = load_with_rng_consumption
    live = run_food_recommendation(cfg, fresh)
    cached_rows = pq.read_table(cached["predictions_path"]).to_pylist()
    live_rows = pq.read_table(live["predictions_path"]).to_pylist()
    assert [row["score"] for row in cached_rows] == [row["score"] for row in live_rows]
    first = torch.load(cached["model_path"], weights_only=True)["ranker_state_dict"]
    second = torch.load(live["model_path"], weights_only=True)["ranker_state_dict"]
    assert all(torch.equal(first[name], second[name]) for name in first)


@pytest.mark.parametrize("problem", ["checkpoint", "data", "preprocess", "missing", "identity",
                                      "shape", "nan", "gradient", "keys"])
def test_frozen_cache_rejects_wrong_bindings_or_invalid_embeddings(tmp_path, problem):
    foundation, inputs = FakeFoundation(), food_inputs()
    cfg = config(tmp_path)
    cfg["split"] = "validation"
    cache = prepare_frozen_embeddings(foundation, inputs, cfg["checkpoint"]["path"])
    key = id(inputs.batches["train"][0])
    if problem == "checkpoint":
        cache.checkpoint_sha256 = "0" * 64
    elif problem == "data":
        cache.data_manifest_sha256 = "0" * 64
    elif problem == "preprocess":
        cache.preprocess_fingerprint = "0" * 64
    elif problem == "missing":
        del cache.embeddings[key]
        del cache.batches[key]
    elif problem == "identity":
        cache.batches[key] = copy.deepcopy(cache.batches[key])
    elif problem == "shape":
        cache.embeddings[key]["short_term"] = torch.zeros(2, 3)
    elif problem == "nan":
        cache.embeddings[key]["short_term"][0, 0] = float("nan")
    elif problem == "gradient":
        cache.embeddings[key]["long_term"].requires_grad_()
    else:
        cache.embeddings[key]["unexpected"] = torch.zeros(2, 4)
    dependency = services(foundation, inputs)
    dependency.frozen_embeddings = cache
    with pytest.raises(FoodRunError) as error:
        run_food_recommendation(cfg, dependency)
    assert error.value.code == 4


def test_frozen_cache_cannot_be_used_for_finetuning_or_unencoded_test(tmp_path):
    foundation, inputs = FakeFoundation(), food_inputs()
    cfg = config(tmp_path)
    dependency = services(foundation, inputs)
    dependency.frozen_embeddings = prepare_frozen_embeddings(foundation, inputs, cfg["checkpoint"]["path"])
    cfg["freeze_foundation"] = False
    with pytest.raises(FoodRunError, match="while fine-tuning"):
        run_food_recommendation(cfg, dependency)
    cfg["freeze_foundation"] = True
    with pytest.raises(FoodRunError, match="missing batch"):
        run_food_recommendation(cfg, dependency)
