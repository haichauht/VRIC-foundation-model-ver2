"""Food contract tests with a public-API encoder fake and the real shared writer."""

from __future__ import annotations

import copy
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from torch import nn

from src.contracts.output import (
    MANIFEST_KEYS,
    PREDICTION_SCHEMA,
    fingerprint,
    sha256_file,
    validate_and_write_predictions,
    validate_run_manifest,
)
from src.downstream.food_inputs import (
    food_baseline_dataset,
    food_baseline_evaluator,
)
from src.downstream.food_recommendation import (
    FoodBatch,
    FoodInputs,
    FoodRanker,
    FoodRunError,
    FoodServices,
    food_loss,
    load_food_ranker_checkpoint,
    run_food_recommendation,
    validate_food_candidates,
)
from src.foundation_input.types import FoundationBatch
from tests.downstream.test_food_samples import (
    load_catalog_fixture,
    valid_records,
    write_samples_zip,
)


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class FakeFoundation(nn.Module):
    """Expose embeddings only; record gradient/mode behavior without encoder internals."""

    def __init__(self):
        super().__init__()
        self.users = nn.Embedding(8, 4)
        self.calls = []

    def embed_users(self, batch):
        self.calls.append((self.training, torch.is_grad_enabled()))
        identity = self.users(batch.user_ids)
        return {"long_term": identity, "short_term": identity * 0.5}

    def forward(self, *args, **kwargs):
        raise AssertionError("Food must obtain foundation embeddings only through embed_users")


def foundation_batch():
    ids = torch.tensor([[1, 5, 2, 6], [1, 5, 2, 0]], dtype=torch.long)
    return FoundationBatch(
        input_ids=ids, key_ids=torch.zeros_like(ids), modality_ids=torch.zeros_like(ids),
        source_ids=torch.zeros_like(ids), position_ids=torch.zeros_like(ids),
        time_gap_ids=torch.zeros_like(ids), attention_mask=ids.ne(0),
        numeric_values=torch.zeros_like(ids, dtype=torch.float32),
        numeric_missing_mask=torch.zeros_like(ids, dtype=torch.bool),
        user_ids=torch.tensor([1, 2]), targets={},
    )


def candidate(split, request, identifier, positive):
    cutoff = {"train": 100, "validation": 200, "test": 300}[split]
    return {
        "request_id": request, "user_id": "user-1" if request.endswith("1") else "user-2",
        "cutoff_timestamp": cutoff, "candidate_id": identifier, "is_positive": positive,
        "split": split, "candidate_timestamp": cutoff - 10,
        "exposure_timestamp": cutoff, "label_timestamp": cutoff if positive else None,
        "profile_as_of_timestamp": cutoff - 30,
    }


def food_inputs():
    batches, candidates = {}, []
    vocabulary = {"food-a": 1, "food-b": 2, "food-c": 3}
    for split in ("train", "validation", "test"):
        third = "food-test-only" if split == "test" else "food-c"
        groups = [
            [candidate(split, f"{split}-1", "food-a", 1),
             candidate(split, f"{split}-1", "food-b", 0),
             candidate(split, f"{split}-1", third, 1)],
            [candidate(split, f"{split}-2", "food-b", 1),
             candidate(split, f"{split}-2", "food-c", 0)],
        ]
        ids = torch.tensor([[1, 2, vocabulary.get(third, 0)], [2, 3, 0]])
        batches[split] = [FoodBatch(foundation_batch(), ids, groups)]
        candidates.extend(copy.deepcopy(row) for group in groups for row in group)
    manifest = {
        "dataset_version": "food-test-v1", "schema_version": 1,
        "candidate_policy": "frozen-food-candidates-v1",
        "split": {"kind": "temporal", "train_end": 100, "valid_end": 200},
    }
    return FoodInputs(batches, candidates, manifest, "a" * 64, "b" * 64, 4, 4, vocabulary)


def config(tmp_path):
    checkpoint = tmp_path / "foundation_checkpoint-v1.pt"
    checkpoint.write_bytes(b"test reference; FakeFoundation is injected at the public loader")
    return {
        "contract_version": 1, "seed": 42,
        "data": {"path": str(tmp_path / "samples.zip"), "expected_version": "food-test-v1"},
        "checkpoint": {"path": str(checkpoint), "expected_version": "food-pretrain-v1"},
        "freeze_foundation": True, "split": "test",
        "artifact_dir": str(tmp_path / "artifacts"),
        "candidate_policy": "frozen-food-candidates-v1", "evaluate": True,
        "training": {"epochs": 2, "learning_rate": 0.01, "cpu_threads": 1},
        "model": {"hidden_dim": 8},
    }


def services(foundation=None, inputs=None, evaluator=None, writer=None):
    model = foundation if foundation is not None else FakeFoundation()
    data = inputs if inputs is not None else food_inputs()

    def load_checkpoint(path, *, expected_version):
        assert Path(path).is_file() and expected_version == "food-pretrain-v1"
        return model

    return FoodServices(
        load_checkpoint=load_checkpoint, load_inputs=lambda cfg, encoder: data,
        writer=writer or validate_and_write_predictions,
        evaluator=evaluator or (lambda **kwargs: {}),
    )


def test_ranker_shape_finite_scores_and_masked_gradient():
    torch.manual_seed(7)
    ranker = FoodRanker(4, 4, 8)
    scores = ranker(torch.randn(2, 4), torch.randn(2, 4), torch.tensor([[1, 2, 3], [2, 3, 0]]))
    assert scores.shape == (2, 3) and torch.isfinite(scores).all()
    labels = torch.tensor([[1.0, 0.0, 1.0], [1.0, 0.0, float("nan")]])
    mask = torch.tensor([[True, True, True], [True, True, False]])
    loss = food_loss(scores, labels, mask)
    assert torch.isfinite(loss)
    loss.backward()
    assert ranker.candidates.weight.grad is not None
    assert torch.equal(ranker.candidates.weight.grad[0], torch.zeros(4))
    assert ranker.candidates.weight.grad[1:].abs().sum() > 0


def test_default_ranker_loads_legacy_head_layout(tmp_path):
    ranker = FoodRanker(4, 4, 8)
    assert "scorer.2.weight" in ranker.state_dict()
    assert "scorer.3.weight" not in ranker.state_dict()
    checkpoint = tmp_path / "legacy-food-ranker.pt"
    torch.save({
        "format_version": 1, "task": "food_recommendation", "d_model": 4,
        "hidden_dim": 8, "foundation_checkpoint_sha256": "a" * 64,
        "data_version": "food-test-v1",
        "candidate_vocabulary": {"food-a": 1, "food-b": 2, "food-c": 3},
        "ranker_state_dict": ranker.state_dict(),
    }, checkpoint)
    loaded, vocabulary, _ = load_food_ranker_checkpoint(checkpoint,
        expected_foundation_sha256="a" * 64, expected_data_version="food-test-v1")
    assert vocabulary == {"food-a": 1, "food-b": 2, "food-c": 3}
    assert isinstance(loaded.scorer[2], nn.Linear)
    assert all(torch.equal(loaded.state_dict()[name], value)
               for name, value in ranker.state_dict().items())


@pytest.mark.parametrize("positives", [[1.0, 0.0, 0.0], [1.0, 0.0, 1.0]])
def test_listwise_loss_assigns_uniform_mass_to_each_positive(positives):
    scores = torch.tensor([[0.1, -0.3, 0.8, float("nan")]], requires_grad=True)
    labels = torch.tensor([positives + [float("nan")]])
    mask = torch.tensor([[True, True, True, False]])
    loss = food_loss(scores, labels, mask)
    expected = -torch.log_softmax(scores[0, :3], dim=0)[torch.tensor(positives).bool()].mean()
    assert torch.isfinite(loss) and torch.allclose(loss, expected)
    loss.backward()
    assert scores.grad[0, 3] == 0 and torch.isfinite(scores.grad).all()


@pytest.mark.parametrize("bad_label", [-1.0, 2.0, float("nan"), float("inf")])
def test_listwise_loss_rejects_invalid_positive_labels(bad_label):
    with pytest.raises(ValueError, match="binary"):
        food_loss(torch.zeros(1, 2), torch.tensor([[1.0, bad_label]]),
                  torch.ones(1, 2, dtype=torch.bool))


def test_zero_positive_request_cannot_train():
    with pytest.raises(ValueError, match="at least one positive"):
        food_loss(torch.zeros(2, 2), torch.tensor([[1.0, 0.0], [0.0, 0.0]]),
                  torch.ones(2, 2, dtype=torch.bool))
    inputs = food_inputs()
    for row in inputs.candidates:
        if row["request_id"] == "train-1":
            row["is_positive"] = 0
    with pytest.raises(ValueError, match="positive candidate in every request"):
        validate_food_candidates(inputs.candidates)


@pytest.mark.parametrize("field", [
    "candidate_timestamp", "exposure_timestamp", "label_timestamp", "profile_as_of_timestamp",
])
def test_future_candidate_label_or_profile_blocks_training(tmp_path, field):
    model, inputs = FakeFoundation(), food_inputs()
    inputs.candidates[0][field] = inputs.candidates[0]["cutoff_timestamp"] + 1
    with pytest.raises(FoodRunError, match=f"future/invalid {field}") as caught:
        run_food_recommendation(config(tmp_path), services(model, inputs))
    assert caught.value.code == 3 and not model.calls
    assert not (tmp_path / "artifacts").exists()


@pytest.mark.parametrize("problem,match", [
    ("duplicate", "duplicate Food candidate"),
    ("invalid_label", "cutoff/positive invalid"),
    ("label_not_after_cutoff", "label availability"),
    ("user_mismatch", "identity/cutoff mismatch"),
    ("float_cutoff", "cutoff/positive invalid"),
])
def test_invalid_handoff_rejected(problem, match):
    inputs = food_inputs()
    if problem == "duplicate":
        inputs.candidates.append(dict(inputs.candidates[0]))
    elif problem == "invalid_label":
        inputs.candidates[0]["is_positive"] = None
    elif problem == "label_not_after_cutoff":
        inputs.candidates[0]["label_available_at_timestamp"] = inputs.candidates[0]["cutoff_timestamp"]
    elif problem == "user_mismatch":
        inputs.candidates[1]["user_id"] = "other-user"
    else:
        inputs.candidates[0]["cutoff_timestamp"] = 100.0
    with pytest.raises(ValueError, match=match):
        validate_food_candidates(inputs.candidates)


@pytest.mark.parametrize("freeze", [True, False])
def test_frozen_and_fine_tuned_runs_use_only_embed_users(tmp_path, freeze):
    torch.manual_seed(8)
    foundation = FakeFoundation()
    before = foundation.users.weight.detach().clone()
    cfg = config(tmp_path)
    cfg["freeze_foundation"] = freeze
    result = run_food_recommendation(cfg, services(foundation))
    assert torch.equal(before, foundation.users.weight) == freeze
    assert all(parameter.requires_grad == (not freeze) for parameter in foundation.parameters())
    assert foundation.calls == [(not freeze, not freeze), (False, False)] * 2 + [(False, False)]
    reference = json.loads(Path(result["predictions_path"]).with_name(
        "checkpoint_ref.json").read_text(encoding="utf-8"))
    assert reference["freeze_foundation"] is freeze
    assert reference["checkpoint_sha256"] == sha256_file(cfg["checkpoint"]["path"])


def test_exported_food_ranker_round_trip_matches_predictions(tmp_path):
    foundation, inputs = FakeFoundation(), food_inputs()
    cfg = config(tmp_path)
    cfg["export_model"] = True
    cfg["evaluate"] = False
    result = run_food_recommendation(cfg, services(foundation, inputs))
    model_path = Path(result["model_path"])
    assert model_path.is_file()
    ranker, vocabulary, payload = load_food_ranker_checkpoint(model_path,
        expected_foundation_sha256=sha256_file(cfg["checkpoint"]["path"]),
        expected_data_version=cfg["data"]["expected_version"])
    assert vocabulary == inputs.candidate_vocabulary
    assert payload["freeze_foundation"] is True
    assert payload["best_validation_loss"] >= 0
    predicted = {(row["request_id"], row["candidate_id"]): row["score"]
        for row in pq.read_table(result["predictions_path"]).to_pylist()}
    with torch.no_grad():
        for batch in inputs.batches["test"]:
            embeddings = foundation.embed_users(batch.foundation)
            scores = ranker(embeddings["long_term"], embeddings["short_term"], batch.candidate_ids)
            for index, rows in enumerate(batch.rows):
                for column, row in enumerate(rows):
                    assert predicted[row["request_id"], row["candidate_id"]] == float(scores[index, column])
    with pytest.raises(ValueError, match="foundation checkpoint mismatch"):
        load_food_ranker_checkpoint(model_path, expected_foundation_sha256="0" * 64,
            expected_data_version=cfg["data"]["expected_version"])


def test_regularized_batched_head_can_cache_frozen_foundation_and_reload(tmp_path):
    foundation = FakeFoundation()
    cfg = config(tmp_path)
    cfg["export_model"] = True
    cfg["evaluate"] = False
    cfg["model"].update(candidate_dim=2, dropout=0.2)
    cfg["training"].update(batch_size=2, weight_decay=0.001,
                           cache_foundation_embeddings=True, shuffle=True)
    result = run_food_recommendation(cfg, services(foundation))
    assert len(foundation.calls) == 3
    assert all(not is_training and not has_grad for is_training, has_grad in foundation.calls)
    ranker, _, payload = load_food_ranker_checkpoint(result["model_path"],
        expected_foundation_sha256=sha256_file(cfg["checkpoint"]["path"]),
        expected_data_version=cfg["data"]["expected_version"])
    assert ranker.candidates.embedding_dim == payload["candidate_dim"] == 2
    assert isinstance(ranker.scorer[2], nn.Dropout) and ranker.scorer[2].p == 0.2
    assert payload["best_validation_loss"] >= 0
    manifest = json.loads(Path(result["predictions_path"]).with_name(
        "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["selection"]["best_epoch"] == payload["best_epoch"]
    assert 1 <= payload["best_epoch"] <= cfg["training"]["epochs"]
    assert manifest["model_hyperparameters"]["candidate_dim"] == 2
    assert manifest["training_hyperparameters"] == cfg["training"]
    records = [json.loads(line) for line in Path(result["predictions_path"]).with_name(
        "train_metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [record["step"] for record in records] == [1, 1, 2, 2]


def test_caching_cannot_be_used_while_fine_tuning(tmp_path):
    cfg = config(tmp_path)
    cfg["freeze_foundation"] = False
    cfg["training"]["cache_foundation_embeddings"] = True
    with pytest.raises(FoodRunError, match="cannot cache foundation embeddings") as caught:
        run_food_recommendation(cfg, services())
    assert caught.value.code == 2


def test_cpu_end_to_end_real_writer_public_evaluator_and_artifacts(tmp_path):
    calls, inputs = [], food_inputs()

    def evaluator(*, predictions_path, task, data_version, split):
        path = Path(predictions_path)
        report = json.loads(path.with_name("validation_report.json").read_text(encoding="utf-8"))
        assert report["status"] == "passed"  # Validation has completed before the public call.
        calls.append((predictions_path, task, data_version, split))
        return {"ndcg_at_10": 0.75}  # Public API sentinel; no mock metric calculation.

    cfg = config(tmp_path)
    deps = services(inputs=inputs, evaluator=evaluator)
    assert deps.writer is validate_and_write_predictions
    result = run_food_recommendation(cfg, deps)
    assert result["metrics"] == {"ndcg_at_10": 0.75}
    assert calls == [(result["predictions_path"], "food_recommendation", "food-test-v1", "test")]
    path = Path(result["predictions_path"])
    assert path.name == "predictions.test.parquet" and path.parent.name == result["run_id"]
    assert path.parent.parent.name == "food_recommendation"
    assert {file.name for file in path.parent.iterdir()} == {
        "predictions.test.parquet", "run_manifest.json", "validation_report.json",
        "checkpoint_ref.json", "train_metrics.jsonl",
    }
    table = pq.read_table(path)
    assert table.schema == PREDICTION_SCHEMA
    assert not {"is_positive", "label", "clicked", "embedding"} & set(table.column_names)
    rows = table.to_pylist()
    expected = [row for row in inputs.candidates if row["split"] == "test"]
    identity = lambda row: tuple(row[key] for key in (
        "request_id", "candidate_id", "user_id", "cutoff_timestamp"))
    assert {identity(row) for row in rows} == {identity(row) for row in expected}
    assert len(rows) == len(expected) == 5
    assert rows == sorted(rows, key=lambda row: (row["request_id"], -row["score"], row["candidate_id"]))
    assert all(math.isfinite(row["score"]) for row in rows)
    # A test-only candidate is retained canonically and uses the frozen OOV embedding.
    assert inputs.candidate_vocabulary == {"food-a": 1, "food-b": 2, "food-c": 3}
    assert inputs.batches["test"][0].candidate_ids[0, 2] == 0
    assert any(row["candidate_id"] == "food-test-only" for row in rows)
    manifest = json.loads(path.with_name("run_manifest.json").read_text(encoding="utf-8"))
    assert MANIFEST_KEYS <= manifest.keys()
    validate_run_manifest(manifest)
    assert manifest["run_id"] == result["run_id"] and manifest["row_count"] == 5
    assert manifest["request_count"] == 2 and manifest["data_version"] == "food-test-v1"
    assert manifest["config_sha256"] == fingerprint(cfg)
    assert manifest["prediction_sha256"] == sha256_file(path)
    assert manifest["data_manifest_sha256"] == inputs.data_manifest_sha256
    assert manifest["preprocess_fingerprint"] == inputs.preprocess_fingerprint
    assert manifest["candidate_or_exposure_policy"] == cfg["candidate_policy"]
    reference = json.loads(path.with_name("checkpoint_ref.json").read_text(encoding="utf-8"))
    assert reference == manifest["foundation"]
    records = [json.loads(line) for line in path.with_name("train_metrics.jsonl").read_text(
        encoding="utf-8").splitlines()]
    assert len(records) == 4 and {record["split"] for record in records} == {"train", "validation"}
    assert all(math.isfinite(record["total_loss"]) and record["target_counts"]["requests"] == 2
               for record in records)
    assert not list(path.parent.parent.glob(".pending-*"))


@pytest.mark.parametrize("problem", ["missing", "extra", "duplicate", "changed_label", "wrong_id"])
def test_batch_candidate_exactness_is_checked_before_training(tmp_path, problem):
    foundation, inputs = FakeFoundation(), food_inputs()
    group = inputs.batches["train"][0].rows[0]
    if problem == "missing":
        group.pop()
    elif problem == "extra":
        group[0]["candidate_id"] = "invented-food"
    elif problem == "duplicate":
        group[1]["candidate_id"] = group[0]["candidate_id"]
    elif problem == "changed_label":
        group[0]["is_positive"] = 0
    else:
        inputs.batches["train"][0].candidate_ids[0, 0] = 2
    with pytest.raises(FoodRunError) as caught:
        run_food_recommendation(config(tmp_path), services(foundation, inputs))
    assert caught.value.code == 3 and not foundation.calls
    assert not (tmp_path / "artifacts").exists()


@pytest.mark.parametrize("problem", [
    "duplicate", "missing_candidate", "extra_candidate", "nan_score", "forbidden_label",
    "manifest_key", "manifest_hash",
])
def test_real_writer_rejection_blocks_evaluator_and_cleans_staged_files(tmp_path, problem):
    calls = []

    def corrupting_writer(table, **kwargs):
        rows = table.to_pylist()
        if problem == "duplicate":
            rows[1] = dict(rows[0])
        elif problem == "missing_candidate":
            rows.pop()
            kwargs["run_manifest"]["row_count"] = len(rows)
        elif problem == "extra_candidate":
            rows[0]["candidate_id"] = "invented-food"
        elif problem == "nan_score":
            rows[0]["score"] = float("nan")
        elif problem == "manifest_key":
            del kwargs["run_manifest"]["config_sha256"]
        elif problem == "manifest_hash":
            kwargs["run_manifest"]["data_manifest_sha256"] = "not-a-sha256"
        corrupted = pa.Table.from_pylist(rows, schema=PREDICTION_SCHEMA)
        if problem == "forbidden_label":
            corrupted = corrupted.append_column("is_positive", pa.array([1] * len(rows)))
        return validate_and_write_predictions(corrupted, **kwargs)

    with pytest.raises(FoodRunError) as caught:
        run_food_recommendation(config(tmp_path), services(
            writer=corrupting_writer, evaluator=lambda **kwargs: calls.append(kwargs)))
    assert caught.value.code == 5 and not calls
    assert not list(tmp_path.rglob("predictions.*.parquet"))
    assert not list(tmp_path.rglob(".pending-*"))


def test_test_labels_do_not_change_trained_scores(tmp_path):
    first, second = food_inputs(), food_inputs()
    for row in second.candidates:
        if row["split"] == "test":
            row["is_positive"] = 1 - row["is_positive"]
    for group in second.batches["test"][0].rows:
        for row in group:
            row["is_positive"] = 1 - row["is_positive"]
    torch.manual_seed(12)
    deps1 = services(inputs=first)
    torch.manual_seed(12)
    deps2 = services(inputs=second)
    cfg = config(tmp_path)
    result1 = run_food_recommendation(cfg, deps1)
    result2 = run_food_recommendation(cfg, deps2)
    assert result1["run_id"] != result2["run_id"]
    assert pq.read_table(result1["predictions_path"])["score"].to_pylist() == pq.read_table(
        result2["predictions_path"])["score"].to_pylist()


def test_evaluator_failure_preserves_valid_export_with_code_six(tmp_path):
    def evaluator(*, predictions_path, task, data_version, split):
        assert Path(predictions_path).is_file()
        raise RuntimeError("public evaluator unavailable")

    with pytest.raises(FoodRunError, match="after valid export") as caught:
        run_food_recommendation(config(tmp_path), services(evaluator=evaluator))
    assert caught.value.code == 6
    paths = list(tmp_path.rglob("predictions.test.parquet"))
    assert len(paths) == 1
    assert json.loads(paths[0].with_name("validation_report.json").read_text(
        encoding="utf-8"))["status"] == "passed"


@pytest.fixture
def samples_config(tmp_path):
    path, sha = write_samples_zip(tmp_path, valid_records())
    _, catalog_sha = load_catalog_fixture(tmp_path)
    return {
        "data": {"path": str(path), "member": "samples.jsonl", "expected_sha256": sha,
                 "expected_version": "food-v1", "namespace": "test-food"},
        "catalog": {"pois_path": str(tmp_path / "pois.txt"), "expected_sha256": catalog_sha},
        "candidate_policy": "sampled_catalog",
        "evaluation": {"item_kind": "food", "label_availability_policy": "handoff_timestamp",
                       "observation_complete_policy": "assume_complete_diagnostic",
                       "ks": [1, 3, 5, 10]},
    }


def test_baseline_dataset_from_samples_uses_label_time(samples_config):
    dataset = food_baseline_dataset(samples_config)
    label = next(item for item in dataset.labels if item.request_id == "train-1")
    assert label.available_at == datetime(2021, 3, 1, 1, 0, 1, tzinfo=timezone.utc)
    assert dataset.metadata.item_kind == "food"
    assert dataset.metadata.availability_policy == "handoff_timestamp"
    assert dataset.metadata.candidate_policy == "sampled_catalog"
    assert len(dataset.queries) == 4 and len(dataset.candidates) == len(dataset.labels) == 8


@pytest.mark.parametrize("key", ["item_kind", "label_availability_policy", "observation_complete_policy"])
def test_evaluation_policies_are_required(samples_config, key):
    samples_config["evaluation"].pop(key)
    with pytest.raises(ValueError, match=key):
        food_baseline_dataset(samples_config)


@pytest.mark.parametrize("key", ["item_kind", "label_availability_policy", "observation_complete_policy"])
def test_evaluation_policies_reject_other_values(samples_config, key):
    samples_config["evaluation"][key] = "other"
    with pytest.raises(ValueError, match=key):
        food_baseline_dataset(samples_config)


def test_evaluator_uses_configured_ks(samples_config):
    evaluator = food_baseline_evaluator(samples_config)
    assert evaluator.keywords["ks"] == (1, 3, 5, 10)


def test_cached_run_encodes_test_split_only_after_training(tmp_path, monkeypatch):
    import src.downstream.food_recommendation as module
    losses = []
    original = module.food_loss
    monkeypatch.setattr(module, "food_loss", lambda *a: losses.append(1) or original(*a))
    inputs = food_inputs()
    test_batch = inputs.batches["test"][0]
    seen = []

    class Recording(FakeFoundation):
        def embed_users(self, batch):
            seen.append((batch is test_batch.foundation, len(losses)))
            return super().embed_users(batch)

    cfg = config(tmp_path)
    cfg["training"]["cache_foundation_embeddings"] = True
    run_food_recommendation(cfg, services(Recording(), inputs))
    assert [count for is_test, count in seen if is_test] == [len(losses)]


def test_main_exits_three_for_missing_observation_policy(samples_config, tmp_path, capsys):
    import yaml

    from src.downstream.food_recommendation import main
    samples_config["evaluation"].pop("observation_complete_policy")
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(samples_config), encoding="utf-8")
    assert main(["--config", str(path)]) == 3
    assert "observation_complete_policy" in capsys.readouterr().err


def test_main_exits_three_for_data_error_with_request_id(samples_config, tmp_path, capsys):
    import yaml

    from src.downstream.food_recommendation import main
    samples_config["candidate_policy"] = "wrong_policy"
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(samples_config), encoding="utf-8")
    assert main(["--config", str(path)]) == 3
    assert "request_id=" in capsys.readouterr().err


def test_baseline_dataset_contract_error_carries_request_id(samples_config, monkeypatch):
    from src.baselines.contracts import BaselineDataset, ContractError

    def fail(self):
        raise ContractError("request=train-1, candidate=x: duplicate candidate")

    monkeypatch.setattr(BaselineDataset, "validate", fail)
    with pytest.raises(ValueError, match="request_id=train-1"):
        food_baseline_dataset(samples_config)


def test_baseline_dataset_error_keeps_colon_bearing_request_id(samples_config, monkeypatch):
    from src.baselines.contracts import BaselineDataset, ContractError

    def fail(self):
        raise ContractError("request=meituan:order:1, candidate=x: duplicate candidate")

    monkeypatch.setattr(BaselineDataset, "validate", fail)
    with pytest.raises(ValueError, match="request_id=meituan:order:1:"):
        food_baseline_dataset(samples_config)
