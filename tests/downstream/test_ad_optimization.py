"""Ads training contracts, real shared writer and public Baseline integration."""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
import yaml
from torch import nn

from src.contracts.output import validate_and_write_predictions
from src.downstream.ad_optimization import (
    PREDICTION_SCHEMA,
    AdBatch,
    AdInputs,
    AdRanker,
    AdRunError,
    AdServices,
    build_ad_inputs,
    ctr_loss,
    fit_calibration,
    load_ad_ranker_checkpoint,
    main,
    prediction_table,
    run_ad_optimization,
    train_ad_ranker,
    validate_opportunities,
)
from src.foundation_input.types import FoundationBatch
from src.models.encoder import FoundationModel, FoundationModelConfig


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class FakeFoundation(nn.Module):
    def __init__(self):
        super().__init__()
        self.users = nn.Embedding(8, 4)
        self.calls = 0

    def embed_users(self, batch):
        self.calls += 1
        identity = self.users(batch.user_ids)
        return {"long_term": identity, "short_term": identity * 0.5}

    def forward(self, *args, **kwargs):
        raise AssertionError("downstream must only use embed_users")


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


def opportunity(split, request, candidate, clicked):
    cutoff = {"train": 100, "validation": 200, "test": 300}[split]
    return {
        "request_id": request, "user_id": "user-1" if request.endswith("1") else "user-2",
        "cutoff_timestamp": cutoff, "candidate_id": candidate, "clicked": clicked,
        "split": split, "candidate_timestamp": cutoff - 10,
        "exposure_timestamp": cutoff, "click_timestamp": cutoff if clicked else None,
        "profile_as_of_timestamp": cutoff - 30, "history_end_timestamp": cutoff - 1,
    }


def ad_inputs():
    batches, opportunities = {}, []
    for split in ("train", "validation", "test"):
        rows = [
            [opportunity(split, f"{split}-1", "ad-a", 0),
             opportunity(split, f"{split}-1", "ad-b", 1)],
            [opportunity(split, f"{split}-2", "ad-a", 0)],  # valid zero-click request
        ]
        batches[split] = [AdBatch(foundation_batch(), torch.tensor([[1, 2], [1, 0]]), rows)]
        opportunities.extend(row for group in rows for row in group)
    manifest = {
        "dataset_version": "sim-v1", "schema_version": 1,
        "candidate_policy": "observed-exposures-v1",
        "split": {"kind": "temporal", "train_end": 100, "valid_end": 200},
    }
    return AdInputs(batches, opportunities, manifest, "a" * 64, "b" * 64, 3, 4,
                    {"ad-a": 1, "ad-b": 2})


def test_input_bridge_fits_candidates_only_on_train_and_preserves_requests():
    inputs = ad_inputs()
    inputs.opportunities[-1]["candidate_id"] = "new-test-only-ad"
    contexts = {}
    for row in inputs.opportunities:
        batch = foundation_batch()
        single = FoundationBatch(**{
            name: value[:1] if isinstance(value, torch.Tensor) else value
            for name, value in vars(batch).items()
        })
        contexts[row["request_id"]] = (row["user_id"], row["cutoff_timestamp"], single)
    prepared = build_ad_inputs(
        inputs.opportunities, contexts, manifest=inputs.manifest,
        data_manifest_sha256=inputs.data_manifest_sha256,
        preprocess_fingerprint=inputs.preprocess_fingerprint, d_model=4,
    )
    assert prepared.candidate_vocabulary == {"ad-a": 1, "ad-b": 2}
    assert prepared.num_candidates == 3
    assert prepared.batches["test"][-1].candidate_ids.tolist() == [[0]]
    assert len(prepared.opportunities) == 9
    contexts["test-1"] = ("wrong-user", 300, contexts["test-1"][2])
    with pytest.raises(AdRunError, match="alignment mismatch"):
        build_ad_inputs(inputs.opportunities, contexts, manifest=inputs.manifest,
                        data_manifest_sha256="a" * 64, preprocess_fingerprint="b" * 64,
                        d_model=4)


def config(tmp_path):
    checkpoint = tmp_path / "foundation_checkpoint-v1.pt"
    checkpoint.write_bytes(b"test-only checkpoint reference; FakeFoundation loader")
    return {
        "contract_version": 1, "seed": 42,
        "data": {"root": str(tmp_path), "expected_version": "sim-v1"},
        "checkpoint": {"path": str(checkpoint)}, "freeze_foundation": True,
        "split": "test", "artifact_dir": str(tmp_path / "artifacts"),
        "exposure_policy": "observed-exposures-v1", "missing_click_policy": "reject",
        "training": {"epochs": 2, "learning_rate": 0.01},
        "calibration": {"split": "validation", "steps": 10},
        "model": {"hidden_dim": 8},
    }


def test_ranker_shape_finite_loss_and_padding():
    torch.manual_seed(7)
    rank = AdRanker(4, 3, 8)
    logits = rank(torch.randn(2, 4), torch.randn(2, 4), torch.tensor([[1, 2], [1, 0]]))
    assert logits.shape == (2, 2)
    mask = torch.tensor([[True, True], [True, False]])
    labels = torch.tensor([[0.0, 1.0], [0.0, float("nan")]])
    loss = ctr_loss(logits, labels, mask)
    assert torch.isfinite(loss)
    loss.backward()
    assert rank.candidates.weight.grad is not None
    assert torch.equal(rank.candidates.weight.grad[0], torch.zeros(4))


@pytest.mark.parametrize("bad", [-1.0, 2.0, float("nan"), float("inf")])
def test_ctr_rejects_invalid_labels(bad):
    with pytest.raises(ValueError, match="binary"):
        ctr_loss(torch.zeros(1, 1), torch.tensor([[bad]]), torch.ones(1, 1, dtype=torch.bool))


@pytest.mark.parametrize("freeze", [True, False])
def test_train_freeze_and_finetune_use_public_api(freeze):
    torch.manual_seed(8)
    foundation = FakeFoundation()
    before = foundation.users.weight.detach().clone()
    records = train_ad_ranker(
        AdRanker(4, 3, 8), foundation, ad_inputs().batches["train"],
        epochs=2, learning_rate=0.01, freeze_foundation=freeze,
    )
    assert len(records) == 2 and records[-1]["target_counts"]["ctr"] == 3
    assert records[-1]["class_prevalence"] == pytest.approx(1 / 3)
    assert torch.equal(before, foundation.users.weight) == freeze
    assert all(parameter.requires_grad == (not freeze) for parameter in foundation.parameters())
    assert foundation.calls == 2


def test_zero_click_request_trains():
    batch = ad_inputs().batches["train"][0]
    for group in batch.rows:
        for row in group:
            row["clicked"] = 0
    records = train_ad_ranker(
        AdRanker(4, 3, 8), FakeFoundation(), [batch], epochs=2, learning_rate=0.01,
    )
    assert records[-1]["class_prevalence"] == 0
    assert records[-1]["total_loss"] > 0


@pytest.mark.parametrize("field", ["candidate_timestamp", "exposure_timestamp", "click_timestamp",
                                   "profile_as_of_timestamp", "history_end_timestamp"])
def test_future_data_rejected(field):
    inputs = ad_inputs()
    inputs.opportunities[0][field] = 101
    with pytest.raises(AdRunError, match=f"future {field}") as caught:
        validate_opportunities(inputs.opportunities, inputs.manifest, "sim-v1",
                               "observed-exposures-v1")
    assert caught.value.code == 3


@pytest.mark.parametrize("problem", ["duplicate", "missing_click", "wrong_version", "wrong_split",
                                     "wrong_policy", "float_timestamp", "user_mismatch"])
def test_handoff_rejections(problem):
    inputs = ad_inputs()
    if problem == "duplicate":
        inputs.opportunities.append(dict(inputs.opportunities[0]))
    elif problem == "missing_click":
        inputs.opportunities[0]["clicked"] = None
    elif problem == "wrong_version":
        inputs.manifest["dataset_version"] = "sim-v2"
    elif problem == "wrong_split":
        inputs.opportunities[0]["split"] = "test"
    elif problem == "wrong_policy":
        inputs.manifest["candidate_policy"] = "resampled-negatives"
    elif problem == "float_timestamp":
        inputs.opportunities[0]["cutoff_timestamp"] = 100.0
    else:
        inputs.opportunities[1]["user_id"] = "other"
    with pytest.raises(AdRunError) as caught:
        validate_opportunities(inputs.opportunities, inputs.manifest, "sim-v1",
                               "observed-exposures-v1")
    assert caught.value.code == 3


def test_prediction_schema_order_probabilities_and_no_labels():
    inputs = ad_inputs()
    rows = [row for row in inputs.opportunities if row["split"] == "test"]
    table = prediction_table([(row, 0.0) for row in reversed(rows)], split="test",
                             data_version="sim-v1", run_id="test-id", temperature=2, bias=0)
    assert table.schema == PREDICTION_SCHEMA
    assert table.num_rows == 3
    assert "clicked" not in table.column_names and "label" not in table.column_names
    assert table["score"].to_pylist() == [0.5, 0.5, 0.5]
    assert table["candidate_id"].to_pylist() == ["ad-a", "ad-b", "ad-a"]


@pytest.mark.parametrize("raw", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_prediction_rejected(raw):
    row = opportunity("test", "test-1", "ad-a", 0)
    with pytest.raises(AdRunError, match="non-finite"):
        prediction_table([(row, raw)], split="test", data_version="sim-v1",
                         run_id="test-id", temperature=1, bias=0)


def test_duplicate_prediction_rejected():
    row = opportunity("test", "test-1", "ad-a", 0)
    with pytest.raises(AdRunError, match="duplicate"):
        prediction_table([(row, 0), (row, 0)], split="test", data_version="sim-v1",
                         run_id="test-id", temperature=1, bias=0)


def test_calibration_is_finite_and_requires_both_classes():
    temperature, bias = fit_calibration(torch.tensor([-2.0, 1.0]), torch.tensor([0.0, 1.0]),
                                        steps=10)
    assert temperature > 0 and torch.isfinite(torch.tensor([temperature, bias])).all()
    with pytest.raises(AdRunError, match="both click classes"):
        fit_calibration(torch.ones(2), torch.zeros(2), steps=10)


class SharedWriterSpy:
    """Test-only publisher representing an already-validated shared writer result."""

    def __init__(self):
        self.calls = []

    def __call__(self, table, *, expected_candidates, run_manifest, train_metrics,
                 checkpoint_ref, artifact_dir):
        self.calls.append((table, expected_candidates, run_manifest, checkpoint_ref))
        assert table.schema == PREDICTION_SCHEMA
        assert {(row["request_id"], row["candidate_id"]) for row in table.to_pylist()} == {
            (row["request_id"], row["candidate_id"]) for row in expected_candidates
        }
        run_dir = Path(artifact_dir) / "ad_optimization" / run_manifest["run_id"]
        run_dir.mkdir(parents=True, exist_ok=False)
        predictions = run_dir / "predictions.test.parquet"
        pq.write_table(table, predictions)
        manifest = dict(run_manifest)
        manifest["prediction_sha256"] = hashlib.sha256(predictions.read_bytes()).hexdigest()
        (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (run_dir / "validation_report.json").write_text(
            json.dumps({"status": "passed"}), encoding="utf-8"
        )
        (run_dir / "checkpoint_ref.json").write_text(json.dumps(checkpoint_ref), encoding="utf-8")
        (run_dir / "train_metrics.jsonl").write_text(
            "\n".join(json.dumps(record, allow_nan=False) for record in train_metrics),
            encoding="utf-8",
        )
        return predictions


def services(inputs=None, writer=None, evaluator=None):
    foundation = FakeFoundation()
    data = inputs or ad_inputs()
    def load_checkpoint(path, *, expected_version):
        assert Path(path).is_file() and expected_version == "sim-v1"
        return foundation

    return AdServices(load_checkpoint, lambda config, model: data,
                      writer or SharedWriterSpy(), evaluator or (lambda **kwargs: {}))


def test_cpu_end_to_end_public_evaluator_and_run_artifacts(tmp_path):
    writer, calls = SharedWriterSpy(), []
    def evaluator(*, predictions_path, task, data_version, split):
        calls.append((predictions_path, task, data_version, split))
        assert Path(predictions_path).is_file()
        return {"roc_auc": 0.75}  # sentinel returned by public API, no mock metric logic

    cfg = config(tmp_path)
    result = run_ad_optimization(cfg, services(writer=writer, evaluator=evaluator))
    assert result["metrics"] == {"roc_auc": 0.75}
    assert calls == [(result["predictions_path"], "ad_optimization", "sim-v1", "test")]
    table, expected, manifest, reference = writer.calls[0]
    assert table.num_rows == len(expected) == 3  # includes zero-click request
    assert manifest["missing_click_policy"] == "reject" and manifest["negative_sampling"] == "none"
    assert reference["freeze_foundation"] is True
    assert reference["checkpoint_sha256"] == hashlib.sha256(
        Path(cfg["checkpoint"]["path"]).read_bytes()
    ).hexdigest()
    assert manifest["class_prevalence"] == pytest.approx(1 / 3)
    assert manifest["calibration"]["split"] == "validation"


@pytest.mark.parametrize("rule", ["score_range", "duplicate", "missing_candidate", "extra_candidate",
                                  "nan_inf", "data_version", "manifest_key"])
def test_shared_output_validator_failure_blocks_evaluator(tmp_path, rule):
    calls = []
    def rejecting_writer(*args, **kwargs):
        # The shared validator owns these checks. Assert rejection propagation;
        # do not reimplement its logic or claim the absent validator was tested.
        raise ValueError(f"Output Contract v1: {rule}")

    with pytest.raises(AdRunError, match=rule) as caught:
        run_ad_optimization(config(tmp_path), services(
            writer=rejecting_writer, evaluator=lambda **kwargs: calls.append(kwargs)
        ))
    assert caught.value.code == 5 and not calls
    assert not list(tmp_path.rglob("predictions.*.parquet"))


@pytest.mark.parametrize("problem", ["missing", "extra", "duplicate", "changed_label"])
def test_batch_exactness_before_training(tmp_path, problem):
    inputs = ad_inputs()
    # Independent reference rows prevent the loader from silently editing the handoff.
    inputs.opportunities = copy.deepcopy(inputs.opportunities)
    group = inputs.batches["train"][0].rows[0]
    if problem == "missing":
        group.pop()
    elif problem == "extra":
        group[0]["candidate_id"] = "invented-ad"
    elif problem == "duplicate":
        group[1]["candidate_id"] = group[0]["candidate_id"]
    else:
        group[0]["clicked"] = 1
    with pytest.raises(AdRunError) as caught:
        run_ad_optimization(config(tmp_path), services(inputs=inputs))
    assert caught.value.code == 3


def test_no_test_labels_used_to_train_or_calibrate(tmp_path):
    writer1, writer2 = SharedWriterSpy(), SharedWriterSpy()
    inputs1, inputs2 = ad_inputs(), ad_inputs()
    for row in inputs2.opportunities:
        if row["split"] == "test":
            row["clicked"] = 1 - row["clicked"]
    # Reset FakeFoundation's own initialization as well as the runner's seed.
    torch.manual_seed(12)
    deps1 = services(inputs=inputs1, writer=writer1)
    torch.manual_seed(12)
    deps2 = services(inputs=inputs2, writer=writer2)
    cfg = config(tmp_path)
    run_ad_optimization(cfg, deps1)
    run_ad_optimization(cfg, deps2)
    assert writer1.calls[0][0]["score"].to_pylist() == writer2.calls[0][0]["score"].to_pylist()


def test_evaluator_failure_has_code_six_after_valid_export(tmp_path):
    def evaluator(**kwargs):
        raise RuntimeError("external evaluator unavailable")

    with pytest.raises(AdRunError, match="after valid export") as caught:
        run_ad_optimization(config(tmp_path), services(evaluator=evaluator))
    assert caught.value.code == 6
    assert len(list(tmp_path.rglob("predictions.test.parquet"))) == 1


def test_calibration_eval_split_overlap_rejected(tmp_path):
    cfg = config(tmp_path)
    cfg["split"] = "validation"
    with pytest.raises(AdRunError, match="separate test split") as caught:
        run_ad_optimization(cfg, services())
    assert caught.value.code == 2


def test_missing_checkpoint_rejected(tmp_path):
    cfg = config(tmp_path)
    cfg["checkpoint"]["path"] = str(tmp_path / "missing.pt")
    with pytest.raises(AdRunError, match="Person 3") as caught:
        run_ad_optimization(cfg, services())
    assert caught.value.code == 4


def test_cli_missing_person_three_dependency(tmp_path, capsys):
    cfg = config(tmp_path)
    cfg["integration"] = {
        "load_checkpoint": "src.training.not_implemented:load_foundation_for_ads",
        "load_inputs": "src.training.not_implemented:load_ad_inputs",
        "validate_and_write_predictions": "src.contracts.output:validate_and_write_predictions",
        "evaluate_predictions": "src.evaluation.metrics:evaluate_predictions",
    }
    path = tmp_path / "ad.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    assert main(["--config", str(path)]) == 4
    assert "required team handoff" in capsys.readouterr().err
    assert not (tmp_path / "artifacts").exists()


def test_cli_invalid_yaml_has_config_exit_code(tmp_path, capsys):
    path = tmp_path / "bad.yaml"
    path.write_text("invalid: [", encoding="utf-8")
    assert main(["--config", str(path)]) == 2
    assert "YAML syntax invalid" in capsys.readouterr().err


@pytest.mark.parametrize("section,field,value", [
    ("training", "learning_rate", float("nan")),
    ("training", "epochs", 0),
    ("calibration", "steps", 0),
    ("model", "hidden_dim", -1),
])
def test_invalid_config_fails_before_checkpoint_loader(tmp_path, section, field, value):
    cfg = config(tmp_path)
    cfg[section][field] = value
    deps = services()
    def forbidden_loader(*args, **kwargs):
        raise AssertionError("invalid config must fail before checkpoint loading")
    deps.load_checkpoint = forbidden_loader
    with pytest.raises(AdRunError) as caught:
        run_ad_optimization(cfg, deps)
    assert caught.value.code == 2


def test_actual_person_two_foundation_trains_through_embed_users():
    foundation = FoundationModel(FoundationModelConfig(
        d_model=4, num_heads=2, num_layers=1, dropout=0, max_length=8,
        value_vocab_size=10, key_vocab_size=8, modality_vocab_size=4,
        source_vocab_size=3, time_gap_vocab_size=4, num_user_buckets=8,
        categorical_pair_buckets=17, sequence_source_id=1, numeric_modality_id=3,
    ))
    records = train_ad_ranker(
        AdRanker(4, 3, 8), foundation, ad_inputs().batches["train"],
        epochs=2, learning_rate=0.01,
    )
    assert records[-1]["target_counts"]["ctr"] == 3
    assert torch.isfinite(torch.tensor(records[-1]["total_loss"]))


def test_cached_regularized_head_round_trip_matches_exported_probabilities(tmp_path):
    inputs, foundation = ad_inputs(), FakeFoundation()
    cfg = config(tmp_path)
    cfg["model"].update(candidate_dim=2, dropout=0.1)
    cfg["training"].update(cache_foundation_embeddings=True, shuffle=True, weight_decay=0.001)
    deps = services(inputs, writer=validate_and_write_predictions)
    deps.load_checkpoint = lambda *args, **kwargs: foundation
    result = run_ad_optimization(cfg, deps)
    # Train and validation encoded once, plus test inference once.
    assert foundation.calls == 3
    rank, vocabulary, payload = load_ad_ranker_checkpoint(result["model_path"],
        expected_foundation_sha256=hashlib.sha256(Path(cfg["checkpoint"]["path"]).read_bytes()).hexdigest(),
        expected_data_version="sim-v1")
    assert vocabulary == inputs.candidate_vocabulary
    assert torch.equal(rank.candidates.weight[0], torch.zeros(2))
    assert payload["selection"]["best_epoch"] in {1, 2}
    batch = inputs.batches["test"][0]
    with torch.no_grad():
        embeddings = foundation.embed_users(batch.foundation)
        logits = rank(embeddings["long_term"], embeddings["short_term"], batch.candidate_ids)
    scored = [(row, float(logits[i, j])) for i, group in enumerate(batch.rows) for j, row in enumerate(group)]
    reconstructed = prediction_table(scored, split="test", data_version="sim-v1", run_id=result["run_id"],
        temperature=payload["calibration"]["temperature"], bias=payload["calibration"]["bias"])
    assert pq.read_table(result["predictions_path"]).equals(reconstructed)
    with pytest.raises(ValueError, match="foundation checkpoint mismatch"):
        load_ad_ranker_checkpoint(result["model_path"], expected_foundation_sha256="0" * 64,
                                 expected_data_version="sim-v1")


def _real_evaluator(inputs):
    from src.baselines.contracts import BaselineDataset, Candidate, DatasetMetadata, Label, Query
    from src.evaluation.evaluator import DatasetRegistry, evaluate_predictions
    queries, candidates, labels = {}, [], []
    for row in inputs.opportunities:
        cutoff = datetime.fromtimestamp(row["cutoff_timestamp"], timezone.utc)
        queries[row["request_id"]] = Query(row["request_id"], row["user_id"], cutoff, row["split"])
        candidates.append(Candidate(row["request_id"], row["candidate_id"]))
        labels.append(Label(row["request_id"], row["candidate_id"], int(row["clicked"]), cutoff))
    registry = DatasetRegistry()
    registry.register(BaselineDataset(
        DatasetMetadata("test-ads", "sim-v1", "ad_optimization", "ad", "UTC",
                        "observed-exposures-v1", "at_cutoff_diagnostic"),
        tuple(queries.values()), tuple(candidates), tuple(labels)))
    return partial(evaluate_predictions, registry=registry)


def test_all_zero_clicks_remain_valid_through_real_writer_and_evaluator(tmp_path):
    inputs = ad_inputs()
    for row in inputs.opportunities:
        row["clicked"] = 0
        row["click_timestamp"] = None
    cfg = config(tmp_path)
    cfg["calibration"]["enabled"] = False
    result = run_ad_optimization(cfg, services(inputs, writer=validate_and_write_predictions,
                                               evaluator=_real_evaluator(inputs)))
    assert result["metrics"]["roc_auc"] is None
    assert result["metrics"]["pr_auc"] is None
    assert result["metrics"]["discrimination_defined"] == 0
    assert result["metrics"]["log_loss"] > 0
    assert result["metrics"]["brier_score"] >= 0
    assert "NaN" not in Path(result["predictions_path"]).with_name("evaluation.json").read_text()
    assert pq.read_table(result["predictions_path"]).num_rows == 3


@pytest.mark.parametrize("problem", ["score_low", "score_high", "nan", "inf", "duplicate",
                                      "missing", "extra", "data_version", "manifest_key"])
def test_real_output_validator_rejects_bad_ads_before_evaluation(tmp_path, problem):
    calls = []
    def corrupt(table, **kwargs):
        rows = table.to_pylist()
        if problem in {"score_low", "score_high", "nan", "inf"}:
            rows[0]["score"] = {"score_low": -0.1, "score_high": 1.1,
                                "nan": float("nan"), "inf": float("inf")}[problem]
        elif problem == "duplicate":
            rows[1] = dict(rows[0])
        elif problem == "missing":
            rows.pop()
        elif problem == "extra":
            rows[0]["candidate_id"] = "not-in-handoff"
        elif problem == "data_version":
            rows[0]["data_version"] = "wrong-version"
        elif problem == "manifest_key":
            del kwargs["run_manifest"]["seed"]
        return validate_and_write_predictions(pa.Table.from_pylist(rows, schema=PREDICTION_SCHEMA), **kwargs)
    with pytest.raises(AdRunError) as caught:
        run_ad_optimization(config(tmp_path), services(writer=corrupt,
                                                       evaluator=lambda **kwargs: calls.append(kwargs)))
    assert caught.value.code == 5 and calls == []
    assert not list(tmp_path.rglob("predictions.test.parquet"))


def test_cached_training_rejects_fine_tuning(tmp_path):
    cfg = config(tmp_path)
    cfg["freeze_foundation"] = False
    cfg["training"]["cache_foundation_embeddings"] = True
    with pytest.raises(AdRunError, match="cannot cache"):
        run_ad_optimization(cfg, services())
