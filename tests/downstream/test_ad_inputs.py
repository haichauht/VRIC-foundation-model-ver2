"""Real handoff/tokenizer/public-evaluator tests for the Ads-owned adapter."""

import json
import math
from bisect import bisect_left
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from src.contracts.output import fingerprint, sha256_file
from src.downstream.ad_inputs import (
    EXPOSURE_POLICY,
    NAMESPACE,
    _canonical,
    _index_events,
    ad_baseline_dataset,
    ad_baseline_evaluator,
    ad_readiness_report,
    load_ad_foundation,
    load_ad_inputs,
    read_ad_handoff,
)
from src.downstream.ad_optimization import prediction_table
from src.foundation_input.tokenizer import TokenizerConfig
from src.foundation_input.vocab import VocabularyConfig, fit_vocabulary_from_samples
from src.models.encoder import FoundationModel, FoundationModelConfig
from src.training.checkpoint import save_checkpoint


def _publish(root, manifest, tables):
    for name, rows in tables.items():
        path = root / f"{name}.parquet"
        pq.write_table(pa.Table.from_pylist(rows), path)
        manifest["files"][name] = {"path": path.name, "format": "parquet",
                                    "sha256": sha256_file(path), "row_count": len(rows)}
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def handoff(tmp_path):
    root = tmp_path / "handoff"
    root.mkdir()
    user = "taobao:user:1"
    times = [10, 10, 20, 30, 40, 50]
    candidates = ["taobao:ad:a", "taobao:ad:b", "taobao:ad:a",
                  "taobao:ad:c", "taobao:ad:b", "taobao:ad:c"]
    catalog = {candidate: min(time for time, item in zip(times, candidates) if item == candidate)
               for candidate in candidates}
    events, rows = [], []
    for index, (timestamp, candidate) in enumerate(zip(times, candidates)):
        request = f"taobao:impression:{index}"
        stop = bisect_left(times, timestamp)
        events.append({"event_id": request, "source_row_index": index, "user_id": user,
                       "timestamp": timestamp, "event_type": "ad_impression", "service": "ads",
                       "entity_id": candidate, "value": candidate, "event_index": index})
        rows.append({"request_id": request, "source_row_index": index, "user_id": user,
                     "cutoff_timestamp": timestamp, "candidate_id": candidate,
                     "clicked": int(index in {1, 3}),
                     "split": "train" if timestamp <= 20 else "validation" if timestamp <= 40 else "test",
                     "exposure_timestamp": timestamp, "candidate_timestamp": catalog[candidate],
                     "profile_as_of_timestamp": 10, "history_start": max(0, stop - 3),
                     "history_stop": stop, "history_end_timestamp": times[stop - 1] if stop else None})
    tables = {"ad_opportunities": rows, "events": events,
              "profiles": [{"user_id": user, "as_of_timestamp": 10}],
              "ad_catalog": [{"candidate_id": key, "first_observed_timestamp": value}
                             for key, value in catalog.items()]}
    manifest = {"dataset_version": "taobao-test-v1", "schema_version": 1, "generator_version": "test-v1",
                "seed": 42, "timezone": "Asia/Shanghai", "timestamp_unit": "seconds",
                "files": {}, "split": {"kind": "temporal", "train_end": 20, "valid_end": 40},
                "source_dataset": NAMESPACE, "candidate_policy": EXPOSURE_POLICY,
                "missing_click_policy": "reject", "negative_sampling": "none", "row_count": len(rows),
                "history_policy": {"timestamp_rule": "strictly < cutoff", "target_click_in_history": False,
                                   "equal_timestamp_events_excluded": True, "max_events": 3}}
    _publish(root, manifest, tables)
    config = {"data": {"root": str(root), "expected_version": manifest["dataset_version"]},
              "checkpoint": {"path": str(tmp_path / "foundation.pt")},
              "exposure_policy": EXPOSURE_POLICY, "training": {"batch_size": 2},
              "evaluation": {"label_availability_policy": "at_cutoff_diagnostic"}}
    return config, manifest, tables


def checkpoint(config, manifest, tables):
    events, _ = _index_events(tables)
    samples = [_canonical(row, manifest, events) for row in tables["ad_opportunities"] if row["split"] == "train"]
    manifest_sha = sha256_file(Path(config["data"]["root"]) / "manifest.json")
    vocab = fit_vocabulary_from_samples(samples, VocabularyConfig(
        dataset_name=NAMESPACE, text_hash_buckets=16, source_fingerprint=manifest_sha,
    ), TokenizerConfig())
    dimensions = {key: value for key, value in vocab.model_cardinalities().items()
                  if key.endswith("_vocab_size")}
    model_config = FoundationModelConfig(
        d_model=8, num_heads=2, num_layers=1, dropout=0, max_length=64,
        num_user_buckets=16, categorical_pair_buckets=17,
        sequence_source_id=vocab.source_to_id["sequence"],
        numeric_modality_id=vocab.modality_to_id.get("numeric", 3), **dimensions,
    )
    model = FoundationModel(model_config)
    preprocess = {"format_version": 1, "dataset_version": manifest["dataset_version"],
                  "dataset_namespace": NAMESPACE, "source_manifest_sha256": manifest_sha,
                  "vocabulary_sha256": fingerprint(vocab.to_metadata()),
                  "sequence": {"max_tokens": 64, "max_event_count": 3}}
    save_checkpoint(config["checkpoint"]["path"], model, model_config, vocab, preprocess_metadata=preprocess)
    return model


def test_reader_and_readiness_preserve_all_observed_exposures(tmp_path):
    config, manifest, tables = handoff(tmp_path)
    loaded_manifest, loaded_tables, digest = read_ad_handoff(config)
    assert loaded_manifest == manifest and loaded_tables == tables
    assert digest == sha256_file(Path(config["data"]["root"]) / "manifest.json")
    report = ad_readiness_report(config)
    assert report["splits"]["train"]["exposures"] == 3
    assert report["splits"]["train"]["empty_history_requests"] == 2
    assert report["splits"]["test"]["unseen_candidate_exposures"] == 1
    assert report["splits"]["test"]["missing_label_availability"] == 1
    assert report["ranking_metrics_informative"] is False


@pytest.mark.parametrize("problem,match", [
    ("hash", "checksum"), ("row_count", "row count"), ("traversal", "in-root"),
    ("namespace", "namespace"), ("version", "data_version"), ("unit", "timestamp unit"),
    ("policy", "exposure policy"), ("negative_sampling", "policy mismatch"),
    ("history_policy", "unsafe history policy"),
])
def test_bad_manifest_fails_before_encoding(tmp_path, problem, match):
    config, manifest, _ = handoff(tmp_path)
    root = Path(config["data"]["root"])
    if problem == "hash":
        manifest["files"]["events"]["sha256"] = "0" * 64
    elif problem == "row_count":
        manifest["files"]["events"]["row_count"] += 1
    elif problem == "traversal":
        manifest["files"]["events"]["path"] = "../events.parquet"
    elif problem == "namespace":
        manifest["source_dataset"] = "meituan"
    elif problem == "version":
        manifest["dataset_version"] = "other-v1"
    elif problem == "unit":
        manifest["timestamp_unit"] = "milliseconds"
    elif problem == "policy":
        manifest["candidate_policy"] = "invented-negatives"
    elif problem == "negative_sampling":
        manifest["negative_sampling"] = "resampled"
    else:
        manifest["history_policy"]["target_click_in_history"] = True
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        read_ad_handoff(config)


@pytest.mark.parametrize("problem,match", [
    ("equal_time_history", "history pointer"), ("history_end", "history_end"),
    ("future_candidate", "future candidate"), ("profile_join", "availability mismatch"),
    ("event_join", "exposure identity mismatch"), ("duplicate_event", "duplicate"),
    ("label_feature", "target click leaked"), ("profile_feature", "identity-only"),
    ("index_gap", "contiguous"), ("label_before_exposure", "label availability precedes"),
])
def test_content_validation_even_when_file_hashes_match(tmp_path, problem, match):
    config, manifest, tables = handoff(tmp_path)
    if problem == "equal_time_history":
        tables["ad_opportunities"][1]["history_stop"] = 1
    elif problem == "history_end":
        tables["ad_opportunities"][2]["history_end_timestamp"] = 9
    elif problem == "future_candidate":
        tables["ad_opportunities"][0]["candidate_timestamp"] = 11
    elif problem == "profile_join":
        tables["profiles"][0]["as_of_timestamp"] = 9
    elif problem == "event_join":
        tables["events"][0]["entity_id"] = "taobao:ad:wrong"
    elif problem == "duplicate_event":
        tables["events"][1]["event_id"] = tables["events"][0]["event_id"]
    elif problem == "label_feature":
        for event in tables["events"]:
            event["clicked"] = 0
    elif problem == "profile_feature":
        tables["profiles"][0]["age"] = 30
    elif problem == "index_gap":
        tables["events"][1]["event_index"] = 100
    else:
        for row in tables["ad_opportunities"]:
            row["label_available_at_timestamp"] = row["cutoff_timestamp"] - 1
    _publish(Path(config["data"]["root"]), manifest, tables)
    with pytest.raises(ValueError, match=match):
        read_ad_handoff(config)


def test_real_checkpoint_encode_batches_train_only_candidate_vocabulary(tmp_path):
    config, manifest, tables = handoff(tmp_path)
    checkpoint(config, manifest, tables)
    model = load_ad_foundation(config["checkpoint"]["path"], expected_version=manifest["dataset_version"])
    inputs = load_ad_inputs(config, model)
    assert inputs.candidate_vocabulary == {"taobao:ad:a": 1, "taobao:ad:b": 2}
    assert inputs.batches["test"][0].candidate_ids.tolist() == [[0]]
    assert [len(batch.rows) for batch in inputs.batches["train"]] == [2, 1]
    assert inputs.batches["train"][0].foundation.input_ids.shape[0] == 2
    actual = [row for batches in inputs.batches.values() for batch in batches for group in batch.rows for row in group]
    assert sorted(actual, key=lambda row: row["request_id"]) == tables["ad_opportunities"]
    with torch.no_grad():
        embeddings = model.embed_users(inputs.batches["train"][0].foundation)
    assert embeddings["long_term"].shape == (2, 8)
    assert torch.isfinite(embeddings["short_term"]).all()


def test_temporal_context_excludes_ties_and_labels(tmp_path):
    _, manifest, tables = handoff(tmp_path)
    events, _ = _index_events(tables)
    first, tied, later = [_canonical(row, manifest, events) for row in tables["ad_opportunities"][:3]]
    assert first.events == tied.events == ()
    assert len(later.events) == 2
    assert [event.context for event in later.events] == [
        {"product": "taobao:ad:a"}, {"product": "taobao:ad:b"},
    ]
    assert all(event.action == {"type": "ad_impression"} for event in later.events)
    assert set(later.profile) == {"as_of_time"}


def test_checkpoint_version_namespace_and_provenance_rejected(tmp_path):
    config, manifest, tables = handoff(tmp_path)
    model = checkpoint(config, manifest, tables)
    with pytest.raises(ValueError, match="dataset version"):
        load_ad_foundation(config["checkpoint"]["path"], expected_version="other")
    path = config["checkpoint"]["path"]
    payload = torch.load(path, weights_only=True)
    payload["preprocess_metadata"]["dataset_namespace"] = "meituan"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="namespace"):
        load_ad_inputs(config, model)
    payload["preprocess_metadata"]["dataset_namespace"] = NAMESPACE
    payload["preprocess_metadata"]["source_manifest_sha256"] = "f" * 64
    torch.save(payload, path)
    with pytest.raises(ValueError, match="preprocessing source"):
        load_ad_inputs(config, model)


def test_input_labels_do_not_enter_encoded_features(tmp_path):
    config, manifest, tables = handoff(tmp_path)
    model = checkpoint(config, manifest, tables)
    original = load_ad_inputs(config, model)
    for row in tables["ad_opportunities"]:
        row["clicked"] = 1 - row["clicked"]
    _publish(Path(config["data"]["root"]), manifest, tables)
    # This fixture re-exports the matching manifest provenance without fitting a
    # new vocabulary; it verifies the adapter uses labels only as supervised data.
    payload = torch.load(config["checkpoint"]["path"], weights_only=True)
    manifest_sha = sha256_file(Path(config["data"]["root"]) / "manifest.json")
    payload["preprocess_metadata"]["source_manifest_sha256"] = manifest_sha
    payload["vocabularies"]["source_fingerprint"] = manifest_sha
    payload["preprocess_metadata"]["vocabulary_sha256"] = fingerprint(payload["vocabularies"])
    torch.save(payload, config["checkpoint"]["path"])
    changed = load_ad_inputs(config, model)
    for split in original.batches:
        for before, after in zip(original.batches[split], changed.batches[split]):
            assert torch.equal(before.foundation.input_ids, after.foundation.input_ids)
            assert torch.equal(before.foundation.user_ids, after.foundation.user_ids)


def test_baseline_bridge_requires_explicit_policy_and_real_audited_timestamps(tmp_path):
    config, manifest, tables = handoff(tmp_path)
    config["evaluation"] = {}
    with pytest.raises(ValueError, match="explicit label_availability_policy"):
        ad_baseline_dataset(config)
    config["evaluation"] = {"label_availability_policy": "handoff_timestamp"}
    with pytest.raises(ValueError, match="label_available_at_timestamp missing"):
        ad_baseline_dataset(config)
    for row in tables["ad_opportunities"]:
        row["label_available_at_timestamp"] = row["cutoff_timestamp"] + 5
    _publish(Path(config["data"]["root"]), manifest, tables)
    dataset = ad_baseline_dataset(config)
    for query, label in zip(dataset.queries, dataset.labels):
        assert (label.available_at - query.cutoff_time).total_seconds() == 5


def test_actual_baseline_evaluator_accepts_all_zero_click_test(tmp_path):
    config, _, tables = handoff(tmp_path)
    evaluator = ad_baseline_evaluator(config)
    selected = [row for row in tables["ad_opportunities"] if row["split"] == "test"]
    table = prediction_table([(row, -2.0) for row in selected], split="test", data_version="taobao-test-v1",
                             run_id="test-ads", temperature=1.0, bias=0.0)
    path = tmp_path / "predictions.test.parquet"
    pq.write_table(table, path)
    result = evaluator(predictions_path=str(path), task="ad_optimization", data_version="taobao-test-v1", split="test")
    assert result["discrimination_defined"] == 0
    assert result["clicked_request_count"] == 0
    assert math.isnan(result["roc_auc"])
    assert math.isfinite(result["log_loss"]) and math.isfinite(result["brier_score"])
