"""B1 + foundation features: train-only fitting, leave-one-out prototypes, artifacts."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from src.baselines.contracts import (
    BaselineDataset,
    Candidate,
    ContractError,
    DatasetMetadata,
    Label,
    Query,
)
from src.baselines.features import FeatureRow
from src.baselines.foundation_features import FoundationFeatures
from src.baselines.lightgbm import B1Classifier
from src.baselines.runner import run

T = datetime(2026, 9, 1, tzinfo=timezone.utc)
SPLITS = [("train", -2), ("validation", 0), ("test", 2)]


@pytest.fixture
def dataset():
    queries, candidates, labels = [], [], []
    for split, day in SPLITS:
        for i in range(32):
            request, cutoff = f"{split}-{i}", T + timedelta(days=day)
            queries.append(Query(request, f"u{i % 4}", cutoff, split))
            candidates.append(Candidate(request, str(i % 8), {"price": i % 2}))
            labels.append(Label(request, str(i % 8), i % 2, cutoff + timedelta(hours=1)))
    return BaselineDataset(
        DatasetMetadata("fixture", "v1", "ad_optimization", "ad", "UTC", "logged_impressions",
                        "explicit-v1"), tuple(queries), tuple(candidates), tuple(labels))


def write_embeddings(path: Path, dataset, *, seed=0, change=None, drop=None):
    rng = np.random.default_rng(seed)
    ids = [q.request_id for q in dataset.queries if q.request_id != drop]
    long_term, short_term = rng.normal(size=(len(ids), 4)), rng.normal(size=(len(ids), 4))
    if change:
        for row, request in enumerate(ids):
            if request.startswith(change):
                long_term[row] += 100.0
    np.savez(path, request_id=np.array(ids), long_term=long_term, short_term=short_term)
    path.with_suffix(".json").write_text(json.dumps({"init": "pretrained"}), encoding="utf-8")
    return path


def unit(features, request):
    return features.units[features.index[request]]


def test_pca_and_prototypes_are_fitted_on_train_only(dataset, tmp_path):
    base = FoundationFeatures(write_embeddings(tmp_path / "a.npz", dataset), pca_components=3)
    moved = FoundationFeatures(write_embeddings(tmp_path / "b.npz", dataset, change="test-"),
                               pca_components=3)
    base.fit(dataset)
    moved.fit(dataset)
    assert np.allclose(base.mean, moved.mean) and np.allclose(base.components, moved.components)
    assert base.prototypes.keys() == moved.prototypes.keys() == {"1", "3", "5", "7"}
    assert base.names == ("foundation_pca_0", "foundation_pca_1", "foundation_pca_2",
                          "foundation_proto_similarity")


def test_train_rows_leave_their_own_request_out(dataset, tmp_path):
    features = FoundationFeatures(write_embeddings(tmp_path / "e.npz", dataset),
                                  pca_components=0).fit(dataset)
    rows = (FeatureRow("train-1", "1", "train", {}), FeatureRow("test-1", "1", "test", {}),
            FeatureRow("test-0", "0", "test", {}))
    train_row, test_row, no_prototype = features.extend(rows)
    others = unit(features, "train-9") + unit(features, "train-17") + unit(features, "train-25")
    assert train_row.values["foundation_proto_similarity"] == pytest.approx(
        float(unit(features, "train-1") @ others / np.linalg.norm(others)))
    every = others + unit(features, "train-1")
    assert test_row.values["foundation_proto_similarity"] == pytest.approx(
        float(unit(features, "test-1") @ every / np.linalg.norm(every)))
    assert no_prototype.values["foundation_proto_similarity"] is None  # "0" is never positive


def test_missing_embedding_or_unfitted_use_is_rejected(dataset, tmp_path):
    features = FoundationFeatures(write_embeddings(tmp_path / "m.npz", dataset, drop="test-3"))
    with pytest.raises(ContractError, match="test-3"):
        features.fit(dataset)
    with pytest.raises(ContractError, match="fitted"):
        features.extend((FeatureRow("train-1", "1", "train", {}),))
    with pytest.raises(ContractError, match="no feature"):
        FoundationFeatures(tmp_path / "m.npz", pca_components=0, prototype_similarity=False)


def test_b1_with_foundation_features_round_trips(dataset, tmp_path):
    features = FoundationFeatures(write_embeddings(tmp_path / "r.npz", dataset), pca_components=2)
    model = B1Classifier(n_estimators=20, learning_rate=0.2, num_leaves=3, min_child_samples=2,
                         stopping_rounds=3, foundation_features=features)
    model.fit(dataset, training_cutoff=T)
    assert {"foundation_pca_0", "foundation_pca_1", "foundation_proto_similarity"} <= set(
        model.encoder.columns)
    table = model.predict(dataset, "test", "run", frozen_cutoff=T + timedelta(days=2))
    model.save(tmp_path / "model")
    restored = B1Classifier.load(tmp_path / "model")
    assert restored.foundation_features is not None
    assert table.equals(restored.predict(dataset, "test", "run", frozen_cutoff=T + timedelta(days=2)))


def test_runner_records_foundation_feature_provenance(dataset, tmp_path):
    payload = {"metadata": {"dataset_name": "fixture", "data_version": "v1",
                            "task": "ad_optimization", "item_kind": "ad", "timezone": "UTC",
                            "candidate_policy": "logged_impressions",
                            "availability_policy": "explicit-v1"},
               "queries": [{"request_id": q.request_id, "user_id": q.user_id, "split": q.split,
                            "cutoff_time": q.cutoff_time.isoformat()} for q in dataset.queries],
               "candidates": [{"request_id": c.request_id, "candidate_id": c.candidate_id,
                               "attributes": c.attributes} for c in dataset.candidates],
               "labels": [{"request_id": label.request_id, "candidate_id": label.candidate_id,
                           "target": label.target, "available_at": label.available_at.isoformat()}
                          for label in dataset.labels]}
    (tmp_path / "data.json").write_text(json.dumps(payload), encoding="utf-8")
    write_embeddings(tmp_path / "ads_pretrained.npz", dataset)
    root = Path(__file__).resolve().parents[2]
    settings = {"dataset_path": "data.json", "artifact_dir": str(root / "experiments"),
                "model_kind": "B1", "split": "test", "fit_cutoff": T.isoformat(),
                "prediction_cutoff": (T + timedelta(days=2)).isoformat(),
                "parameters": {"n_estimators": 5, "min_child_samples": 2, "stopping_rounds": 2},
                "ks": [10], "zero_positive_policy": "include",
                "foundation_features": {"embeddings": "ads_pretrained.npz", "pca_components": 2}}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(settings), encoding="utf-8")
    result = run(path)
    manifest = json.loads((Path(result["artifact_dir"]) / "run_manifest.json").read_text())
    assert len(manifest["foundation"]["embeddings_sha256"]) == 64
    assert manifest["foundation"]["embeddings_provenance"] == {"init": "pretrained"}
    assert manifest["foundation"]["feature_names"][-1] == "foundation_proto_similarity"
    settings["model_kind"] = "B0"
    settings["parameters"] = {"variant": "global_ctr"}
    path.write_text(json.dumps(settings), encoding="utf-8")
    with pytest.raises(ContractError, match="foundation_features"):
        run(path, validate_only=True)
