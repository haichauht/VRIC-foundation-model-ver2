"""Ads on the simulator package (``data.format: sim_v1``) and the random-init control arm."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
import yaml

from src.contracts.output import sha256_file, validate_and_write_predictions
from src.downstream.ad_inputs import (
    _index_sim_events,
    _sim_canonical,
    ad_baseline_dataset,
    ad_baseline_evaluator,
    load_ad_foundation,
    load_ad_inputs,
    read_ad_handoff,
)
from src.downstream.ad_optimization import (
    AdRunError,
    AdServices,
    _validate_config,
    ad_data_format,
    main,
    run_ad_optimization,
    validate_opportunities,
)
from src.downstream.sim_events import SESSION_START_TYPES, ads_window_events, sim_canonical_sample
from src.foundation_input.loader import iter_normalized_samples
from src.foundation_input.manifest import load_manifest
from src.foundation_input.vocab import VocabularyConfig, fit_vocabulary_from_samples
from src.models.encoder import FoundationModel, FoundationModelConfig
from src.training.checkpoint import load_checkpoint, save_checkpoint
from src.training.pretrain import pretrain_preprocess_metadata, tokenizer_config_for_manifest

SIM_POLICY = "sim-served-impression-single-candidate-v1"
TAOBAO_POLICY = "taobao-observed-impression-single-candidate-v1"
NOT_CLICKED_MATURITY_SECONDS = 27  # contract "Baseline": exposure + 27 s when not clicked


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@dataclass(frozen=True)
class SimAds:
    packages: object  # tests/conftest.py SimPackages
    manifest: dict  # ads/manifest.json
    checkpoint: Path


@pytest.fixture(scope="module")
def sim_ads(sim_tiny_packages, tmp_path_factory) -> SimAds:
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
    checkpoint = tmp_path_factory.mktemp("sim_ads") / "best_checkpoint.pt"
    save_checkpoint(
        checkpoint, FoundationModel(config), config, vocab,
        preprocess_metadata=pretrain_preprocess_metadata(
            pretrain, vocab, max_length=512, max_event_count=30),
    )
    manifest = json.loads((sim_tiny_packages.ads / "manifest.json").read_text(encoding="utf-8"))
    return SimAds(sim_tiny_packages, manifest, checkpoint)


def sim_config(sim_ads: SimAds, tmp_path, *, root=None, checkpoint=None, init="pretrained"):
    return {
        "contract_version": 1, "seed": 42,
        "data": {"format": "sim_v1", "root": str(root or sim_ads.packages.ads),
                 "expected_version": sim_ads.manifest["dataset_version"]},
        "checkpoint": {"path": str(checkpoint or sim_ads.checkpoint)},
        "foundation": {"init": init},
        "freeze_foundation": True, "split": "test",
        "artifact_dir": str(tmp_path / "artifacts"),
        "exposure_policy": SIM_POLICY, "missing_click_policy": "reject",
        "export_model": False, "evaluate": True,
        "evaluation": {"label_availability_policy": "handoff_timestamp"},
        "model": {"hidden_dim": 8, "candidate_dim": 4},
        "training": {"epochs": 1, "learning_rate": 0.01, "batch_size": 256,
                     "cache_foundation_embeddings": True, "cpu_threads": 1},
        "calibration": {"enabled": False, "split": "validation"},
    }


def load_sim_foundation(sim_ads: SimAds, path=None):
    return load_ad_foundation(
        path or sim_ads.checkpoint, expected_version=sim_ads.manifest["dataset_version"],
        expected_sim_config_sha256=sim_ads.manifest["sim_config_sha256"],
    )


def real_services(config, load_inputs=load_ad_inputs, evaluator=None):
    return AdServices(load_ad_foundation, load_inputs, validate_and_write_predictions,
                      evaluator or ad_baseline_evaluator(config))


def ads_rows(root: Path, name: str) -> list[dict]:
    return pq.read_table(root / f"{name}.parquet").to_pylist()


def write_ads_package(source: Path, target: Path, *, manifest=None, **tables) -> Path:
    """Copy an Ads package, replacing whole tables; the manifest pins the rewritten files."""
    target.mkdir(parents=True)
    manifest = copy.deepcopy(manifest or json.loads(
        (source / "manifest.json").read_text(encoding="utf-8")))
    for name, metadata in manifest["files"].items():
        path = target / metadata["path"]
        if name in tables:
            schema = pq.read_schema(source / metadata["path"])
            pq.write_table(pa.Table.from_pylist(tables[name], schema=schema), path)
            payload = path.read_bytes()
            metadata.update(sha256=hashlib.sha256(payload).hexdigest(),
                            row_count=len(tables[name]), bytes=len(payload))
        else:
            shutil.copyfile(source / metadata["path"], path)
    (target / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return target


def read_manifest(root: Path) -> dict:
    return json.loads((root / "manifest.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def small_ads(sim_ads, tmp_path_factory) -> Path:
    """Every fifth request of the tiny package (same events), to keep end-to-end runs quick."""
    rows = ads_rows(sim_ads.packages.ads, "ad_opportunities")[::5]
    manifest = copy.deepcopy(sim_ads.manifest)
    manifest["row_count"] = len(rows)
    manifest["request_counts"] = {
        split: sum(row["split"] == split for row in rows) for split in manifest["request_counts"]}
    return write_ads_package(sim_ads.packages.ads, tmp_path_factory.mktemp("small") / "ads",
                             manifest=manifest, ad_opportunities=rows)


@pytest.fixture(scope="module")
def sim_inputs(sim_ads, tmp_path_factory):
    config = sim_config(sim_ads, tmp_path_factory.mktemp("unused"))
    return load_ad_inputs(config, load_sim_foundation(sim_ads))


def test_sim_ads_inputs_build(sim_ads, sim_inputs):
    ads = sim_ads.packages.ads
    counts = sim_ads.manifest["request_counts"]
    assert {split: sum(len(batch.rows) for batch in batches)
            for split, batches in sim_inputs.batches.items()} == counts
    clicked = {row["request_id"]: bool(row["clicked"])
               for row in ads_rows(ads, "ad_opportunities")}
    labels = {}
    for split, batches in sim_inputs.batches.items():
        for batch in batches:
            values, mask = batch.labels_and_mask()
            assert mask.all()
            for index, rows in enumerate(batch.rows):
                assert len(rows) == 1 and rows[0]["split"] == split
                assert rows[0]["user_id"].startswith("sim:user:")
                labels[rows[0]["request_id"]] = bool(values[index, 0])
    assert labels == clicked
    assert sim_inputs.data_manifest_sha256 == sha256_file(ads / "manifest.json")
    assert sim_inputs.manifest["source_dataset"] == "sim_v1"
    train_ads = sorted({row["candidate_id"] for row in sim_inputs.opportunities
                        if row["split"] == "train"})
    assert dict(sim_inputs.candidate_vocabulary) == {
        candidate: index + 1 for index, candidate in enumerate(train_ads)}
    foundation = load_sim_foundation(sim_ads).eval()
    for split, batches in sim_inputs.batches.items():
        with torch.no_grad():
            embedded = foundation.embed_users(batches[0].foundation)
        assert all(value.shape == (len(batches[0].rows), 16) and torch.isfinite(value).all()
                   for value in embedded.values()), split


def test_sim_history_goes_through_shared_converter(sim_ads, tmp_path):
    manifest, tables, _ = read_ad_handoff(sim_config(sim_ads, tmp_path))
    events_by_user, _ = _index_sim_events(tables)
    profiles = {row["user_id"]: row for row in tables["profiles"]}
    seen = {}
    for row in tables["ad_opportunities"]:
        window = events_by_user[row["user_id"]][row["history_start"]:row["history_stop"]]
        kind = ("empty" if not window else "start"
                if window[0]["event_type"] in SESSION_START_TYPES else "mid_session")
        if kind in seen:
            continue
        seen[kind] = row
        sample = _sim_canonical(row, manifest, events_by_user, profiles)
        expected = sim_canonical_sample(
            sample_id=row["request_id"], dataset_name="sim_v1", split=row["split"],
            user_id=row["user_id"],
            cutoff_time=datetime.fromtimestamp(row["cutoff_timestamp"], timezone.utc),
            profile={}, events=ads_window_events(window),
            history_truncated=row["history_start"] > 0,
            history_group_truncated=kind == "mid_session",
        )
        assert sample.events == expected.events
        assert [event.event_id for event in sample.events] == [e["event_id"] for e in window]
        assert sample.user_id == row["user_id"] and sample.dataset_name == "sim_v1"
        assert sample.history_truncated == (row["history_start"] > 0)
        assert sample.history_group_truncated == (kind == "mid_session")  # R3
        assert sample.cutoff_time == expected.cutoff_time
        profile = profiles[row["user_id"]]
        assert sample.profile == {
            "as_of_time": datetime.fromtimestamp(profile["as_of_timestamp"], timezone.utc)
            .replace(tzinfo=None).isoformat(),
            "age_group": profile["age_group"],
            "saved_home_geohash": profile["saved_home_geohash"],
            "saved_work_geohash": profile["saved_work_geohash"],
        }
    assert set(seen) == {"empty", "start", "mid_session"}


def test_sim_label_availability(sim_ads, tmp_path):
    ads = sim_ads.packages.ads
    config = sim_config(sim_ads, tmp_path)
    _, tables, _ = read_ad_handoff(config)
    events = ads_rows(ads, "events")
    rows = tables["ad_opportunities"]
    clicked = [row for row in rows if row["clicked"]]
    plain = [row for row in rows if not row["clicked"]]
    assert clicked and plain
    for row in clicked:
        click = events[row["source_row_index"] + 1]
        assert (click["user_id"], click["event_type"], click["entity_id"]) == (
            row["user_id"], "ad_click", row["candidate_id"])
        assert row["label_available_at_timestamp"] == click["timestamp"]
        assert click["timestamp"] > row["exposure_timestamp"]
    for row in plain:
        assert row["label_available_at_timestamp"] == (
            row["exposure_timestamp"] + NOT_CLICKED_MATURITY_SECONDS)
    dataset = ad_baseline_dataset(config)
    assert dataset.metadata.dataset_name == "sim_v1"
    assert dataset.metadata.availability_policy == "handoff_timestamp"
    available = {label.request_id: label.available_at for label in dataset.labels}
    assert available == {
        row["request_id"]: datetime.fromtimestamp(row["label_available_at_timestamp"],
                                                  timezone.utc)
        for row in rows}
    assert {label.request_id: label.target for label in dataset.labels} == {
        row["request_id"]: int(row["clicked"]) for row in rows}


def test_sim_split_bounds_are_exclusive(sim_ads, small_ads, tmp_path, capsys):
    rows = ads_rows(small_ads, "ad_opportunities")
    first_validation = min(row["cutoff_timestamp"] for row in rows
                           if row["split"] == "validation")
    manifest = read_manifest(small_ads)
    manifest["split"]["train_end"] = first_validation
    package = write_ads_package(small_ads, tmp_path / "ads", manifest=manifest)
    config = sim_config(sim_ads, tmp_path, root=package)
    _, tables, _ = read_ad_handoff(config)
    boundary = [row for row in tables["ad_opportunities"]
                if row["cutoff_timestamp"] == manifest["split"]["train_end"]]
    assert boundary and {row["split"] for row in boundary} == {"validation"}
    version = manifest["dataset_version"]
    validate_opportunities(tables["ad_opportunities"], manifest, version, SIM_POLICY,
                           exclusive_split_ends=True)
    with pytest.raises(AdRunError, match="split expected train") as caught:  # Taobao rule
        validate_opportunities(tables["ad_opportunities"], manifest, version, SIM_POLICY)
    assert caught.value.code == 3
    # The CLI run (default bindings) re-validates the handoff with the sim rule.
    path = tmp_path / "ads.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    assert main(["--config", str(path)]) == 0, capsys.readouterr().err
    result = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert {"roc_auc", "log_loss", "brier_score"} <= set(result["metrics"])


@pytest.mark.parametrize(("mutation", "match"), [
    ("other_sim_config", "sim_config_sha256"),
    ("no_provenance", "sim_config_sha256"),
    ("other_namespace", "dataset_name"),
])
def test_ads_sim_checkpoint_provenance_mismatch_is_rejected(
    sim_ads, tmp_path, monkeypatch, mutation, match
):
    payload = torch.load(sim_ads.checkpoint, map_location="cpu", weights_only=False)
    if mutation == "other_sim_config":
        payload["preprocess_metadata"]["dataset_provenance"]["sim_config_sha256"] = "0" * 64
    elif mutation == "no_provenance":
        del payload["preprocess_metadata"]["dataset_provenance"]
    else:
        payload["vocabularies"]["dataset_name"] = "taobao_display_ads"
    bad = tmp_path / "other_world.pt"
    torch.save(payload, bad)
    with pytest.raises(ValueError, match=match):
        load_sim_foundation(sim_ads, bad)
    # The input loader re-checks the binding before any request is encoded.
    from src.downstream import ad_inputs

    built = []
    monkeypatch.setattr(ad_inputs, "encode_sample", lambda *args: built.append(args))
    config = sim_config(sim_ads, tmp_path, checkpoint=bad)
    with pytest.raises(ValueError, match=match):
        load_ad_inputs(config, load_sim_foundation(sim_ads))
    assert built == []
    with pytest.raises(AdRunError, match=match) as caught:
        run_ad_optimization(config, real_services(config, evaluator=lambda **kwargs: {}))
    assert caught.value.code == 4
    assert built == [] and not (tmp_path / "artifacts").exists()


@pytest.mark.parametrize(("problem", "match"), [
    ("click_label", "clicked disagrees with the event stream"),
    ("impression_pointer", "exposure identity mismatch"),
    ("profile_population", "population"),
    ("request_counts", "request_counts"),
])
def test_sim_content_validation(sim_ads, tmp_path, problem, match):
    ads = sim_ads.packages.ads
    rows = ads_rows(ads, "ad_opportunities")
    tables, manifest = {}, None
    if problem == "click_label":
        row = next(row for row in rows if not row["clicked"])
        row["clicked"] = True
        tables["ad_opportunities"] = rows
    elif problem == "impression_pointer":
        events = ads_rows(ads, "events")
        row = next(row for row in rows if row["source_row_index"] + 1 < len(events)
                   and events[row["source_row_index"] + 1]["event_type"] != "ad_impression")
        row["source_row_index"] += 1  # the next event row, not this impression
        tables["ad_opportunities"] = rows
    elif problem == "profile_population":
        profiles = ads_rows(ads, "profiles")
        tables["profiles"] = [p for p in profiles if p["user_id"] != rows[0]["user_id"]]
    else:
        manifest = copy.deepcopy(sim_ads.manifest)
        manifest["request_counts"]["train"] += 1
    package = write_ads_package(ads, tmp_path / "ads", manifest=manifest, **tables)
    with pytest.raises(ValueError, match=match):
        read_ad_handoff(sim_config(sim_ads, tmp_path, root=package))


@pytest.mark.parametrize("arm", ["pretrained", "random"])
def test_sim_run_evaluates_and_records_foundation_arm(sim_ads, small_ads, tmp_path, arm):
    # Same config for both arms but foundation.init, embedding cache included.
    config = sim_config(sim_ads, tmp_path, root=small_ads, init=arm)
    config["export_model"] = True
    used = []

    def load_inputs(cfg, foundation):
        used.append(foundation)
        return load_ad_inputs(cfg, foundation)

    result = run_ad_optimization(config, real_services(config, load_inputs=load_inputs))
    assert {"roc_auc", "log_loss", "brier_score"} <= set(result["metrics"])
    manifest = json.loads(Path(result["predictions_path"]).with_name(
        "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["foundation"]["init"] == arm
    assert manifest["candidate_or_exposure_policy"] == SIM_POLICY
    assert manifest["request_count"] == read_manifest(small_ads)["request_counts"]["test"]
    assert manifest["split_cutoffs"] == sim_ads.manifest["split"]
    assert torch.load(result["model_path"], weights_only=True)["foundation_init"] == arm
    pretrained = load_checkpoint(sim_ads.checkpoint)[0].state_dict()
    state = used[0].state_dict()
    unchanged = all(torch.equal(state[name], pretrained[name]) for name in state)
    assert unchanged == (arm == "pretrained")
    assert used[0].config == load_checkpoint(sim_ads.checkpoint, load_model=False)[1]


@pytest.mark.parametrize(("change", "match"), [
    (lambda c: c["data"].update(format="parquet"), "data.format"),
    (lambda c: c.update(foundation={"init": "zeros"}), "foundation.init"),
    (lambda c: c["checkpoint"].update(expected_version="sim-v1"), "checkpoint.expected_version"),
])
def test_sim_config_errors_exit_two(sim_ads, tmp_path, change, match):
    config = sim_config(sim_ads, tmp_path)
    change(config)

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid config must fail before loading")

    with pytest.raises(AdRunError, match=match) as caught:
        run_ad_optimization(config, AdServices(forbidden, forbidden, forbidden, forbidden))
    assert caught.value.code == 2


def test_taobao_defaults_unchanged(sim_ads, tmp_path):
    config = sim_config(sim_ads, tmp_path)
    del config["data"]["format"], config["foundation"]
    assert ad_data_format(config) == "taobao"
    config["exposure_policy"] = TAOBAO_POLICY
    # Every Taobao check still applies: the simulator package is not a Taobao handoff ...
    with pytest.raises(ValueError, match="namespace mismatch"):
        read_ad_handoff(config)
    # ... nor is a sim checkpoint a Taobao one (version/namespace/manifest binding).
    with pytest.raises(ValueError, match="dataset version"):
        load_ad_foundation(sim_ads.checkpoint, expected_version=config["data"]["expected_version"])
    # Split bounds stay inclusive by default.
    rows = [{"request_id": "r", "user_id": "u", "candidate_id": "a", "cutoff_timestamp": 20,
             "clicked": 0, "split": "train"}]
    manifest = {"dataset_version": "v", "schema_version": 1, "candidate_policy": "p",
                "split": {"kind": "temporal", "train_end": 20, "valid_end": 40}}
    validate_opportunities(rows, manifest, "v", "p")
    with pytest.raises(AdRunError, match="split expected validation"):
        validate_opportunities(rows, manifest, "v", "p", exclusive_split_ends=True)
    # The sim format, in turn, refuses a Taobao manifest.
    sim = sim_config(sim_ads, tmp_path)
    sim["exposure_policy"] = TAOBAO_POLICY
    with pytest.raises(ValueError, match="exposure policy"):
        read_ad_handoff(sim)


def test_checked_in_sim_config():
    sim = yaml.safe_load(Path("configs/ad_optimization_sim.yaml").read_text(encoding="utf-8"))
    assert sim["data"] == {"format": "sim_v1", "root": "experiments/sim/sim-smoke-v1/ads",
                           "expected_version": "sim-smoke-v1"}
    assert sim["checkpoint"] == {"path": "experiments/pretrain_sim/checkpoints/best_checkpoint.pt"}
    assert sim["exposure_policy"] == SIM_POLICY
    assert sim["evaluate"] is True
    assert sim["evaluation"] == {"label_availability_policy": "handoff_timestamp"}
    assert sim["foundation"] == {"init": "pretrained"}
    _validate_config(sim)
