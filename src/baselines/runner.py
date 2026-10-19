"""Local B0/B1 runs from explicit canonical handoffs; no Git operations."""

import hashlib
import json
import math
import os
import tempfile
import time
import uuid
from dataclasses import asdict
from pathlib import Path

from ..evaluation.evaluator import DatasetRegistry, evaluate_predictions
from ..evaluation.predictions import write_predictions
from .adapters import parse_time
from .baseline import B0Scorer
from .contracts import (
    BaselineDataset,
    Candidate,
    ContractError,
    DatasetMetadata,
    Interaction,
    Label,
    Query,
)


def checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".json")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_dataset(path: Path) -> BaselineDataset:
    """Read canonical JSON, not raw Meituan/Taobao files or inferred fields."""
    data = json.loads(path.read_text(encoding="utf-8"))
    try:
        queries = tuple(Query(**{**row, "cutoff_time": parse_time(row["cutoff_time"])})
                        for row in data["queries"])
        labels = tuple(Label(**{**row, "available_at": parse_time(row["available_at"])})
                       for row in data["labels"])
        history = tuple(Interaction(**{**row, "event_time": parse_time(row["event_time"]),
                                       "available_at": parse_time(row["available_at"])})
                        for row in data.get("history", []))
        dataset = BaselineDataset(DatasetMetadata(**data["metadata"]), queries,
                                  tuple(Candidate(**row) for row in data["candidates"]),
                                  labels, history)
    except (KeyError, TypeError) as error:
        raise ContractError(f"invalid canonical dataset structure: {error}") from error
    dataset.validate()
    return dataset


def prepare(config_path: str | Path) -> tuple[dict, Path, BaselineDataset]:
    path = Path(config_path).resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {"dataset_path", "artifact_dir", "model_kind", "split", "fit_cutoff",
                "parameters", "ks", "zero_positive_policy"}
    if not isinstance(config, dict) or not required <= config.keys():
        raise ContractError("run config missing required keys")
    if config["model_kind"] not in {"B0", "B1", "B2", "B3", "B4"} or config["split"] not in {"validation", "test"}:
        raise ContractError("invalid model_kind or evaluation split")
    if not isinstance(config["parameters"], dict):
        raise ContractError("parameters must be object")
    if (not isinstance(config["ks"], list) or not config["ks"]
            or any(type(k) is not int or k <= 0 for k in config["ks"])):
        raise ContractError("ks requires positive integers")
    if config["zero_positive_policy"] not in {"include", "exclude"}:
        raise ContractError("invalid zero_positive_policy")
    extra = config.get("foundation_features")
    if extra is not None and (config["model_kind"] != "B1" or not isinstance(extra, dict)
                              or not isinstance(extra.get("embeddings"), str)
                              or not set(extra) <= {"embeddings", "pca_components",
                                                    "prototype_similarity"}):
        raise ContractError("foundation_features is a B1 object with embeddings path and options")
    parse_time(config["fit_cutoff"])
    dataset_path = (path.parent / config["dataset_path"]).resolve()
    return config, dataset_path, load_dataset(dataset_path)


def run(config_path: str | Path, *, validate_only: bool = False) -> dict:
    config, dataset_path, dataset = prepare(config_path)
    cutoff = parse_time(config["fit_cutoff"])
    selected = [q for q in dataset.queries if q.split == config["split"]]
    if not selected or any(q.cutoff_time < cutoff for q in selected):
        raise ContractError("evaluation queries missing or precede fit cutoff")
    if validate_only:
        if config['model_kind'] in {'B2', 'B3', 'B4'}:
            from .sequence import SequenceClassifier, sequence_samples

            raw = json.loads(dataset_path.read_text(encoding='utf-8'))
            parameters = dict(config['parameters'])
            if parameters.get('checkpoint_path'):
                parameters['checkpoint_path'] = str((Path(config_path).resolve().parent /
                                                     parameters['checkpoint_path']).resolve())
            model = SequenceClassifier(model_kind=config['model_kind'], **parameters)
            samples = sequence_samples(dataset, 'train', profiles=raw.get('sequence_profiles', {}))
            model._initialize(dataset, samples)
            for sample in samples:
                model._batch(sample)
            sequence_samples(dataset, config['split'], parse_time(config['prediction_cutoff']),
                             raw.get('sequence_profiles', {}))
        return {"status": "passed", "data_version": dataset.metadata.data_version,
                "query_count": len(dataset.queries), "candidate_count": len(dataset.candidates)}
    root = Path(__file__).resolve().parents[2]
    artifact_root = (Path(config_path).resolve().parent / config["artifact_dir"]).resolve()
    if not artifact_root.is_relative_to(root / "experiments"):
        raise ContractError("artifact_dir must stay inside workspace experiments directory")
    run_id = str(uuid.uuid4())
    output = artifact_root / run_id / "artifacts"
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    try:
        if config["model_kind"] == "B0":
            model = B0Scorer(**config["parameters"]).fit(dataset, cutoff)
            table = model.predict(dataset, config["split"], run_id)
            model.save(output / "model.json")
        elif config['model_kind'] == 'B1':
            from .lightgbm import B1Classifier

            history_start = (parse_time(config["history_start"])
                             if config.get("history_start") else None)
            foundation_features = None
            if config.get("foundation_features"):
                from .foundation_features import FoundationFeatures

                options = dict(config["foundation_features"])
                foundation_features = FoundationFeatures(
                    Path(config_path).resolve().parent / options.pop("embeddings"), **options)
            model = B1Classifier(**config["parameters"],
                                 foundation_features=foundation_features).fit(
                dataset, training_cutoff=cutoff, history_start=history_start)
            frozen = parse_time(config["prediction_cutoff"])
            table = model.predict(dataset, config["split"], run_id, frozen_cutoff=frozen,
                                  history_start=history_start)
            model.save(output / "model")
        else:
            from .sequence import SequenceClassifier

            raw = json.loads(dataset_path.read_text(encoding='utf-8'))
            parameters = dict(config['parameters'])
            if parameters.get('checkpoint_path'):
                parameters['checkpoint_path'] = str((Path(config_path).resolve().parent /
                                                     parameters['checkpoint_path']).resolve())
            profiles = raw.get('sequence_profiles', {})
            model = SequenceClassifier(model_kind=config['model_kind'], **parameters).fit(
                dataset, training_cutoff=cutoff, profiles=profiles)
            table = model.predict(dataset, config['split'], run_id,
                                  frozen_cutoff=parse_time(config['prediction_cutoff']),
                                  profiles=profiles)
            model.save(output / 'model')
        prediction_path = output / f"predictions.{config['split']}.parquet"
        write_predictions(prediction_path, table, dataset, config["split"])
        registry = DatasetRegistry()
        registry.register(dataset)
        metrics = evaluate_predictions(str(prediction_path), dataset.metadata.task,
                                       dataset.metadata.data_version, config["split"],
                                       registry=registry, ks=tuple(config["ks"]),
                                       zero_positive_policy=config["zero_positive_policy"])
        # Single-class discrimination metrics are explicitly undefined, strict JSON uses null.
        clean_metrics = {key: value if math.isfinite(value) else None for key, value in metrics.items()}
        files = {str(file.relative_to(output)): checksum(file) for file in output.rglob("*")
                 if file.is_file()}
        manifest = {"contract_version": "baseline-run-v1", "run_id": run_id,
                    "model_kind": config["model_kind"], "task": dataset.metadata.task,
                    "split": config["split"], "data_version": dataset.metadata.data_version,
                    "foundation": None, "code_revision": "unknown",
                    "dataset_sha256": checksum(dataset_path),
                    "config_sha256": checksum(Path(config_path)),
                    "candidate_or_exposure_policy": dataset.metadata.candidate_policy,
                    "seed": config["parameters"].get("seed", 42),
                    "elapsed_seconds": time.perf_counter() - started,
                    "row_count": len(table), "request_count": len(selected), "files": files}
        if getattr(model, "foundation_features", None) is not None:
            features = model.foundation_features
            manifest["foundation"] = {"embeddings_sha256": features.sha256,
                                      "feature_names": list(features.names),
                                      "embeddings_provenance": features.provenance}
        if config['model_kind'] in {'B2', 'B3', 'B4'}:
            manifest.update(foundation=model.foundation, resources=model.resources,
                            history_policy='train_prefix_eval_frozen',
                            preprocess_fingerprint=checksum(output / 'model' / 'vocabulary.json'),
                            candidate_counts=[sum(c.request_id == q.request_id for c in dataset.candidates)
                                              for q in selected],
                            fixture_only=model.config['fixture_only'])
            write_json(output / 'training_log.json', {'epochs': model.training_log})
        write_json(output / "config.json", config)
        write_json(output / "dataset_metadata.json", asdict(dataset.metadata))
        write_json(output / "metrics.json", clean_metrics)
        write_json(output / "run_manifest.json", manifest)
        write_json(output / "validation_report.json", {"status": "passed", "row_count": len(table),
                                                       "request_count": len(selected)})
        report = "# Baseline run\n\n" + f"{config['model_kind']} / {dataset.metadata.task}\n\n"
        report += "| Metric | Value |\n|---|---|\n"
        report += "".join(f"| {key} | {value if value is not None else 'undefined'} |\n"
                          for key, value in clean_metrics.items())
        (output / "report.md").write_text(report, encoding="utf-8")
        return {"status": "passed", "run_id": run_id, "artifact_dir": str(output),
                "metrics": clean_metrics}
    except (ContractError, ValueError, TypeError, KeyError, OSError) as error:
        write_json(output / "failure.json", {"status": "failed", "error_type": type(error).__name__,
                                              "message": str(error)})
        raise
