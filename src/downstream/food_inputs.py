"""Load Food ranking inputs from samples.jsonl; no Data-team files are rewritten.

``data.format`` selects the package: ``meituan`` (default; samples zip + pois.txt + Person 3
checkpoint) or ``sim_v1`` (``food/`` of the simulator: manifest.json, samples.jsonl,
merchants.parquet; checkpoint bound by vocabulary namespace and simulator config hash).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields
from functools import partial
from pathlib import Path

from src.contracts.output import fingerprint, sha256_file
from src.downstream.candidate_features import (
    attribute_names,
    encode_attributes,
    fit_attribute_vocabularies,
)
from src.downstream.food_samples import (
    MAPPING_VERSION,
    SIM_MAPPING_VERSION,
    SPLITS,
    candidate_rows,
    load_merchant_catalog,
    load_poi_catalog,
    read_food_samples,
    read_sim_food_manifest,
    sim_candidate_rows,
    sim_to_canonical_sample,
    to_canonical_sample,
    validate_food_samples,
)
from src.downstream.foundation_init import check_sim_checkpoint_binding
from src.foundation_input.dataset import DatasetConfig, encode_sample, foundation_collate
from src.foundation_input.tokenizer import TokenizerConfig
from src.foundation_input.validation import validate_sample
from src.training.checkpoint import load_checkpoint

FOOD_DATA_FORMATS = ("meituan", "sim_v1")
FOOD_SAMPLES_CAVEATS = (
    "History của samples chỉ có đơn hàng, không có click như lúc pretrain; profile đếm event thấp hơn.",
    "long_term là vector ngẫu nhiên: UserIdAdapter không nhận gradient trong pretraining hiện tại.",
    ("Checkpoint được pretrain trên event đến 21/03, trùng giai đoạn validation/test của samples"
    " → metric chỉ mang tính chẩn đoán."),
    ("Candidate có lối tắt theo restaurant_id (positive luôn cùng một quán; 99,6% negative khác quán)."
    " FoodRanker chấm từng món độc lập nên không khai thác được lối tắt này,"
    " nhưng baseline nào nhìn cả danh sách có thể khai thác."),
)
SIM_FOOD_CAVEATS = (
    ("Ứng viên chỉ dùng merchant_id: FoodRanker không đọc cuisine, price_level, geohash"
     " của catalog."),
)


def food_data_format(config):
    """The configured ``data.format`` (default ``"meituan"``), validated."""
    data = config["data"]
    if not isinstance(data, dict):
        raise ValueError("data must be a mapping")  # noqa: TRY004
    value = data.get("format", "meituan")
    if value not in FOOD_DATA_FORMATS:
        raise ValueError(f"data.format must be one of {list(FOOD_DATA_FORMATS)}, got {value!r}")
    return value


@dataclass(frozen=True)
class FoodSource:
    """One read and validated Food package plus the row/sample mappers of its format."""

    records: list
    dataset_version: str
    namespace: str
    policy: str
    data_sha256: str
    provenance: dict
    mapping_version: str
    caveats: tuple
    rows_of: Callable
    to_sample: Callable
    sim_manifest: dict | None = None
    catalog: dict | None = None  # merchant_id -> merchants.parquet row (sim_v1)


def _meituan_source(config):
    data, policy = config["data"], config["candidate_policy"]
    records, samples_sha = read_food_samples(data["path"], data["member"], data["expected_sha256"])
    validate_food_samples(records, policy)
    catalog, pois_sha = load_poi_catalog(
        config["catalog"]["pois_path"], config["catalog"]["expected_sha256"])
    namespace = data["namespace"]
    return FoodSource(
        records, data["expected_version"], namespace, policy, samples_sha,
        {"samples_sha256": samples_sha, "pois_sha256": pois_sha}, MAPPING_VERSION,
        FOOD_SAMPLES_CAVEATS, partial(candidate_rows, catalog=catalog),
        partial(to_canonical_sample, catalog=catalog, namespace=namespace),
    )


def _sim_source(config):
    data = config["data"]
    root = Path(data["root"])
    manifest, manifest_sha = read_sim_food_manifest(root, data["expected_version"])
    samples, merchants = manifest["files"]["samples"], manifest["files"]["merchants"]
    records, samples_sha = read_food_samples(
        root / samples["path"], samples["path"], samples["sha256"])
    policy = manifest["candidate_policy"]
    validate_food_samples(records, policy)
    catalog = load_merchant_catalog(root / merchants["path"], merchants["sha256"])
    namespace = manifest["source_dataset"]
    provenance = {"samples_sha256": samples_sha, "merchants_sha256": merchants["sha256"],
                  "food_manifest_sha256": manifest_sha,
                  "sim_config_sha256": manifest["sim_config_sha256"]}
    return FoodSource(
        records, manifest["dataset_version"], namespace, policy, manifest_sha, provenance,
        SIM_MAPPING_VERSION, SIM_FOOD_CAVEATS, partial(sim_candidate_rows, catalog=catalog),
        partial(sim_to_canonical_sample, dataset_name=namespace), manifest, catalog,
    )


def read_food_source(config):
    """Read and validate the configured Food package (``data.format``)."""
    if food_data_format(config) == "sim_v1":
        return _sim_source(config)
    return _meituan_source(config)


def _read_person3_handoff(path, expected_version, handoff):
    """Bind the untouched checkpoint to the exact record/vocabulary supplied with it."""
    required = {"experiment_record_path", "expected_record_sha256", "vocab_path",
                "expected_vocab_sha256", "expected_checkpoint_sha256"}
    if not isinstance(handoff, dict) or not required <= set(handoff):
        raise ValueError("Food checkpoint handoff missing required binding fields")
    for source, expected, label in (
        (path, handoff["expected_checkpoint_sha256"], "checkpoint"),
        (handoff["experiment_record_path"], handoff["expected_record_sha256"], "experiment record"),
        (handoff["vocab_path"], handoff["expected_vocab_sha256"], "vocabulary"),
    ):
        if not isinstance(expected, str) or len(expected) != 64 or sha256_file(source) != expected:
            raise ValueError(f"Food handoff {label} SHA-256 mismatch")
    record = json.loads(Path(handoff["experiment_record_path"]).read_text(encoding="utf-8"))
    vocabulary = json.loads(Path(handoff["vocab_path"]).read_text(encoding="utf-8"))
    if record.get("best_checkpoint_sha256") != handoff["expected_checkpoint_sha256"]:
        raise ValueError("Food handoff record does not identify this best checkpoint")
    if record.get("dataset_version") != expected_version:
        raise ValueError("Food checkpoint dataset version mismatch")
    return record, vocabulary


# FoundationModelConfig fields added after the Person 3 record format was frozen (branch
# feat/sim-foundation); only these may be absent from a record.
_POST_RECORD_MODEL_FIELDS = frozenset({"long_term_injection", "user_embedding_init_std"})


def _record_matches_model_config(recorded, model_config):
    """A record may omit a field added after it was written, only when the checkpoint has its default."""
    if not isinstance(recorded, dict):
        return False
    actual = asdict(model_config)
    defaults = {item.name: item.default for item in fields(model_config)}
    return set(recorded) <= set(actual) and all(
        recorded[name] == value if name in recorded
        else name in _POST_RECORD_MODEL_FIELDS and value == defaults[name]
        for name, value in actual.items()
    )


def _validate_person3_payload(payload, model_config, vocabulary, record, external_vocabulary):
    metadata = payload.get("preprocess_metadata", {})
    if not _record_matches_model_config(record.get("model_config"), model_config):
        raise ValueError("Food handoff model configuration mismatch")
    if (record.get("dataset_name") != vocabulary.dataset_name
            or metadata.get("dataset_name") != vocabulary.dataset_name):
        raise ValueError("Food handoff dataset namespace mismatch")
    if external_vocabulary != payload["vocabularies"]:
        raise ValueError("Food handoff vocabulary differs from checkpoint vocabulary")
    if ("dataset_version" in metadata
            and metadata["dataset_version"] != record["dataset_version"]):
        raise ValueError("Food handoff version conflicts with checkpoint metadata")
    tokenizer_from_vocabulary(vocabulary)


def load_food_foundation(path, *, expected_version, handoff=None, expected_sim_config_sha256=None):
    """Load the public encoder, with an optional hash-bound Person 3 handoff.

    With ``expected_sim_config_sha256`` (``data.format: sim_v1``) the checkpoint is bound to the
    Food package by vocabulary namespace and simulator config hash instead; the input loader
    matches ``expected_version`` against the Food manifest.
    """
    if expected_sim_config_sha256 is not None:
        if handoff is not None:
            raise ValueError("Food sim_v1 checkpoints take no Person 3 handoff")
        model, _, vocabulary, payload = load_checkpoint(path, device="cpu")
        check_sim_checkpoint_binding(
            vocabulary, payload.get("preprocess_metadata", {}), expected_sim_config_sha256,
            package="Food")
        return model
    # Verify bindings before the public loader deserializes the external file.
    bound = None if handoff is None else _read_person3_handoff(path, expected_version, handoff)
    model, model_config, vocabulary, payload = load_checkpoint(path, device="cpu")
    if bound is None:
        if payload.get("preprocess_metadata", {}).get("dataset_version") != expected_version:
            raise ValueError("Food checkpoint dataset version mismatch")
    else:
        _validate_person3_payload(payload, model_config, vocabulary, *bound)
    return model


def _valid_sequence(sequence, model_config):
    return (isinstance(sequence, dict)
            and type(sequence.get("max_tokens")) is int
            and type(sequence.get("max_event_count")) is int
            and 2 <= sequence["max_tokens"] <= model_config.max_length
            and sequence["max_event_count"] >= 1)


def _sim_checkpoint_metadata(config, checkpoint, model_config, vocabulary, sim_manifest):
    """Sim binding plus the sequence policy: the checkpoint's (a config ``sequence`` must match)."""
    metadata = checkpoint.get("preprocess_metadata", {})
    check_sim_checkpoint_binding(vocabulary, metadata, sim_manifest["sim_config_sha256"],
                                 package="Food")
    sequence = metadata.get("sequence")
    if "sequence" in config and config["sequence"] != sequence:
        raise ValueError(
            f"Food config sequence {config['sequence']!r} differs from the checkpoint's "
            f"preprocess_metadata.sequence {sequence!r}")
    if not _valid_sequence(sequence, model_config):
        raise ValueError("Food sim_v1 requires valid sequence settings in the checkpoint")
    return {
        "checkpoint_preprocess_metadata": metadata,
        "sim_checkpoint_binding": {
            "dataset_name": vocabulary.dataset_name,
            "sim_config_sha256": sim_manifest["sim_config_sha256"],
            "food_dataset_version": sim_manifest["dataset_version"],
        },
        "downstream_sequence": {
            "max_tokens": sequence["max_tokens"], "max_event_count": sequence["max_event_count"],
        },
    }


def resolve_food_checkpoint_metadata(config, checkpoint, model_config, vocabulary, *,
                                     sim_manifest=None):
    """Return auditable preprocessing provenance without rewriting checkpoint metadata.

    Person 3's runner stores dataset version in experiment_record.json but omits
    sequence settings from its checkpoint. An explicit downstream sequence policy
    therefore describes this run's inputs, never an inferred pretraining setting.
    """
    if food_data_format(config) == "sim_v1":
        if sim_manifest is None:
            sim_manifest, _ = read_sim_food_manifest(
                config["data"]["root"], config["data"]["expected_version"])
        return _sim_checkpoint_metadata(config, checkpoint, model_config, vocabulary, sim_manifest)
    settings = config["checkpoint"]
    expected_version = settings.get("expected_version", config["data"]["expected_version"])
    if "handoff" not in settings:
        metadata = checkpoint.get("preprocess_metadata", {})
        if metadata.get("dataset_version") != expected_version:
            raise ValueError("Food checkpoint dataset version mismatch")
        return metadata
    handoff = settings["handoff"]
    record, external_vocabulary = _read_person3_handoff(settings["path"], expected_version, handoff)
    _validate_person3_payload(checkpoint, model_config, vocabulary, record, external_vocabulary)
    sequence = config.get("sequence", {})
    if not _valid_sequence(sequence, model_config):
        raise ValueError("Food Person 3 handoff requires explicit valid downstream sequence settings")
    return {
        "checkpoint_preprocess_metadata": checkpoint.get("preprocess_metadata", {}),
        "person3_handoff": {
            **handoff,
            "dataset_name": record["dataset_name"],
            "dataset_version": record["dataset_version"],
            "checkpoint_epoch": checkpoint.get("epoch"),
            "checkpoint_step": checkpoint.get("step"),
            "pretrain_code_revision": record.get("code_revision", "unknown"),
        },
        "downstream_sequence": {
            "max_tokens": sequence["max_tokens"], "max_event_count": sequence["max_event_count"],
        },
    }


def tokenizer_from_vocabulary(vocabulary):
    contract = vocabulary.tokenizer_contract
    if not contract:
        raise ValueError("Food checkpoint lacks tokenizer contract")
    tokenizer = TokenizerConfig(
        context_allowlist={key: tuple(value) for key, value in contract["context_allowlist"].items()},
        max_text_chars=int(contract["max_text_chars"]),
        include_action_content=bool(contract["include_action_content"]),
        numeric_fields=frozenset(tuple(value) for value in contract["numeric_fields"]),
        numeric_transform_policy={
            (key.rsplit(":", 3)[-3], key.rsplit(":", 3)[-1]): value
            for key, value in vocabulary.numeric_transform_policy.items()
        },
        time_gap_boundaries_seconds=tuple(contract["time_gap_boundaries_seconds"]),
    )
    vocabulary.validate_tokenizer_config(tokenizer)
    return tokenizer


def _with_request_id(error, request_id):
    message = str(error)
    if request_id is None or "request_id=" in message:
        return ValueError(message)
    return ValueError(f"request_id={request_id}: {message}")


EVALUATION_POLICIES = {
    "item_kind": "food",
    "label_availability_policy": "handoff_timestamp",
    "observation_complete_policy": "assume_complete_diagnostic",
}


def food_baseline_dataset(config):
    """Build the Baseline handoff from samples.jsonl with explicit evaluation policies."""
    from datetime import datetime, timezone

    from src.baselines.contracts import BaselineDataset, Candidate, DatasetMetadata, Label, Query

    evaluation = config.get("evaluation")
    evaluation = evaluation if isinstance(evaluation, dict) else {}
    for key, required in EVALUATION_POLICIES.items():
        if evaluation.get(key) != required:
            raise ValueError(f"evaluation.{key} must be {required!r}")
    source = read_food_source(config)

    def moment(seconds):
        return datetime.fromtimestamp(seconds, timezone.utc)

    queries, candidates, labels = [], [], []
    for record in source.records:
        try:
            rows = source.rows_of(record)
        except ValueError as error:
            raise _with_request_id(error, record.get("sample_id")) from error
        first = rows[0]
        queries.append(Query(first["request_id"], first["user_id"],
                             moment(first["cutoff_timestamp"]), first["split"]))
        for row in rows:
            candidates.append(Candidate(row["request_id"], row["candidate_id"]))
            labels.append(Label(row["request_id"], row["candidate_id"], int(row["is_positive"]),
                                moment(row["label_available_at_timestamp"])))
    dataset = BaselineDataset(
        DatasetMetadata(
            dataset_name=source.namespace, data_version=source.dataset_version,
            task="food_recommendation", item_kind="food", timezone="UTC",
            candidate_policy=source.policy, availability_policy="handoff_timestamp",
        ),
        tuple(queries), tuple(candidates), tuple(labels),
    )
    try:
        dataset.validate()
    except ValueError as error:
        found = re.search(r"request=(\S+?)(?:,|: |$)", str(error))
        raise _with_request_id(error, found.group(1) if found else None) from error
    return dataset


def food_baseline_evaluator(config):
    """Bind Baseline's public evaluator to a validated, explicit Food registry."""
    from src.evaluation.evaluator import DatasetRegistry, evaluate_predictions

    registry = DatasetRegistry()
    registry.register(food_baseline_dataset(config))
    return partial(evaluate_predictions, registry=registry, ks=tuple(config["evaluation"]["ks"]))


def _unseen_rate(rows, vocabulary, positive_only):
    selected = [row for row in rows if row["is_positive"] or not positive_only]
    if not selected:
        return 0.0
    return sum(row["candidate_id"] not in vocabulary for row in selected) / len(selected)


def load_food_inputs(config, foundation):
    from src.downstream.food_recommendation import FoodBatch, FoodInputs, validate_food_candidates

    source = read_food_source(config)
    all_rows = [source.rows_of(record) for record in source.records]
    flat_rows = [row for rows in all_rows for row in rows]
    validate_food_candidates(flat_rows)
    _, model_config, vocabulary, checkpoint = load_checkpoint(
        config["checkpoint"]["path"], device="cpu", load_model=False
    )
    effective_metadata = resolve_food_checkpoint_metadata(
        config, checkpoint, model_config, vocabulary, sim_manifest=source.sim_manifest)
    if model_config != foundation.config:
        raise ValueError("Food checkpoint/model configuration mismatch")
    namespace = source.namespace
    if namespace != vocabulary.dataset_name:
        raise ValueError("Food namespace incompatible with checkpoint; no implicit cross-dataset ID mapping")
    training_ids = sorted({row["candidate_id"] for row in flat_rows if row["split"] == "train"})
    candidate_vocabulary = {candidate: index + 1 for index, candidate in enumerate(training_ids)}
    names = attribute_names(config.get("model", {}))
    attribute_vocabularies = None
    if names:
        if source.catalog is None:
            raise ValueError("model.candidate_attributes needs the sim_v1 merchant catalog")
        attribute_vocabularies = fit_attribute_vocabularies(source.catalog, names, training_ids)
    tokenizer = tokenizer_from_vocabulary(vocabulary)
    sequence = effective_metadata.get("downstream_sequence", effective_metadata.get("sequence"))
    if sequence is None:
        raise ValueError("Food checkpoint lacks sequence settings")
    dataset = DatasetConfig(max_tokens=sequence["max_tokens"], max_event_count=sequence["max_event_count"])
    batches = {split: [] for split in SPLITS}
    pending = {split: [] for split in batches}
    batch_size = config["training"].get("batch_size", 1)

    def emit(split):
        groups = pending[split]
        if not groups:
            return
        import torch
        samples = [sample for sample, _ in groups]
        request_rows = [rows for _, rows in groups]
        candidate_ids = torch.zeros((len(groups), max(len(rows) for rows in request_rows)), dtype=torch.long)
        for index, rows in enumerate(request_rows):
            candidate_ids[index, :len(rows)] = torch.tensor(
                [candidate_vocabulary.get(row["candidate_id"], 0) for row in rows], dtype=torch.long
            )
        attributes = None if not names else encode_attributes(
            request_rows, source.catalog, names, attribute_vocabularies, candidate_ids.shape[1])
        batches[split].append(FoodBatch(foundation_collate(samples), candidate_ids, request_rows,
                                        attributes))
        groups.clear()

    requests_by_split = dict.fromkeys(SPLITS, 0)
    in_vocab_by_split = dict.fromkeys(SPLITS, 0)
    users, users_in_vocab = set(), set()
    requests = sorted(zip(source.records, all_rows), key=lambda pair: pair[0]["sample_id"])
    for record, rows in requests:
        split = rows[0]["split"]
        sample = validate_sample(source.to_sample(record))
        encoded = encode_sample(sample, vocabulary, tokenizer, dataset)
        known = encoded.user_id_value_id is not None
        requests_by_split[split] += 1
        in_vocab_by_split[split] += int(known)
        users.add(rows[0]["user_id"])
        if known:
            users_in_vocab.add(rows[0]["user_id"])
        pending[split].append((encoded, rows))
        if len(pending[split]) == batch_size:
            emit(split)
    for split in batches:
        emit(split)
    by_split = {split: [row for row in flat_rows if row["split"] == split] for split in SPLITS}
    effective_metadata = {**effective_metadata, "food_samples": {
        "mapping_version": source.mapping_version,
        **source.provenance,
        "requests_by_split": requests_by_split,
        "requests_with_user_in_vocab_by_split": in_vocab_by_split,
        "users_total": len(users),
        "users_in_vocab": len(users_in_vocab),
        "unseen_candidate_rate_by_split": {
            split: _unseen_rate(rows, candidate_vocabulary, False) for split, rows in by_split.items()},
        "unseen_positive_rate_by_split": {
            split: _unseen_rate(rows, candidate_vocabulary, True) for split, rows in by_split.items()},
        "caveats": list(source.caveats),
    }}
    manifest = {
        "dataset_version": source.dataset_version, "candidate_policy": source.policy,
        "dataset_namespace": namespace, "timezone": "UTC", **source.provenance,
    }
    return FoodInputs(batches, flat_rows, manifest, source.data_sha256,
                      fingerprint(effective_metadata), len(candidate_vocabulary) + 1,
                      model_config.d_model, candidate_vocabulary, preprocess_metadata=effective_metadata,
                      candidate_attributes=names, attribute_vocabularies=attribute_vocabularies)
