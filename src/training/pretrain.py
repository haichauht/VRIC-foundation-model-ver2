"""Foundation model pretraining script supporting local and GCP (Compute Engine, Vertex AI) training."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import random

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import time
from collections.abc import Iterator
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim import AdamW

from src.foundation_input.dataset import (
    DatasetConfig,
    FoundationPretrainDataset,
    foundation_collate,
)
from src.foundation_input.loader import iter_normalized_samples
from src.foundation_input.manifest import DatasetManifest, load_manifest
from src.foundation_input.tokenizer import (
    DEFAULT_CONTEXT_ALLOWLIST,
    TokenizerConfig,
    semantic_tokens,
)
from src.foundation_input.types import SPECIAL_TOKEN_IDS, CanonicalSample, FoundationBatch
from src.foundation_input.vocab import (
    VocabularyArtifacts,
    VocabularyConfig,
    fit_vocabulary_from_samples,
)
from src.models.encoder import FoundationModel, FoundationModelConfig
from src.models.objectives import PretrainObjectives, PretrainObjectivesConfig

from .checkpoint import save_checkpoint
from .gcp import (
    download_from_gcs,
    get_vertex_ai_environment,
    is_gcs_uri,
    resolve_device,
    upload_to_gcs,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("pretrain")


def tokenizer_config_for_manifest(manifest: DatasetManifest) -> TokenizerConfig:
    """Tokenizer config whose context allowlist is extended by the package's own allowlist."""
    package = manifest.config.get("context_allowlist") or {}
    if not package:
        return TokenizerConfig()
    allowlist = {**DEFAULT_CONTEXT_ALLOWLIST}
    allowlist.update({str(k): tuple(v) for k, v in package.items()})
    return TokenizerConfig(context_allowlist=allowlist)


def check_vocab_allowlist(
    vocabularies: VocabularyArtifacts, tokenizer_config: TokenizerConfig
) -> None:
    """Reject a reused vocabulary fitted without every context key the package allows."""
    fitted = (vocabularies.tokenizer_contract or {}).get("context_allowlist", {})
    problems = []
    for scenario, keys in sorted(tokenizer_config.context_allowlist.items()):
        missing = sorted(set(keys) - set(fitted.get(scenario, ())))
        if missing:
            problems.append(f"{scenario}: {missing}")
    if problems:
        raise ValueError(
            "Vocabulary was fitted without context keys the package allows ("
            + "; ".join(problems)
            + "). Refit the vocabulary against this package."
        )


def user_id_key_id(vocabularies: Any, tokenizer_config: TokenizerConfig) -> int:
    """Key id of the ``user_id`` token exactly as ``semantic_tokens`` emits it."""

    probe = CanonicalSample(
        sample_id="probe",
        dataset_name=vocabularies.dataset_name,
        split="train",
        user_id="probe",
        cutoff_time=datetime(1970, 1, 1, tzinfo=timezone.utc),
        cutoff_inclusive=True,
        history_order_known=True,
        history_truncated=False,
        history_group_truncated=False,
        profile={},
        events=(),
    )
    token = semantic_tokens(probe, tokenizer_config)[0]
    key_id = vocabularies.encode_key(token)
    if key_id == SPECIAL_TOKEN_IDS["UNK"]:
        raise ValueError("user_id key is missing from the vocabulary; cannot protect it from masking")
    return key_id


def pretrain_preprocess_metadata(
    manifest: DatasetManifest,
    vocabularies: VocabularyArtifacts,
    *,
    max_length: int,
    max_event_count: int,
) -> dict[str, Any]:
    """Checkpoint preprocess metadata including package provenance and sequence budget."""
    manifest_bytes = (Path(manifest.root) / "manifest.json").read_bytes()
    return {
        "dataset_name": vocabularies.dataset_name,
        "source_schema_version": vocabularies.source_schema_version,
        "source_fingerprint": vocabularies.source_fingerprint,
        "source_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "sequence": {"max_tokens": max_length, "max_event_count": max_event_count},
        "dataset_provenance": {
            k: manifest.data[k]
            for k in ("dataset_version", "generator_version", "sim_config_sha256")
            if k in manifest.data
        },
    }


def optimizer_param_groups(
    model: FoundationModel,
    objectives: PretrainObjectives,
    *,
    learning_rate: float,
    weight_decay: float,
    user_adapter_lr_multiplier: float = 1.0,
) -> list[dict[str, Any]]:
    """AdamW groups: the usual decay / no-decay split, with ``user_adapter`` at lr × multiplier.

    Each user embedding row is updated only when its user is in the batch (about ten times per
    epoch), so it may need a larger step than the shared weights. A scheduler that scales group
    learning rates (e.g. ``LambdaLR``) keeps the ratio.
    """
    multiplier = user_adapter_lr_multiplier
    if (isinstance(multiplier, bool) or not isinstance(multiplier, (int, float))
            or not math.isfinite(multiplier) or multiplier <= 0):
        raise ValueError(
            f"training.user_adapter_lr_multiplier must be a positive finite number, got {multiplier!r}"
        )
    user_ids = {id(parameter) for parameter in model.user_adapter.parameters()}
    no_decay = ("bias", "LayerNorm.weight")
    buckets: dict[tuple[bool, bool], list[torch.nn.Parameter]] = {}
    for name, parameter in [*model.named_parameters(), *objectives.named_parameters()]:
        key = (id(parameter) in user_ids, any(part in name for part in no_decay))
        buckets.setdefault(key, []).append(parameter)
    groups = []
    for (is_user, is_no_decay), group_name in (
        ((False, False), "decay"),
        ((False, True), "no_decay"),
        ((True, False), "user_adapter"),
        ((True, True), "user_adapter_no_decay"),
    ):
        if buckets.get((is_user, is_no_decay)):
            groups.append({
                "name": group_name,
                "params": buckets[(is_user, is_no_decay)],
                "weight_decay": 0.0 if is_no_decay else weight_decay,
                "lr": learning_rate * (multiplier if is_user else 1.0),
            })
    return groups


def set_seed(seed: int) -> None:
    """Set seeds across standard libraries for deterministic training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Parse YAML config without mandatory external pyyaml dependency if absent."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        import yaml  # type: ignore[import-untyped]

        with config_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        # Fallback simple line parser for key-value / basic JSON
        logger.warning("PyYAML not found, attempting JSON/basic YAML parse...")
        try:
            return json.loads(config_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.debug("Failed to parse config as JSON: %s", exc)

        # Basic key-value parser fallback
        data: dict[str, Any] = {}
        curr_section: dict[str, Any] | None = None
        for line in config_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.endswith(":") and ":" not in line[:-1]:
                section_name = line[:-1].strip()
                data[section_name] = {}
                curr_section = data[section_name]
            elif ":" in line:
                k, v = line.split(":", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                # Type inference
                val: Any = v
                if v.lower() == "true":
                    val = True
                elif v.lower() == "false":
                    val = False
                elif v.lower() == "null" or v == "":
                    val = None
                else:
                    try:
                        val = int(v)
                    except ValueError:
                        try:
                            val = float(v)
                        except ValueError:
                            pass
                if curr_section is not None and not line.startswith(" "):
                    curr_section[k] = val
                else:
                    data[k] = val
        return data


def batch_generator(
    dataset: FoundationPretrainDataset,
    batch_size: int,
    max_samples: int | None = None,
) -> Iterator[FoundationBatch]:
    """Collate streamed EncodedSample records into FoundationBatch tensors."""
    buffer = []
    for count, sample in enumerate(dataset, start=1):
        buffer.append(sample)
        if len(buffer) == batch_size:
            yield foundation_collate(buffer)
            buffer = []
        if max_samples is not None and count >= max_samples:
            break
    if buffer:
        yield foundation_collate(buffer)


def evaluate(
    model: FoundationModel,
    objectives: PretrainObjectives,
    val_dataset: FoundationPretrainDataset,
    device: torch.device,
    batch_size: int,
    max_samples: int | None = None,
) -> dict[str, float]:
    """Run validation loop and calculate loss and metrics."""
    model.eval()
    objectives.eval()

    total_loss = 0.0
    total_cat_loss = 0.0
    total_num_loss = 0.0
    total_action_loss = 0.0
    total_cat_acc = 0.0
    batch_count = 0

    with torch.no_grad():
        for batch in batch_generator(val_dataset, batch_size=batch_size, max_samples=max_samples):
            # Apply deterministic masking for evaluation
            masked_batch, targets = objectives.masking_policy.apply_mask(batch)

            # Move tensors to device
            masked_batch_dev = FoundationBatch(
                input_ids=masked_batch.input_ids.to(device),
                key_ids=masked_batch.key_ids.to(device),
                modality_ids=masked_batch.modality_ids.to(device),
                source_ids=masked_batch.source_ids.to(device),
                position_ids=masked_batch.position_ids.to(device),
                time_gap_ids=masked_batch.time_gap_ids.to(device),
                attention_mask=masked_batch.attention_mask.to(device),
                numeric_values=masked_batch.numeric_values.to(device),
                numeric_missing_mask=masked_batch.numeric_missing_mask.to(device),
                user_ids=masked_batch.user_ids.to(device),
                targets={k: v.to(device) for k, v in masked_batch.targets.items()},
            )
            targets_dev = {k: v.to(device) for k, v in targets.items()}

            output = model(masked_batch_dev, attention_mode="bidirectional")
            loss_dict = objectives(output, targets_dev, batch_targets=masked_batch_dev.targets)

            total_loss += loss_dict["loss"].item()
            total_cat_loss += loss_dict["masked_categorical_loss"].item()
            total_num_loss += loss_dict["masked_numeric_loss"].item()
            total_action_loss += loss_dict["next_action_loss"].item()
            total_cat_acc += loss_dict["masked_categorical_accuracy"].item()
            batch_count += 1

    if batch_count == 0:
        return {"loss": 0.0, "batches": 0}

    return {
        "loss": total_loss / batch_count,
        "masked_loss": (total_cat_loss + total_num_loss) / batch_count,
        "next_token_loss": total_action_loss / batch_count,
        "masked_categorical_loss": total_cat_loss / batch_count,
        "masked_numeric_loss": total_num_loss / batch_count,
        "next_action_loss": total_action_loss / batch_count,
        "masked_categorical_accuracy": total_cat_acc / batch_count,
        "batches": batch_count,
    }


def resolve_targets(config: dict[str, Any], overrides: dict[str, Any]) -> list[str]:
    """Resolve dataset targets; the config's data_dir applies only when no dataset is requested."""
    explicit_data_dir = overrides.get("data_dir")
    requested_dataset = overrides.get("dataset") or config.get("dataset")
    if not explicit_data_dir and not requested_dataset:
        explicit_data_dir = config.get("data_dir")
    return discover_target_datasets(requested_dataset, explicit_data_dir=explicit_data_dir)


def check_for_git_lfs_pointers(data_dir: Path) -> list[str]:
    """Check if any parquet files in dataset directory are Git LFS text pointers."""
    lfs_files: list[str] = []
    if not data_dir.is_dir():
        return lfs_files
    for p in data_dir.glob("**/*.parquet"):
        try:
            with p.open("rb") as f:
                header = f.read(30)
                if b"version https://git-lfs" in header:
                    lfs_files.append(str(p.name))
                    if len(lfs_files) >= 5:
                        break
        except OSError:
            pass
    return lfs_files


def discover_target_datasets(
    requested: str | None,
    explicit_data_dir: str | None = None,
    default_root: str = "data/pretrain_data",
) -> list[str]:
    """Resolve list of target dataset paths or URIs."""
    if explicit_data_dir:
        return [explicit_data_dir]

    if not requested:
        target = f"{default_root}/meituan_data"
        if not Path(target).is_dir() and Path("data/gcs_cache/meituan_data").is_dir():
            return ["data/gcs_cache/meituan_data"]
        return [target]

    req = requested.strip()
    if is_gcs_uri(req):
        return [req]

    if req.lower() in ("all", "both"):
        base = Path(default_root)
        candidates = []
        if base.is_dir():
            for child in sorted(base.iterdir()):
                if child.is_dir() and (child / "manifest.json").is_file():
                    candidates.append(str(child))
        if candidates:
            return candidates
        cache_base = Path("data/gcs_cache")
        if cache_base.is_dir():
            cache_candidates = [
                str(child) for child in sorted(cache_base.iterdir())
                if child.is_dir() and (child / "manifest.json").is_file()
            ]
            if cache_candidates:
                return cache_candidates
        return [f"{default_root}/meituan_data", f"{default_root}/OmniBehavior_data"]

    if req.lower() in ("meituan", "meituan_data"):
        for candidate in (
            f"{default_root}/meituan_data",
            "data/gcs_cache/meituan_data",
            "data/meituan_data",
        ):
            if Path(candidate).is_dir():
                return [candidate]
        return [f"{default_root}/meituan_data"]

    if req.lower() in ("omnibehavior", "omnibehavior_data"):
        for candidate in (
            f"{default_root}/OmniBehavior_data",
            "data/gcs_cache/OmniBehavior_data",
            "data/OmniBehavior_data",
        ):
            if Path(candidate).is_dir():
                return [candidate]
        return [f"{default_root}/OmniBehavior_data"]

    p = Path(req)
    if p.is_dir() and not (p / "manifest.json").is_file():
        sub_datasets = [
            str(child)
            for child in sorted(p.iterdir())
            if child.is_dir() and (child / "manifest.json").is_file()
        ]
        if sub_datasets:
            return sub_datasets

    return [req]


def pretrain_single_dataset(
    config: dict[str, Any],
    raw_data_dir: str,
    local_output_dir: Path,
    gcs_target_output: str | None,
    overrides: dict[str, Any],
    device: torch.device,
    hardware_info: dict[str, Any],
    seed: int,
    is_multi: bool = False,
) -> dict[str, Any]:
    """Execute pretraining on a single dataset directory and return experiment record."""
    dataset_key = Path(raw_data_dir.rstrip("/")).name
    logger.info("==================================================================")
    logger.info(">>> Launching Pretraining for Dataset: %s", dataset_key)
    logger.info("==================================================================")

    # Download data from GCS if specified as gs:// URI
    if is_gcs_uri(raw_data_dir):
        local_data_dir = Path("data/gcs_cache") / Path(raw_data_dir[len("gs://") :]).name
        download_from_gcs(raw_data_dir, local_data_dir)
    else:
        local_data_dir = Path(raw_data_dir)

    # Check for Git LFS text pointers
    lfs_files = check_for_git_lfs_pointers(local_data_dir)
    if lfs_files:
        err_msg = (
            f"Dataset '{dataset_key}' at '{local_data_dir}' contains Git LFS pointer files "
            f"({', '.join(lfs_files)}...). Please run 'git lfs pull' or install git-lfs to download actual Parquet data files."
        )
        logger.error(err_msg)
        raise RuntimeError(err_msg)

    local_output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = local_output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # Training parameters
    training_cfg = config.get("training", {})
    batch_size = overrides.get("batch_size") or training_cfg.get("batch_size", 32)
    epochs = overrides.get("epochs") or training_cfg.get("epochs", 10)
    learning_rate = overrides.get("learning_rate") or float(training_cfg.get("learning_rate", 1e-4))
    weight_decay = float(training_cfg.get("weight_decay", 0.01))
    user_adapter_lr_multiplier = float(training_cfg.get("user_adapter_lr_multiplier", 1.0))
    grad_clip_norm = float(training_cfg.get("grad_clip_norm", 1.0))
    use_amp = bool(training_cfg.get("mixed_precision", device.type == "cuda"))

    # Sequence / Dataset parameters
    seq_cfg = config.get("sequence", {})
    max_length = int(seq_cfg.get("max_length", 256))
    max_event_count = int(seq_cfg.get("max_event_count", 30))
    next_action_prediction = bool(config.get("objectives", {}).get("next_action_prediction", True))

    dataset_config = DatasetConfig(
        max_tokens=max_length,
        max_event_count=max_event_count,
        next_action_prediction=next_action_prediction,
    )

    # Smoke test mode
    is_smoke = overrides.get("smoke") or config.get("smoke", False)
    max_train_samples = 32 if is_smoke else config.get("max_train_samples")
    max_val_samples = 16 if is_smoke else config.get("max_val_samples")
    if is_smoke:
        epochs = min(epochs, 2)
        logger.info("[%s] Running in SMOKE mode: epochs=%d, max_samples=%s", dataset_key, epochs, max_train_samples)

    # Inspect manifest and determine validation strategy
    manifest = load_manifest(local_data_dir, validate_shard_files=False)
    train_snapshot_count = sum(s.get("snapshots_by_split", {}).get("train", 0) for s in manifest.shards)
    val_snapshot_count = sum(s.get("snapshots_by_split", {}).get("validation", 0) for s in manifest.shards)
    test_snapshot_count = sum(s.get("snapshots_by_split", {}).get("test", 0) for s in manifest.shards)
    has_native_val = val_snapshot_count > 0

    if has_native_val:
        val_split_strategy = "native"
        train_filter = None
        val_filter = None
        train_split_name = "train"
        val_split_name = "validation"
        logger.info(
            "[%s] Dataset manifest has NATIVE 'validation' split (train: %d, val: %d, test: %d snapshots)",
            dataset_key,
            train_snapshot_count,
            val_snapshot_count,
            test_snapshot_count,
        )
    else:
        val_split_strategy = "fallback_from_train (deterministic 10% held-out split)"
        logger.info(
            "[%s] Dataset manifest has NO 'validation' split (%d train snapshots total). "
            "Automatically partitioning 10%% deterministic held-out split from 'train' for validation.",
            dataset_key,
            train_snapshot_count,
        )

        def _is_val_sample(s: Any) -> bool:
            token = f"{s.user_id}:{s.sample_id}"
            return int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % 10 == 0

        train_filter = lambda s: not _is_val_sample(s)
        val_filter = _is_val_sample
        train_split_name = "train"
        val_split_name = "train"

    tokenizer_config = tokenizer_config_for_manifest(manifest)

    # Load or fit Vocabulary
    vocab_path = overrides.get("vocab_path") or config.get("vocab_path")
    if vocab_path and Path(vocab_path).is_file():
        logger.info("[%s] Loading existing vocabulary from %s", dataset_key, vocab_path)
        vocabularies = VocabularyArtifacts.load(vocab_path)
        check_vocab_allowlist(vocabularies, tokenizer_config)
    else:
        logger.info("[%s] Fitting vocabulary from training split in %s...", dataset_key, local_data_dir)
        raw_train_samples = iter_normalized_samples(local_data_dir, split=train_split_name, validate=False)
        train_samples_stream = (s for s in raw_train_samples if train_filter(s)) if train_filter else raw_train_samples
        if is_smoke:
            import itertools
            train_samples_stream = itertools.islice(train_samples_stream, max(max_train_samples * 4, 128))
        vocab_config = VocabularyConfig(dataset_name=manifest.dataset_name)
        vocabularies = fit_vocabulary_from_samples(
            train_samples_stream, vocab_config, tokenizer_config
        )
        vocab_save_path = local_output_dir / "vocab.json"
        vocabularies.save(vocab_save_path)
        logger.info("[%s] Fitted vocabulary saved to %s", dataset_key, vocab_save_path)

    cardinalities = vocabularies.model_cardinalities()
    logger.info("[%s] Vocabulary cardinalities: %s", dataset_key, cardinalities)

    preprocess_metadata = pretrain_preprocess_metadata(
        manifest, vocabularies, max_length=max_length, max_event_count=max_event_count
    )

    # Construct Model Config
    model_cfg_dict = config.get("model", {})
    source_vocab = vocabularies.source_to_id
    modality_vocab = vocabularies.modality_to_id
    seq_source_id = source_vocab.get("sequence", 1)
    num_modality_id = modality_vocab.get("numeric", 2)

    model_config = FoundationModelConfig(
        d_model=int(model_cfg_dict.get("d_model", 128)),
        num_heads=int(model_cfg_dict.get("num_heads", 4)),
        num_layers=int(model_cfg_dict.get("num_layers", 4)),
        dropout=float(model_cfg_dict.get("dropout", 0.1)),
        max_length=max_length,
        value_vocab_size=cardinalities["value_vocab_size"],
        key_vocab_size=cardinalities["key_vocab_size"],
        modality_vocab_size=cardinalities["modality_vocab_size"],
        source_vocab_size=cardinalities["source_vocab_size"],
        time_gap_vocab_size=cardinalities["time_gap_vocab_size"],
        num_user_buckets=int(model_cfg_dict.get("num_user_buckets", 50_000)),
        categorical_pair_buckets=int(model_cfg_dict.get("categorical_pair_buckets", 100_003)),
        sequence_source_id=seq_source_id,
        numeric_modality_id=num_modality_id,
        default_attention_mode=model_cfg_dict.get("default_attention_mode", "bidirectional"),
        long_term_injection=str(model_cfg_dict.get("long_term_injection", "none")),
        user_embedding_init_std=(
            None if model_cfg_dict.get("user_embedding_init_std") is None
            else float(model_cfg_dict["user_embedding_init_std"])),
    )
    unmaskable_key_ids = (
        (user_id_key_id(vocabularies, tokenizer_config),)
        if model_config.long_term_injection == "cls"
        else ()
    )

    # Instantiate Model & Objectives
    model = FoundationModel(model_config).to(device)

    obj_cfg_dict = config.get("objectives", {})
    objectives_config = PretrainObjectivesConfig(
        mask_prob=float(obj_cfg_dict.get("mask_prob", 0.15)),
        mask_token_prob=float(obj_cfg_dict.get("mask_token_prob", 0.8)),
        random_token_prob=float(obj_cfg_dict.get("random_token_prob", 0.1)),
        keep_token_prob=float(obj_cfg_dict.get("keep_token_prob", 0.1)),
        masked_categorical_weight=float(obj_cfg_dict.get("masked_categorical_weight", 1.0)),
        masked_numeric_weight=float(obj_cfg_dict.get("masked_numeric_weight", 1.0)),
        next_action_weight=float(obj_cfg_dict.get("next_action_weight", 0.5)),
        unmaskable_key_ids=unmaskable_key_ids,
    )
    objectives = PretrainObjectives(model_config, objectives_config).to(device)

    # Optimizer & Scheduler
    all_params = list(model.parameters()) + list(objectives.parameters())
    optimizer_grouped_parameters = optimizer_param_groups(
        model,
        objectives,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        user_adapter_lr_multiplier=user_adapter_lr_multiplier,
    )
    logger.info(
        "[%s] user_adapter: lr multiplier %s, init std %s",
        dataset_key,
        user_adapter_lr_multiplier,
        model_config.user_embedding_init_std,
    )
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and device.type == "cuda"))

    # Datasets
    train_dataset = FoundationPretrainDataset(
        dataset_root=str(local_data_dir),
        split=train_split_name,
        artifacts=vocabularies,
        dataset_config=dataset_config,
        sample_filter=train_filter,
    )
    val_dataset = FoundationPretrainDataset(
        dataset_root=str(local_data_dir),
        split=val_split_name,
        artifacts=vocabularies,
        dataset_config=dataset_config,
        sample_filter=val_filter,
    )

    # Logging setup
    metrics_log_path = local_output_dir / "metrics.jsonl"
    experiment_record_path = local_output_dir / "experiment_record.json"

    best_val_loss = float("inf")
    best_checkpoint_sha = ""
    latest_checkpoint_sha = ""
    global_step = 0
    training_history = []

    logger.info("[%s] Starting training loop: %d epochs...", dataset_key, epochs)
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        objectives.train()

        epoch_train_loss = 0.0
        epoch_cat_loss = 0.0
        epoch_num_loss = 0.0
        epoch_action_loss = 0.0
        epoch_batches = 0

        for batch in batch_generator(train_dataset, batch_size=batch_size, max_samples=max_train_samples):
            global_step += 1
            optimizer.zero_grad()

            masked_batch, targets = objectives.masking_policy.apply_mask(batch)

            masked_batch_dev = FoundationBatch(
                input_ids=masked_batch.input_ids.to(device),
                key_ids=masked_batch.key_ids.to(device),
                modality_ids=masked_batch.modality_ids.to(device),
                source_ids=masked_batch.source_ids.to(device),
                position_ids=masked_batch.position_ids.to(device),
                time_gap_ids=masked_batch.time_gap_ids.to(device),
                attention_mask=masked_batch.attention_mask.to(device),
                numeric_values=masked_batch.numeric_values.to(device),
                numeric_missing_mask=masked_batch.numeric_missing_mask.to(device),
                user_ids=masked_batch.user_ids.to(device),
                targets={k: v.to(device) for k, v in masked_batch.targets.items()},
            )
            targets_dev = {k: v.to(device) for k, v in targets.items()}

            if use_amp and device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    output = model(masked_batch_dev, attention_mode="bidirectional")
                    loss_dict = objectives(output, targets_dev, batch_targets=masked_batch_dev.targets)
                    loss = loss_dict["loss"]
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(all_params, grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(masked_batch_dev, attention_mode="bidirectional")
                loss_dict = objectives(output, targets_dev, batch_targets=masked_batch_dev.targets)
                loss = loss_dict["loss"]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(all_params, grad_clip_norm)
                optimizer.step()

            epoch_train_loss += loss.item()
            epoch_cat_loss += loss_dict["masked_categorical_loss"].item()
            epoch_num_loss += loss_dict["masked_numeric_loss"].item()
            epoch_action_loss += loss_dict["next_action_loss"].item()
            epoch_batches += 1

            if global_step % 20 == 0:
                logger.info(
                    "[%s] Epoch %d | Step %d | Train Loss: %.4f (Masked: %.4f, NextToken: %.4f)",
                    dataset_key,
                    epoch,
                    global_step,
                    loss.item(),
                    loss_dict["masked_loss"].item(),
                    loss_dict["next_token_loss"].item(),
                )

        avg_train_loss = epoch_train_loss / max(epoch_batches, 1)

        # Validation
        val_metrics = evaluate(
            model,
            objectives,
            val_dataset,
            device,
            batch_size=batch_size,
            max_samples=max_val_samples,
        )

        epoch_duration = time.time() - epoch_start
        logger.info(
            "=== [%s] Epoch %d Completed (%.2fs) | Train Loss: %.4f | Val Loss: %.4f (Masked: %.4f, NextToken: %.4f, batches=%d) ===",
            dataset_key,
            epoch,
            epoch_duration,
            avg_train_loss,
            val_metrics["loss"],
            val_metrics.get("masked_loss", 0.0),
            val_metrics.get("next_token_loss", 0.0),
            val_metrics.get("batches", 0),
        )

        epoch_record = {
            "dataset_key": dataset_key,
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": avg_train_loss,
            "val_loss": val_metrics["loss"],
            "val_metrics": val_metrics,
            "epoch_duration_seconds": epoch_duration,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        training_history.append(epoch_record)

        # Append to JSONL metrics log
        with metrics_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(epoch_record) + "\n")

        # Save latest checkpoint
        latest_ckpt_path = checkpoints_dir / "latest_checkpoint.pt"
        latest_checkpoint_sha = save_checkpoint(
            latest_ckpt_path,
            model,
            model_config,
            vocabularies,
            optimizer=optimizer,
            epoch=epoch,
            step=global_step,
            metrics=val_metrics,
            preprocess_metadata=preprocess_metadata,
        )

        # Save best checkpoint
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_ckpt_path = checkpoints_dir / "best_checkpoint.pt"
            best_checkpoint_sha = save_checkpoint(
                best_ckpt_path,
                model,
                model_config,
                vocabularies,
                optimizer=optimizer,
                epoch=epoch,
                step=global_step,
                metrics=val_metrics,
                preprocess_metadata=preprocess_metadata,
            )
            logger.info("[%s] Saved new BEST checkpoint (loss=%.4f, sha256=%s)", dataset_key, best_val_loss, best_checkpoint_sha)

    total_training_time = time.time() - start_time

    # Construct final experiment record
    experiment_record = {
        "dataset_key": dataset_key,
        "dataset_name": manifest.dataset_name,
        "dataset_version": config.get("dataset_version", "v1"),
        "dataset_source_path": str(local_data_dir),
        "dataset_split_info": {
            "validation_strategy": val_split_strategy,
            "manifest_train_snapshots": train_snapshot_count,
            "manifest_validation_snapshots": val_snapshot_count,
            "manifest_test_snapshots": test_snapshot_count,
        },
        "seed": seed,
        "hardware_info": hardware_info,
        "total_training_time_seconds": total_training_time,
        "epochs_completed": epochs,
        "total_steps": global_step,
        "best_val_loss": best_val_loss,
        "best_checkpoint_sha256": best_checkpoint_sha,
        "latest_checkpoint_sha256": latest_checkpoint_sha,
        "model_config": asdict(model_config),
        "objectives_config": asdict(objectives_config),
        "user_adapter_lr_multiplier": user_adapter_lr_multiplier,
        "training_history": training_history,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    experiment_record_path.write_text(json.dumps(experiment_record, indent=2) + "\n", encoding="utf-8")
    logger.info("[%s] Experiment record saved to %s", dataset_key, experiment_record_path)

    # Sync to GCS if requested
    if gcs_target_output:
        logger.info("[%s] Syncing outputs to Google Cloud Storage: %s", dataset_key, gcs_target_output)
        upload_to_gcs(local_output_dir, gcs_target_output)

    return experiment_record


def pretrain(
    config: dict[str, Any],
    cli_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute the full pretraining pipeline across one or more datasets."""
    overrides = cli_overrides or {}

    seed = overrides.get("seed") or config.get("seed", 42)
    set_seed(seed)

    raw_output_dir = overrides.get("output_dir") or config.get("output_dir") or "experiments/pretrain"

    # Handle Vertex AI environment variables if present
    vertex_env = get_vertex_ai_environment()
    if "AIP_MODEL_DIR" in vertex_env:
        logger.info("Detected Vertex AI AIP_MODEL_DIR=%s", vertex_env["AIP_MODEL_DIR"])
        if is_gcs_uri(vertex_env["AIP_MODEL_DIR"]):
            raw_output_dir = vertex_env["AIP_MODEL_DIR"]

    gcs_target_output = raw_output_dir if is_gcs_uri(raw_output_dir) else None
    local_output_root = Path("experiments/pretrain_local") if gcs_target_output else Path(raw_output_dir)

    # Determine requested dataset targets
    targets = resolve_targets(config, overrides)
    logger.info("Discovered %d dataset target(s): %s", len(targets), targets)

    # Resolve device and accelerators (GCP GPU detection)
    requested_device = overrides.get("device") or config.get("device", "auto")
    device, hardware_info = resolve_device(requested_device)
    if device.type == "cpu":
        import os
        num_threads = os.cpu_count() or 4
        torch.set_num_threads(num_threads)
        hardware_info["cpu_threads"] = num_threads
    logger.info("Training on device=%s, hardware=%s", device, hardware_info)

    if len(targets) == 1:
        target_path = targets[0]
        return pretrain_single_dataset(
            config=config,
            raw_data_dir=target_path,
            local_output_dir=local_output_root,
            gcs_target_output=gcs_target_output,
            overrides=overrides,
            device=device,
            hardware_info=hardware_info,
            seed=seed,
            is_multi=False,
        )

    # Multiple datasets (e.g. 'all' or both datasets)
    local_output_root.mkdir(parents=True, exist_ok=True)
    summary_report: dict[str, Any] = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "total_datasets": len(targets),
        "results": {},
    }

    for target_path in targets:
        ds_key = Path(target_path.rstrip("/")).name
        ds_output = local_output_root / ds_key
        ds_gcs = f"{gcs_target_output.rstrip('/')}/{ds_key}" if gcs_target_output else None
        try:
            record = pretrain_single_dataset(
                config=config,
                raw_data_dir=target_path,
                local_output_dir=ds_output,
                gcs_target_output=ds_gcs,
                overrides=overrides,
                device=device,
                hardware_info=hardware_info,
                seed=seed,
                is_multi=True,
            )
            summary_report["results"][ds_key] = {
                "status": "success",
                "dataset_name": record["dataset_name"],
                "best_val_loss": record["best_val_loss"],
                "val_strategy": record["dataset_split_info"]["validation_strategy"],
                "best_checkpoint_sha256": record["best_checkpoint_sha256"],
                "completed_at": record["completed_at"],
            }
        except RuntimeError as exc:
            logger.error("[%s] Pretraining failed or skipped: %s", ds_key, exc)
            summary_report["results"][ds_key] = {
                "status": "error",
                "error": str(exc),
            }

    summary_file = local_output_root / "summary_comparison.json"
    summary_file.write_text(json.dumps(summary_report, indent=2) + "\n", encoding="utf-8")
    logger.info("Multi-dataset summary comparison saved to %s", summary_file)
    if gcs_target_output:
        upload_to_gcs(summary_file, f"{gcs_target_output.rstrip('/')}/summary_comparison.json")

    return summary_report


def parse_args() -> tuple[dict[str, Any], dict[str, Any]]:
    """Parse CLI arguments and load configuration."""
    parser = argparse.ArgumentParser(
        description="Foundation model multi-task pretraining on local or GCP."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pretrain.yaml",
        help="Path to YAML/JSON configuration file.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Target dataset name or mode: 'meituan', 'omnibehavior', or 'all'.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Pretraining dataset root path or GCS URI (gs://...).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save checkpoints, logs, and metadata (or gs://...).",
    )
    parser.add_argument(
        "--vocab_path",
        type=str,
        default=None,
        help="Optional pre-fitted vocabulary JSON path.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size per training step.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Peak learning rate.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Execution device ('cuda', 'mps', 'cpu', or 'auto').",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run quick smoke test (2 epochs, minimal sample count).",
    )

    args = parser.parse_args()
    config = load_yaml_config(args.config) if Path(args.config).is_file() else {}

    overrides = {
        k: v
        for k, v in vars(args).items()
        if v is not None and k != "config"
    }
    if args.smoke:
        overrides["smoke"] = True

    return config, overrides


def main() -> None:
    config, overrides = parse_args()
    pretrain(config, overrides)


if __name__ == "__main__":
    main()

