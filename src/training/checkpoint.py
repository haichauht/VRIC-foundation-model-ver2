"""Portable checkpoint serialization matching the team foundation contract."""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from src.foundation_input.vocab import VocabularyArtifacts
from src.models.encoder import FoundationModel, FoundationModelConfig

CHECKPOINT_FORMAT_VERSION = 1


def compute_file_sha256(path: str | Path) -> str:
    """Compute sha256 checksum of a file."""
    path = Path(path)
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def save_checkpoint(
    path: str | Path,
    model: FoundationModel,
    config: FoundationModelConfig,
    vocabularies: VocabularyArtifacts,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    lr_scheduler: Any | None = None,
    epoch: int = 0,
    step: int = 0,
    metrics: dict[str, Any] | None = None,
    preprocess_metadata: dict[str, Any] | None = None,
) -> str:
    """Save a portable checkpoint complying with the project contract.

    Args:
        path: Target filepath to save checkpoint to.
        model: FoundationModel instance.
        config: Model configuration.
        vocabularies: VocabularyArtifacts used during preprocessing/tokenization.
        optimizer: Optional optimizer for training resumption.
        lr_scheduler: Optional learning rate scheduler.
        epoch: Current training epoch.
        step: Current training step.
        metrics: Optional dictionary of evaluation metrics.
        preprocess_metadata: Optional data pipeline metadata.

    Returns:
        SHA-256 hash string of the written checkpoint file.
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    metadata = preprocess_metadata or {
        "dataset_name": vocabularies.dataset_name,
        "source_schema_version": vocabularies.source_schema_version,
        "source_fingerprint": vocabularies.source_fingerprint,
    }

    # Extract model state dict onto CPU to prevent device binding issues
    cpu_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}

    payload: dict[str, Any] = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "epoch": epoch,
        "step": step,
        "model_config": asdict(config),
        "vocabularies": vocabularies.to_metadata(),
        "preprocess_metadata": metadata,
        "model_state_dict": cpu_state_dict,
        "metrics": metrics or {},
    }

    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if lr_scheduler is not None and hasattr(lr_scheduler, "state_dict"):
        payload["lr_scheduler_state_dict"] = lr_scheduler.state_dict()

    # Save to a temporary file first then replace to guarantee atomic write
    tmp_path = dest.with_suffix(dest.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(dest)

    return compute_file_sha256(dest)


def load_checkpoint(
    path: str | Path,
    device: torch.device | str = "cpu",
    load_model: bool = True,
) -> tuple[FoundationModel | None, FoundationModelConfig, VocabularyArtifacts, dict[str, Any]]:
    """Load a checkpoint and reconstruct FoundationModel and VocabularyArtifacts.

    Args:
        path: Filepath of the checkpoint.
        device: Device to place the model on ('cpu', 'cuda', etc.).
        load_model: If True, instantiates and loads weights into FoundationModel.

    Returns:
        tuple of (model, model_config, vocab_artifacts, raw_payload)
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    payload = torch.load(path, map_location=device, weights_only=False)

    version = payload.get("format_version")
    if version != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint format_version: {version!r}; "
            f"expected {CHECKPOINT_FORMAT_VERSION}"
        )

    for field in ("model_config", "vocabularies", "model_state_dict"):
        if field not in payload:
            raise KeyError(f"Checkpoint is missing required field: {field!r}")

    model_config = FoundationModelConfig(**payload["model_config"])

    # Reconstruct VocabularyArtifacts from metadata dictionary
    vocab_meta = payload["vocabularies"]
    vocabularies = VocabularyArtifacts(
        dataset_name=vocab_meta["dataset_name"],
        value_to_id={
            str(k): {str(val): int(vid) for val, vid in mapping.items()}
            for k, mapping in vocab_meta["value_to_id"].items()
        },
        numeric_boundaries={
            str(k): tuple(float(x) for x in bounds)
            for k, bounds in vocab_meta["numeric_boundaries"].items()
        },
        key_to_id={str(k): int(v) for k, v in vocab_meta["key_to_id"].items()},
        modality_to_id={str(k): int(v) for k, v in vocab_meta["modality_to_id"].items()},
        source_to_id={str(k): int(v) for k, v in vocab_meta["source_to_id"].items()},
        text_hash_buckets=int(vocab_meta["text_hash_buckets"]),
        fit_split=str(vocab_meta.get("fit_split", "train")),
        source_schema_version=str(vocab_meta.get("source_schema_version", "omnibehavior_pretraining_v2")),
        source_fingerprint=vocab_meta.get("source_fingerprint"),
        include_action_content=bool(vocab_meta.get("include_action_content", False)),
        numeric_normalization={
            str(k): (float(center), float(scale))
            for k, (center, scale) in vocab_meta.get("numeric_normalization", {}).items()
        },
        numeric_field_registry=tuple(
            (str(src), str(field_name))
            for src, field_name in vocab_meta.get("numeric_field_registry", [])
        ),
        numeric_transform_policy={
            str(k): str(p) for k, p in vocab_meta.get("numeric_transform_policy", {}).items()
        },
        time_gap_boundaries_seconds=tuple(
            float(x) for x in vocab_meta.get("time_gap_boundaries_seconds", [])
        ),
        tokenizer_contract=dict(vocab_meta.get("tokenizer_contract", {})),
    )

    model = None
    if load_model:
        model = FoundationModel(model_config)
        model.load_state_dict(payload["model_state_dict"])
        model.to(device)

    return model, model_config, vocabularies, payload
