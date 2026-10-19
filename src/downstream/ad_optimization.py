"""Person 4: supervised CTR on Grab-style long/short user embeddings.

Reference: https://engineering.grab.com/user-foundation-models-for-grab
Run: python -m src.downstream.ad_optimization --config configs/ad_optimization.yaml

The foundation is accessed exclusively through ``embed_users``. Training never
changes the exposure set. No evaluator or prediction-file writer lives here.

Integration boundary (Person 3 / Data / Baseline teams):
* load_checkpoint(path, expected_version=...) -> an nn.Module with embed_users;
* load_inputs(config, foundation) -> AdInputs. The handoff adapter validates
  checksums and uses Person 1's tokenizer/collator with checkpoint metadata,
  never refitting. ``build_ad_inputs`` provides the Ads bridge and train-only
  candidate vocabulary (0 = OOV); the Data/input owner supplies encoded contexts.
* validate_and_write_predictions(table, *, expected_candidates, run_manifest,
  train_metrics, checkpoint_ref, artifact_dir) -> Path. This binding MUST call
  src.contracts.output's validator/atomic writer and emit the v1 run artifacts.
* evaluate_predictions uses the public Baseline API in the implementation spec.

The Ads adapter binds Person 3's checkpoint implementation and shared writer
to the public Baseline evaluator. Evaluation requires an explicit policy for
label availability; export-only runs can leave evaluation disabled.

``data.format`` selects the handoff: ``taobao`` (default, unchanged) or ``sim_v1``
(simulator ``ads/`` package; split bounds are exclusive and the checkpoint is bound
by ``sim_config_sha256``). ``foundation.init: random`` is the control arm: the
checkpoint's config and vocabulary with fresh weights seeded by ``seed``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import torch
from torch import nn
from torch.nn import functional as F

from src.downstream.foundation_init import foundation_init_arm, reinitialize_foundation
from src.downstream.head_training import head_optimizer
from src.foundation_input.types import FoundationBatch

TASK = "ad_optimization"
AD_DATA_FORMATS = ("taobao", "sim_v1")
PREDICTION_SCHEMA = pa.schema([
    ("schema_version", pa.int32()), ("task", pa.string()), ("split", pa.string()),
    ("data_version", pa.string()), ("run_id", pa.string()), ("request_id", pa.string()),
    ("user_id", pa.string()), ("cutoff_timestamp", pa.int64()),
    ("candidate_id", pa.string()), ("score", pa.float64()), ("raw_logit", pa.float64()),
])


class AdRunError(ValueError):
    """A contract failure with the shared specification's process exit code."""

    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code


def ad_data_format(config):
    """The configured ``data.format`` (default ``"taobao"``), validated."""
    data = config["data"]
    if not isinstance(data, dict):
        raise ValueError("data must be a mapping")  # noqa: TRY004
    value = data.get("format", "taobao")
    if value not in AD_DATA_FORMATS:
        raise ValueError(f"data.format must be one of {list(AD_DATA_FORMATS)}, got {value!r}")
    return value


def is_sha256(value):
    return (isinstance(value, str) and len(value) == 64
            and all(char in "0123456789abcdef" for char in value))


class AdRanker(nn.Module):
    """Fuse both user representations and an ad/campaign embedding into logits."""

    def __init__(self, d_model: int, num_candidates: int, hidden_dim: int = 64,
                 candidate_dim: int | None = None, dropout: float = 0.0):
        super().__init__()
        candidate_dim = d_model if candidate_dim is None else candidate_dim
        if min(d_model, num_candidates, hidden_dim, candidate_dim) <= 0 or not 0 <= dropout < 1:
            raise ValueError("d_model, num_candidates and hidden_dim must be positive")
        self.d_model = d_model
        # Train-only vocabulary: unknown ads have a neutral, fixed embedding.
        self.candidates = nn.Embedding(num_candidates, candidate_dim, padding_idx=0)
        layers = [nn.Linear(2 * d_model + candidate_dim, hidden_dim), nn.ReLU()]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.scorer = nn.Sequential(*layers, nn.Linear(hidden_dim, 1))

    def forward(self, long_term, short_term, ad_candidate_ids):
        if ad_candidate_ids.ndim != 2 or ad_candidate_ids.dtype != torch.long:
            raise ValueError("ad_candidate_ids: expected torch.long [B,C]")
        expected = (ad_candidate_ids.shape[0], self.d_model)
        for name, value in (("long_term", long_term), ("short_term", short_term)):
            if value.shape != expected or value.dtype != torch.float32:
                raise ValueError(f"{name}: expected float32 {expected}, got {value.shape}")
            if value.device != ad_candidate_ids.device or not torch.isfinite(value).all():
                raise ValueError(f"{name}: expected finite embedding on candidate device")
        if ((ad_candidate_ids < 0) | (ad_candidate_ids >= self.candidates.num_embeddings)).any():
            raise ValueError("ad_candidate_ids outside train-fitted vocabulary")
        count = ad_candidate_ids.shape[1]
        features = torch.cat([
            long_term[:, None, :].expand(-1, count, -1),
            short_term[:, None, :].expand(-1, count, -1),
            self.candidates(ad_candidate_ids),
        ], dim=-1)
        return self.scorer(features).squeeze(-1)


def ctr_loss(logits, clicked, candidate_mask):
    """Exposure-weighted BCE, including valid requests with no clicks."""
    if logits.shape != clicked.shape or candidate_mask.shape != logits.shape:
        raise ValueError("CTR logits, clicked and mask must share [B,C] shape")
    if candidate_mask.dtype != torch.bool or not candidate_mask.any():
        raise ValueError("CTR requires a bool mask and at least one exposure")
    labels, scores = clicked[candidate_mask], logits[candidate_mask]
    if not torch.isfinite(scores).all() or not ((labels == 0) | (labels == 1)).all():
        raise ValueError("CTR requires finite logits and binary, non-missing click labels")
    return F.binary_cross_entropy_with_logits(scores, labels.to(scores.dtype))


@dataclass
class AdBatch:
    """One encoded request per foundation row; padding is excluded from CTR/output.

    ``rows[i][j]`` preserves the original canonical request/candidate strings.
    The loader must align foundation row i to that request's cutoff/context.
    IDs must use a frozen train-fitted candidate vocabulary, with 0 for OOV.
    """

    foundation: FoundationBatch
    candidate_ids: torch.Tensor
    rows: Sequence[Sequence[Mapping[str, Any]]]
    embeddings: tuple[torch.Tensor, torch.Tensor] | None = None

    def labels_and_mask(self):
        shape = self.candidate_ids.shape
        if self.candidate_ids.ndim != 2 or self.candidate_ids.dtype != torch.long:
            raise AdRunError(3, "candidate_ids expected torch.long [B,C]")
        if len(self.rows) != shape[0] or self.foundation.user_ids.shape != (shape[0],):
            raise AdRunError(3, "request rows must match foundation batch size")
        mask = torch.zeros(shape, dtype=torch.bool, device=self.candidate_ids.device)
        labels = torch.zeros(shape, dtype=torch.float32, device=mask.device)
        for index, rows in enumerate(self.rows):
            if not rows or len(rows) > shape[1]:
                raise AdRunError(3, "each request requires at least one supplied exposure")
            first = rows[0]
            for row in rows:
                if any(row[key] != first[key] for key in
                       ("request_id", "user_id", "cutoff_timestamp", "split")):
                    raise AdRunError(3, "foundation row must represent exactly one request")
            mask[index, :len(rows)] = True
            labels[index, :len(rows)] = torch.tensor(
                [row["clicked"] for row in rows], dtype=torch.float32, device=mask.device
            )
        return labels, mask


@dataclass
class AdInputs:
    """Validated handoff plus independent, complete expected opportunity rows."""

    batches: Mapping[str, Sequence[AdBatch]]
    opportunities: Sequence[Mapping[str, Any]]
    manifest: Mapping[str, Any]
    data_manifest_sha256: str
    preprocess_fingerprint: str
    num_candidates: int
    d_model: int
    candidate_vocabulary: Mapping[str, int]


def build_ad_inputs(opportunities, encoded_requests, *, manifest, data_manifest_sha256,
                    preprocess_fingerprint, d_model):
    """Bridge validated Data rows to checkpoint-encoded requests, fitting IDs on train.

    ``encoded_requests`` maps request_id to (user_id, cutoff_timestamp,
    FoundationBatch). Each batch must contain exactly one request encoded by
    Person 1 with the checkpoint's frozen metadata and validated as-of history.
    The loader must validate source file checksums before calling this bridge.
    No raw profile/event normalization or synthetic data is performed here.
    """
    rows = [dict(row) for row in opportunities]
    validate_opportunities(rows, manifest, manifest.get("dataset_version"),
                           manifest.get("candidate_policy"))
    train_candidates = sorted({row["candidate_id"] for row in rows if row["split"] == "train"})
    if not train_candidates:
        raise AdRunError(3, "train candidate vocabulary cannot be empty")
    vocabulary = {candidate: index + 1 for index, candidate in enumerate(train_candidates)}
    grouped = {}
    for row in rows:
        grouped.setdefault(row["request_id"], []).append(row)
    if set(grouped) != set(encoded_requests):
        raise AdRunError(3, "encoded request set must exactly match opportunity handoff")
    batches = {split: [] for split in ("train", "validation", "test")}
    for request in sorted(grouped):
        candidates = grouped[request]
        user, cutoff, foundation = encoded_requests[request]
        first = candidates[0]
        if user != first["user_id"] or cutoff != first["cutoff_timestamp"]:
            raise AdRunError(3, f"request_id={request}: encoded user/cutoff alignment mismatch")
        if foundation.input_ids.shape[0] != 1 or foundation.user_ids.shape != (1,):
            raise AdRunError(3, f"request_id={request}: expected one encoded request")
        ids = torch.tensor([[vocabulary.get(row["candidate_id"], 0) for row in candidates]],
                           dtype=torch.long)
        batches[first["split"]].append(AdBatch(foundation, ids, [candidates]))
    return AdInputs(batches, rows, manifest, data_manifest_sha256,
                    preprocess_fingerprint, len(vocabulary) + 1, d_model, vocabulary)


@dataclass
class AdServices:
    """Public integrations; tests may inject FakeFoundation and evaluator spies."""

    load_checkpoint: Callable
    load_inputs: Callable
    validate_and_write_predictions: Callable
    evaluate_predictions: Callable


def _timestamp(value, field, request):
    if type(value) is not int or not -(2**63) <= value < 2**63:
        raise AdRunError(3, f"request_id={request}: {field} expected int64 epoch seconds")
    return value


def validate_opportunities(rows, manifest, expected_version, exposure_policy, *,
                           exclusive_split_ends=False):
    """Reject malformed/duplicate/future records; never drop or repair rows.

    Optional provenance timestamps are checked when supplied. The Data manifest
    remains responsible for proving candidate/exposure availability, profile
    as-of and temporal split correctness, including raw source checksums.
    ``train_end``/``valid_end`` are inclusive upper bounds (Taobao) unless
    ``exclusive_split_ends`` (simulator manifests).
    """
    if manifest.get("dataset_version") != expected_version or manifest.get("schema_version") != 1:
        raise AdRunError(3, "data_version/schema_version does not match handoff contract")
    if manifest.get("candidate_policy") != exposure_policy:
        raise AdRunError(3, "exposure_policy does not match manifest.candidate_policy")
    boundary = manifest.get("split", {})
    if boundary.get("kind") != "temporal":
        raise AdRunError(3, "manifest.split.kind expected temporal")
    train_end = _timestamp(boundary.get("train_end"), "train_end", "manifest")
    valid_end = _timestamp(boundary.get("valid_end"), "valid_end", "manifest")
    if train_end >= valid_end or not rows:
        raise AdRunError(3, "invalid temporal boundaries or empty opportunity handoff")
    seen, requests = set(), {}
    for row in rows:
        request = row.get("request_id")
        for field in ("request_id", "user_id", "candidate_id"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise AdRunError(3, f"request_id={request}: {field} expected canonical string")
        cutoff = _timestamp(row.get("cutoff_timestamp"), "cutoff_timestamp", request)
        split = row.get("split")
        if exclusive_split_ends:
            actual = "train" if cutoff < train_end else "validation" if cutoff < valid_end else "test"
        else:
            actual = "train" if cutoff <= train_end else "validation" if cutoff <= valid_end else "test"
        if split != actual:
            raise AdRunError(3, f"request_id={request}: split expected {actual}, got {split}")
        if type(row.get("clicked")) not in (int, bool) or row["clicked"] not in (0, 1):
            raise AdRunError(3, f"request_id={request}: missing/non-binary clicked; policy=reject")
        key = (request, row["candidate_id"])
        if key in seen:
            raise AdRunError(3, f"request_id={request}: duplicate candidate")
        seen.add(key)
        identity = (row["user_id"], cutoff, split)
        if requests.setdefault(request, identity) != identity:
            raise AdRunError(3, f"request_id={request}: inconsistent user/cutoff/split")
        for field in ("candidate_timestamp", "exposure_timestamp", "click_timestamp",
                      "profile_as_of_timestamp", "history_end_timestamp"):
            if row.get(field) is not None:
                observed = _timestamp(row[field], field, request)
                if observed > cutoff:
                    raise AdRunError(3, f"request_id={request}: future {field} after cutoff")


def _embeddings(foundation, batch, freeze):
    with torch.set_grad_enabled(not freeze):
        embeddings = foundation.embed_users(batch)
    if set(embeddings) != {"long_term", "short_term"}:
        raise AdRunError(4, "embed_users expected exactly long_term and short_term")
    return embeddings["long_term"], embeddings["short_term"]


def _score(rank, foundation, batch, freeze):
    embeddings = batch.embeddings if freeze and batch.embeddings is not None else _embeddings(
        foundation, batch.foundation, freeze
    )
    return rank(*embeddings, batch.candidate_ids)


def train_ad_ranker(rank, foundation, batches, *, epochs, learning_rate,
                    freeze_foundation=True, seed=42, validation_batches=None,
                    weight_decay=0.0, shuffle=False, cache_embeddings=False):
    """Train unchanged exposures and select the head using validation BCE only."""
    if not batches or epochs < 1 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise AdRunError(2, "training requires batches, positive epochs and learning_rate")
    if cache_embeddings and not freeze_foundation:
        raise AdRunError(2, "cannot cache foundation embeddings while fine-tuning")
    for parameter in foundation.parameters():
        parameter.requires_grad_(not freeze_foundation)
    foundation.train(not freeze_foundation)
    # Never reuse cached embeddings from another run/checkpoint.
    for batch in list(batches) + list(validation_batches or []):
        batch.embeddings = None
        if cache_embeddings:
            batch.embeddings = _embeddings(foundation, batch.foundation, True)
    parameters = list(rank.parameters())
    if not freeze_foundation:
        parameters += list(foundation.parameters())
    optimizer = head_optimizer(parameters, learning_rate=learning_rate, weight_decay=weight_decay)
    records, step, best, best_loss = [], 0, None, float("inf")
    for epoch in range(epochs):
        splits = [("train", batches)]
        if validation_batches:
            splits.append(("validation", validation_batches))
        for split, current in splits:
            training = split == "train"
            rank.train(training)
            foundation.train(training and not freeze_foundation)
            if training and shuffle:
                current = [current[index] for index in torch.randperm(len(current)).tolist()]
            total, count, positives = 0.0, 0, 0
            for batch in current:
                labels, mask = batch.labels_and_mask()
                with torch.set_grad_enabled(training):
                    loss = ctr_loss(_score(rank, foundation, batch,
                                           freeze_foundation or not training), labels, mask)
                    if not torch.isfinite(loss):
                        raise AdRunError(5, "CTR training produced non-finite loss")
                    if training:
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        optimizer.step()
                        step += 1
                exposures = int(mask.sum())
                total += float(loss.detach()) * exposures
                count += exposures
                positives += int(labels[mask].sum())
            records.append({
                "epoch": epoch, "step": step, "split": split, "total_loss": total / count,
                "ctr_loss": total / count, "target_counts": {"ctr": count},
                "class_prevalence": positives / count, "seed": seed, "timestamp_utc": _utc(),
            })
            print(f"epoch={epoch + 1} split={split} ctr_loss={total / count:.6f} exposures={count}", flush=True)
            if not training and total / count < best_loss:
                best_loss = total / count
                best = (copy.deepcopy(rank.state_dict()),
                        None if freeze_foundation else copy.deepcopy(foundation.state_dict()))
    if best is not None:
        rank.load_state_dict(best[0])
        if best[1] is not None:
            foundation.load_state_dict(best[1])
    foundation.eval()
    rank.eval()
    return records


def fit_calibration(logits, labels, *, steps=100, learning_rate=0.05):
    """Validation-only affine logit calibration; never fit on evaluated/test rows."""
    if (steps < 1 or not math.isfinite(learning_rate) or learning_rate <= 0
            or not torch.isfinite(logits).all() or not ((labels == 0) | (labels == 1)).all()):
        raise AdRunError(2, "calibration requires positive steps and finite binary samples")
    if logits.shape != labels.shape or logits.numel() == 0 or labels.unique().numel() != 2:
        raise AdRunError(3, "calibration requires both click classes in a held-out split")
    log_temperature = nn.Parameter(torch.zeros((), device=logits.device))
    bias = nn.Parameter(torch.zeros((), device=logits.device))
    optimizer = torch.optim.Adam([log_temperature, bias], lr=learning_rate)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        calibrated = logits.detach() / log_temperature.exp() + bias
        loss = F.binary_cross_entropy_with_logits(calibrated, labels.to(logits.dtype))
        if not torch.isfinite(loss):
            raise AdRunError(5, "calibration produced non-finite loss")
        loss.backward()
        optimizer.step()
    temperature, offset = float(log_temperature.exp().detach()), float(bias.detach())
    if not math.isfinite(temperature) or temperature <= 0 or not math.isfinite(offset):
        raise AdRunError(5, "calibration parameters are non-finite")
    return temperature, offset


def prediction_table(scored_rows, *, split, data_version, run_id, temperature, bias):
    """Create a typed in-memory table; the shared writer owns validation/publication."""
    if split not in {"validation", "test"} or not data_version or not run_id:
        raise AdRunError(5, "prediction split/data_version/run_id invalid")
    if temperature <= 0 or not math.isfinite(temperature) or not math.isfinite(bias):
        raise AdRunError(5, "invalid calibration parameters")
    output, seen = [], set()
    for row, raw in scored_rows:
        if row["split"] != split or not math.isfinite(raw):
            raise AdRunError(5, f"request_id={row['request_id']}: wrong split/non-finite raw_logit")
        key = (row["request_id"], row["candidate_id"])
        if key in seen:
            raise AdRunError(5, f"request_id={key[0]}: duplicate prediction candidate")
        seen.add(key)
        calibrated = raw / temperature + bias
        if not math.isfinite(calibrated):
            raise AdRunError(5, "non-finite calibrated logit")
        probability = float(torch.sigmoid(torch.tensor(calibrated, dtype=torch.float64)))
        output.append({
            "schema_version": 1, "task": TASK, "split": split, "data_version": data_version,
            "run_id": run_id, "request_id": row["request_id"], "user_id": row["user_id"],
            "cutoff_timestamp": row["cutoff_timestamp"], "candidate_id": row["candidate_id"],
            "score": probability, "raw_logit": raw,
        })
    if not output:
        raise AdRunError(5, "prediction table cannot be empty")
    output.sort(key=lambda row: (row["request_id"], -row["score"], row["candidate_id"]))
    return pa.Table.from_pylist(output, schema=PREDICTION_SCHEMA)


def _utc():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_config(config):
    try:
        for field in ("root", "expected_version"):
            if not config["data"][field]:
                raise ValueError(field)
        for field in ("artifact_dir", "exposure_policy"):
            if not config[field]:
                raise ValueError(field)
        if not config["checkpoint"]["path"] or type(config["seed"]) is not int:
            raise ValueError("checkpoint.path/seed")
        foundation_init_arm(config)
        if ad_data_format(config) == "sim_v1" and "expected_version" in config["checkpoint"]:
            raise ValueError("checkpoint.expected_version must be omitted for data.format "
                             "sim_v1; the checkpoint is bound by sim_config_sha256")
        if config.get("contract_version") != 1 or config["split"] not in {"validation", "test"}:
            raise ValueError("contract_version/split")
        if type(config.get("freeze_foundation", True)) is not bool:
            raise ValueError("freeze_foundation")
        if config.get("missing_click_policy") != "reject":
            raise ValueError("missing_click_policy must be reject; negatives are never resampled")
        calibration = config.get("calibration", {"enabled": False})
        enabled = calibration.get("enabled", True)
        if type(enabled) is not bool:
            raise ValueError("calibration.enabled must be boolean")
        if enabled and (calibration.get("split") != "validation" or config["split"] == "validation"):
            raise ValueError("calibration uses validation; evaluate on separate test split")
        if type(config["training"]["epochs"]) is not int or config["training"]["epochs"] < 1:
            raise ValueError("training.epochs")
        if config["training"]["learning_rate"] <= 0:
            raise ValueError("training.learning_rate")
        if not math.isfinite(config["training"]["learning_rate"]):
            raise ValueError("training.learning_rate must be finite")
        steps = calibration.get("steps", 100)
        if type(steps) is not int or steps < 1:
            raise ValueError("calibration.steps must be a positive integer")
        hidden = config.get("model", {}).get("hidden_dim", 64)
        if type(hidden) is not int or hidden < 1:
            raise ValueError("model.hidden_dim must be a positive integer")
        for name in ("batch_size", "cpu_threads"):
            value = config["training"].get(name, 1)
            if type(value) is not int or value < 1:
                raise ValueError(f"training.{name} must be a positive integer")
        weight_decay = config["training"].get("weight_decay", 0.0)
        if not math.isfinite(weight_decay) or weight_decay < 0:
            raise ValueError("training.weight_decay must be finite and nonnegative")
        candidate_dim = config.get("model", {}).get("candidate_dim", 16)
        dropout = config.get("model", {}).get("dropout", 0.0)
        if type(candidate_dim) is not int or candidate_dim < 1:
            raise ValueError("model.candidate_dim must be a positive integer")
        if not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("model.dropout must be in [0,1)")
        for name in ("cache_foundation_embeddings", "shuffle", "initialize_bias_from_train"):
            if type(config["training"].get(name, False)) is not bool:
                raise ValueError(f"training.{name} must be boolean")
        if (config["training"].get("cache_foundation_embeddings", False)
                and not config.get("freeze_foundation", True)):
            raise ValueError("cannot cache foundation embeddings while fine-tuning")
        for name in ("evaluate", "export_model"):
            if type(config.get(name, True)) is not bool:
                raise ValueError(f"{name} must be boolean")
    except (KeyError, TypeError, ValueError) as error:
        raise AdRunError(2, f"config field/rule invalid: {error}") from error


def _sim_config_sha256(root):
    """The simulator config hash of a sim Ads package, the key a sim checkpoint is bound by."""
    manifest = json.loads((Path(root) / "manifest.json").read_text(encoding="utf-8"))
    value = manifest.get("sim_config_sha256") if isinstance(manifest, dict) else None
    if not is_sha256(value):
        raise ValueError("Ads sim_v1 manifest requires a SHA-256 sim_config_sha256")
    return value


def _collect(rank, foundation, batches):
    scored, labels = [], []
    with torch.no_grad():
        for batch in batches:
            targets, mask = batch.labels_and_mask()
            logits = _score(rank, foundation, batch, True)
            if not torch.isfinite(logits[mask]).all():
                raise AdRunError(5, "inference logits contain NaN/Inf")
            for index, rows in enumerate(batch.rows):
                scored.extend((row, float(logits[index, column]))
                              for column, row in enumerate(rows))
            labels.extend(targets[mask].tolist())
    return scored, torch.tensor(labels, dtype=torch.float32)


def load_ad_ranker_checkpoint(path, *, expected_foundation_sha256, expected_data_version):
    """Reload the head and calibration together with its frozen candidate vocabulary."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1 or payload.get("task") != TASK:
        raise ValueError("Ads ranker checkpoint format/task mismatch")
    if payload.get("foundation_checkpoint_sha256") != expected_foundation_sha256:
        raise ValueError("Ads foundation checkpoint mismatch")
    if payload.get("data_version") != expected_data_version:
        raise ValueError("Ads data version mismatch")
    vocabulary = payload.get("candidate_vocabulary")
    if (not isinstance(vocabulary, dict) or not vocabulary
            or not all(isinstance(key, str) and key.strip() and type(value) is int and value > 0
                       for key, value in vocabulary.items())
            or set(vocabulary.values()) != set(range(1, len(vocabulary) + 1))):
        raise ValueError("Ads candidate vocabulary invalid")
    calibration = payload["calibration"]
    if (not math.isfinite(calibration["temperature"]) or calibration["temperature"] <= 0
            or not math.isfinite(calibration["bias"])):
        raise ValueError("Ads calibration invalid")
    ranker = AdRanker(payload["d_model"], len(vocabulary) + 1, **payload["model_hyperparameters"])
    ranker.load_state_dict(payload["ranker_state_dict"], strict=True)
    if not all(torch.isfinite(parameter).all() for parameter in ranker.parameters()):
        raise ValueError("Ads ranker has non-finite weights")
    ranker.eval()
    return ranker, vocabulary, payload


def _export_ranker(path, rank, foundation, inputs, manifest):
    payload = {
        "format_version": 1, "task": TASK, "run_id": manifest["run_id"],
        "data_version": manifest["data_version"], "data_manifest_sha256": inputs.data_manifest_sha256,
        "foundation_checkpoint_sha256": manifest["foundation"]["checkpoint_sha256"],
        "freeze_foundation": manifest["foundation"]["freeze_foundation"],
        "foundation_init": manifest["foundation"]["init"],
        "d_model": inputs.d_model, "model_hyperparameters": manifest["model_hyperparameters"],
        "candidate_vocabulary": dict(inputs.candidate_vocabulary),
        "ranker_state_dict": rank.state_dict(), "calibration": manifest["calibration"],
        "selection": manifest["selection"], "seed": manifest["seed"],
    }
    if not payload["freeze_foundation"]:
        payload["foundation_state_dict"] = foundation.state_dict()
    descriptor, temporary = tempfile.mkstemp(prefix=".ad-ranker-", suffix=".pt", dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        load_ad_ranker_checkpoint(temporary,
            expected_foundation_sha256=payload["foundation_checkpoint_sha256"],
            expected_data_version=payload["data_version"])
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _json_metrics(metrics):
    """Represent only the public evaluator's explicitly undefined metrics as null."""
    if not isinstance(metrics, dict):
        raise ValueError("evaluator expected dict[str, float]")  # noqa: TRY004
    result = {}
    for name, value in metrics.items():
        if not isinstance(name, str) or not isinstance(value, (int, float)):
            raise ValueError("evaluator expected dict[str, float]")  # noqa: TRY004
        undefined = (name in {"roc_auc", "pr_auc"} and metrics.get("discrimination_defined") == 0
                     or name.startswith(("ndcg_at_", "recall_at_"))
                     and metrics.get("clicked_request_count") == 0)
        if math.isnan(value) and undefined:
            result[name] = None
        elif not math.isfinite(value):
            raise ValueError(f"unexpected non-finite evaluator metric: {name}")
        else:
            result[name] = value
    return result


def run_ad_optimization(config, services: AdServices):
    """Train, calibrate, validate/export via shared service, then call evaluator.

    Services must enforce the full shared output contract, including manifest
    hashes, candidate exactness, atomic publication and validation_report.json.
    The returned path is checked to prevent evaluation of an unrelated artifact.
    """
    _validate_config(config)
    data_format, init = ad_data_format(config), foundation_init_arm(config)
    run_id = str(uuid.uuid4())
    context = f"run_id={run_id} task={TASK} split={config['split']}"
    seed = config["seed"]
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(config["training"].get("cpu_threads", 1))
    checkpoint = Path(config["checkpoint"]["path"])
    if not checkpoint.is_file():
        raise AdRunError(4, f"{context}: checkpoint.path missing; Person 3 must export v1")
    checkpoint_hash = _sha(checkpoint)
    options = {"expected_version": config["checkpoint"].get(
        "expected_version", config["data"]["expected_version"])}
    if data_format == "sim_v1":
        try:
            options["expected_sim_config_sha256"] = _sim_config_sha256(config["data"]["root"])
        except (ValueError, OSError) as error:
            raise AdRunError(3, f"{context}: data handoff invalid: {error}") from error
    try:
        foundation = services.load_checkpoint(str(checkpoint), **options)
    except (ValueError, OSError) as error:
        raise AdRunError(4, f"{context}: checkpoint/preprocess incompatibility: {error}") from error
    if not isinstance(foundation, nn.Module) or not callable(getattr(foundation, "embed_users", None)):
        raise AdRunError(4, f"{context}: loader must return nn.Module with embed_users")
    if init == "random":  # control arm: checkpoint vocabulary and config, fresh weights
        try:
            foundation = reinitialize_foundation(foundation, seed)
        except (AttributeError, TypeError, ValueError, RuntimeError) as error:
            raise AdRunError(4, f"{context}: foundation.init random needs a FoundationModel "
                                f"checkpoint: {error}") from error
    try:
        inputs = services.load_inputs(config, foundation)
        validate_opportunities(inputs.opportunities, inputs.manifest,
                               config["data"]["expected_version"], config["exposure_policy"],
                               exclusive_split_ends=data_format == "sim_v1")
        if inputs.num_candidates < 1 or inputs.d_model < 1:
            raise AdRunError(3, "invalid train-fitted candidate vocabulary/model dimensions")
        train_candidates = sorted({row["candidate_id"] for row in inputs.opportunities
                                   if row["split"] == "train"})
        vocabulary = {candidate: index + 1 for index, candidate in enumerate(train_candidates)}
        if dict(inputs.candidate_vocabulary) != vocabulary or inputs.num_candidates != len(vocabulary) + 1:
            raise AdRunError(3, "candidate vocabulary must be fitted on train only; ID 0 is OOV")
        for field in ("data_manifest_sha256", "preprocess_fingerprint"):
            if not is_sha256(getattr(inputs, field)):
                raise AdRunError(3, f"{field} expected SHA-256 fingerprint")
        expected = {(row["request_id"], row["candidate_id"]): dict(row)
                    for row in inputs.opportunities}
        actual = {}
        for split, batches in inputs.batches.items():
            for batch in batches:
                batch.labels_and_mask()
                for index, rows in enumerate(batch.rows):
                    for column, row in enumerate(rows):
                        key = (row["request_id"], row["candidate_id"])
                        if row["split"] != split or key in actual:
                            raise AdRunError(3, "batch duplicate/wrong split")
                        if int(batch.candidate_ids[index, column]) != vocabulary.get(row["candidate_id"], 0):
                            raise AdRunError(3, "candidate ID does not match frozen train vocabulary")
                        actual[key] = dict(row)
        if actual != expected:
            raise AdRunError(3, "batch rows must equal complete Data handoff; no drops/additions")
        for split in ("train", "validation", config["split"]):
            if not inputs.batches.get(split):
                raise AdRunError(3, f"required {split} exposure batches missing")
    except AdRunError as error:
        raise AdRunError(error.code, f"{context}: {error}") from error
    except (ValueError, KeyError, OSError) as error:
        raise AdRunError(3, f"{context}: data handoff invalid: {error}") from error
    # CPU is the MVP target; loaders must supply CPU FoundationBatch tensors.
    foundation.cpu()
    rank = AdRanker(inputs.d_model, inputs.num_candidates,
                    config.get("model", {}).get("hidden_dim", 64),
                    config.get("model", {}).get("candidate_dim"),
                    config.get("model", {}).get("dropout", 0.0))
    train_rows = [row for row in inputs.opportunities if row["split"] == "train"]
    if config["training"].get("initialize_bias_from_train", False):
        prior = min(1 - 1e-6, max(1e-6, sum(row["clicked"] for row in train_rows) / len(train_rows)))
        with torch.no_grad():
            rank.scorer[-1].bias.fill_(math.log(prior / (1 - prior)))
    try:
        records = train_ad_ranker(
            rank, foundation, inputs.batches["train"],
            freeze_foundation=config.get("freeze_foundation", True), seed=seed,
            epochs=config["training"]["epochs"],
            learning_rate=config["training"]["learning_rate"],
            validation_batches=inputs.batches["validation"],
            weight_decay=config["training"].get("weight_decay", 0.0),
            shuffle=config["training"].get("shuffle", False),
            cache_embeddings=config["training"].get("cache_foundation_embeddings", False),
        )
        calibration = config.get("calibration", {"enabled": False})
        temperature, bias = 1.0, 0.0
        if calibration.get("enabled", True):
            calibration_rows, calibration_labels = _collect(rank, foundation, inputs.batches["validation"])
            temperature, bias = fit_calibration(
                torch.tensor([value for _, value in calibration_rows]), calibration_labels,
                steps=calibration.get("steps", 100),
            )
        scored, _ = _collect(rank, foundation, inputs.batches[config["split"]])
        table = prediction_table(
            scored, split=config["split"], data_version=config["data"]["expected_version"],
            run_id=run_id, temperature=temperature, bias=bias,
        )
    except AdRunError as error:
        raise AdRunError(error.code, f"{context}: {error}") from error
    except ValueError as error:
        raise AdRunError(5, f"{context}: embedding/CTR/prediction field invalid: {error}") from error
    run_dir = Path(config["artifact_dir"]) / TASK / run_id
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unknown"
    selected = [row for row in inputs.opportunities if row["split"] == config["split"]]
    reference = {
        "checkpoint_path": str(checkpoint), "checkpoint_sha256": checkpoint_hash,
        "format_version": 1, "freeze_foundation": config.get("freeze_foundation", True),
        "init": init,
    }
    best_validation = min((record for record in records if record["split"] == "validation"),
                          key=lambda record: record["ctr_loss"])
    manifest = {
        "contract_version": 1, "run_id": run_id, "task": TASK, "split": config["split"],
        "data_version": config["data"]["expected_version"],
        "data_manifest_sha256": inputs.data_manifest_sha256,
        "preprocess_fingerprint": inputs.preprocess_fingerprint,
        "config_sha256": hashlib.sha256(json.dumps(
            config, sort_keys=True, allow_nan=False
        ).encode()).hexdigest(), "code_revision": revision, "seed": seed,
        "created_at_utc": _utc(), "foundation": reference,
        "candidate_or_exposure_policy": config["exposure_policy"],
        "row_count": table.num_rows, "request_count": len({row["request_id"] for row in selected}),
        # Shared writer computes prediction_sha256 from actual Parquet bytes before validation.
        "missing_click_policy": "reject", "negative_sampling": "none",
        "class_prevalence": sum(row["clicked"] for row in train_rows) / len(train_rows),
        "split_cutoffs": dict(inputs.manifest["split"]), "artifact_path": str(run_dir),
        "calibration": {"enabled": calibration.get("enabled", True),
                        "split": "validation" if calibration.get("enabled", True) else None,
                        "temperature": temperature, "bias": bias},
        "evaluation_policy": dict(config.get("evaluation", {})),
        "data_limitations": list(inputs.manifest.get("limitations", [])),
        "model_hyperparameters": {"hidden_dim": config.get("model", {}).get("hidden_dim", 64),
                                  "candidate_dim": rank.candidates.embedding_dim,
                                  "dropout": config.get("model", {}).get("dropout", 0.0)},
        "training_hyperparameters": dict(config["training"]),
        "selection": {"metric": "validation_ctr_loss", "best_epoch": best_validation["epoch"] + 1,
                      "best_validation_loss": best_validation["ctr_loss"]},
    }
    try:
        predictions = Path(services.validate_and_write_predictions(
            table, expected_candidates=selected, run_manifest=manifest,
            train_metrics=records, checkpoint_ref=reference, artifact_dir=config["artifact_dir"],
        ))
        required = [run_dir / name for name in
                    ("run_manifest.json", "validation_report.json", "checkpoint_ref.json",
                     "train_metrics.jsonl")]
        if predictions.resolve() != (run_dir / f"predictions.{config['split']}.parquet").resolve():
            raise ValueError("shared writer returned unexpected prediction artifact path")
        if not predictions.is_file() or not all(path.is_file() for path in required):
            raise ValueError("shared writer did not publish complete Output Contract v1 artifacts")
        report = json.loads((run_dir / "validation_report.json").read_text(encoding="utf-8"))
        written = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        if report.get("status") != "passed" or written.get("prediction_sha256") != _sha(predictions):
            raise ValueError("shared writer validation status/prediction_sha256 invalid")
    except (ValueError, OSError, TypeError) as error:
        raise AdRunError(5, f"{context}: Output Contract v1 validation/export failed: {error}") from error
    model_path = None
    if config.get("export_model", True):
        model_path = run_dir / "ad_ranker-v1.pt"
        _export_ranker(model_path, rank, foundation, inputs, manifest)
    if not config.get("evaluate", True):
        return {"run_id": run_id, "predictions_path": str(predictions),
                "model_path": str(model_path) if model_path else None,
                "evaluation_status": "not_requested", "metrics": None}
    try:
        metrics = services.evaluate_predictions(
            predictions_path=str(predictions), task=TASK,
            data_version=config["data"]["expected_version"], split=config["split"],
        )
        metrics = _json_metrics(metrics)
    except Exception as error:
        raise AdRunError(6, f"{context}: evaluator failed after valid export: {error}") from error
    result = {"run_id": run_id, "predictions_path": str(predictions),
              "model_path": str(model_path) if model_path else None, "metrics": metrics,
              "undefined_metrics": [name for name, value in metrics.items() if value is None],
              "evaluation_policy": manifest["evaluation_policy"]}
    (run_dir / "evaluation.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    return result


def _binding(path, code):
    try:
        module, name = path.split(":", 1)
        value = getattr(importlib.import_module(module), name)
        if not callable(value):
            raise TypeError("binding must be callable")
        return value
    except (ImportError, AttributeError, TypeError, ValueError) as error:
        raise AdRunError(code, f"integration {path!r} unavailable; required team handoff: {error}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    try:
        import yaml
        try:
            config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        except yaml.YAMLError as error:
            raise AdRunError(2, "config YAML syntax invalid") from error
        _validate_config(config)
        bindings = config.get("integration", {
            "load_checkpoint": "src.downstream.ad_inputs:load_ad_foundation",
            "load_inputs": "src.downstream.ad_inputs:load_ad_inputs",
            "validate_and_write_predictions": "src.contracts.output:validate_and_write_predictions",
        })
        writer = bindings["validate_and_write_predictions"]
        if not isinstance(writer, str) or not writer.startswith("src.contracts.output:"):
            raise AdRunError(2, "prediction writer must come from src.contracts.output")
        services = AdServices(
            _binding(bindings["load_checkpoint"], 4), _binding(bindings["load_inputs"], 3),
            _binding(bindings["validate_and_write_predictions"], 2),
            None,
        )
        if config.get("evaluate", True):
            from src.downstream.ad_inputs import ad_baseline_evaluator
            services.evaluate_predictions = ad_baseline_evaluator(config)
        print(json.dumps(run_ad_optimization(config, services), allow_nan=False))
        return 0
    except AdRunError as error:
        print(f"task={TASK}: {error}", file=sys.stderr)
        return error.code
    except (ImportError, OSError, KeyError, TypeError, ValueError) as error:
        print(f"task={TASK}: config/dependency error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
