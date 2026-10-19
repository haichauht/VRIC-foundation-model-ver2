"""Food ranking on frozen long/short embeddings, using shared Output Contract v1."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import torch
import yaml
from torch import nn
from torch.nn import functional as F

from src.contracts.output import (
    PREDICTION_SCHEMA,
    fingerprint,
    sha256_file,
    utc_now,
    validate_and_write_predictions,
)
from src.downstream.candidate_features import (
    attribute_dim,
    attribute_names,
    attribute_sizes,
    check_candidate_attributes,
    validate_attribute_vocabularies,
)
from src.downstream.food_inputs import (
    food_baseline_evaluator,
    food_data_format,
    load_food_foundation,
    load_food_inputs,
)
from src.downstream.food_samples import read_sim_food_manifest
from src.downstream.foundation_init import foundation_init_arm, reinitialize_foundation
from src.downstream.head_training import embed_users_on_device, encoder_device, head_optimizer


class FoodRunError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


class FoodRanker(nn.Module):
    """Candidate scorer; normalization affects frozen user vectors, not item embeddings.

    ``mlp_dot`` adds a low-rank, scaled user/item interaction to the usual MLP.
    ``attribute_sizes`` adds one embedding table per catalog attribute (0 = missing/unseen); the
    item vector is then [candidate-ID embedding, attribute embeddings].
    Defaults preserve the parameter names and behavior of previously exported heads.
    """

    def __init__(self, d_model, num_candidates, hidden_dim=64, candidate_dim=None, dropout=0.0,
                 normalization="none", architecture="mlp", interaction_dim=None,
                 attribute_sizes=(), attribute_dim=8):
        super().__init__()
        candidate_dim = d_model if candidate_dim is None else candidate_dim
        if (min(d_model, num_candidates, hidden_dim, candidate_dim) < 1
                or not isinstance(dropout, (int, float)) or not 0 <= dropout < 1):
            raise ValueError("FoodRanker dimensions must be positive")
        if normalization not in {"none", "layernorm", "l2"}:
            raise ValueError("normalization must be none, layernorm, or l2")
        if architecture not in {"mlp", "mlp_dot"}:
            raise ValueError("architecture must be mlp or mlp_dot")
        interaction_dim = candidate_dim if interaction_dim is None else interaction_dim
        if type(interaction_dim) is not int or interaction_dim < 1:
            raise ValueError("interaction_dim must be a positive integer")
        attribute_sizes = tuple(attribute_sizes)
        if (not all(type(size) is int and size >= 1 for size in attribute_sizes)
                or type(attribute_dim) is not int or attribute_dim < 1):
            raise ValueError("attribute sizes and attribute_dim must be positive integers")
        self.d_model = d_model
        self.normalization = normalization
        self.architecture = architecture
        self.interaction_dim = interaction_dim
        self.attribute_sizes = attribute_sizes
        self.attribute_dim = attribute_dim
        self.candidates = nn.Embedding(num_candidates, candidate_dim)
        item_dim = candidate_dim
        if attribute_sizes:
            self.attributes = nn.ModuleList(
                nn.Embedding(size, attribute_dim, padding_idx=0) for size in attribute_sizes)
            item_dim += attribute_dim * len(attribute_sizes)
        layers = [nn.Linear(2 * d_model + item_dim, hidden_dim), nn.ReLU()]
        if dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, 1))
        self.scorer = nn.Sequential(*layers)
        self.long_norm = nn.LayerNorm(d_model) if normalization == "layernorm" else nn.Identity()
        self.short_norm = nn.LayerNorm(d_model) if normalization == "layernorm" else nn.Identity()
        if architecture == "mlp_dot":
            self.user_projection = nn.Linear(2 * d_model, interaction_dim, bias=False)
            self.candidate_projection = nn.Linear(item_dim, interaction_dim, bias=False)

    def forward(
        self,
        long_term: torch.Tensor,
        short_term: torch.Tensor,
        food_candidate_ids: torch.LongTensor,
        candidate_attributes: torch.LongTensor | None = None,
    ) -> torch.Tensor:
        if food_candidate_ids.ndim != 2 or food_candidate_ids.dtype != torch.long:
            raise ValueError("food_candidate_ids expected torch.long [B,C]")
        check_candidate_attributes(candidate_attributes, food_candidate_ids, self.attribute_sizes)
        for name, value in (("long_term", long_term), ("short_term", short_term)):
            if value.shape != (food_candidate_ids.shape[0], self.d_model) or value.dtype != torch.float32:
                raise ValueError(f"{name}: expected float32 [B,D]")
            if not torch.isfinite(value).all() or value.device != food_candidate_ids.device:
                raise ValueError(f"{name}: expected finite embedding on candidate device")
        if ((food_candidate_ids < 0) | (food_candidate_ids >= self.candidates.num_embeddings)).any():
            raise ValueError("Food candidate ID outside train-fitted vocabulary")
        long_term, short_term = self.long_norm(long_term), self.short_norm(short_term)
        if self.normalization == "l2":
            long_term = F.normalize(long_term, p=2, dim=-1)
            short_term = F.normalize(short_term, p=2, dim=-1)
        count = food_candidate_ids.shape[1]
        candidates = self.candidates(food_candidate_ids)
        if self.attribute_sizes:
            candidates = torch.cat([candidates] + [
                table(candidate_attributes[..., index])
                for index, table in enumerate(self.attributes)], -1)
        values = torch.cat([long_term[:, None].expand(-1, count, -1),
                            short_term[:, None].expand(-1, count, -1), candidates], -1)
        scores = self.scorer(values).squeeze(-1)
        if self.architecture == "mlp_dot":
            user = self.user_projection(torch.cat([long_term, short_term], dim=-1))
            item = self.candidate_projection(candidates)
            scores = scores + (user[:, None] * item).sum(-1) / math.sqrt(self.interaction_dim)
        return scores


def load_food_ranker_checkpoint(path, *, expected_foundation_sha256, expected_data_version):
    """Load a downstream head with its train-fitted candidate vocabulary."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1 or payload.get("task") != "food_recommendation":
        raise ValueError("Food ranker checkpoint format/task mismatch")
    if payload.get("foundation_checkpoint_sha256") != expected_foundation_sha256:
        raise ValueError("Food ranker foundation checkpoint mismatch")
    if payload.get("data_version") != expected_data_version:
        raise ValueError("Food ranker data version mismatch")
    vocabulary = payload.get("candidate_vocabulary")
    if not isinstance(vocabulary, dict) or not all(
        isinstance(key, str) and type(value) is int and value > 0
        for key, value in vocabulary.items()
    ) or set(vocabulary.values()) != set(range(1, len(vocabulary) + 1)):
        raise ValueError("Food ranker candidate vocabulary invalid")
    names = tuple(payload.get("candidate_attributes", ()))
    vocabularies = payload.get("attribute_vocabularies", {})
    validate_attribute_vocabularies(names, vocabularies)
    ranker = FoodRanker(payload["d_model"], len(vocabulary) + 1, payload["hidden_dim"],
                        payload.get("candidate_dim", payload["d_model"]), payload.get("dropout", 0.0),
                        payload.get("normalization", "none"), payload.get("architecture", "mlp"),
                        payload.get("interaction_dim"), attribute_sizes(names, vocabularies),
                        payload.get("attribute_dim", 8))
    ranker.load_state_dict(payload["ranker_state_dict"], strict=True)
    ranker.eval()
    return ranker, vocabulary, payload


def food_loss(scores, positives, mask):
    """Listwise CE; multiple positives receive equal target mass, padding receives none."""
    if scores.ndim != 2 or scores.shape != positives.shape or mask.shape != scores.shape or mask.dtype != torch.bool:
        raise ValueError("Food loss expects scores/positives/bool mask [B,C]")
    if not torch.isfinite(scores[mask]).all() or not ((positives[mask] == 0) | (positives[mask] == 1)).all():
        raise ValueError("Food scores must be finite and positives binary")
    target = positives.masked_fill(~mask, 0).to(scores.dtype)
    count = target.sum(-1)
    if (count < 1).any():
        raise ValueError("Food requests require at least one positive candidate")
    log_probabilities = F.log_softmax(scores.masked_fill(~mask, float("-inf")), -1)
    return -(target / count[:, None] * log_probabilities.masked_fill(~mask, 0)).sum(-1).mean()


def code_revision():
    """Record the checked-out commit, or the module hash outside a Git checkout."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2],
            capture_output=True, text=True, check=True, timeout=5,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return sha256_file(__file__)


@dataclass
class FoodBatch:
    foundation: object
    candidate_ids: torch.LongTensor
    rows: list
    candidate_attributes: torch.LongTensor | None = None  # [B,C,K] with model.candidate_attributes

    def labels_and_mask(self):
        shape = self.candidate_ids.shape
        if self.candidate_ids.ndim != 2 or self.candidate_ids.dtype != torch.long:
            raise FoodRunError(3, "candidate_ids expected torch.long [B,C]")
        if len(self.rows) != shape[0] or self.foundation.user_ids.shape != (shape[0],):
            raise FoodRunError(3, "request rows must match foundation batch size")
        mask = torch.zeros(shape, dtype=torch.bool, device=self.candidate_ids.device)
        labels = torch.zeros(shape, dtype=torch.float32, device=self.candidate_ids.device)
        for index, rows in enumerate(self.rows):
            if not rows or len(rows) > shape[1]:
                raise FoodRunError(3, "each request requires supplied candidates")
            first = rows[0]
            for row in rows:
                if any(row[key] != first[key] for key in
                       ("request_id", "user_id", "cutoff_timestamp", "split")):
                    raise FoodRunError(3, "foundation row must represent exactly one request")
            mask[index, :len(rows)] = True
            labels[index, :len(rows)] = torch.tensor(
                [row["is_positive"] for row in rows], dtype=torch.float32,
                device=self.candidate_ids.device,
            )
        return labels, mask


@dataclass
class FoodInputs:
    batches: dict
    candidates: list
    manifest: dict
    data_manifest_sha256: str
    preprocess_fingerprint: str
    num_candidates: int
    d_model: int
    candidate_vocabulary: dict
    preprocess_metadata: dict | None = None
    candidate_attributes: tuple = ()
    attribute_vocabularies: dict | None = None


def _validate_food_embeddings(embeddings, batch, d_model):
    if not isinstance(embeddings, dict) or set(embeddings) != {"long_term", "short_term"}:
        raise ValueError("Food embeddings require exactly long_term and short_term")
    for value in embeddings.values():
        if (not isinstance(value, torch.Tensor)
                or value.shape != (len(batch.rows), d_model)
                or value.dtype != torch.float32 or value.device != batch.candidate_ids.device
                or not torch.isfinite(value).all()):
            raise ValueError("Food embeddings must be finite float32 [B,D] on candidate device")


@dataclass
class FrozenFoodEmbeddings:
    """In-memory cache bound to one checkpoint and one unchanged batch collection."""

    checkpoint_sha256: str
    data_manifest_sha256: str
    preprocess_fingerprint: str
    embeddings: dict
    batches: dict

    def validate(self, inputs, checkpoint_path, splits):
        if (self.checkpoint_sha256 != sha256_file(checkpoint_path)
                or self.data_manifest_sha256 != inputs.data_manifest_sha256
                or self.preprocess_fingerprint != inputs.preprocess_fingerprint):
            raise ValueError("Frozen Food embedding cache checkpoint/data/preprocessing mismatch")
        if set(self.embeddings) != set(self.batches):
            raise ValueError("Frozen Food embedding cache keys mismatch")
        for split in dict.fromkeys(splits):
            for batch in inputs.batches[split]:
                key = id(batch)
                if self.batches.get(key) is not batch or key not in self.embeddings:
                    raise ValueError("Frozen Food embedding cache missing batch or batch identity changed")
                _validate_food_embeddings(self.embeddings[key], batch, inputs.d_model)
                if any(value.requires_grad for value in self.embeddings[key].values()):
                    raise ValueError("Frozen Food embedding cache must contain detached tensors")


def prepare_frozen_embeddings(foundation, inputs, checkpoint_path, splits=("train", "validation")):
    """Encode selected splits once through ``embed_users``; leave test untouched by default.

    Callers must reuse the same FoodInputs objects. Cache bindings reject accidental
    reuse with another checkpoint, data manifest, preprocessing, or batch collection.
    """
    embeddings, batches = {}, {}
    previous_mode = foundation.training
    foundation.eval()
    try:
        with torch.no_grad():
            for split in dict.fromkeys(splits):
                if not inputs.batches.get(split):
                    raise ValueError(f"Food embedding cache has no batches for {split}")
                processed = 0
                for index, batch in enumerate(inputs.batches[split]):
                    values = embed_users_on_device(foundation, batch.foundation)
                    _validate_food_embeddings(values, batch, inputs.d_model)
                    embeddings[id(batch)] = {name: value.detach().clone() for name, value in values.items()}
                    batches[id(batch)] = batch
                    processed += len(batch.rows)
                    if (index + 1) % 100 == 0 or index + 1 == len(inputs.batches[split]):
                        print(f"embedding split={split} processed_requests={processed}", flush=True)
    finally:
        foundation.train(previous_mode)
    return FrozenFoodEmbeddings(sha256_file(checkpoint_path), inputs.data_manifest_sha256,
                                inputs.preprocess_fingerprint, embeddings, batches)


@dataclass
class FoodServices:
    load_checkpoint: object = load_food_foundation
    load_inputs: object = load_food_inputs
    writer: object = validate_and_write_predictions
    evaluator: object = None
    frozen_embeddings: FrozenFoodEmbeddings | None = None


def validate_food_candidates(rows):
    seen, identity, positives = set(), {}, {}
    for row in rows:
        for field in ("request_id", "user_id", "candidate_id"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"Food {field} must be canonical non-empty string")
        request, cutoff = row["request_id"], row["cutoff_timestamp"]
        if type(cutoff) is not int or type(row.get("is_positive")) not in (int, bool) or row["is_positive"] not in (0, 1):
            raise ValueError(f"request_id={request}: cutoff/positive invalid")
        split = row.get("split")
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"request_id={request}: invalid split")
        available_at = row.get("label_available_at_timestamp")
        if available_at is not None and (type(available_at) is not int or available_at <= cutoff):
            raise ValueError(f"request_id={request}: label availability must be after cutoff")
        key = (request, row["candidate_id"])
        if key in seen:
            raise ValueError(f"request_id={request}: duplicate Food candidate")
        seen.add(key)
        value = (row["user_id"], cutoff, split)
        if identity.setdefault(request, value) != value:
            raise ValueError(f"request_id={request}: identity/cutoff mismatch")
        positives[request] = positives.get(request, 0) + row["is_positive"]
        for field in ("candidate_timestamp", "exposure_timestamp", "label_timestamp", "profile_as_of_timestamp", "history_end_timestamp"):
            if row.get(field) is not None and (type(row[field]) is not int or row[field] > cutoff):
                raise ValueError(f"request_id={request}: future/invalid {field}")
    if not rows or any(number == 0 for number in positives.values()):
        raise ValueError("Food handoff requires a positive candidate in every request")


def run_food_recommendation(config, services=None):
    services = services or FoodServices()
    try:
        if config["contract_version"] != 1 or config["split"] not in {"validation", "test"}:
            raise ValueError("contract/split")
        if type(config["seed"]) is not int or type(config["training"]["epochs"]) is not int:
            raise ValueError("seed/epochs must be integers")
        if (config["training"]["epochs"] < 1 or config["training"]["learning_rate"] <= 0
                or not math.isfinite(config["training"]["learning_rate"])):
            raise ValueError("training epochs/learning_rate")
        training = config["training"]
        model_config = config.get("model", {})
        threads = training.get("cpu_threads", 1)
        hidden = model_config.get("hidden_dim", 64)
        candidate_dim = model_config.get("candidate_dim")
        dropout = model_config.get("dropout", 0.0)
        normalization = model_config.get("normalization", "none")
        architecture = model_config.get("architecture", "mlp")
        interaction_dim = model_config.get("interaction_dim")
        batch_size = training.get("batch_size", 1)
        weight_decay = training.get("weight_decay", 0.0)
        for name, number in (("cpu_threads", threads), ("hidden_dim", hidden), ("batch_size", batch_size)):
            if type(number) is not int or number < 1:
                raise ValueError(f"{name} must be a positive integer")
        if candidate_dim is not None and (type(candidate_dim) is not int or candidate_dim < 1):
            raise ValueError("candidate_dim must be a positive integer")
        if normalization not in {"none", "layernorm", "l2"}:
            raise ValueError("normalization must be none, layernorm, or l2")
        if architecture not in {"mlp", "mlp_dot"}:
            raise ValueError("architecture must be mlp or mlp_dot")
        if interaction_dim is not None and (type(interaction_dim) is not int or interaction_dim < 1):
            raise ValueError("interaction_dim must be a positive integer")
        if not isinstance(dropout, (int, float)) or not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        if not isinstance(weight_decay, (int, float)) or not math.isfinite(weight_decay) or weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        for name in ("cache_foundation_embeddings", "shuffle"):
            if type(training.get(name, False)) is not bool:
                raise ValueError(f"{name} must be boolean")
        if training.get("cache_foundation_embeddings", False) and not config.get("freeze_foundation", True):
            raise ValueError("cannot cache foundation embeddings while fine-tuning")
        names, attributes_dim = attribute_names(model_config), attribute_dim(model_config)
        device = encoder_device(training.get("encoder_device", "cpu"))
        if services.frozen_embeddings is not None and not config.get("freeze_foundation", True):
            raise ValueError("cannot use frozen embedding cache while fine-tuning")
        data_format = food_data_format(config)
        if names and data_format != "sim_v1":
            raise ValueError("model.candidate_attributes needs data.format sim_v1 (merchant catalog)")
        if data_format == "sim_v1":  # policy and catalog come from the package manifest
            for name, present in (("candidate_policy", "candidate_policy" in config),
                                  ("catalog", "catalog" in config),
                                  ("checkpoint.handoff", "handoff" in config["checkpoint"])):
                if present:
                    raise ValueError(f"{name} must be omitted for data.format sim_v1")
            required = (config["data"]["root"], config["data"]["expected_version"],
                        config["checkpoint"]["path"])
        else:
            required = (config["data"]["path"], config["data"]["expected_version"],
                        config["checkpoint"]["path"], config["candidate_policy"])
        for value in required:
            if not isinstance(value, str) or not value:
                raise ValueError("data/checkpoint/policy must be non-empty")
        init = foundation_init_arm(config)
        if services.frozen_embeddings is not None and init == "random":
            raise ValueError("cannot use frozen embedding cache with foundation.init random")
        if (not config["artifact_dir"] or type(config.get("freeze_foundation", True)) is not bool
                or type(config.get("export_model", False)) is not bool):
            raise ValueError("artifact_dir/freeze_foundation/export_model")
    except (ValueError, KeyError, TypeError) as error:
        raise FoodRunError(2, f"Food config invalid: {error}") from error
    identifier = str(uuid.uuid4())
    context = f"run_id={identifier} task=food_recommendation split={config['split']}"
    torch.manual_seed(config["seed"])
    torch.set_num_threads(config["training"].get("cpu_threads", 1))
    sim_manifest = None
    if data_format == "sim_v1":
        try:
            sim_manifest, _ = read_sim_food_manifest(
                config["data"]["root"], config["data"]["expected_version"])
        except (ValueError, OSError) as error:
            raise FoodRunError(3, f"{context}: {error}") from error
    policy = (config["candidate_policy"] if sim_manifest is None
              else sim_manifest["candidate_policy"])
    try:
        checkpoint_options = {"expected_version": config["checkpoint"].get(
            "expected_version", config["data"]["expected_version"])}
        if "handoff" in config["checkpoint"]:
            checkpoint_options["handoff"] = config["checkpoint"]["handoff"]
        if sim_manifest is not None:
            checkpoint_options["expected_sim_config_sha256"] = sim_manifest["sim_config_sha256"]
        foundation = services.load_checkpoint(config["checkpoint"]["path"], **checkpoint_options)
        if init == "random":  # control arm: checkpoint vocabulary and config, fresh weights
            foundation = reinitialize_foundation(foundation, config["seed"])
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as error:
        raise FoodRunError(4, f"{context}: {error}") from error
    try:
        inputs = services.load_inputs(config, foundation)
        validate_food_candidates(inputs.candidates)
        if inputs.manifest["dataset_version"] != config["data"]["expected_version"]:
            raise ValueError("Food data_version mismatch")
        if inputs.manifest["candidate_policy"] != policy:
            raise ValueError("Food candidate policy mismatch")
        vocabulary = {value: index + 1 for index, value in enumerate(sorted({
            row["candidate_id"] for row in inputs.candidates if row["split"] == "train"}))}
        if inputs.candidate_vocabulary != vocabulary or inputs.num_candidates != len(vocabulary) + 1:
            raise ValueError("Food vocabulary must fit train only")
        expected = {(row["request_id"], row["candidate_id"]): row for row in inputs.candidates}
        actual = {}
        for split, batches in inputs.batches.items():
            for batch in batches:
                for index, group in enumerate(batch.rows):
                    for column, row in enumerate(group):
                        key = (row["request_id"], row["candidate_id"])
                        if key in actual or row["split"] != split:
                            raise ValueError("Food batch duplicate/wrong split")
                        if int(batch.candidate_ids[index, column]) != vocabulary.get(row["candidate_id"], 0):
                            raise ValueError("Food candidate IDs differ from frozen vocabulary")
                        actual[key] = row
        if actual != expected or not all(inputs.batches.get(split) for split in ("train", "validation", config["split"])):
            raise ValueError("Food batches differ from complete handoff")
        if tuple(inputs.candidate_attributes) != names:
            raise ValueError("Food inputs candidate attributes differ from model.candidate_attributes")
        sizes = attribute_sizes(names, inputs.attribute_vocabularies) if names else ()
        for batches in inputs.batches.values():
            for batch in batches:
                check_candidate_attributes(batch.candidate_attributes, batch.candidate_ids, sizes)
    except (ValueError, KeyError, OSError) as error:
        raise FoodRunError(getattr(error, "code", 3), f"{context}: {error}") from error
    model_config = config.get("model", {})
    training_config = config["training"]
    # Checkpoint construction/loading must not change downstream initialization.
    torch.manual_seed(config["seed"])
    rank = FoodRanker(inputs.d_model, inputs.num_candidates, model_config.get("hidden_dim", 64),
                      model_config.get("candidate_dim"), model_config.get("dropout", 0.0),
                      model_config.get("normalization", "none"), model_config.get("architecture", "mlp"),
                      model_config.get("interaction_dim"), sizes, attributes_dim)
    foundation.to(device)  # only embed_users runs there; the head and batches stay on CPU
    freeze = config.get("freeze_foundation", True)
    for parameter in foundation.parameters():
        parameter.requires_grad_(not freeze)
    parameters = list(rank.parameters()) + ([] if freeze else list(foundation.parameters()))
    optimizer = head_optimizer(parameters, learning_rate=training_config["learning_rate"],
                               weight_decay=training_config.get("weight_decay", 0.0))
    cache = {}
    if services.frozen_embeddings is not None:
        try:
            services.frozen_embeddings.validate(inputs, config["checkpoint"]["path"],
                                                 ("train", "validation", config["split"]))
        except (ValueError, KeyError, TypeError, OSError) as error:
            raise FoodRunError(4, f"{context}: {error}") from error
        cache = services.frozen_embeddings.embeddings
    elif training_config.get("cache_foundation_embeddings", False):
        foundation.eval()
        with torch.no_grad():
            for split in ("train", "validation"):  # scored split is encoded after model selection
                processed = 0
                for index, batch in enumerate(inputs.batches[split]):
                    cache[id(batch)] = embed_users_on_device(foundation, batch.foundation)
                    processed += len(batch.rows)
                    if (index + 1) % 100 == 0 or index + 1 == len(inputs.batches[split]):
                        print(f"embedding split={split} processed_requests={processed}", flush=True)

    def embed(batch):
        if id(batch) in cache:
            return cache[id(batch)]
        return embed_users_on_device(foundation, batch.foundation)

    logs, best, best_loss, best_epoch, step = [], None, float("inf"), None, 0
    for epoch in range(config["training"]["epochs"]):
        for split in ("train", "validation"):
            training = split == "train"
            rank.train(training)
            foundation.train(training and not freeze)
            total, count = 0.0, 0
            batches = inputs.batches[split]
            if training_config.get("shuffle", False) and training:
                batches = [batches[index] for index in torch.randperm(len(batches)).tolist()]
            for batch in batches:
                labels, mask = batch.labels_and_mask()
                with torch.set_grad_enabled(training and not freeze):
                    embedding = embed(batch)
                if set(embedding) != {"long_term", "short_term"}:
                    raise FoodRunError(4, f"{context}: embed_users keys incompatible")
                with torch.set_grad_enabled(training):
                    scores = rank(embedding["long_term"], embedding["short_term"], batch.candidate_ids,
                                  batch.candidate_attributes)
                    loss = food_loss(scores, labels, mask)
                    if training:
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        optimizer.step()
                        step += 1
                total += float(loss.detach()) * len(batch.rows)
                count += len(batch.rows)
                if count % 5000 == 0:
                    print(f"epoch={epoch} split={split} processed_requests={count}", flush=True)
            logs.append({"epoch": epoch, "step": step, "split": split, "total_loss": total / count,
                         "listwise_loss": total / count, "target_counts": {"requests": count},
                         "seed": config["seed"], "timestamp_utc": utc_now()})
            print(f"epoch={epoch} split={split} loss={total / count:.6f} requests={count}", flush=True)
            if not training and total / count < best_loss:
                import copy
                best_loss = total / count
                best_epoch = epoch + 1
                best = (copy.deepcopy(rank.state_dict()), copy.deepcopy(foundation.state_dict()))
    rank.load_state_dict(best[0])
    foundation.load_state_dict(best[1])
    rank.eval()
    foundation.eval()
    predictions = []
    with torch.no_grad():
        for batch in inputs.batches[config["split"]]:
            embedding = embed(batch)
            scores = rank(embedding["long_term"], embedding["short_term"], batch.candidate_ids,
                          batch.candidate_attributes)
            for index, group in enumerate(batch.rows):
                for column, row in enumerate(group):
                    predictions.append({"schema_version": 1, "task": "food_recommendation",
                        "split": config["split"], "data_version": config["data"]["expected_version"],
                        "run_id": identifier, **{key: row[key] for key in
                        ("request_id", "user_id", "cutoff_timestamp", "candidate_id")},
                        "score": float(scores[index, column])})
    predictions.sort(key=lambda row: (row["request_id"], -row["score"], row["candidate_id"]))
    reference = {"checkpoint_path": config["checkpoint"]["path"],
                 "checkpoint_sha256": sha256_file(config["checkpoint"]["path"]),
                 "format_version": 1, "freeze_foundation": freeze, "init": init}
    if "handoff" in config["checkpoint"]:
        reference["handoff"] = dict(config["checkpoint"]["handoff"])
    selected = [row for row in inputs.candidates if row["split"] == config["split"]]
    manifest = {"contract_version": 1, "run_id": identifier, "task": "food_recommendation",
        "split": config["split"], "data_version": config["data"]["expected_version"],
        "data_manifest_sha256": inputs.data_manifest_sha256, "preprocess_fingerprint": inputs.preprocess_fingerprint,
        "config_sha256": fingerprint(config), "code_revision": code_revision(), "seed": config["seed"],
        "created_at_utc": utc_now(), "foundation": reference,
        "candidate_or_exposure_policy": policy,
        "row_count": len(predictions), "request_count": len({row["request_id"] for row in selected}),
        "positive_policy": "uniform target mass across all supplied positive candidates",
        "model_hyperparameters": {"hidden_dim": model_config.get("hidden_dim", 64),
                                  "candidate_dim": rank.candidates.embedding_dim,
                                  "dropout": model_config.get("dropout", 0.0),
                                  "normalization": rank.normalization,
                                  "normalization_scope": "long_term_and_short_term_only",
                                  "architecture": rank.architecture,
                                  "interaction_dim": rank.interaction_dim,
                                  **({"candidate_attributes": list(names),
                                      "attribute_dim": attributes_dim} if names else {})},
        "training_hyperparameters": dict(training_config),
        "selection": {"metric": "validation_listwise_loss", "best_epoch": best_epoch,
                      "best_validation_loss": best_loss}}
    if inputs.preprocess_metadata is not None:
        manifest["preprocessing"] = inputs.preprocess_metadata
    if services.frozen_embeddings is not None:
        manifest["frozen_embedding_cache"] = {
            "checkpoint_sha256": services.frozen_embeddings.checkpoint_sha256,
            "data_manifest_sha256": services.frozen_embeddings.data_manifest_sha256,
            "preprocess_fingerprint": services.frozen_embeddings.preprocess_fingerprint,
        }
    if config.get("evaluate", True):
        manifest["evaluation_policy"] = config.get("evaluation", {})
    if "handoff" in config["checkpoint"]:
        manifest["downstream_sequence"] = dict(config["sequence"])
    try:
        path = services.writer(pa.Table.from_pylist(predictions, schema=PREDICTION_SCHEMA),
            expected_candidates=selected, run_manifest=manifest, train_metrics=logs,
            checkpoint_ref=reference, artifact_dir=config["artifact_dir"])
    except (ValueError, OSError) as error:
        raise FoodRunError(5, f"{context}: {error}") from error
    model_artifact = None
    if config.get("export_model", False):
        model_artifact = path.parent / "food_ranker-v1.pt"
        payload = {
            "format_version": 1, "task": "food_recommendation", "run_id": identifier,
            "data_version": config["data"]["expected_version"],
            "data_manifest_sha256": inputs.data_manifest_sha256,
            "foundation_checkpoint_sha256": reference["checkpoint_sha256"],
            "freeze_foundation": freeze, "foundation_init": init, "seed": config["seed"],
            "best_validation_loss": best_loss, "best_epoch": best_epoch, "d_model": inputs.d_model,
            "hidden_dim": config.get("model", {}).get("hidden_dim", 64),
            "candidate_dim": rank.candidates.embedding_dim,
            "dropout": config.get("model", {}).get("dropout", 0.0),
            "normalization": rank.normalization,
            "architecture": rank.architecture,
            "interaction_dim": rank.interaction_dim,
            "preprocess_fingerprint": inputs.preprocess_fingerprint,
            "preprocess_metadata": inputs.preprocess_metadata,
            "candidate_vocabulary": inputs.candidate_vocabulary,
            "ranker_state_dict": rank.state_dict(),
        }
        if names:
            payload.update(candidate_attributes=list(names), attribute_dim=attributes_dim,
                           attribute_vocabularies=inputs.attribute_vocabularies)
        if not freeze:
            payload["foundation_state_dict"] = foundation.state_dict()
        descriptor, temporary = tempfile.mkstemp(prefix=".food-ranker-", suffix=".pt", dir=path.parent)
        os.close(descriptor)
        try:
            torch.save(payload, temporary)
            os.replace(temporary, model_artifact)
            load_food_ranker_checkpoint(model_artifact,
                expected_foundation_sha256=reference["checkpoint_sha256"],
                expected_data_version=config["data"]["expected_version"])
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    if not config.get("evaluate", True):
        return {"run_id": identifier, "predictions_path": str(path),
                "model_path": str(model_artifact) if model_artifact else None,
                "evaluation_status": "not_requested"}
    if services.evaluator is None:
        raise FoodRunError(6, f"{context}: public Baseline evaluator not supplied; predictions are validated")
    try:
        metrics = services.evaluator(predictions_path=str(path), task="food_recommendation",
                                     data_version=config["data"]["expected_version"], split=config["split"])
        if not isinstance(metrics, dict) or any(not isinstance(value, (int, float)) or not math.isfinite(value)
                                               for value in metrics.values()):
            raise ValueError("evaluator must return finite metrics")
    except (ValueError, RuntimeError, OSError, TypeError) as error:
        raise FoodRunError(6, f"{context}: evaluator failed after valid export: {error}") from error
    return {"run_id": identifier, "predictions_path": str(path),
            "model_path": str(model_artifact) if model_artifact else None, "metrics": metrics,
            **({"evaluation_status": "diagnostic"} if config.get("evaluation", {}).get(
                "observation_complete_policy") == "assume_complete_diagnostic" else {})}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    try:
        config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        services = FoodServices()
        if config.get("evaluate", True):
            try:
                services.evaluator = food_baseline_evaluator(config)
            except (ValueError, KeyError) as error:
                raise FoodRunError(3, f"Baseline evaluation dataset invalid: {error}") from error
            except (ImportError, AttributeError) as error:
                raise FoodRunError(6, f"public Baseline evaluator setup failed: {error}") from error
        print(json.dumps(run_food_recommendation(config, services), allow_nan=False))
        return 0
    except (ValueError, OSError, KeyError, TypeError, yaml.YAMLError) as error:
        print(f"task=food_recommendation: {error}", file=sys.stderr)
        return getattr(error, "code", 2)


if __name__ == "__main__":
    sys.exit(main())
