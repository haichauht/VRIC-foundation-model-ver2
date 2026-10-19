"""Train-only vocabularies, numeric buckets, and deterministic metadata."""

from __future__ import annotations

import hashlib
import json
import math
from bisect import bisect_right
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .types import SPECIAL_TOKEN_IDS, KeyValueToken


@dataclass(frozen=True)
class VocabularyConfig:
    dataset_name: str
    max_values_per_key: int = 50_000
    numeric_bucket_count: int = 16
    text_hash_buckets: int = 4096
    fit_split: str = "train"
    source_schema_version: str = "omnibehavior_pretraining_v2"
    source_fingerprint: str | None = None
    include_action_content: bool = False
    numeric_transform_policy: dict[tuple[str, str], str] = field(default_factory=dict)
    time_gap_boundaries_seconds: tuple[float, ...] = ()


def _value_key(value: Any) -> str:
    if value is None:
        return "<MISSING>"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Numeric value must be finite, got {value!r}")
        return format(value, ".17g")
    return str(value)


def _namespace(dataset_name: str, token: KeyValueToken) -> str:
    return f"{dataset_name}:{token.source}:{token.modality}:{token.key}"


def _default_numeric_policy(key: str, values: list[float]) -> str:
    lowered = key.lower()
    skewed_markers = ("count", "_cnt", "duration", "price", "amount", "_amt")
    if all(value >= 0 for value in values) and any(marker in lowered for marker in skewed_markers):
        return "log1p_zscore"
    return "zscore"


def _tokenizer_contract(config: Any) -> dict[str, object]:
    return {
        "context_allowlist": {
            key: list(config.context_allowlist[key]) for key in sorted(config.context_allowlist)
        },
        "include_action_content": config.include_action_content,
        "max_text_chars": config.max_text_chars,
        "numeric_fields": [list(field) for field in sorted(config.numeric_fields)],
        "time_gap_boundaries_seconds": list(config.time_gap_boundaries_seconds),
    }


@dataclass
class VocabularyArtifacts:
    dataset_name: str
    value_to_id: dict[str, dict[str, int]]
    numeric_boundaries: dict[str, tuple[float, ...]]
    key_to_id: dict[str, int]
    modality_to_id: dict[str, int]
    source_to_id: dict[str, int]
    text_hash_buckets: int
    fit_split: str = "train"
    source_schema_version: str = "omnibehavior_pretraining_v2"
    source_fingerprint: str | None = None
    include_action_content: bool = False
    numeric_normalization: dict[str, tuple[float, float]] = field(default_factory=dict)
    numeric_field_registry: tuple[tuple[str, str], ...] = ()
    numeric_transform_policy: dict[str, str] = field(default_factory=dict)
    time_gap_boundaries_seconds: tuple[float, ...] = ()
    tokenizer_contract: dict[str, object] = field(default_factory=dict)

    def validate_tokenizer_config(self, config: Any) -> None:
        if self.tokenizer_contract and self.tokenizer_contract != _tokenizer_contract(config):
            raise ValueError("TokenizerConfig does not match vocabulary tokenizer contract")

    def model_cardinalities(self) -> dict[str, int]:
        value_max = max(SPECIAL_TOKEN_IDS.values(), default=0)
        for values in self.value_to_id.values():
            value_max = max(value_max, max(values.values(), default=0))
        for boundaries in self.numeric_boundaries.values():
            value_max = max(value_max, SPECIAL_TOKEN_IDS["MASK"] + 2 + len(boundaries))
        if self.text_hash_buckets:
            value_max = max(
                value_max,
                SPECIAL_TOKEN_IDS["MASK"] + 2 + self.text_hash_buckets - 1,
            )
        special_max = max(SPECIAL_TOKEN_IDS.values(), default=0)
        return {
            "value_vocab_size": value_max + 1,
            "key_vocab_size": max(special_max, max(self.key_to_id.values(), default=0)) + 1,
            "modality_vocab_size": max(special_max, max(self.modality_to_id.values(), default=0)) + 1,
            "source_vocab_size": max(special_max, max(self.source_to_id.values(), default=0)) + 1,
            "time_gap_vocab_size": max(special_max, len(self.time_gap_boundaries_seconds)) + 1,
            "text_hash_buckets": self.text_hash_buckets,
        }

    def normalize_numeric(self, token: KeyValueToken) -> tuple[float, bool]:
        namespace = _namespace(self.dataset_name, token)
        try:
            value = float(token.value)
        except (TypeError, ValueError):
            return 0.0, True
        if not math.isfinite(value):
            return 0.0, True
        center, scale = self.numeric_normalization.get(namespace, (0.0, 1.0))
        policy = self.numeric_transform_policy.get(namespace, "zscore")
        if policy == "log1p_zscore":
            if value < 0:
                return 0.0, True
            value = math.log1p(value)
        elif policy != "zscore":
            raise ValueError(f"Unsupported numeric transform policy: {policy!r}")
        return (value - center) / max(scale, 1e-6), False

    def encode_value(self, token: KeyValueToken) -> int:
        namespace = _namespace(self.dataset_name, token)
        if token.modality == "numeric":
            try:
                value = float(token.value)
            except (TypeError, ValueError):
                return SPECIAL_TOKEN_IDS["UNK"]
            if not math.isfinite(value) or namespace not in self.numeric_boundaries:
                return SPECIAL_TOKEN_IDS["UNK"]
            bucket = bisect_right(self.numeric_boundaries[namespace], value)
            return SPECIAL_TOKEN_IDS["MASK"] + 2 + bucket
        if token.modality == "text":
            digest = hashlib.sha256(f"{namespace}\0{_value_key(token.value)}".encode()).digest()
            return (
                SPECIAL_TOKEN_IDS["MASK"]
                + 2
                + int.from_bytes(digest[:8], "big") % self.text_hash_buckets
            )
        return self.value_to_id.get(namespace, {}).get(
            _value_key(token.value), SPECIAL_TOKEN_IDS["UNK"]
        )

    def encode_key(self, token: KeyValueToken) -> int:
        return self.key_to_id.get(_namespace(self.dataset_name, token), SPECIAL_TOKEN_IDS["UNK"])

    def encode_modality(self, token: KeyValueToken) -> int:
        return self.modality_to_id.get(token.modality, SPECIAL_TOKEN_IDS["UNK"])

    def encode_source(self, token: KeyValueToken) -> int:
        return self.source_to_id.get(token.source, SPECIAL_TOKEN_IDS["UNK"])

    def to_metadata(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "special_token_ids": SPECIAL_TOKEN_IDS,
            "value_to_id": self.value_to_id,
            "numeric_boundaries": {
                key: list(values) for key, values in self.numeric_boundaries.items()
            },
            "key_to_id": self.key_to_id,
            "modality_to_id": self.modality_to_id,
            "source_to_id": self.source_to_id,
            "text_hash_buckets": self.text_hash_buckets,
            "fit_split": self.fit_split,
            "source_schema_version": self.source_schema_version,
            "source_fingerprint": self.source_fingerprint,
            "include_action_content": self.include_action_content,
            "numeric_normalization": {
                key: [center, scale]
                for key, (center, scale) in self.numeric_normalization.items()
            },
            "numeric_field_registry": [list(item) for item in self.numeric_field_registry],
            "numeric_transform_policy": self.numeric_transform_policy,
            "time_gap_boundaries_seconds": list(self.time_gap_boundaries_seconds),
            "tokenizer_contract": self.tokenizer_contract,
        }

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.to_metadata(), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> VocabularyArtifacts:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("special_token_ids") != SPECIAL_TOKEN_IDS:
            raise ValueError("Vocabulary special token IDs do not match the contract")
        return cls(
            dataset_name=payload["dataset_name"],
            value_to_id={
                str(namespace): {str(value): int(value_id) for value, value_id in values.items()}
                for namespace, values in payload["value_to_id"].items()
            },
            numeric_boundaries={
                str(namespace): tuple(float(value) for value in values)
                for namespace, values in payload["numeric_boundaries"].items()
            },
            key_to_id={str(key): int(value) for key, value in payload["key_to_id"].items()},
            modality_to_id={
                str(key): int(value) for key, value in payload["modality_to_id"].items()
            },
            source_to_id={str(key): int(value) for key, value in payload["source_to_id"].items()},
            text_hash_buckets=int(payload["text_hash_buckets"]),
            fit_split=str(payload.get("fit_split", "train")),
            source_schema_version=str(
                payload.get("source_schema_version", "omnibehavior_pretraining_v2")
            ),
            source_fingerprint=payload.get("source_fingerprint"),
            include_action_content=bool(payload.get("include_action_content", False)),
            numeric_normalization={
                str(namespace): (float(values[0]), float(values[1]))
                for namespace, values in payload.get("numeric_normalization", {}).items()
            },
            numeric_field_registry=tuple(
                (str(source), str(key))
                for source, key in payload.get("numeric_field_registry", [])
            ),
            numeric_transform_policy={
                str(namespace): str(policy)
                for namespace, policy in payload.get("numeric_transform_policy", {}).items()
            },
            time_gap_boundaries_seconds=tuple(
                float(value) for value in payload.get("time_gap_boundaries_seconds", [])
            ),
            tokenizer_contract=dict(payload.get("tokenizer_contract", {})),
        )


def fit_vocabulary(
    train_tokens: Iterable[KeyValueToken],
    config: VocabularyConfig,
    *,
    numeric_field_registry: Iterable[tuple[str, str]] = (),
) -> VocabularyArtifacts:
    """Fit metadata from train tokens only; the iterable is never mutated."""

    if config.fit_split != "train":
        raise ValueError("Vocabulary fitting must use the train split")

    categorical_counts: dict[str, Counter[str]] = defaultdict(Counter)
    numeric_values: dict[str, list[float]] = defaultdict(list)
    keys: set[str] = set()
    modalities: set[str] = set()
    sources: set[str] = set()
    registry = tuple(sorted(set(numeric_field_registry)))
    for token in train_tokens:
        namespace = _namespace(config.dataset_name, token)
        keys.add(namespace)
        modalities.add(token.modality)
        sources.add(token.source)
        if token.modality == "numeric":
            try:
                value = float(token.value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                numeric_values[namespace].append(value)
        elif token.modality != "text":
            categorical_counts[namespace][_value_key(token.value)] += 1

    value_to_id: dict[str, dict[str, int]] = {}
    for namespace, counts in categorical_counts.items():
        ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        value_to_id[namespace] = {
            value: SPECIAL_TOKEN_IDS["MASK"] + 2 + index
            for index, (value, _) in enumerate(ordered[: config.max_values_per_key])
        }

    numeric_boundaries: dict[str, tuple[float, ...]] = {}
    for namespace, values in numeric_values.items():
        low, high = min(values), max(values)
        if low == high:
            numeric_boundaries[namespace] = (low,)
        else:
            step = (high - low) / config.numeric_bucket_count
            numeric_boundaries[namespace] = tuple(
                low + step * index for index in range(1, config.numeric_bucket_count)
            )

    numeric_normalization: dict[str, tuple[float, float]] = {}
    numeric_transform_policy: dict[str, str] = {}
    registry_set = set(registry)
    for namespace, values in numeric_values.items():
        _, source, _, key = namespace.split(":", 3)
        field_key = (source, key)
        policy = config.numeric_transform_policy.get(field_key)
        if policy is None:
            policy = _default_numeric_policy(key, values)
        if policy not in {"zscore", "log1p_zscore"}:
            raise ValueError(f"Unsupported numeric transform policy: {policy!r}")
        transformed = [math.log1p(value) for value in values] if policy == "log1p_zscore" else values
        center = sum(transformed) / len(transformed)
        variance = sum((value - center) ** 2 for value in transformed) / len(transformed)
        numeric_normalization[namespace] = (center, math.sqrt(variance) or 1.0)
        numeric_transform_policy[namespace] = policy
        registry_set.add(field_key)

    return VocabularyArtifacts(
        dataset_name=config.dataset_name,
        value_to_id=value_to_id,
        numeric_boundaries=numeric_boundaries,
        key_to_id={key: index for index, key in enumerate(sorted(keys), start=1)},
        modality_to_id={value: index for index, value in enumerate(sorted(modalities), start=1)},
        source_to_id={value: index for index, value in enumerate(sorted(sources), start=1)},
        text_hash_buckets=config.text_hash_buckets,
        fit_split=config.fit_split,
        source_schema_version=config.source_schema_version,
        source_fingerprint=config.source_fingerprint,
        include_action_content=config.include_action_content,
        numeric_normalization=numeric_normalization,
        numeric_field_registry=tuple(sorted(registry_set)),
        numeric_transform_policy=numeric_transform_policy,
        time_gap_boundaries_seconds=config.time_gap_boundaries_seconds,
    )


def fit_vocabulary_from_samples(
    samples: Iterable[Any],
    config: VocabularyConfig,
    tokenizer_config: Any | None = None,
) -> VocabularyArtifacts:
    """Fit a vocabulary from canonical train samples and reject other splits."""

    if config.fit_split != "train":
        raise ValueError("Vocabulary fitting must use the train split")
    from .tokenizer import TokenizerConfig, semantic_tokens

    token_config = tokenizer_config or TokenizerConfig()
    tokens: list[KeyValueToken] = []
    for sample in samples:
        if sample.split != config.fit_split:
            raise ValueError(
                f"Vocabulary fitting requires the train split; got {sample.split!r}"
            )
        if sample.dataset_name != config.dataset_name:
            raise ValueError(
                f"Vocabulary dataset {config.dataset_name!r} does not match "
                f"sample dataset {sample.dataset_name!r}"
            )
        tokens.extend(semantic_tokens(sample, token_config))
    config_policy = config.numeric_transform_policy
    token_policy = token_config.numeric_transform_policy
    if config_policy and token_policy and config_policy != token_policy:
        raise ValueError("VocabularyConfig and TokenizerConfig numeric transform policies differ")
    config = replace(
        config,
        numeric_transform_policy=token_policy or config_policy,
        time_gap_boundaries_seconds=tuple(token_config.time_gap_boundaries_seconds),
        include_action_content=bool(token_config.include_action_content),
    )
    artifacts = fit_vocabulary(
        tokens,
        config,
        numeric_field_registry=token_config.numeric_fields,
    )
    return replace(artifacts, tokenizer_contract=_tokenizer_contract(token_config))
