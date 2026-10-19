"""Static candidate attributes (catalog side information) for the downstream heads.

``model.candidate_attributes`` names catalog columns (Food ``merchants.parquet``: e.g. cuisine,
price_level, geohash; Ads ``ad_catalog.parquet``: e.g. category, advertiser). Each attribute gets
its own vocabulary fitted on the catalog rows of train candidates; 0 = missing or unseen value.
Catalog attributes are request-independent metadata, so a candidate keeps its attributes even
when its ID is outside the train-fitted ID vocabulary.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence

import torch


def attribute_names(model_config: Mapping) -> tuple[str, ...]:
    """Validated ``model.candidate_attributes`` (empty tuple when absent)."""
    names = model_config.get("candidate_attributes", [])
    if (not isinstance(names, list) or len(set(names)) != len(names)
            or not all(isinstance(name, str) and name.strip() for name in names)):
        raise ValueError("model.candidate_attributes must be a list of unique column names")
    return tuple(names)


def attribute_dim(model_config: Mapping) -> int:
    value = model_config.get("attribute_dim", 8)
    if type(value) is not int or value < 1:
        raise ValueError("model.attribute_dim must be a positive integer")
    return value


def _key(value) -> str | None:
    return None if value is None else json.dumps(value, ensure_ascii=False, sort_keys=True)


def fit_attribute_vocabularies(catalog: Mapping[str, Mapping], names: Sequence[str],
                               train_candidates: Iterable[str]) -> dict[str, dict[str, int]]:
    """Per-attribute value vocabularies over the catalog rows of train candidates (1-based)."""
    rows = []
    for candidate in sorted(set(train_candidates)):
        if candidate not in catalog:
            raise ValueError(f"candidate {candidate!r} missing from catalog")
        rows.append(catalog[candidate])
    vocabularies = {}
    for name in names:
        if not any(name in row for row in catalog.values()):
            raise ValueError(f"candidate attribute {name!r} is not a catalog column")
        values = sorted({key for row in rows if (key := _key(row.get(name))) is not None})
        vocabularies[name] = {value: index + 1 for index, value in enumerate(values)}
    return vocabularies


def attribute_sizes(names: Sequence[str], vocabularies: Mapping[str, Mapping[str, int]]):
    """Embedding table sizes (vocabulary + the 0 = missing/unseen row)."""
    return tuple(len(vocabularies[name]) + 1 for name in names)


def encode_attributes(rows: Sequence[Sequence[Mapping]], catalog: Mapping[str, Mapping],
                      names: Sequence[str], vocabularies: Mapping[str, Mapping[str, int]],
                      width: int) -> torch.LongTensor:
    """``[B, width, K]`` attribute IDs for request rows (padding columns stay 0)."""
    values = torch.zeros((len(rows), width, len(names)), dtype=torch.long)
    for index, request in enumerate(rows):
        for column, row in enumerate(request):
            entry = catalog.get(row["candidate_id"])
            if entry is None:
                raise ValueError(f"request_id={row['request_id']}: candidate missing from catalog")
            for position, name in enumerate(names):
                values[index, column, position] = vocabularies[name].get(_key(entry.get(name)), 0)
    return values


def validate_attribute_vocabularies(names: Sequence[str], vocabularies) -> None:
    """Shape check for vocabularies restored from an exported head."""
    if not isinstance(vocabularies, dict) or list(vocabularies) != list(names):
        raise ValueError("candidate attribute vocabularies do not match attribute names")
    for vocabulary in vocabularies.values():
        if (not isinstance(vocabulary, dict)
                or not all(isinstance(key, str) and type(value) is int for key, value in vocabulary.items())
                or set(vocabulary.values()) != set(range(1, len(vocabulary) + 1))):
            raise ValueError("candidate attribute vocabulary invalid")


def check_candidate_attributes(attributes, candidate_ids: torch.Tensor, sizes: Sequence[int]) -> None:
    """Heads with attribute tables need ``[B, C, K]`` long IDs in range; others need ``None``."""
    if not sizes:
        if attributes is not None:
            raise ValueError("candidate_attributes given to a head without attribute tables")
        return
    if (not isinstance(attributes, torch.Tensor) or attributes.dtype != torch.long
            or attributes.shape != (*candidate_ids.shape, len(sizes))
            or attributes.device != candidate_ids.device):
        raise ValueError("candidate_attributes expected torch.long [B,C,K] on candidate device")
    bounds = torch.tensor(tuple(sizes), device=attributes.device)
    if ((attributes < 0) | (attributes >= bounds)).any():
        raise ValueError("candidate attribute ID outside train-fitted vocabulary")
