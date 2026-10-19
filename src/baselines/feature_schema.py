"""Train-only encoding, fixed columns, explicit missing/unknown values."""

import json
from math import isfinite
from pathlib import Path

import numpy as np

from .contracts import ContractError
from .features import FeatureRow

CATEGORICAL = frozenset({"context_meal_period", "context_placement", "context_zone",
                         "item_category", "item_brand"})


class FeatureEncoder:
    """Numeric values use zero plus missing indicator; category 0=missing, 1=OOV."""

    def __init__(self) -> None:
        self.columns: tuple[str, ...] = ()
        self.vocabularies: dict[str, dict[str, int]] = {}

    @staticmethod
    def _category(value: object) -> str:
        if not isinstance(value, (str, int, float, bool)):
            raise ContractError("categorical feature must be scalar")
        if isinstance(value, float) and not isfinite(value):
            raise ContractError("categorical feature must be finite")
        return json.dumps(value, ensure_ascii=False)

    def fit(self, rows: tuple[FeatureRow, ...]) -> "FeatureEncoder":
        if not rows or any(row.split != "train" for row in rows):
            raise ContractError("feature encoder fits non-empty train rows only")
        columns = tuple(sorted(rows[0].values))
        if not columns or any(set(row.values) != set(columns) for row in rows):
            raise ContractError("inconsistent feature schema")
        self.columns = columns
        self.vocabularies = {}
        for key in columns:
            if key in CATEGORICAL:
                observed = sorted({self._category(row.values[key]) for row in rows
                                   if row.values[key] is not None})
                self.vocabularies[key] = {value: i + 2 for i, value in enumerate(observed)}
        self.transform(rows)
        return self

    @property
    def categorical_indices(self) -> tuple[int, ...]:
        return tuple(i for i, name in enumerate(self.output_columns) if name in CATEGORICAL)

    @property
    def output_columns(self) -> tuple[str, ...]:
        return tuple(name for key in self.columns for name in
                     ((key,) if key in CATEGORICAL else (key, f"{key}__missing")))

    def transform(self, rows: tuple[FeatureRow, ...]) -> np.ndarray:
        if not self.columns:
            raise ContractError("encoder must be fitted")
        matrix = []
        for row in rows:
            if set(row.values) != set(self.columns):
                raise ContractError(f"request={row.request_id}: feature schema mismatch")
            encoded = []
            for key in self.columns:
                value = row.values[key]
                if key in CATEGORICAL:
                    encoded.append(0 if value is None else
                                   self.vocabularies[key].get(self._category(value), 1))
                else:
                    if value is not None and not isinstance(value, (int, float)):
                        raise ContractError(f"{key}: numeric scalar required")
                    missing = value is None or not isfinite(value)
                    encoded.extend((0.0 if missing else float(value), float(missing)))
            matrix.append(encoded)
        return np.asarray(matrix, dtype=np.float64).reshape(len(rows), len(self.output_columns))

    def save(self, path: str | Path) -> None:
        if not self.columns:
            raise ContractError("cannot save unfitted encoder")
        Path(path).write_text(json.dumps({"format_version": 1, "columns": self.columns,
                                         "vocabularies": self.vocabularies}, allow_nan=False),
                              encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "FeatureEncoder":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("format_version") != 1:
            raise ContractError("unsupported feature encoder artifact version")
        model = cls()
        model.columns = tuple(data["columns"])
        model.vocabularies = data["vocabularies"]
        if not model.columns or set(model.vocabularies) != set(model.columns) & CATEGORICAL:
            raise ContractError("invalid encoder schema/vocabularies")
        return model
