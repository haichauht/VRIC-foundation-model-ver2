"""Frozen B0 scorers; fitting never consumes evaluation labels."""

import hashlib
import json
from collections import Counter
from datetime import datetime, timedelta
from math import isfinite
from pathlib import Path

import pyarrow as pa

from ..evaluation.predictions import PREDICTION_SCHEMA
from .contracts import BaselineDataset, Candidate, ContractError, Query, eligible_history


class B0Scorer:
    """Fit one frozen state per split boundary, then score future queries.

    Food counts raw order interactions. CTR counts labeled train impressions
    whose labels were available strictly before the fit boundary.
    """

    FOOD = frozenset({"random", "popularity", "meal_popularity", "recent"})
    ADS = frozenset({"global_ctr", "ad_ctr", "category_ctr"})

    def __init__(self, variant: str, *, seed: int = 42, window_days: int = 7,
                 alpha: float = 10.0, empty_prior: float = 0.5) -> None:
        if variant not in self.FOOD | self.ADS:
            raise ContractError(f"unknown B0 variant: {variant}")
        if type(seed) is not int or type(window_days) is not int or window_days <= 0:
            raise ContractError("seed must be integer; window_days must be positive integer")
        if not isfinite(alpha) or alpha <= 0 or not isfinite(empty_prior) or not 0 <= empty_prior <= 1:
            raise ContractError("positive finite alpha and empty_prior in [0,1] required")
        self.variant, self.seed, self.window_days = variant, seed, window_days
        self.alpha, self.empty_prior = float(alpha), float(empty_prior)
        self.state: dict = {}

    def fit(self, dataset: BaselineDataset, cutoff: datetime) -> "B0Scorer":
        dataset.validate()
        history = eligible_history(dataset.history, cutoff)
        task = dataset.metadata.task
        if ((task == "food_recommendation" and self.variant not in self.FOOD)
                or (task == "ad_optimization" and self.variant not in self.ADS)):
            raise ContractError("B0 variant/task mismatch")
        state = {"dataset_name": dataset.metadata.dataset_name, "task": task,
                 "item_kind": dataset.metadata.item_kind, "cutoff": cutoff.isoformat(),
                 "counts": {}, "meal": {}, "impressions": {}, "clicks": {},
                 "category_impressions": {}, "category_clicks": {}, "prior": self.empty_prior}
        if task == "food_recommendation":
            orders = [event for event in history if event.event_type == "order"]
            if self.variant == "recent":
                start = cutoff - timedelta(days=self.window_days)
                orders = [event for event in orders if event.event_time >= start]
            state["counts"] = dict(Counter(event.entity_id for event in orders))
            meal: dict[str, Counter] = {}
            for event in orders:
                period = event.attributes.get("meal_period")
                if period is not None:
                    meal.setdefault(str(period), Counter())[event.entity_id] += 1
            state["meal"] = {key: dict(counts) for key, counts in meal.items()}
        else:
            queries = {q.request_id: q for q in dataset.queries}
            candidates = {(c.request_id, c.candidate_id): c for c in dataset.candidates}
            total, clicks = 0, 0
            for label in dataset.labels:
                query = queries[label.request_id]
                if (query.split != "train" or query.cutoff_time >= cutoff
                        or label.available_at >= cutoff):
                    continue
                candidate = candidates[(label.request_id, label.candidate_id)]
                item = candidate.candidate_id
                for name, increment in (("impressions", 1), ("clicks", label.target)):
                    state[name][item] = state[name].get(item, 0) + increment
                category = candidate.attributes.get("category")
                if category is not None:
                    for name, increment in (("category_impressions", 1),
                                            ("category_clicks", label.target)):
                        key = str(category)
                        state[name][key] = state[name].get(key, 0) + increment
                total += 1
                clicks += label.target
            state["prior"] = clicks / total if total else self.empty_prior
        self.state = state
        return self

    def score(self, query: Query, candidate: Candidate) -> float:
        if not self.state:
            raise ContractError("B0 must be fitted before scoring")
        if query.request_id != candidate.request_id:
            raise ContractError("query/candidate request mismatch")
        # Also validates timezone without inferring it.
        eligible_history((), query.cutoff_time)
        if query.cutoff_time < datetime.fromisoformat(self.state["cutoff"]):
            raise ContractError("query precedes frozen fit cutoff")
        item = candidate.candidate_id
        if self.variant == "random":
            payload = json.dumps([self.seed, self.state["dataset_name"], query.request_id, item])
            value = int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big")
            return (value >> 11) / (1 << 53)
        if self.variant in {"popularity", "recent"}:
            return float(self.state["counts"].get(item, 0))
        if self.variant == "meal_popularity":
            counts = self.state["meal"].get(str(query.context.get("meal_period")), {})
            return float(counts.get(item, self.state["counts"].get(item, 0)))
        prior = self.state["prior"]
        if self.variant == "global_ctr":
            return float(prior)
        category = candidate.attributes.get("category")
        key = str(category)
        category_prior = (
            self.state["category_clicks"].get(key, 0) + self.alpha * prior
        ) / (self.state["category_impressions"].get(key, 0) + self.alpha)
        if self.variant == "category_ctr":
            return float(category_prior)
        return float((self.state["clicks"].get(item, 0) + self.alpha * category_prior)
                     / (self.state["impressions"].get(item, 0) + self.alpha))

    def predict(self, dataset: BaselineDataset, split: str, run_id: str) -> pa.Table:
        dataset.validate()
        if not self.state or any(self.state[key] != getattr(dataset.metadata, key)
                                 for key in ("dataset_name", "task", "item_kind")):
            raise ContractError("unfitted scorer or dataset namespace/task mismatch")
        if split not in {"validation", "test"} or not isinstance(run_id, str) or not run_id.strip():
            raise ContractError("prediction requires validation/test split and non-empty run_id")
        queries = {q.request_id: q for q in dataset.queries if q.split == split}
        rows = []
        for candidate in dataset.candidates:
            if candidate.request_id not in queries:
                continue
            query = queries[candidate.request_id]
            if query.cutoff_time.microsecond:
                raise ContractError("Output Contract v1 requires cutoff precision in seconds")
            rows.append({"schema_version": 1, "task": dataset.metadata.task, "split": split,
                         "data_version": dataset.metadata.data_version, "run_id": run_id,
                         "request_id": query.request_id, "user_id": query.user_id,
                         "cutoff_timestamp": int(query.cutoff_time.timestamp()),
                         "candidate_id": candidate.candidate_id,
                         "score": self.score(query, candidate)})
        if not rows:
            raise ContractError("requested split has no candidates")
        rows.sort(key=lambda row: (row["request_id"], -row["score"], row["candidate_id"]))
        return pa.Table.from_pylist(rows, schema=PREDICTION_SCHEMA)

    def save(self, path: str | Path) -> None:
        if not self.state:
            raise ContractError("cannot save unfitted scorer")
        payload = {"format_version": 1, "variant": self.variant, "seed": self.seed,
                   "window_days": self.window_days, "alpha": self.alpha,
                   "empty_prior": self.empty_prior, "state": self.state}
        Path(path).write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "B0Scorer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format_version") != 1:
            raise ContractError("unsupported B0 artifact version")
        model = cls(**{key: payload[key] for key in
                       ("variant", "seed", "window_days", "alpha", "empty_prior")})
        model.state = payload["state"]
        return model
