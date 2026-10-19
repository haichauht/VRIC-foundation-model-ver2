"""B1 binary classifier with train-only encoding and validation early stopping."""

import json
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pyarrow as pa

from ..evaluation.predictions import PREDICTION_SCHEMA, validate_predictions
from .contracts import BaselineDataset, ContractError, eligible_history
from .feature_schema import FeatureEncoder
from .features import FeatureBuilder


class B1Classifier:
    def __init__(self, *, seed: int = 42, n_estimators: int = 200,
                 learning_rate: float = 0.05, num_leaves: int = 31,
                 min_child_samples: int = 20, stopping_rounds: int = 20) -> None:
        if (type(seed) is not int or any(type(v) is not int or v <= 0 for v in
                (n_estimators, num_leaves, min_child_samples, stopping_rounds))
                or not 0 < learning_rate <= 1 or num_leaves < 2):
            raise ContractError("invalid B1 hyperparameters")
        self.config = {"seed": seed, "n_estimators": n_estimators, "learning_rate": learning_rate,
                       "num_leaves": num_leaves, "min_child_samples": min_child_samples,
                       "stopping_rounds": stopping_rounds}
        self.encoder = FeatureEncoder()
        self.booster = None
        self.identity: dict[str, str] = {}
        self.training_cutoff: datetime | None = None

    def fit(self, dataset: BaselineDataset, *, training_cutoff: datetime,
            history_start: datetime | None = None) -> "B1Classifier":
        dataset.validate()
        eligible_history((), training_cutoff)
        queries = {q.request_id: q for q in dataset.queries}
        train_queries = [q for q in dataset.queries if q.split == "train"]
        valid_queries = [q for q in dataset.queries if q.split == "validation"]
        if not train_queries or not valid_queries:
            raise ContractError("B1 requires train and validation queries")
        if (max(q.cutoff_time for q in train_queries) >= training_cutoff
                or min(q.cutoff_time for q in valid_queries) < training_cutoff):
            raise ContractError("train/validation are not chronological at training cutoff")
        for label in dataset.labels:
            if queries[label.request_id].split == "train" and label.available_at >= training_cutoff:
                raise ContractError("train label not available before training cutoff")
        builder = FeatureBuilder()
        train = builder.build(dataset, "train", history_start=history_start)
        valid = builder.build(dataset, "validation", frozen_cutoff=training_cutoff,
                              history_start=history_start)
        labels = {(label.request_id, label.candidate_id): label.target for label in dataset.labels}
        y_train = [labels[(row.request_id, row.candidate_id)] for row in train]
        y_valid = [labels[(row.request_id, row.candidate_id)] for row in valid]
        if set(y_train) != {0, 1}:
            raise ContractError("B1 training requires both target classes")
        encoder = FeatureEncoder().fit(train)
        params = {k: v for k, v in self.config.items()
                  if k not in {"n_estimators", "stopping_rounds"}}
        params.update(objective="binary", metric="binary_logloss", num_threads=1,
                      deterministic=True, force_col_wise=True, verbosity=-1)
        training = lgb.Dataset(encoder.transform(train), label=y_train,
                               categorical_feature=list(encoder.categorical_indices),
                               feature_name=list(encoder.output_columns))
        validation = lgb.Dataset(encoder.transform(valid), label=y_valid, reference=training)
        booster = lgb.train(params, training, num_boost_round=self.config["n_estimators"],
                            valid_sets=[validation],
                            callbacks=[lgb.early_stopping(self.config["stopping_rounds"],
                                                         verbose=False)])
        self.encoder, self.booster = encoder, booster
        self.training_cutoff = training_cutoff
        self.identity = {key: getattr(dataset.metadata, key)
                         for key in ("dataset_name", "task", "item_kind")}
        return self

    def predict(self, dataset: BaselineDataset, split: str, run_id: str, *,
                frozen_cutoff: datetime, history_start: datetime | None = None) -> pa.Table:
        if self.booster is None or self.training_cutoff is None:
            raise ContractError("B1 must be fitted before prediction")
        if split not in {"validation", "test"}:
            raise ContractError("prediction requires validation/test split")
        if any(getattr(dataset.metadata, key) != value for key, value in self.identity.items()):
            raise ContractError("B1 dataset namespace/task mismatch")
        eligible_history((), frozen_cutoff)
        if frozen_cutoff < self.training_cutoff:
            raise ContractError("prediction history cutoff precedes training cutoff")
        features = FeatureBuilder().build(dataset, split, frozen_cutoff=frozen_cutoff,
                                          history_start=history_start)
        scores = self.booster.predict(self.encoder.transform(features), num_threads=1)
        if not np.isfinite(scores).all() or not ((scores >= 0) & (scores <= 1)).all():
            raise ContractError("invalid B1 probabilities")
        queries = {q.request_id: q for q in dataset.queries}
        rows = []
        for row, score in zip(features, scores, strict=True):
            query = queries[row.request_id]
            rows.append({"schema_version": 1, "task": dataset.metadata.task, "split": split,
                         "data_version": dataset.metadata.data_version, "run_id": run_id,
                         "request_id": row.request_id, "user_id": query.user_id,
                         "cutoff_timestamp": int(query.cutoff_time.timestamp()),
                         "candidate_id": row.candidate_id, "score": float(score)})
        rows.sort(key=lambda row: (row["request_id"], -row["score"], row["candidate_id"]))
        table = pa.Table.from_pylist(rows, schema=PREDICTION_SCHEMA)
        validate_predictions(table, dataset, split)
        return table

    def save(self, directory: str | Path) -> None:
        if self.booster is None:
            raise ContractError("cannot save unfitted B1")
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=False)
        self.encoder.save(path / "encoder.json")
        self.booster.save_model(str(path / "model.txt"))
        (path / "metadata.json").write_text(json.dumps({
            "format_version": 1, "config": self.config, "identity": self.identity,
            "training_cutoff": self.training_cutoff.isoformat(),
            "lightgbm_version": lgb.__version__,
        }, allow_nan=False), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path) -> "B1Classifier":
        path = Path(directory)
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("format_version") != 1:
            raise ContractError("unsupported B1 artifact version")
        model = cls(**metadata["config"])
        model.encoder = FeatureEncoder.load(path / "encoder.json")
        model.booster = lgb.Booster(model_file=str(path / "model.txt"))
        if model.booster.feature_name() != list(model.encoder.output_columns):
            raise ContractError("B1 model/encoder feature mismatch")
        model.identity = metadata["identity"]
        model.training_cutoff = datetime.fromisoformat(metadata["training_cutoff"])
        eligible_history((), model.training_cutoff)
        return model
