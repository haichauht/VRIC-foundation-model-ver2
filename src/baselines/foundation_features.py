"""Foundation user embeddings as extra B1 features ("B1 + foundation", Grab-style evaluation).

Embeddings come from ``python -m src.downstream.user_embeddings``: one row per request, encoded
from history strictly before that request's cutoff. Everything below is fitted on train only:

* ``foundation_pca_<k>``: the first k principal components of ``[long_term, short_term]``.
* ``foundation_proto_similarity``: cosine between the request's unit embedding and the
  candidate's prototype, the mean unit embedding of the train requests in which that candidate
  is positive. A train row leaves its own request out; no prototype gives a missing value.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from .contracts import BaselineDataset, ContractError
from .features import FeatureRow

STATE_FILE = "foundation_features.npz"
METADATA_FILE = "foundation_features.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class FoundationFeatures:
    def __init__(self, embeddings_path: str | Path, *, pca_components: int = 32,
                 prototype_similarity: bool = True) -> None:
        if type(pca_components) is not int or pca_components < 0:
            raise ContractError("foundation_features.pca_components must be a nonnegative integer")
        if type(prototype_similarity) is not bool:
            raise ContractError("foundation_features.prototype_similarity must be boolean")
        if pca_components == 0 and not prototype_similarity:
            raise ContractError("foundation_features adds no feature")
        self.path = Path(embeddings_path).resolve()
        if not self.path.is_file():
            raise ContractError(f"foundation embeddings {self.path} not found")
        self.sha256 = _sha256(self.path)
        with np.load(self.path, allow_pickle=False) as data:
            request_ids = [str(value) for value in data["request_id"]]
            vectors = np.concatenate([data["long_term"], data["short_term"]], 1).astype(np.float64)
        if (vectors.ndim != 2 or len(request_ids) != len(vectors)
                or len(set(request_ids)) != len(request_ids) or not np.isfinite(vectors).all()):
            raise ContractError("foundation embeddings need unique request ids and finite vectors")
        self.index = {request: row for row, request in enumerate(request_ids)}
        self.vectors = vectors
        self.units = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
        sidecar = self.path.with_suffix(".json")
        self.provenance = (json.loads(sidecar.read_text(encoding="utf-8"))
                           if sidecar.is_file() else None)
        self.pca_components = pca_components
        self.prototype_similarity = prototype_similarity
        self.mean = self.components = None
        self.prototypes: dict[str, tuple[np.ndarray, int]] = {}
        self.train_positives: set[tuple[str, str]] = set()

    def _row(self, request_id: str) -> int:
        if request_id not in self.index:
            raise ContractError(f"request={request_id}: no foundation embedding")
        return self.index[request_id]

    def fit(self, dataset: BaselineDataset) -> "FoundationFeatures":
        for query in dataset.queries:
            self._row(query.request_id)
        train = {q.request_id for q in dataset.queries if q.split == "train"}
        rows = np.array([self.index[request] for request in sorted(train)])
        if self.pca_components:
            if len(rows) < 2:
                raise ContractError("foundation PCA needs at least two train requests")
            matrix = self.vectors[rows]
            self.mean = matrix.mean(0)
            _, _, vt = np.linalg.svd(matrix - self.mean, full_matrices=False)
            self.components = vt[:self.pca_components]
        self.prototypes, self.train_positives = {}, set()
        if self.prototype_similarity:
            for label in dataset.labels:
                if label.request_id in train and label.target > 0:
                    self.train_positives.add((label.request_id, label.candidate_id))
            for request, candidate in sorted(self.train_positives):
                total, count = self.prototypes.get(candidate, (0.0, 0))
                self.prototypes[candidate] = (total + self.units[self.index[request]], count + 1)
        return self

    @property
    def names(self) -> tuple[str, ...]:
        names = tuple(f"foundation_pca_{k}" for k in range(
            0 if self.components is None else len(self.components)))
        return names + (("foundation_proto_similarity",) if self.prototype_similarity else ())

    def _similarity(self, request_id: str, candidate_id: str, split: str) -> float | None:
        if candidate_id not in self.prototypes:
            return None
        unit = self.units[self.index[request_id]]
        total, count = self.prototypes[candidate_id]
        if split == "train" and (request_id, candidate_id) in self.train_positives:
            total, count = total - unit, count - 1
        if count < 1:
            return None
        norm = float(np.linalg.norm(total))
        return None if norm < 1e-12 else float(unit @ total / norm)

    def extend(self, rows: tuple[FeatureRow, ...]) -> tuple[FeatureRow, ...]:
        if self.pca_components and self.components is None:
            raise ContractError("foundation features must be fitted before use")
        requests = sorted({row.request_id for row in rows})
        projected = {}
        if self.components is not None and requests:
            matrix = self.vectors[[self._row(request) for request in requests]] - self.mean
            projected = dict(zip(requests, matrix @ self.components.T, strict=True))
        out = []
        for row in rows:
            extra = {}
            if self.components is not None:
                extra.update({f"foundation_pca_{k}": float(value)
                              for k, value in enumerate(projected[row.request_id])})
            if self.prototype_similarity:
                self._row(row.request_id)
                extra["foundation_proto_similarity"] = self._similarity(
                    row.request_id, row.candidate_id, row.split)
            if set(extra) & set(row.values):
                raise ContractError("foundation feature names collide with B1 features")
            out.append(FeatureRow(row.request_id, row.candidate_id, row.split,
                                  {**row.values, **extra}))
        return tuple(out)

    def save(self, directory: Path) -> None:
        candidates = sorted(self.prototypes)
        np.savez(directory / STATE_FILE,
                 mean=np.zeros(0) if self.mean is None else self.mean,
                 components=np.zeros((0, 0)) if self.components is None else self.components,
                 prototype_candidates=np.array(candidates, dtype=str),
                 prototype_sums=np.array([self.prototypes[c][0] for c in candidates]).reshape(
                     len(candidates), -1),
                 prototype_counts=np.array([self.prototypes[c][1] for c in candidates], dtype=np.int64),
                 train_positives=np.array(sorted(self.train_positives), dtype=str).reshape(-1, 2))
        (directory / METADATA_FILE).write_text(json.dumps({
            "format_version": 1, "embeddings_path": str(self.path), "embeddings_sha256": self.sha256,
            "pca_components": self.pca_components,
            "prototype_similarity": self.prototype_similarity, "provenance": self.provenance,
        }, allow_nan=False), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> "FoundationFeatures":
        metadata = json.loads((directory / METADATA_FILE).read_text(encoding="utf-8"))
        if metadata.get("format_version") != 1:
            raise ContractError("unsupported foundation feature artifact version")
        features = cls(metadata["embeddings_path"], pca_components=metadata["pca_components"],
                       prototype_similarity=metadata["prototype_similarity"])
        if features.sha256 != metadata["embeddings_sha256"]:
            raise ContractError("foundation embeddings changed since the B1 model was fitted")
        with np.load(directory / STATE_FILE, allow_pickle=False) as state:
            if state["components"].size:
                features.mean, features.components = state["mean"], state["components"]
            features.prototypes = {
                str(candidate): (total, int(count)) for candidate, total, count in zip(
                    state["prototype_candidates"], state["prototype_sums"], state["prototype_counts"],
                    strict=True)}
            features.train_positives = {(str(a), str(b)) for a, b in state["train_positives"]}
        return features
