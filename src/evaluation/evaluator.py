"""Shared evaluator; dataset resolution is explicit, never filename-based."""

import pyarrow.parquet as pq

from ..baselines.contracts import BaselineDataset, ContractError
from .metrics import advertising_metrics, ranking_metrics
from .predictions import validate_predictions


class DatasetRegistry:
    """Inject audited handoffs before calling the public evaluation API."""

    def __init__(self) -> None:
        self._datasets: dict[tuple[str, str], BaselineDataset] = {}

    def register(self, dataset: BaselineDataset) -> None:
        dataset.validate()
        key = (dataset.metadata.task, dataset.metadata.data_version)
        if key in self._datasets:
            raise ContractError(f"dataset version already registered: {key}")
        self._datasets[key] = dataset

    def resolve(self, task: str, version: str) -> BaselineDataset:
        try:
            return self._datasets[(task, version)]
        except KeyError as error:
            raise ContractError(f"unregistered dataset: {task}/{version}") from error


DEFAULT_REGISTRY = DatasetRegistry()


def evaluate_predictions(
    predictions_path: str, task: str, data_version: str, split: str,
    *, registry: DatasetRegistry | None = None, ks: tuple[int, ...] = (10,),
    zero_positive_policy: str = "include",
) -> dict[str, float]:
    if zero_positive_policy not in {"include", "exclude"}:
        raise ContractError("zero_positive_policy must be include or exclude")
    dataset = (registry if registry is not None else DEFAULT_REGISTRY).resolve(task, data_version)
    rows = validate_predictions(pq.read_table(predictions_path), dataset, split)
    labels = {(label.request_id, label.candidate_id): label.target for label in dataset.labels}
    if task == "ad_optimization":
        result = advertising_metrics([labels[(r["request_id"], r["candidate_id"])] for r in rows],
                                     [r["score"] for r in rows])
        groups = {}
        for row in rows:
            groups.setdefault(row['request_id'], []).append(
                (row['candidate_id'], labels[(row['request_id'], row['candidate_id'])], row['score']))
        ranking = [ranking_metrics(group, ks) for group in groups.values()
                   if any(label for _, label, _ in group)]
        result['clicked_request_count'] = float(len(ranking))
        for k in ks:
            for prefix in ('ndcg', 'recall'):
                key = f'{prefix}_at_{k}'
                result[key] = (sum(metric[key] for metric in ranking) / len(ranking)
                               if ranking else float('nan'))
        return result
    groups: dict[str, list[tuple[str, int, float]]] = {}
    for row in rows:
        groups.setdefault(row["request_id"], []).append(
            (row["candidate_id"], labels[(row["request_id"], row["candidate_id"])], row["score"]))
    zero_count = sum(not any(row[1] for row in group) for group in groups.values())
    metrics = [ranking_metrics(group, ks) for group in groups.values()
               if zero_positive_policy == "include" or any(row[1] for row in group)]
    if not metrics:
        raise ContractError("no evaluable requests after zero-positive exclusion")
    result = {name: sum(metric[name] for metric in metrics) / len(metrics)
              for name in metrics[0]}
    result.update(request_count=float(len(groups)), evaluated_request_count=float(len(metrics)),
                  zero_positive_request_count=float(zero_count))
    return result
