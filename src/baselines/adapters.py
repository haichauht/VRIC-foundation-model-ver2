"""Map documented JSON handoffs without inventing availability or labels."""

from datetime import datetime
from typing import Any

from .contracts import BaselineDataset, Candidate, ContractError, DatasetMetadata, Label, Query


def parse_time(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, AttributeError) as error:
        raise ContractError(f"invalid timestamp: {value!r}") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ContractError("timestamp requires explicit timezone offset")
    return parsed


def adapt_records(
    records: list[dict[str, Any]], metadata: DatasetMetadata, split: str,
    label_available_at: dict[tuple[str, str], datetime],
) -> BaselineDataset:
    """Availability mapping must come from an audited source, not prediction time.

    User profile/history are intentionally not mapped into business features here:
    their point-in-time mapping belongs to the feature/history handoff.
    """
    queries, candidates, labels = [], [], []
    try:
        for record in records:
            if metadata.task == "food_recommendation":
                data = record["input"]
                audit = record["label_metadata"]
                request_id = audit["request_id"]
                if audit.get("candidate_source") != metadata.candidate_policy:
                    raise ContractError(f"request={request_id}: candidate policy mismatch")
                if audit.get("observation_complete") is not True:
                    raise ContractError(f"request={request_id}: incomplete label observation")
                queries.append(Query(request_id, data["user_id"],
                                     parse_time(data["cutoff_time"]), split,
                                     dict(data.get("context", {}))))
                for item in data["candidates"]:
                    attributes = {key: value for key, value in item.items() if key != "item_id"}
                    candidates.append(Candidate(request_id, item["item_id"], attributes))
                for label in record["labels"]:
                    key = (request_id, label["item_id"])
                    labels.append(Label(*key, label["relevance"], label_available_at[key]))
            elif metadata.task == "ad_optimization":
                data = record["inputs"]
                request_id = record["impression_id"]
                item = data["ad"]
                key = (request_id, item["ad_id"])
                queries.append(Query(request_id, data["user"]["user_id"],
                                     parse_time(record["prediction_time"]), split,
                                     dict(data.get("context", {}))))
                candidates.append(Candidate(request_id, item["ad_id"],
                                            {k: v for k, v in item.items() if k != "ad_id"}))
                labels.append(Label(*key, record["labels"]["clicked"], label_available_at[key]))
            else:
                raise ContractError(f"unknown task: {metadata.task}")
    except (KeyError, TypeError) as error:
        raise ContractError(f"handoff missing/invalid field or label availability: {error}") from error
    dataset = BaselineDataset(metadata, tuple(queries), tuple(candidates), tuple(labels))
    dataset.validate()
    return dataset
