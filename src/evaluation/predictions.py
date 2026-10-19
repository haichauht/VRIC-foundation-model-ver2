"""Exact-set Output Contract v1 validation and atomic Parquet export."""

import os
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ..baselines.contracts import BaselineDataset, ContractError, validate_score

PREDICTION_SCHEMA = pa.schema([
    ("schema_version", pa.int32()), ("task", pa.string()), ("split", pa.string()),
    ("data_version", pa.string()), ("run_id", pa.string()), ("request_id", pa.string()),
    ("user_id", pa.string()), ("cutoff_timestamp", pa.int64()),
    ("candidate_id", pa.string()), ("score", pa.float64()),
])


def validate_predictions(
    table: pa.Table, dataset: BaselineDataset, split: str,
) -> list[dict]:
    dataset.validate()
    if split not in {"validation", "test"}:
        raise ContractError("predictions: split must be validation or test")
    names = set(table.column_names)
    optional = {"raw_logit"}
    if not set(PREDICTION_SCHEMA.names) <= names or names - set(PREDICTION_SCHEMA.names) - optional:
        raise ContractError("predictions: unexpected or missing columns")
    for field in PREDICTION_SCHEMA:
        if table.schema.field(field.name).type != field.type:
            raise ContractError(f"predictions: incorrect dtype for {field.name}")
    if "raw_logit" in names and table.schema.field("raw_logit").type != pa.float64():
        raise ContractError("predictions: raw_logit must be float64")
    queries = {q.request_id: q for q in dataset.queries if q.split == split}
    expected = {(c.request_id, c.candidate_id) for c in dataset.candidates
                if c.request_id in queries}
    if not expected:
        raise ContractError("predictions: requested split has no candidates")
    rows = table.to_pylist()
    seen, runs = set(), set()
    for row in rows:
        if any(row[field] is None for field in PREDICTION_SCHEMA.names):
            raise ContractError("predictions: required field is null")
        key = (row["request_id"], row["candidate_id"])
        if key in seen or key not in expected:
            raise ContractError(f"predictions: duplicate/extra candidate {key}")
        seen.add(key)
        query = queries[row["request_id"]]
        if (row["schema_version"] != 1 or row["task"] != dataset.metadata.task
                or row["split"] != split or row["data_version"] != dataset.metadata.data_version
                or row["user_id"] != query.user_id
                or query.cutoff_time.microsecond != 0
                or row["cutoff_timestamp"] != int(query.cutoff_time.timestamp())):
            raise ContractError(f"predictions: metadata/user/cutoff mismatch {key}")
        if not row["run_id"].strip():
            raise ContractError("predictions: empty run_id")
        runs.add(row["run_id"])
        validate_score(row["score"], dataset.metadata.task)
        if "raw_logit" in row:
            validate_score(row["raw_logit"], "food_recommendation")
    if seen != expected or len(runs) != 1:
        raise ContractError("predictions: missing candidate or inconsistent run_id")
    return rows


def write_predictions(
    path: str | Path, table: pa.Table, dataset: BaselineDataset, split: str,
) -> None:
    rows = validate_predictions(table, dataset, split)
    rows.sort(key=lambda row: (row["request_id"], -row["score"], row["candidate_id"]))
    ordered = pa.Table.from_pylist(rows, schema=table.schema)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(suffix=".parquet", dir=destination.parent)
    os.close(descriptor)
    try:
        pq.write_table(ordered, temporary)
        validate_predictions(pq.read_table(temporary), dataset, split)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
