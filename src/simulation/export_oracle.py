"""Oracle package ``oracle/`` (spec §4.7, §6.5): true probabilities, never a model input.

``food_oracle.parquet``: ``request_id, candidate_id, true_probability`` = softmax of the true
Food logits (``behavior.food_logits``) over exactly the request's candidates, so each request sums
to 1. ``ads_oracle.parquet``: ``request_id, true_click_probability`` = the stored impression
probability.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pyarrow as pa

from .behavior import Simulation, food_logits
from .export_ads import AdRequest
from .export_food import FoodRequest
from .export_pretrain import _write_parquet

FOOD_SCHEMA = pa.schema([
    ("request_id", pa.string()),
    ("candidate_id", pa.string()),
    ("true_probability", pa.float64()),
])
ADS_SCHEMA = pa.schema([
    ("request_id", pa.string()),
    ("true_click_probability", pa.float64()),
])


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = np.exp(logits - np.max(logits))
    return shifted / shifted.sum()


def export_oracle(sim: Simulation, food_requests: Sequence[FoodRequest],
                  ad_requests: Sequence[AdRequest], out_dir) -> dict:
    """Write ``out_dir/oracle/``; returns per-file ``{path, rows, bytes, sha256}``."""
    root = Path(out_dir) / "oracle"
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {root}")
    root.mkdir(parents=True, exist_ok=True)

    request_ids: list[str] = []
    candidate_ids: list[str] = []
    probabilities: list[float] = []
    for r in food_requests:
        probs = softmax(food_logits(sim.world, r.session, r.candidate_ids))
        request_ids.extend([r.request_id] * len(r.candidate_ids))
        candidate_ids.extend(r.candidate_ids)
        probabilities.extend(float(p) for p in probs)
    food = pa.table({"request_id": pa.array(request_ids, pa.string()),
                     "candidate_id": pa.array(candidate_ids, pa.string()),
                     "true_probability": pa.array(probabilities, pa.float64())},
                    schema=FOOD_SCHEMA)
    ads = pa.table({"request_id": pa.array([r.request_id for r in ad_requests], pa.string()),
                    "true_click_probability": pa.array(
                        [float(r.impression.true_probability) for r in ad_requests],
                        pa.float64())},
                   schema=ADS_SCHEMA)
    result = {}
    for name, table in (("food_oracle", food), ("ads_oracle", ads)):
        size, sha = _write_parquet(table, root / f"{name}.parquet")
        result[name] = {"path": f"oracle/{name}.parquet", "rows": table.num_rows,
                        "bytes": size, "sha256": sha}
    return result
