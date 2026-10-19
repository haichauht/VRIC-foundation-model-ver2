"""Difficulty report (spec §5, §8) computed from the EXPORTED files of one run.

``difficulty_report(out_dir)`` reads ``food/``, ``ads/`` and ``oracle/`` under ``out_dir`` (the
``<out>/<dataset_version>`` directory) and scores the **test** split with simple rankers:

Food (NDCG@10 per request via ``src.evaluation.metrics.ranking_metrics``, averaged):

* ``popularity``: the merchant's ``order`` count in weeks 1..weeks-2 (``ads/events.parquet``
  orders before ``food/manifest.json:popularity_end_exclusive``; the same count the candidate
  sampler uses).
* ``candidate_only``: ``log(rate(cuisine, price_level)) + log1p(popularity)``, where ``rate`` is
  the positive rate of the ``(cuisine, price_level)`` pair among the **train** candidates,
  smoothed towards the global rate with ``PAIR_PRIOR_CANDIDATES`` pseudo-candidates. Candidate
  attributes + catalog + popularity only (no history, no profile): the spec §8 shortcut guard.
* ``oracle``: ``oracle/food_oracle.parquet`` true probabilities.
* Profile rankers (ruling R7, reported, never capped): ``distance_to_home`` = minus the haversine
  distance between the candidate's geohash and the profile's ``saved_home_geohash``;
  ``home_or_work_by_hour`` = the same distance to ``saved_work_geohash`` on weekdays 9:00-17:00
  (context ``weekday`` < 5 and 9 <= ``hour`` < 17) when the user has one, else to home.

Ads (ROC AUC via ``src.evaluation.metrics.advertising_metrics``):

* ``popularity``: per-ad train CTR, ``(clicks + m * ctr_train) / (impressions + m)`` with
  ``m = AD_PRIOR_IMPRESSIONS``.
* ``oracle``: ``oracle/ads_oracle.parquet`` true click probabilities.

The Food block carries ``experiment_grade = n_merchants >= FULL_DENSITY_MIN_MERCHANTS``: a sparse
catalog (smoke, 60 merchants) cannot support the candidate zone structure, so its Food numbers are
diagnostics only (ruling R9). ``targets`` restate spec §5 (meant for the ``full`` preset).
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ..evaluation.metrics import advertising_metrics, ranking_metrics

SPLIT = "test"
K = 10
PAIR_PRIOR_CANDIDATES = 20.0
AD_PRIOR_IMPRESSIONS = 20.0
FULL_DENSITY_MIN_MERCHANTS = 300
WORK_HOURS = (9, 17)  # [start, end) local hour on weekdays (Mon=0..Fri=4)
FOOD_TARGETS = {"popularity_ndcg_at_10": (0.40, 0.55), "oracle_minus_popularity_min": 0.20}
ADS_TARGETS = {"popularity_auc": (0.55, 0.65), "oracle_auc_min": 0.75}
_BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"
_EARTH_RADIUS_KM = 6371.0088


# ---- geometry -----------------------------------------------------------------------------


def geohash_decode(geohash: str) -> tuple[float, float]:
    """Centre ``(lat, lon)`` of a geohash cell (inverse of ``world.geohash_encode``)."""
    lat_lo, lat_hi, lon_lo, lon_hi = -90.0, 90.0, -180.0, 180.0
    even = True
    for char in geohash:
        value = _BASE32.index(char)
        for shift in range(4, -1, -1):
            bit = (value >> shift) & 1
            if even:
                mid = (lon_lo + lon_hi) / 2
                lon_lo, lon_hi = (mid, lon_hi) if bit else (lon_lo, mid)
            else:
                mid = (lat_lo + lat_hi) / 2
                lat_lo, lat_hi = (mid, lat_hi) if bit else (lat_lo, mid)
            even = not even
    return (lat_lo + lat_hi) / 2, (lon_lo + lon_hi) / 2


class _Distances:
    """Haversine distance between geohash cell centres, cached per pair."""

    def __init__(self) -> None:
        self._points: dict[str, tuple[float, float]] = {}
        self._cache: dict[tuple[str, str], float] = {}

    def _point(self, geohash: str) -> tuple[float, float]:
        point = self._points.get(geohash)
        if point is None:
            lat, lon = geohash_decode(geohash)
            point = self._points[geohash] = (math.radians(lat), math.radians(lon))
        return point

    def km(self, a: str, b: str) -> float:
        key = (a, b)
        value = self._cache.get(key)
        if value is None:
            (p1, l1), (p2, l2) = self._point(a), self._point(b)
            h = (math.sin((p2 - p1) / 2) ** 2
                 + math.cos(p1) * math.cos(p2) * math.sin((l2 - l1) / 2) ** 2)
            value = self._cache[key] = 2 * _EARTH_RADIUS_KM * math.asin(math.sqrt(h))
        return value


# ---- helpers ------------------------------------------------------------------------------


def _epoch(iso: str) -> int:
    return int(datetime.fromisoformat(iso).timestamp())


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _round(value: float | None) -> float | None:
    return None if value is None or math.isnan(value) else round(float(value), 6)


def train_order_counts(root: Path) -> dict[str, int]:
    """Orders per merchant before ``food/manifest.json:popularity_end_exclusive``, read from
    ``ads/events.parquet`` (every event of every service)."""
    manifest = json.loads((root / "food" / "manifest.json").read_text(encoding="utf-8"))
    end = _epoch(manifest["popularity_end_exclusive"])
    events = pq.read_table(root / "ads" / "events.parquet",
                           columns=["timestamp", "service", "event_type", "entity_id"])
    mask = pc.and_(pc.and_(pc.equal(events["service"], "food"),
                           pc.equal(events["event_type"], "order")),
                   pc.less(events["timestamp"], end))
    counts = pc.value_counts(events.filter(mask)["entity_id"])
    return {str(row["values"]): int(row["counts"]) for row in counts.to_pylist()}


def _ndcg(rows: list[tuple[str, int, float]]) -> float:
    return ranking_metrics(rows, ks=(K,))[f"ndcg_at_{K}"]


# ---- Food ---------------------------------------------------------------------------------


def _food_difficulty(root: Path) -> dict:
    counts = train_order_counts(root)
    n_merchants = pq.read_metadata(root / "food" / "merchants.parquet").num_rows
    pair_n: dict[tuple[str, int], int] = {}
    pair_pos: dict[tuple[str, int], int] = {}
    tests: list[dict] = []
    with (root / "food" / "samples.jsonl").open("rb") as fh:
        for line in fh:
            row = json.loads(line)
            split = row["label_metadata"]["split"]
            if split == "train":
                for cand, label in zip(row["input"]["candidates"], row["labels"]):
                    key = (cand["cuisine"], cand["price_level"])
                    pair_n[key] = pair_n.get(key, 0) + 1
                    pair_pos[key] = pair_pos.get(key, 0) + int(label["relevance"])
            elif split == SPLIT:
                inp = row["input"]
                tests.append({
                    "request_id": row["sample_id"],
                    "candidates": inp["candidates"],
                    "relevance": {lb["item_id"]: int(lb["relevance"]) for lb in row["labels"]},
                    "home": inp["profile"]["saved_home_geohash"],
                    "work": inp["profile"]["saved_work_geohash"],
                    "context": inp["context"],
                })
    total_n = sum(pair_n.values())
    global_rate = sum(pair_pos.values()) / total_n if total_n else 0.1

    def pair_rate(cuisine: str, price_level: int) -> float:
        key = (cuisine, price_level)
        return ((pair_pos.get(key, 0) + PAIR_PRIOR_CANDIDATES * global_rate)
                / (pair_n.get(key, 0) + PAIR_PRIOR_CANDIDATES))

    wanted = {t["request_id"] for t in tests}
    oracle_table = pq.read_table(root / "oracle" / "food_oracle.parquet")
    oracle: dict[tuple[str, str], float] = {}
    for rid, cid, p in zip(oracle_table["request_id"].to_pylist(),
                           oracle_table["candidate_id"].to_pylist(),
                           oracle_table["true_probability"].to_pylist()):
        if rid in wanted:
            oracle[(rid, cid)] = p

    dist = _Distances()
    scores: dict[str, list[float]] = {name: [] for name in (
        "popularity", "candidate_only", "oracle", "distance_to_home", "home_or_work_by_hour")}
    for t in tests:
        context = t["context"]
        at_work = (t["work"] is not None and context["weekday"] < 5
                   and WORK_HOURS[0] <= context["hour"] < WORK_HOURS[1])
        anchor = t["work"] if at_work else t["home"]
        rankers = {
            "popularity": lambda c: float(counts.get(c["item_id"], 0)),
            "candidate_only": lambda c: (math.log(pair_rate(c["cuisine"], c["price_level"]))
                                         + math.log1p(counts.get(c["item_id"], 0))),
            "oracle": lambda c, rid=t["request_id"]: oracle[(rid, c["item_id"])],
            "distance_to_home": lambda c, home=t["home"]: -dist.km(c["geohash"], home),
            "home_or_work_by_hour": lambda c, a=anchor: -dist.km(c["geohash"], a),
        }
        for name, score in rankers.items():
            scores[name].append(_ndcg([(c["item_id"], t["relevance"][c["item_id"]], score(c))
                                       for c in t["candidates"]]))
    popularity = _mean(scores["popularity"])
    oracle_ndcg = _mean(scores["oracle"])
    gap = None if popularity is None or oracle_ndcg is None else oracle_ndcg - popularity
    lo, hi = FOOD_TARGETS["popularity_ndcg_at_10"]
    report = {
        "requests": len(tests),
        "merchants": n_merchants,
        "experiment_grade": n_merchants >= FULL_DENSITY_MIN_MERCHANTS,
        **{f"{name}_ndcg_at_{K}": _round(_mean(values)) for name, values in scores.items()},
        "oracle_minus_popularity": _round(gap),
        "targets": {"popularity_ndcg_at_10": [lo, hi],
                    "oracle_minus_popularity_min": FOOD_TARGETS["oracle_minus_popularity_min"]},
        "within_targets": None if gap is None else bool(
            lo <= popularity <= hi and gap >= FOOD_TARGETS["oracle_minus_popularity_min"]),
    }
    if not report["experiment_grade"]:
        report["note"] = (f"sparse merchant catalog (< {FULL_DENSITY_MIN_MERCHANTS} merchants): "
                          "Food numbers are diagnostics only, not experiment-grade (ruling R9)")
    return report


# ---- Ads ----------------------------------------------------------------------------------


def _ads_difficulty(root: Path) -> dict:
    opp = pq.read_table(root / "ads" / "ad_opportunities.parquet",
                        columns=["request_id", "candidate_id", "clicked", "split"])
    splits = opp["split"].to_pylist()
    ads = opp["candidate_id"].to_pylist()
    clicked = [int(c) for c in opp["clicked"].to_pylist()]
    request_ids = opp["request_id"].to_pylist()
    impressions: dict[str, int] = {}
    clicks: dict[str, int] = {}
    for split, ad, y in zip(splits, ads, clicked):
        if split == "train":
            impressions[ad] = impressions.get(ad, 0) + 1
            clicks[ad] = clicks.get(ad, 0) + y
    n_train = sum(impressions.values())
    ctr = sum(clicks.values()) / n_train if n_train else 0.0

    oracle_table = pq.read_table(root / "oracle" / "ads_oracle.parquet")
    oracle = dict(zip(oracle_table["request_id"].to_pylist(),
                      oracle_table["true_click_probability"].to_pylist()))
    targets: list[int] = []
    popularity: list[float] = []
    truth: list[float] = []
    for rid, split, ad, y in zip(request_ids, splits, ads, clicked):
        if split != SPLIT:
            continue
        targets.append(y)
        popularity.append((clicks.get(ad, 0) + AD_PRIOR_IMPRESSIONS * ctr)
                          / (impressions.get(ad, 0) + AD_PRIOR_IMPRESSIONS))
        truth.append(oracle[rid])

    def auc(scores: list[float]) -> float | None:
        if not targets:
            return None
        return _round(advertising_metrics(targets, scores)["roc_auc"])

    pop_auc, oracle_auc = auc(popularity), auc(truth)
    lo, hi = ADS_TARGETS["popularity_auc"]
    return {
        "requests": len(targets),
        "test_ctr": _round(_mean([float(y) for y in targets])),
        "train_ctr": _round(ctr),
        "popularity_auc": pop_auc,
        "oracle_auc": oracle_auc,
        "targets": {"popularity_auc": [lo, hi], "oracle_auc_min": ADS_TARGETS["oracle_auc_min"]},
        "within_targets": None if pop_auc is None or oracle_auc is None else bool(
            lo <= pop_auc <= hi and oracle_auc >= ADS_TARGETS["oracle_auc_min"]),
    }


def difficulty_report(out_dir: Path) -> dict:
    """Difficulty numbers of one exported run on the test split (see module docstring)."""
    root = Path(out_dir)
    return {"split": SPLIT, "food": _food_difficulty(root), "ads": _ads_difficulty(root)}
