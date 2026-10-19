"""Simulator configuration: SimConfig, NoiseConfig, presets and static constants."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

# Version of the generator code written into every exported package's provenance.
GENERATOR_VERSION = "src.simulation 1.0"

CUISINES: tuple[str, ...] = (
    "cơm văn phòng",
    "phở/bún",
    "healthy/salad",
    "gà rán",
    "pizza",
    "lẩu",
    "đồ Nhật",
    "đồ Hàn",
    "trà sữa",
    "cà phê/bánh",
    "đồ chay",
    "ăn vặt đêm",
)

AD_CATEGORIES: tuple[str, ...] = (
    "đồ ăn",
    "thể thao",
    "du lịch",
    "khách sạn",
    "thời trang",
    "làm đẹp",
    "điện tử",
    "tài chính",
    "giáo dục",
    "game",
    "mẹ & bé",
    "nhà cửa",
    "xe cộ",
    "sức khỏe",
    "giải trí",
)

PERSONA_SHARES: dict[str, float] = {
    "office": 0.40,
    "student": 0.20,
    "night_shift": 0.10,
    "home": 0.15,
    "traveler": 0.15,
}

ZONE_TYPE_COUNTS: dict[str, int] = {
    "residential": 20,
    "office": 12,
    "university": 4,
    "nightlife": 6,
    "mall": 5,
    "transport_hub": 3,
}


@dataclass(frozen=True)
class NoiseConfig:
    """Coefficients of the true-probability formulas (Task 2).

    The defaults are calibrated on the ``full`` preset against the spec §5 difficulty targets
    (Task 8; achieved values in docs/simulation_data_contract.md). Changed from the Task 2
    starting values: ``food_distance_weight`` 2.0 -> 1.25 (2.0 put 99.5% of full-preset orders
    within 3 km of the current location), ``ad_intent_weight`` 1.5 -> 1.0 (keeps the test week's
    payday CTR bump inside 4.5-5.5%) and ``ad_logit_noise_sd`` 0.5 -> 1.25 (lowers the expected AUC
    of the per-ad CTR ranker from ~0.634 to ~0.62, away from the 0.65 ceiling).
    """

    food_cuisine_weight: float = 2.0
    food_price_weight: float = 0.6
    food_distance_weight: float = 1.25
    food_popularity_weight: float = 1.0
    food_logit_noise_sd: float = 0.5
    ad_interest_weight: float = 3.0
    ad_intent_weight: float = 1.0
    ad_zone_weight: float = 0.8
    ad_cross_service_weight: float = 1.2
    ad_logit_noise_sd: float = 1.25


# The smoke catalog is sparse (60 merchants over 50 zones, ~1 per zone), so the smoke preset
# keeps the stronger pre-calibration distance weight: with few nearby merchants the calibrated
# 1.25 lets the Food choice drift away from the current location (the 24-merchant test world
# derived from smoke drops from 62% to 53% of orders within 3 km). Smoke Food numbers are
# diagnostics only (ruling R9).
SMOKE_FOOD_DISTANCE_WEIGHT = 2.0


@dataclass(frozen=True)
class SimConfig:
    dataset_version: str
    seed: int
    n_users: int
    n_merchants: int
    n_ads: int
    n_drivers: int
    n_zones: int = 50
    dishes_per_merchant: int = 8
    n_ad_categories: int = 15
    start_date: str = "2026-01-05"
    weeks: int = 8
    utc_offset_hours: int = 7
    target_events_per_user: int = 200
    food_requests: dict[str, int] | None = None
    ads_requests: dict[str, int] | None = None
    target_ctr: float = 0.05
    snapshot_every_days: int = 3
    max_history_events: int = 30
    # Food candidate sampling (ruling R10): length scale of the exp(-d/tau) distance decay used
    # for the ordered cluster's latent centre and for drawing every cluster's negatives.
    # Calibrated in Task 8 (1.5 -> 2.0 km): a wider decay blurs the zone clusters, which lowers
    # what a learned candidate-only ranker gains over popularity; of the values tried, 2.0 km is
    # the widest that keeps the spec §5 "negatives near the ordered merchant / the anchors"
    # guarantees as tested in tests/simulation/test_export_food.py (2.5 km fails them).
    food_candidate_tau_km: float = 2.0
    noise: NoiseConfig = field(default_factory=NoiseConfig)


PRESETS: dict[str, SimConfig] = {
    "smoke": SimConfig(
        dataset_version="sim-smoke-v1",
        seed=42,
        n_users=500,
        n_merchants=60,
        n_ads=150,
        n_drivers=300,
        food_requests=None,
        ads_requests=None,
        noise=NoiseConfig(food_distance_weight=SMOKE_FOOD_DISTANCE_WEIGHT),
    ),
    "full": SimConfig(
        dataset_version="sim-v1",
        seed=42,
        n_users=20000,
        n_merchants=600,
        n_ads=1500,
        n_drivers=3000,
        food_requests={"train": 60000, "validation": 10000, "test": 10000},
        ads_requests={"train": 200000, "validation": 30000, "test": 30000},
    ),
}


def load_config(path: str | Path) -> SimConfig:
    """Load a YAML file with a ``preset:`` key plus optional override keys."""
    data: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    preset = data.pop("preset", None)
    if preset not in PRESETS:
        raise ValueError(f"unknown or missing preset {preset!r}; expected one of {sorted(PRESETS)}")
    noise_over = data.pop("noise", None)
    config = PRESETS[preset]
    if noise_over:
        config = replace(config, noise=replace(config.noise, **noise_over))
    return replace(config, **data)


def config_sha256(config: SimConfig) -> str:
    """SHA-256 of the canonical (sorted-key) JSON form of the config."""
    payload = json.dumps(asdict(config), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
