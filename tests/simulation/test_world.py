from collections import Counter
from dataclasses import asdict, replace
from itertools import combinations

import numpy as np

from src.simulation.config import (
    PERSONA_SHARES,
    PRESETS,
    SMOKE_FOOD_DISTANCE_WEIGHT,
    ZONE_TYPE_COUNTS,
    NoiseConfig,
    SimConfig,
    config_sha256,
    load_config,
)
from src.simulation.world import build_world, geohash_encode, haversine_km


def test_presets_match_spec():
    s, f = PRESETS["smoke"], PRESETS["full"]
    assert (s.dataset_version, s.n_users, s.n_merchants, s.n_ads, s.n_drivers) == (
        "sim-smoke-v1", 500, 60, 150, 300)
    assert (f.dataset_version, f.n_users, f.n_merchants, f.n_ads, f.n_drivers) == (
        "sim-v1", 20000, 600, 1500, 3000)
    assert f.food_requests == {"train": 60000, "validation": 10000, "test": 10000}
    assert f.ads_requests == {"train": 200000, "validation": 30000, "test": 30000}
    assert s.seed == f.seed == 42 and s.snapshot_every_days == 3 and s.max_history_events == 30


def test_noise_defaults_are_calibrated():
    """Task 8: the values calibrated on the full preset against the spec §5 targets
    (docs/simulation_data_contract.md, "Độ khó"); changing one invalidates that table."""
    assert asdict(NoiseConfig()) == {
        "food_cuisine_weight": 2.0,
        "food_price_weight": 0.6,
        "food_distance_weight": 1.25,
        "food_popularity_weight": 1.0,
        "food_logit_noise_sd": 0.5,
        "ad_interest_weight": 3.0,
        "ad_intent_weight": 1.0,
        "ad_zone_weight": 0.8,
        "ad_cross_service_weight": 1.2,
        "ad_logit_noise_sd": 1.25,
    }
    assert SimConfig.__dataclass_fields__["food_candidate_tau_km"].default == 2.0
    full, smoke = PRESETS["full"], PRESETS["smoke"]
    assert full.noise == NoiseConfig() and full.food_candidate_tau_km == 2.0
    # the sparse smoke catalog keeps only the stronger distance weight (see config.py)
    assert SMOKE_FOOD_DISTANCE_WEIGHT == 2.0
    assert smoke.noise == replace(NoiseConfig(), food_distance_weight=2.0)
    assert smoke.food_candidate_tau_km == 2.0


def test_world_is_deterministic_and_shaped(tiny_config):
    a = build_world(tiny_config, np.random.default_rng(tiny_config.seed))
    b = build_world(tiny_config, np.random.default_rng(tiny_config.seed))
    assert a == b
    assert len(a.zones) == 50 and Counter(z.zone_type for z in a.zones) == ZONE_TYPE_COUNTS
    assert len(a.users) == 80 and len(a.merchants) == 24 and len(a.dishes) == 24 * 8
    assert all(len(u.cuisine_pref) == 12 and abs(sum(u.cuisine_pref) - 1) < 1e-9 for u in a.users)
    assert all((u.work_zone is None) == (u.persona == "home") for u in a.users)
    assert a.users[0].user_id == "sim:user:0" and a.merchants[0].merchant_id == "sim:merchant:0"


def test_geohash_known_value():
    assert geohash_encode(57.64911, 10.40744, 6) == "u4pruy"
    hcmc = geohash_encode(10.7769, 106.7009, 6)
    assert len(hcmc) == 6 and hcmc.startswith("w3")
    assert set(hcmc) <= set("0123456789bcdefghjkmnpqrstuvwxyz")


def test_persona_shares_on_smoke():
    world = build_world(PRESETS["smoke"], np.random.default_rng(42))
    shares = Counter(u.persona for u in world.users)
    assert all(abs(shares[p] / 500 - q) < 0.06 for p, q in PERSONA_SHARES.items())


def test_load_config_and_sha(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("preset: smoke\nn_users: 100\n", encoding="utf-8")
    c = load_config(p)
    assert c.n_users == 100 and c.dataset_version == "sim-smoke-v1"
    assert config_sha256(c) == config_sha256(load_config(p))
    assert config_sha256(c) != config_sha256(PRESETS["smoke"])
    assert load_config("configs/simulation/full.yaml") == PRESETS["full"]


def test_zone_geohashes_are_distinct_on_presets():
    # one zone per 6-char geohash, otherwise history cannot tell two zones apart
    for config in (PRESETS["smoke"], PRESETS["full"]):
        zones = build_world(config, np.random.default_rng(config.seed)).zones
        assert len({z.geohash for z in zones}) == len(zones) == 50


def test_zones_are_distinct_and_separated_for_many_seeds(tiny_config):
    for seed in range(25):
        zones = build_world(replace(tiny_config, seed=seed), np.random.default_rng(seed)).zones
        assert len({z.geohash for z in zones}) == 50
        assert min(haversine_km(a, b) for a, b in combinations(zones, 2)) >= 1.0
