"""Shared fixtures: a tiny simulator world exported as the three dataset packages."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest

from src.simulation.behavior import simulate
from src.simulation.config import PRESETS
from src.simulation.export_ads import build_ad_requests, export_ads
from src.simulation.export_food import build_food_requests, export_food
from src.simulation.export_pretrain import export_pretrain
from src.simulation.world import build_world


@dataclass(frozen=True)
class SimPackages:
    root: Path
    pretrain: Path
    food: Path
    ads: Path


@pytest.fixture(scope="session")
def sim_tiny_packages(tmp_path_factory) -> SimPackages:
    config = replace(PRESETS["smoke"], n_users=80, n_merchants=24, n_ads=40, n_drivers=40)
    rng = np.random.default_rng(config.seed)
    world = build_world(config, rng)
    sim = simulate(world, rng)
    food_requests = build_food_requests(sim, rng)
    ad_requests = build_ad_requests(sim, rng)
    root = tmp_path_factory.mktemp("sim")
    export_pretrain(sim, root)
    export_food(sim, food_requests, root)
    export_ads(sim, ad_requests, root)
    return SimPackages(root=root, pretrain=root / "pretrain", food=root / "food", ads=root / "ads")
