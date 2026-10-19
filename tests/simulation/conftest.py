from dataclasses import replace

import pytest

from src.simulation.config import PRESETS

# Tiny world used by the CLI tests: the smoke preset with fewer users, merchants, ads, drivers.
TINY_CONFIG_YAML = "preset: smoke\nn_users: 80\nn_merchants: 24\nn_ads: 40\nn_drivers: 40\n"


@pytest.fixture(scope="session")
def tiny_config():
    return replace(PRESETS["smoke"], n_users=80, n_merchants=24, n_ads=40, n_drivers=40)


@pytest.fixture(scope="session")
def tiny_config_yaml():
    return TINY_CONFIG_YAML


@pytest.fixture
def tiny_config_file(tmp_path, tiny_config_yaml):
    """YAML ``{preset: smoke, n_users: 80, n_merchants: 24, n_ads: 40, n_drivers: 40}``."""
    path = tmp_path / "tiny.yaml"
    path.write_text(tiny_config_yaml, encoding="utf-8")
    return path
