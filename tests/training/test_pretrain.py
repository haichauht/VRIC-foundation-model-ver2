from __future__ import annotations

import tempfile
from pathlib import Path

from src.training.pretrain import (
    discover_target_datasets,
    load_yaml_config,
    parse_args,
    set_seed,
)


def test_set_seed():
    set_seed(42)


def test_load_yaml_config():
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write("seed: 123\nepochs: 5\nlearning_rate: 0.001\n")
        path = f.name
    try:
        cfg = load_yaml_config(path)
        assert cfg["seed"] == 123
        assert cfg["epochs"] == 5
        assert cfg["learning_rate"] == 0.001
    finally:
        Path(path).unlink(missing_ok=True)


def test_discover_target_datasets():
    datasets = discover_target_datasets("meituan", default_root="data/pretrain_data")
    assert len(datasets) >= 1
    assert "meituan" in datasets[0].lower()


def test_parse_args_with_smoke(monkeypatch):
    monkeypatch.setattr("sys.argv", ["pretrain.py", "--smoke", "--epochs", "2"])
    _config, overrides = parse_args()
    assert overrides.get("smoke") is True
    assert overrides.get("epochs") == 2
