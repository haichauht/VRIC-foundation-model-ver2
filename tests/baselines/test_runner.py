import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.baselines.contracts import ContractError
from src.baselines.runner import run


@pytest.fixture
def config(tmp_path):
    dataset = {"metadata": {"dataset_name": "fixture", "data_version": "fixture-v1",
                            "task": "ad_optimization", "item_kind": "ad", "timezone": "UTC",
                            "candidate_policy": "logged_impressions", "availability_policy": "v1"},
               "queries": [], "candidates": [], "labels": []}
    for split, day in [("train", 1), ("validation", 3), ("test", 5)]:
        for i in range(24):
            q = f"{split}-{i}"
            dataset["queries"].append({"request_id": q, "user_id": "u", "split": split,
                                       "cutoff_time": f"2026-09-0{day}T00:00:00+00:00"})
            dataset["candidates"].append({"request_id": q, "candidate_id": str(i),
                                          "attributes": {"price": i % 2}})
            dataset["labels"].append({"request_id": q, "candidate_id": str(i), "target": i % 2,
                                      "available_at": f"2026-09-0{day}T01:00:00+00:00"})
    (tmp_path / "data.json").write_text(json.dumps(dataset), encoding="utf-8")
    root = Path(__file__).resolve().parents[2]
    settings = {"dataset_path": "data.json", "artifact_dir": str(root / "experiments"),
                "model_kind": "B0", "split": "test", "fit_cutoff": "2026-09-03T00:00:00Z",
                "prediction_cutoff": "2026-09-05T00:00:00Z",
                "parameters": {"variant": "global_ctr"}, "ks": [10],
                "zero_positive_policy": "include"}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(settings), encoding="utf-8")
    return path, settings


@pytest.mark.parametrize("kind", ["B0", "B1"])
def test_runner_artifacts(config, kind):
    path, settings = config
    settings["model_kind"] = kind
    if kind == "B1":
        settings["parameters"] = {"n_estimators": 5, "min_child_samples": 2, "stopping_rounds": 2}
    path.write_text(json.dumps(settings), encoding="utf-8")
    assert run(path, validate_only=True)["status"] == "passed"
    result = run(path)
    output = Path(result["artifact_dir"])
    assert result["status"] == "passed"
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["foundation"] is None
    assert len(manifest["dataset_sha256"]) == 64
    assert (output / "metrics.json").exists()
    assert (output / "report.md").exists()
    assert "NaN" not in (output / "metrics.json").read_text()


def test_cli_validate_and_failure(config):
    path, settings = config
    result = subprocess.run([sys.executable, "-m", "src.baselines", "validate", "--config", str(path)],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0
    assert json.loads(result.stdout)["status"] == "passed"
    settings["dataset_path"] = "missing.json"
    path.write_text(json.dumps(settings), encoding="utf-8")
    result = subprocess.run([sys.executable, "-m", "src.baselines", "validate", "--config", str(path)],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 2


def test_refuse_external_artifact_directory(config, tmp_path):
    path, settings = config
    settings["artifact_dir"] = str(tmp_path / "outside")
    path.write_text(json.dumps(settings), encoding="utf-8")
    with pytest.raises(ContractError, match="workspace experiments"):
        run(path)
    assert not (tmp_path / "outside").exists()
