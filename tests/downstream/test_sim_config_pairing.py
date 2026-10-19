"""Each downstream sim config must pair with the pretrain config that produces its checkpoint."""
from pathlib import PurePosixPath

import pytest
import yaml

TRIOS = {
    "local": ("pretrain_sim", "food_recommendation_sim", "ad_optimization_sim", None),
    "kaggle": ("pretrain_sim_kaggle", "food_recommendation_sim_kaggle",
               "ad_optimization_sim_kaggle", 4),
}


def _load(name):
    from pathlib import Path

    return yaml.safe_load(Path(f"configs/{name}.yaml").read_text(encoding="utf-8"))


@pytest.mark.parametrize("trio", sorted(TRIOS))
@pytest.mark.parametrize("downstream", ["food", "ads"])
def test_downstream_sim_config_pairs_with_pretrain_config(trio, downstream):
    pretrain_name, food_name, ads_name, threads = TRIOS[trio]
    pretrain = _load(pretrain_name)
    config = _load(food_name if downstream == "food" else ads_name)
    version = pretrain["dataset_version"]
    root = PurePosixPath(config["data"]["root"])
    assert root.name == downstream and root.parent.name == version
    assert PurePosixPath(pretrain["data_dir"]).parent.name == version
    assert config["data"]["expected_version"] == version
    assert config["checkpoint"]["path"] == (
        pretrain["output_dir"].rstrip("/") + "/checkpoints/best_checkpoint.pt")
    if threads is not None:
        assert config["training"]["cpu_threads"] == threads


def test_kaggle_downstream_roots_read_the_attached_generator_kernel():
    for name, leaf in (("food_recommendation_sim_kaggle", "food"),
                       ("ad_optimization_sim_kaggle", "ads")):
        config = _load(name)
        assert config["data"]["root"] == f"/kaggle/input/vric-sim-generate/sim-v1/{leaf}"
        assert config["artifact_dir"].startswith("/kaggle/working/")
