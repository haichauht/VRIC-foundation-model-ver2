from src import __version__, baselines, downstream, evaluation, models, simulation


def test_package_skeleton_imports():
    assert __version__ == "0.1.0"
    assert all(module is not None for module in (baselines, downstream, evaluation, models, simulation))
