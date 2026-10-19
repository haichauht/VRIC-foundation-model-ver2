"""Foundation checkpoint of a downstream run: initialisation arm and sim-world binding.

The ``random`` arm keeps the checkpoint's vocabulary and ``model_config`` and only replaces the
weights, so it answers "does pretraining help" with everything else held fixed.

A ``sim_v1`` package (Food or Ads) accepts a checkpoint only from the same simulated world:
vocabulary ``dataset_name`` is ``sim_v1`` and the pretrain provenance carries the package's
``sim_config_sha256`` (spec §5-§6).
"""

from __future__ import annotations

import torch

from src.downstream.sim_events import SIM_DATASET_NAME
from src.models.encoder import FoundationModel

FOUNDATION_INITS = ("pretrained", "random")


def foundation_init_arm(config: dict) -> str:
    """The configured ``foundation.init`` (default ``"pretrained"``), validated."""
    settings = config.get("foundation", {})
    if not isinstance(settings, dict):
        raise ValueError("foundation must be a mapping")  # noqa: TRY004
    arm = settings.get("init", "pretrained")
    if arm not in FOUNDATION_INITS:
        raise ValueError(f"foundation.init must be pretrained or random, got {arm!r}")
    return arm


def reinitialize_foundation(model: FoundationModel, seed: int) -> FoundationModel:
    """A new ``FoundationModel(model.config)`` built under ``torch.manual_seed(seed)``.

    Same device as ``model``, in ``eval()`` mode; ``model`` itself is left untouched. The global
    CPU RNG state is restored afterwards so the caller's own seeding is not disturbed.
    """
    device = next(model.parameters()).device
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        fresh = FoundationModel(model.config)
    return fresh.to(device).eval()


def check_sim_checkpoint_binding(vocabulary, preprocess_metadata, expected_sim_config_sha256, *,
                                 package):
    """A sim checkpoint must come from the simulated world of the ``package`` (Food, Ads) data."""
    if vocabulary.dataset_name != SIM_DATASET_NAME:
        raise ValueError(
            f"{package} sim_v1 checkpoint vocabulary dataset_name is {vocabulary.dataset_name!r}, "
            f"expected {SIM_DATASET_NAME!r}"
        )
    metadata = preprocess_metadata if isinstance(preprocess_metadata, dict) else {}
    provenance = metadata.get("dataset_provenance")
    actual = provenance.get("sim_config_sha256") if isinstance(provenance, dict) else None
    if actual != expected_sim_config_sha256:
        raise ValueError(
            f"{package} checkpoint sim_config_sha256 mismatch: checkpoint {actual!r}, "
            f"{package} package {expected_sim_config_sha256!r}"
        )
