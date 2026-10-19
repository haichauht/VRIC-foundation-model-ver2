"""Optimizer and encoder placement shared by the downstream ranking heads (Food, Ads)."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable

import torch

ENCODER_DEVICES = ("cpu", "cuda", "auto")


def head_optimizer(parameters: Iterable[torch.nn.Parameter], *, learning_rate: float,
                   weight_decay: float = 0.0) -> torch.optim.Optimizer:
    """AdamW: weight decay decoupled from the gradient.

    ``torch.optim.Adam(weight_decay=...)`` adds ``weight_decay * param`` to the gradient of every
    parameter at every step, including candidate-embedding rows absent from the batch; Adam's
    per-parameter normalisation turns that tiny term into a near-``lr`` pull toward zero. With
    1,500 ads each in ~4% of batches, the Ads head's ad embeddings collapsed to ~0 and the head
    could not learn per-ad CTR. Decoupled decay shrinks each row by ``lr * weight_decay`` per step.
    """
    return torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)


def encoder_device(name: str = "cpu") -> torch.device:
    """Device of the foundation forward pass (``training.encoder_device``).

    Heads always stay on CPU; only ``embed_users`` moves. ``auto`` picks CUDA when present.
    """
    if name not in ENCODER_DEVICES:
        raise ValueError(f"training.encoder_device must be one of {list(ENCODER_DEVICES)}")
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("training.encoder_device is cuda but CUDA is unavailable")
    return torch.device(name)


def _to(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _to(item, device) for key, item in value.items()}
    return value


def embed_users_on_device(foundation: torch.nn.Module, batch) -> dict[str, torch.Tensor]:
    """``foundation.embed_users(batch)`` on the device holding the foundation; CPU outputs.

    ``.to("cpu")`` keeps the autograd graph, so fine-tuning through a GPU encoder still works.
    """
    parameter = next(foundation.parameters(), None)
    if parameter is None or parameter.device.type == "cpu":
        return foundation.embed_users(batch)
    moved = dataclasses.replace(batch, **{field.name: _to(getattr(batch, field.name), parameter.device)
                                          for field in dataclasses.fields(batch)})
    return {name: value.to("cpu") for name, value in foundation.embed_users(moved).items()}
