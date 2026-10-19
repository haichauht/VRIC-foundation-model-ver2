"""Optimizer shared by the downstream ranking heads (Food, Ads)."""

from __future__ import annotations

from collections.abc import Iterable

import torch


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
