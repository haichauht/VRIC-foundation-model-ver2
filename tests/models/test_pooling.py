from __future__ import annotations

import torch

from src.models.pooling import SequencePooler


def test_pooler_excludes_tabular_and_padding_tokens() -> None:
    hidden = torch.tensor([[[100.0, 100.0], [2.0, 2.0], [4.0, 4.0], [999.0, 999.0]]])
    valid = torch.tensor([[True, True, True, False]])
    source = torch.tensor([[0, 2, 2, 2]])

    result = SequencePooler(sequence_source_id=2)(hidden, valid, source)

    assert torch.equal(result, torch.tensor([[3.0, 3.0]]))


def test_pooler_uses_cls_when_user_has_no_event() -> None:
    hidden = torch.tensor([[[7.0, 8.0], [2.0, 2.0]]])
    valid = torch.tensor([[True, True]])
    source = torch.tensor([[0, 1]])

    result = SequencePooler(sequence_source_id=2)(hidden, valid, source)

    assert torch.equal(result, torch.tensor([[7.0, 8.0]]))
