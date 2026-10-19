"""Catalog attributes for the downstream heads, and where the foundation forward pass runs."""

from __future__ import annotations

import pytest
import torch

from src.downstream.candidate_features import (
    attribute_names,
    attribute_sizes,
    check_candidate_attributes,
    encode_attributes,
    fit_attribute_vocabularies,
)
from src.downstream.food_recommendation import FoodRanker
from src.downstream.head_training import embed_users_on_device, encoder_device

CATALOG = {
    "m1": {"cuisine": "phở", "price_level": 1},
    "m2": {"cuisine": "bún", "price_level": 2},
    "m3": {"cuisine": "phở", "price_level": None},
    "m4": {"cuisine": "pizza", "price_level": 3},  # never a train candidate
}


def rows(*candidates):
    return [{"request_id": "r", "candidate_id": candidate} for candidate in candidates]


def test_vocabularies_fit_on_train_candidates_only():
    vocabularies = fit_attribute_vocabularies(CATALOG, ("cuisine", "price_level"), ["m2", "m1", "m3"])
    assert vocabularies == {"cuisine": {'"bún"': 1, '"phở"': 2}, "price_level": {"1": 1, "2": 2}}
    assert attribute_sizes(("cuisine", "price_level"), vocabularies) == (3, 3)


def test_out_of_vocabulary_candidates_keep_known_attribute_values():
    names = ("cuisine", "price_level")
    vocabularies = fit_attribute_vocabularies(CATALOG, names, ["m1", "m2"])
    encoded = encode_attributes([rows("m3", "m4"), rows("m1")], CATALOG, names, vocabularies, 2)
    assert encoded.dtype == torch.long and encoded.shape == (2, 2, 2)
    assert encoded[0, 0].tolist() == [2, 0]  # m3: unseen ID, known cuisine, missing price
    assert encoded[0, 1].tolist() == [0, 0]  # m4: value never seen on a train candidate
    assert encoded[1].tolist() == [[2, 1], [0, 0]]  # padding column stays 0


@pytest.mark.parametrize(("catalog", "names", "train", "match"), [
    (CATALOG, ("zone",), ["m1"], "not a catalog column"),
    (CATALOG, ("cuisine",), ["m9"], "missing from catalog"),
])
def test_vocabulary_errors(catalog, names, train, match):
    with pytest.raises(ValueError, match=match):
        fit_attribute_vocabularies(catalog, names, train)


@pytest.mark.parametrize("value", [["cuisine", "cuisine"], "cuisine", [""], [3]])
def test_attribute_names_must_be_unique_column_names(value):
    with pytest.raises(ValueError, match="candidate_attributes"):
        attribute_names({"candidate_attributes": value})


def test_check_candidate_attributes():
    ids = torch.zeros((2, 3), dtype=torch.long)
    check_candidate_attributes(None, ids, ())
    check_candidate_attributes(torch.ones((2, 3, 2), dtype=torch.long), ids, (2, 2))
    with pytest.raises(ValueError, match="without attribute tables"):
        check_candidate_attributes(torch.ones((2, 3, 2), dtype=torch.long), ids, ())
    with pytest.raises(ValueError, match=r"\[B,C,K\]"):
        check_candidate_attributes(None, ids, (2, 2))
    with pytest.raises(ValueError, match=r"\[B,C,K\]"):
        check_candidate_attributes(torch.ones((2, 3, 1), dtype=torch.long), ids, (2, 2))
    with pytest.raises(ValueError, match="outside"):
        check_candidate_attributes(torch.full((2, 3, 2), 2, dtype=torch.long), ids, (2, 2))


@pytest.mark.parametrize("architecture", ["mlp", "mlp_dot"])
def test_food_ranker_scores_depend_on_attributes(architecture):
    torch.manual_seed(0)
    ranker = FoodRanker(4, 3, 8, 4, architecture=architecture, attribute_sizes=(3, 5),
                        attribute_dim=2).eval()
    long_term, short_term = torch.randn(2, 4), torch.randn(2, 4)
    ids = torch.zeros((2, 2), dtype=torch.long)  # both candidates out of ID vocabulary
    attributes = torch.tensor([[[1, 1], [2, 4]], [[1, 1], [2, 4]]])
    scores = ranker(long_term, short_term, ids, attributes)
    assert scores.shape == (2, 2) and not torch.equal(scores[:, 0], scores[:, 1])
    plain = FoodRanker(4, 3, 8, 4, architecture=architecture)
    assert not any(name.startswith("attributes") for name in plain.state_dict())
    with pytest.raises(ValueError, match="without attribute tables"):
        plain(long_term, short_term, ids, attributes)


def test_encoder_device_resolution(monkeypatch):
    assert encoder_device() == torch.device("cpu")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert encoder_device("auto") == torch.device("cpu")
    with pytest.raises(ValueError, match="CUDA is unavailable"):
        encoder_device("cuda")
    with pytest.raises(ValueError, match="encoder_device"):
        encoder_device("tpu")


def test_embed_users_on_cpu_foundation_is_a_plain_call():
    class Foundation(torch.nn.Module):
        def embed_users(self, batch):
            return {"long_term": batch, "short_term": batch}

    batch = torch.ones(2, 3)
    assert embed_users_on_device(Foundation(), batch)["long_term"] is batch
