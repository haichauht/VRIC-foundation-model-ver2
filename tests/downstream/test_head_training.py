"""Downstream heads keep their candidate embeddings under weight decay (decoupled, AdamW)."""

from __future__ import annotations

import math
from types import SimpleNamespace

import torch
from torch import nn

from src.downstream.ad_optimization import AdBatch, AdRanker, train_ad_ranker
from src.downstream.head_training import head_optimizer


def test_head_optimizer_does_not_shrink_embedding_rows_without_gradient():
    torch.manual_seed(0)
    embedding = nn.Embedding(100, 8)
    before = embedding.weight.detach().clone()
    optimizer = head_optimizer(embedding.parameters(), learning_rate=1e-3, weight_decay=1e-4)
    for _ in range(500):
        optimizer.zero_grad()
        embedding(torch.tensor([0])).sum().backward()
        optimizer.step()
    ratio = embedding.weight[1:].detach().norm(dim=1) / before[1:].norm(dim=1)
    assert float(ratio.min()) > 0.99


class NoiseFoundation(nn.Module):
    """Frozen random user vectors (like the random-init arm): no click information."""

    def __init__(self, users: int, d_model: int):
        super().__init__()
        self.users = nn.Embedding(users, d_model)

    def embed_users(self, batch):
        vectors = self.users(batch.user_ids)
        return {"long_term": vectors, "short_term": vectors}


def _per_ad_ctr_batches(ads=40, exposures=4000, batch_size=64, seed=0):
    """Single-candidate exposures; ads 1..ads/2 click at 30%, the rest at 2%."""
    generator = torch.Generator().manual_seed(seed)
    ad_ids = torch.randint(1, ads + 1, (exposures,), generator=generator)
    rates = torch.where(ad_ids <= ads // 2, 0.30, 0.02)
    clicks = (torch.rand(exposures, generator=generator) < rates).long()
    batches = []
    for start in range(0, exposures, batch_size):
        index = range(start, min(start + batch_size, exposures))
        rows = [[{"request_id": f"r{i}", "user_id": f"u{i}", "cutoff_timestamp": i,
                  "split": "train", "clicked": int(clicks[i])}] for i in index]
        foundation = SimpleNamespace(user_ids=torch.tensor(list(index)))
        batches.append(AdBatch(foundation, ad_ids[list(index)][:, None], rows))
    return batches, float(clicks.float().mean())


def test_ads_head_keeps_rarely_seen_ad_embeddings_under_weight_decay():
    # 600 ads, each in ~6% of batches (the full sim-v1 run: 1,500 ads, ~4%). Coupled Adam decay
    # shrinks the median ad embedding to ~0.54x here (to ~0.0002x over the real 31k steps).
    torch.manual_seed(1)
    batches, ctr = _per_ad_ctr_batches(ads=600, exposures=24000)
    base = -(ctr * math.log(ctr) + (1 - ctr) * math.log(1 - ctr))
    rank = AdRanker(16, 601, hidden_dim=32, candidate_dim=8, dropout=0.1)
    initial = float(rank.candidates.weight[1:].detach().norm(dim=1).median())
    records = train_ad_ranker(rank, NoiseFoundation(24000, 16), batches, epochs=3,
                              learning_rate=1e-3, weight_decay=1e-4, shuffle=True)
    final = float(rank.candidates.weight[1:].detach().norm(dim=1).median())
    assert final > 0.9 * initial
    assert records[-1]["ctr_loss"] < base - 0.003
