import copy
import dataclasses
import hashlib
import json
from collections import Counter
from datetime import datetime
from functools import lru_cache

import numpy as np
import pyarrow.parquet as pq
import pytest

import src.simulation.export_food as export_food_module
from src.evaluation.metrics import ranking_metrics
from src.simulation.behavior import food_logits, simulate
from src.simulation.export_food import (
    CANDIDATE_POLICY,
    FoodRequest,
    build_food_requests,
    candidate_stats,
    cluster_sizes,
    decay_sample,
    export_food,
    popularity_deciles,
    popularity_units,
    zone_groups,
)
from src.simulation.history import PROFILE_FIELDS, last_history
from src.simulation.timeutil import day_start_ts, local_hour, local_weekday, split_of
from src.simulation.world import build_world, haversine_km

# spec §8: a ranker using candidate attributes only may beat popularity by at most this much
SHORTCUT_MARGIN = 0.05

SPLITS = ("train", "validation", "test")
CANDIDATE_KEYS = {"item_id", "cuisine", "price_level", "geohash"}
# public context/action keys of history events (spec §6.1/§6.2); nothing hidden
ATTRIBUTE_KEYS = {
    "pickup_geohash", "dropoff_geohash", "pickup_zone_type", "dropoff_zone_type", "vehicle_type",
    "driver_id", "fare", "distance_km", "wait_minutes",
    "merchant_id", "merchant_name", "cuisine", "price_level", "search_query", "dish_id",
    "dish_name", "order_amount", "delivery_distance_km",
    "product", "industry_main", "industry_sub", "caption",
}


# ---- helpers ------------------------------------------------------------------------------


@lru_cache(maxsize=4)
def _simulate(config):
    rng = np.random.default_rng(config.seed)
    return simulate(build_world(config, rng), rng), rng


def requests_for(config):
    """World + simulation + requests, each from a fresh generator seeded with ``config.seed``."""
    rng = np.random.default_rng(config.seed)
    sim = simulate(build_world(config, rng), rng)
    return build_food_requests(sim, rng)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def zone_of(world):
    return {m.merchant_id: m.zone_id for m in world.merchants}


def nearest_mall_zone(world, zone_id):
    home = world.zones[zone_id]
    malls = [z for z in world.zones if z.zone_type == "mall"]
    return min(malls, key=lambda z: (haversine_km(home, z), z.zone_id)).zone_id


def third_anchor(world, user):
    return user.work_zone if user.work_zone is not None else nearest_mall_zone(
        world, user.home_zone)


def train_counts(sim):
    limit = day_start_ts(43, sim.world.config)
    counts = Counter(s.merchant_id for s in sim.food_sessions if s.order_ts < limit)
    return {m.merchant_id: counts.get(m.merchant_id, 0) for m in sim.world.merchants}


def decile_map(sim):
    """Independent re-computation: order counts of weeks 1-6, ordinal rank, rank*10//n."""
    counts = train_counts(sim)
    merchants = sim.world.merchants
    order = sorted(range(len(merchants)),
                   key=lambda i: (counts[merchants[i].merchant_id], i))
    return {merchants[i].merchant_id: rank * 10 // len(merchants)
            for rank, i in enumerate(order)}


def mean_ndcg(requests, scorer):
    values = []
    for r in requests:
        rows = [(c, int(c == r.positive_id), float(s))
                for c, s in zip(r.candidate_ids, scorer(r))]
        values.append(ranking_metrics(rows, ks=(10,))["ndcg_at_10"])
    return float(np.mean(values))


class CandidateOnlyRankers:
    """Rankers that see only the candidate set + public catalog + week 1-6 order counts."""

    def __init__(self, sim):
        world = sim.world
        self.counts = train_counts(sim)
        self.position = {m.merchant_id: i for i, m in enumerate(world.merchants)}
        self.decile = decile_map(sim)
        self.pair = {m: d // 2 for m, d in self.decile.items()}
        self.centred_window = self._centred_windows()
        self._orders = {}
        self.zone = {m.merchant_id: world.zones[m.zone_id] for m in world.merchants}
        self.zid = {m.merchant_id: m.zone_id for m in world.merchants}
        n_zones = len(world.zones)
        self.zone_dist = np.array([[haversine_km(a, b) for b in world.zones] for a in world.zones])
        self.zone_order = [list(np.lexsort((np.arange(n_zones), self.zone_dist[a])))
                           for a in range(n_zones)]
        self.pair_zone_counts = {}
        for m in world.merchants:
            vec = self.pair_zone_counts.setdefault(self.pair[m.merchant_id], np.zeros(n_zones))
            vec[m.zone_id] += 1
        self.kernel = np.exp(-self.zone_dist / world.config.food_candidate_tau_km)
        self.nearest3 = {}
        for m in world.merchants:
            others = [o for o in world.merchants
                      if o.merchant_id != m.merchant_id
                      and self.pair[o.merchant_id] == self.pair[m.merchant_id]]
            others.sort(key=lambda o: (haversine_km(world.zones[m.zone_id], world.zones[o.zone_id]),
                                       o.zone_id, o.merchant_id))
            self.nearest3[m.merchant_id] = [o.merchant_id for o in others[:3]]

    def popularity(self, r):
        return [self.counts[c] for c in r.candidate_ids]

    def pair_match(self, r):
        """(i) does the candidate's own popularity pair contain the whole set? then popularity."""
        pairs = {self.pair[c] for c in r.candidate_ids}
        return [1e6 * (pairs == {self.pair[c]}) + self.counts[c] for c in r.candidate_ids]

    def centred_window_match(self, r):
        """(i') the I2 leak: would a window centred on the candidate give the set's decile span?

        Merchant-centred window = narrowest decile range around the merchant with >= 10
        merchants, towards the nearer neighbour (mean |rank difference|), then popularity.
        """
        span = (min(self.decile[c] for c in r.candidate_ids),
                max(self.decile[c] for c in r.candidate_ids))
        return [1e6 * (self.centred_window[c] == span) + self.counts[c] for c in r.candidate_ids]

    def _centred_windows(self):
        ranks = sorted(self.decile, key=lambda m: (self.counts[m], self.position[m]))
        rank = {m: i for i, m in enumerate(ranks)}
        windows = {}
        for m, d in self.decile.items():
            for width in range(1, 11):
                options = []
                for lo in range(max(0, d - width + 1), min(d, 10 - width) + 1):
                    members = [rank[o] for o, od in self.decile.items() if lo <= od < lo + width]
                    if len(members) >= 10:
                        spread = sum(abs(x - rank[m]) for x in members) / (len(members) - 1)
                        options.append((spread, lo, lo + width - 1))
                if options:
                    windows[m] = min(options)[1:]
                    break
        return windows

    def nearest3_in_set(self, r):
        """(ii) how many of the candidate's 3 nearest same-pair merchants are in the set."""
        members = set(r.candidate_ids)
        return [sum(o in members for o in self.nearest3[c]) for c in r.candidate_ids]

    def nearest3_all_in_set(self, r):
        return [float(score == 3) for score in self.nearest3_in_set(r)]

    def density(self, radius_km):
        """(iii) number of other candidates within ``radius_km``."""
        def score(r):
            zs = [self.zone[c] for c in r.candidate_ids]
            return [sum(haversine_km(a, b) <= radius_km for b in zs) - 1 for a in zs]
        return score

    def soft_density(self, r):
        zs = [self.zone[c] for c in r.candidate_ids]
        return [sum(np.exp(-haversine_km(a, b)) for b in zs) for a in zs]

    def saturation_depth(self, r):
        """(iv) policy-aware: how many of the same-pair merchants nearest to the candidate's
        zone (whole zones, in the published fill order: centre distance, zone id) are in the
        set before the first zone that is not fully in it."""
        members = set(r.candidate_ids)
        scores = []
        for c in r.candidate_ids:
            depth = 0
            for block in self._fill_order(c):
                inside = sum(m in members for m in block)
                depth += inside
                if inside < len(block):
                    break
            scores.append(depth)
        return scores

    # ---- fix round 3 (R10): replay and centrality rankers ----------------------------------

    def centrality_mean(self, r):
        """(v) the candidate closest on average to the other 9."""
        ids = [self.zid[c] for c in r.candidate_ids]
        return [-np.mean([self.zone_dist[a, b] for j, b in enumerate(ids) if j != i])
                for i, a in enumerate(ids)]

    def centrality_nearest3(self, r):
        """(v') the candidate closest on average to its 3 nearest other candidates."""
        ids = [self.zid[c] for c in r.candidate_ids]
        scores = []
        for i, a in enumerate(ids):
            near = sorted(self.zone_dist[a, b] for j, b in enumerate(ids) if j != i)[:3]
            scores.append(-float(np.mean(near)))
        return scores

    def _replay(self, remaining, got, anchor, quota, limit):
        """Deterministic zone-count fill (the R5/R8 published policy); False when it would
        need more than ``limit`` per zone (pruning) or the pool runs out."""
        need = quota
        for zone in self.zone_order[anchor]:
            have = remaining.get(zone, 0)
            if have:
                take = min(need, have)
                got[zone] += take
                if got[zone] > limit[zone]:
                    return False
                remaining[zone] = have - take
                need -= take
                if not need:
                    return True
        return False

    def replay_consistency(self, r):
        """(vi) the reviewer's probe: for each candidate as the hypothetical positive, count the
        deterministic-fill hypotheses ([(zone, 9)], [(zone, 4), (Z, 5)] for every zone Z, and
        the three-anchor (4,3,3) permutations over candidate zones) that reproduce the other 9
        candidates' zones exactly."""
        counts = self.pair_zone_counts[self.pair[r.candidate_ids[0]]]
        ids = [self.zid[c] for c in r.candidate_ids]
        zones_in_set = sorted(set(ids))
        scores = []
        for i, zc in enumerate(ids):
            actual = Counter(z for j, z in enumerate(ids) if j != i)
            base = {z: int(v) for z, v in enumerate(counts) if v}
            base[zc] -= 1
            hits = 0
            hypotheses = [[(zc, 9)]]
            hypotheses += [[(zc, 4), (z, 5)] for z in range(len(counts)) if z != zc]
            hypotheses += [[(zc, s[0] - 1), (za, s[1]), (zb, s[2])]
                           for za in zones_in_set for zb in zones_in_set
                           if len({zc, za, zb}) == 3 for s in R8_SIZES_3]
            for quotas in hypotheses:
                remaining, got = dict(base), Counter()
                if all(self._replay(remaining, got, a, q, actual) for a, q in quotas):
                    hits += got == actual
            scores.append(hits)
        return scores

    def replay_count(self, r):
        return [1e6 * k + self.counts[c]
                for k, c in zip(self.replay_consistency(r), r.candidate_ids)]

    def replay_binary(self, r):
        return [1e6 * (k > 0) + self.counts[c]
                for k, c in zip(self.replay_consistency(r), r.candidate_ids)]

    def r10_likelihood(self, r, share=0.4):
        """(vii) likelihood of the other 9 zones with the candidate as the positive under the
        R10 model: ordered-cluster centre ~ exp(-d/tau) around the candidate's zone, a share
        of the negatives ~ exp(-d/tau) around that centre over the pair's merchants, the rest
        from clusters around unknown (uniform) centres."""
        counts = self.pair_zone_counts[self.pair[r.candidate_ids[0]]]
        ids = np.array([self.zid[c] for c in r.candidate_ids])
        scores = []
        for i, zc in enumerate(ids):
            others = np.delete(ids, i)
            available = counts.copy()
            available[zc] -= 1
            norm = self.kernel @ available                      # per centre z
            dens = self.kernel[:, others] / norm[:, None]       # [centre, negative]
            background = dens.mean(axis=0)
            prior = self.kernel[zc] / self.kernel[zc].sum()
            loglik = np.log(share * dens + (1 - share) * background[None, :]).sum(axis=1)
            terms = np.log(prior) + loglik
            top = terms.max()
            scores.append(float(top + np.log(np.exp(terms - top).sum())))
        return scores

    def _fill_order(self, c):
        key = (self.pair[c], self.zone[c].zone_id)
        if key not in self._orders:
            blocks = {}
            for m, pair in self.pair.items():
                if pair == key[0]:
                    blocks.setdefault(self.zone[m].zone_id, []).append(m)
            origin = self.zone[c]
            zones = {self.zone[blk[0]].zone_id: self.zone[blk[0]] for blk in blocks.values()}
            order = sorted(blocks, key=lambda z: (haversine_km(origin, zones[z]), z))
            self._orders[key] = [blocks[z] for z in order]
        return self._orders[key]


def _rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# ---- fixtures -----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def food_config(tiny_config):
    # 60 merchants (as in smoke) -> 6 per popularity decile, 12 per decile pair
    return dataclasses.replace(tiny_config, n_merchants=60)


@pytest.fixture(scope="module")
def food_sim(food_config):
    return _simulate(food_config)[0]


@pytest.fixture(scope="module")
def world(food_sim):
    return food_sim.world


@pytest.fixture(scope="module")
def food_requests(food_config):
    return requests_for(food_config)


@pytest.fixture(scope="module")
def popularity_decile(food_sim):
    return decile_map(food_sim)


@pytest.fixture(scope="module")
def dense_config(tiny_config):
    """Full catalog density (600 merchants, 50 zones) with few users so it stays fast."""
    return dataclasses.replace(tiny_config, n_users=300, n_merchants=600)


@pytest.fixture(scope="module")
def dense_sim(dense_config):
    return _simulate(dense_config)[0]


@pytest.fixture(scope="module")
def dense_requests(dense_config):
    return requests_for(dense_config)


@pytest.fixture(scope="module")
def dense_rankers(dense_sim):
    return CandidateOnlyRankers(dense_sim)


@pytest.fixture(scope="module")
def exported(food_sim, food_requests, tmp_path_factory):
    out = tmp_path_factory.mktemp("food_export")
    summary = export_food(food_sim, food_requests, out)
    return out / "food", summary


@pytest.fixture(scope="module")
def food_root(exported):
    return exported[0]


@pytest.fixture(scope="module")
def food_jsonl_rows(food_root):
    return _rows(food_root / "samples.jsonl")


# ---- tests from the brief -----------------------------------------------------------------


def test_candidates_are_ten_unique_popularity_matched(food_requests, popularity_decile):
    assert food_requests
    for r in food_requests:
        assert len(r.candidate_ids) == 10 == len(set(r.candidate_ids))
        assert r.positive_id in r.candidate_ids
        deciles = {popularity_decile[c] for c in r.candidate_ids}
        assert max(deciles) - min(deciles) <= 1  # cùng hoặc kề thập phân vị
        assert len({d // 2 for d in deciles}) == 1  # one fixed decile pair (R6)


def negatives_near(sim, requests, anchor_of, radius_km=3.0):
    """Mean number of negatives within ``radius_km`` of ``anchor_of(request)`` and the mean
    that drawing the 9 negatives uniformly from the same-pair pool would give."""
    world = sim.world
    zones = zone_of(world)
    pair = {m: d // 2 for m, d in decile_map(sim).items()}
    near = uniform = 0.0
    for r in requests:
        anchor = world.zones[anchor_of(r)]

        def close(mid, anchor=anchor):
            return haversine_km(anchor, world.zones[zones[mid]]) <= radius_km

        near += sum(close(c) for c in r.candidate_ids if c != r.positive_id)
        pool = [m for m in zones if pair[m] == pair[r.positive_id] and m != r.positive_id]
        uniform += 9 * sum(close(m) for m in pool) / len(pool)
    return near / len(requests), uniform / len(requests)


def test_positive_zone_is_not_isolated(dense_sim, dense_requests):
    """No "quán lạc khu": the ordered merchant's cluster is drawn around a centre near its
    zone, so negatives sit near that zone far more than uniform (R10: on average). Checked at
    full catalog density: with 12 merchants per pair (smoke) any sampler takes most of the
    few nearby same-pair merchants, so the comparison would say nothing."""
    zones = zone_of(dense_sim.world)
    near, uniform = negatives_near(dense_sim, dense_requests, lambda r: zones[r.positive_id])
    assert near >= 2.0 and near >= 3 * uniform, (near, uniform)


def test_no_location_leak_in_inputs(food_jsonl_rows):
    for row in food_jsonl_rows:
        assert set(row["input"]["context"]) == {"hour", "weekday"}
        assert all(set(c) == CANDIDATE_KEYS for c in row["input"]["candidates"])
        assert set(row["input"]["profile"]) == {"as_of_time", *PROFILE_FIELDS}
        for event in row["input"]["history"]:
            assert set(event) == {"event_id", "event_time", "service", "event_type",
                                  "entity_id", "attributes"}
            assert set(event["attributes"]) <= ATTRIBUTE_KEYS


def test_session_events_and_future_excluded(food_jsonl_rows):
    for row in food_jsonl_rows:
        cutoff = datetime.fromisoformat(row["input"]["cutoff_time"])
        history = row["input"]["history"]
        assert all(datetime.fromisoformat(e["event_time"]) < cutoff for e in history)
        assert datetime.fromisoformat(row["label_metadata"]["label_time"]) > cutoff
        assert row["label_metadata"]["observation_complete"] is True


def test_homebody_uses_mall_group(dense_sim, dense_requests):
    """Users without a work place get a cluster around the mall nearest home (on average)."""
    world = dense_sim.world
    homebodies = {u.user_id: u for u in world.users if u.persona == "home"}
    requests = [r for r in dense_requests if r.session.user_id in homebodies]
    assert len(requests) > 100
    near, uniform = negatives_near(
        dense_sim, requests,
        lambda r: nearest_mall_zone(world, homebodies[r.session.user_id].home_zone))
    assert near >= 2.0 and near >= 3 * uniform, (near, uniform)


def test_sparse_decile_falls_back_deterministically(tiny_config):
    sparse = dataclasses.replace(tiny_config, n_merchants=12)  # thiếu quán cùng thập phân vị
    a, b = requests_for(sparse), requests_for(sparse)
    assert a and [r.candidate_ids for r in a] == [r.candidate_ids for r in b]
    assert all(len(r.candidate_ids) == 10 == len(set(r.candidate_ids)) for r in a)
    assert all(r.positive_id in r.candidate_ids for r in a)


# ---- full catalog density (fix rounds 1-3: R6, R8, R10, spec §8 candidate-only rankers) ----


R8_SIZES_3 = ((4, 3, 3), (3, 4, 3), (3, 3, 4))


def allowed_sizes(n_anchors):
    """R8 cluster sizes (ordered merchant included) for 1-3 distinct anchors."""
    return {1: {(10,)}, 2: {(5, 5)}, 3: set(R8_SIZES_3)}[n_anchors]


def anchors_of(world, users, zones, request):
    user = users[request.session.user_id]
    return list(dict.fromkeys((zones[request.positive_id], user.home_zone,
                               third_anchor(world, user))))


def requests_with_sizes(config):
    """``requests_for(config)`` plus the cluster sizes drawn for each request, recorded by a
    spy on ``export_food.cluster_sizes`` (called once per request, in request order)."""
    drawn = []
    real = export_food_module.cluster_sizes

    def spy(n_anchors, rng):
        sizes = real(n_anchors, rng)
        drawn.append(sizes)
        return sizes

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(export_food_module, "cluster_sizes", spy)
        requests = requests_for(config)
    assert len(drawn) == len(requests)
    return requests, drawn


def assert_allowed_sizes(sim, requests, drawn):
    """Each request drew R8 sizes for its number of distinct anchors."""
    world = sim.world
    zones = zone_of(world)
    users = {u.user_id: u for u in world.users}
    for r, sizes in zip(requests, drawn):
        assert sizes in allowed_sizes(len(anchors_of(world, users, zones, r))), r.request_id


def assert_exchangeable_sizes(drawn):
    """In 3-anchor requests each cluster (ordered, home, work/mall) has size 4 w.p. 1/3."""
    three = [s for s in drawn if len(s) == 3]
    assert len(three) > 300
    shares = [sum(s[k] == 4 for s in three) / len(three) for k in range(3)]
    assert all(abs(share - 1 / 3) <= 0.05 for share in shares), shares


def test_dense_cluster_sizes_follow_r8(dense_config, dense_sim, dense_requests):
    requests, drawn = requests_with_sizes(dense_config)
    assert requests == dense_requests
    assert_allowed_sizes(dense_sim, requests, drawn)
    assert_exchangeable_sizes(drawn)
    assert {len(s) for s in drawn} >= {2, 3}


def test_smoke_catalog_cluster_sizes_follow_r8(food_config, food_sim, food_requests):
    requests, drawn = requests_with_sizes(food_config)
    assert requests == food_requests
    assert_allowed_sizes(food_sim, requests, drawn)
    assert_exchangeable_sizes(drawn)


def anchor_distance_shares(sim, requests, radius_km=3.0):
    """Share of negatives within ``radius_km`` of their request's nearest anchor zone, and the
    same share for every other same-pair merchant (what uniform sampling would give)."""
    world = sim.world
    zones = zone_of(world)
    pair = {m: d // 2 for m, d in decile_map(sim).items()}
    users = {u.user_id: u for u in world.users}
    near = total = near_pool = total_pool = 0
    for r in requests:
        anchors = [world.zones[a] for a in anchors_of(world, users, zones, r)]

        def close(mid, anchors=anchors):
            return min(haversine_km(a, world.zones[zones[mid]]) for a in anchors) <= radius_km

        negatives = [c for c in r.candidate_ids if c != r.positive_id]
        near += sum(close(c) for c in negatives)
        total += len(negatives)
        pool = [m for m in zones if pair[m] == pair[r.positive_id] and m != r.positive_id]
        near_pool += sum(close(m) for m in pool)
        total_pool += len(pool)
    return near / total, near_pool / total_pool


def test_dense_negatives_decay_with_distance_from_anchors(dense_sim, dense_requests):
    """R10: negatives are drawn ~exp(-d/tau) around cluster centres, so they sit near the
    anchors far more often than uniform same-pair merchants (no per-request guarantee)."""
    held_out = [r for r in dense_requests if r.split != "train"]
    share, uniform = anchor_distance_shares(dense_sim, held_out)
    assert share >= 0.5 and share >= 3 * uniform, (share, uniform)


def test_decay_sample_draws_without_replacement_proportional_to_kernel():
    distances = np.array([0.0, 1.0, 2.0, 4.0, 8.0])
    rng = np.random.default_rng(3)
    for k in (1, 3, 5):
        picks = decay_sample(rng, distances, k, 1.5)
        assert len(picks) == k == len(set(picks.tolist()))
    trials = 20000
    first = Counter(int(decay_sample(rng, distances, 1, 1.5)[0]) for _ in range(trials))
    expected = np.exp(-distances / 1.5) / np.exp(-distances / 1.5).sum()
    for i, p in enumerate(expected):
        assert abs(first[i] / trials - p) < 0.015, (i, first[i] / trials, p)
    # the second draw is proportional to the kernel over what is left
    seconds = Counter()
    for _ in range(trials):
        a, b = decay_sample(rng, distances, 2, 1.5).tolist()
        if a == 0:
            seconds[b] += 1
    rest = np.exp(-distances[1:] / 1.5) / np.exp(-distances[1:] / 1.5).sum()
    n = sum(seconds.values())
    for i, p in enumerate(rest, start=1):
        assert abs(seconds[i] / n - p) < 0.02, (i, seconds[i] / n, p)
    with pytest.raises(ValueError):
        decay_sample(rng, distances, 2, 0.0)


def test_dense_candidates_share_one_decile_pair(dense_sim, dense_requests):
    """(b) all 10 candidates in the ordered merchant's decile pair; no last resort needed."""
    decile = decile_map(dense_sim)
    assert all(len({decile[c] // 2 for c in r.candidate_ids}) == 1 for r in dense_requests)
    stats = candidate_stats(dense_sim, dense_requests)
    assert stats["last_resort_requests"] == 0


def test_dense_candidate_only_rankers_do_not_beat_popularity(dense_requests, dense_rankers):
    """spec §8: no candidate-attribute-only ranker beats popularity by more than 0.05."""
    held_out = [r for r in dense_requests if r.split != "train"]
    assert len(held_out) > 500
    rankers = dense_rankers
    popularity = mean_ndcg(held_out, rankers.popularity)
    scores = {
        "pair_match": mean_ndcg(held_out, rankers.pair_match),
        "nearest3_in_set": mean_ndcg(held_out, rankers.nearest3_in_set),
        "nearest3_all_in_set": mean_ndcg(held_out, rankers.nearest3_all_in_set),
        "density_1.5km": mean_ndcg(held_out, rankers.density(1.5)),
        "density_3km": mean_ndcg(held_out, rankers.density(3.0)),
        "soft_density": mean_ndcg(held_out, rankers.soft_density),
    }
    assert all(v <= popularity + SHORTCUT_MARGIN for v in scores.values()), (popularity, scores)


def test_dense_policy_aware_rankers_do_not_beat_popularity(dense_requests, dense_rankers):
    """spec §8 at full density (R9) against rankers that know the published policy: the
    largest-cluster depth (iv), centrality (v, v'), the reviewer's exact replay of a
    deterministic fill (vi, count and binary) and the R10-model likelihood (vii)."""
    held_out = [r for r in dense_requests if r.split != "train"]
    rankers = dense_rankers
    popularity = mean_ndcg(held_out, rankers.popularity)
    scores = {
        "largest_cluster_depth": mean_ndcg(held_out, rankers.saturation_depth),
        "centrality_mean": mean_ndcg(held_out, rankers.centrality_mean),
        "centrality_nearest3": mean_ndcg(held_out, rankers.centrality_nearest3),
        "r10_likelihood": mean_ndcg(held_out, rankers.r10_likelihood),
        # the reverse direction too: the positive must not be systematically off-centre
        "least_central": mean_ndcg(held_out, lambda r: [-s for s in rankers.centrality_mean(r)]),
        "least_likely": mean_ndcg(held_out, lambda r: [-s for s in rankers.r10_likelihood(r)]),
    }
    assert all(v <= popularity + SHORTCUT_MARGIN for v in scores.values()), (popularity, scores)
    # the exact replay is slow: every third held-out request, against popularity on the same
    sample = held_out[::3]
    assert len(sample) > 400
    sample_popularity = mean_ndcg(sample, rankers.popularity)
    replay = {"replay_count": mean_ndcg(sample, rankers.replay_count),
              "replay_binary": mean_ndcg(sample, rankers.replay_binary)}
    assert all(v <= sample_popularity + SHORTCUT_MARGIN for v in replay.values()), (
        sample_popularity, replay)


def test_smoke_catalog_window_rankers_do_not_beat_popularity(tiny_config):
    """The I2 leak at smoke catalog size (60 merchants, 6 per decile): with a shared partition
    neither the candidate's pair nor a window centred on it identifies the positive."""
    config = dataclasses.replace(tiny_config, n_users=300, n_merchants=60)
    sim = _simulate(config)[0]
    held_out = [r for r in requests_for(config) if r.split != "train"]
    assert len(held_out) > 1000
    rankers = CandidateOnlyRankers(sim)
    popularity = mean_ndcg(held_out, rankers.popularity)
    assert mean_ndcg(held_out, rankers.pair_match) == popularity
    centred = mean_ndcg(held_out, rankers.centred_window_match)
    assert centred <= popularity + SHORTCUT_MARGIN, (popularity, centred)


def test_manifest_and_split_counts(food_root, food_requests, world):
    manifest = json.loads((food_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["split"]["kind"] == "temporal"
    assert manifest["candidate_policy"] == "sim-popularity-matched-home-work-v1"
    assert sha256(food_root / "samples.jsonl") == manifest["files"]["samples"]["sha256"]
    assert sha256(food_root / "merchants.parquet") == manifest["files"]["merchants"]["sha256"]
    assert manifest["files"]["samples"]["row_count"] == len(food_requests)
    assert manifest["files"]["merchants"]["row_count"] == len(world.merchants)
    assert manifest["split"]["train_end"] == "2026-02-16T00:00:00+07:00"
    assert manifest["split"]["valid_end"] == "2026-02-23T00:00:00+07:00"
    assert manifest["schema_version"] == 1 and manifest["timezone"] == "Asia/Ho_Chi_Minh"
    assert manifest["dataset_version"] == world.config.dataset_version
    assert manifest["seed"] == world.config.seed
    expected = Counter(r.split for r in food_requests)
    assert manifest["request_counts"] == {s: expected.get(s, 0) for s in SPLITS}
    assert all(expected[s] > 0 for s in SPLITS)


# ---- requests -----------------------------------------------------------------------------


def test_requests_cover_every_ordered_session_with_anchor_at_start(food_sim, food_requests):
    assert food_sim.world.config.food_requests is None
    assert sorted(r.session.session_id for r in food_requests) == sorted(
        s.session_id for s in food_sim.food_sessions)
    assert [r.request_id for r in food_requests] == [
        f"sim:food_req:{i}" for i in range(len(food_requests))]
    assert [r.cutoff_ts for r in food_requests] == sorted(r.cutoff_ts for r in food_requests)
    config = food_sim.world.config
    for r in food_requests:
        assert isinstance(r, FoodRequest)
        assert r.cutoff_ts == r.session.start_ts - 1
        assert r.positive_id == r.session.merchant_id
        assert r.split == split_of(r.cutoff_ts, config)
        assert food_logits(food_sim.world, r.session, r.candidate_ids).shape == (10,)


def test_sampling_takes_configured_count_per_split(food_config, food_sim):
    sampled = dataclasses.replace(food_config, food_requests={"train": 40, "validation": 15,
                                                              "test": 12})
    sim = dataclasses.replace(food_sim, world=dataclasses.replace(food_sim.world, config=sampled))
    a = build_food_requests(sim, np.random.default_rng(7))
    b = build_food_requests(sim, np.random.default_rng(7))
    c = build_food_requests(sim, np.random.default_rng(8))
    assert a == b and a != c
    assert Counter(r.split for r in a) == {"train": 40, "validation": 15, "test": 12}
    sessions = {s.session_id for s in food_sim.food_sessions}
    assert len({r.session.session_id for r in a}) == 67 and all(
        r.session.session_id in sessions for r in a)
    assert [r.request_id for r in a] == [f"sim:food_req:{i}" for i in range(67)]


def test_sampling_rejects_unknown_split(food_config, food_sim):
    bad = dataclasses.replace(food_config, food_requests={"train": 5, "valid": 1})
    sim = dataclasses.replace(food_sim, world=dataclasses.replace(food_sim.world, config=bad))
    with pytest.raises(ValueError, match="valid"):
        build_food_requests(sim, np.random.default_rng(0))


def test_candidate_order_is_shuffled(food_requests):
    positions = Counter(r.candidate_ids.index(r.positive_id) for r in food_requests)
    assert len(positions) == 10 and max(positions.values()) < 0.2 * len(food_requests)


def test_zone_groups_use_exchangeable_cluster_sizes():
    # R8: sizes count the ordered merchant toward its own (first) cluster
    assert zone_groups(1, 2, 3, (4, 3, 3)) == [(1, 3), (2, 3), (3, 3)]
    assert zone_groups(1, 2, 3, (3, 4, 3)) == [(1, 2), (2, 4), (3, 3)]
    assert zone_groups(1, 2, 3, (3, 3, 4)) == [(1, 2), (2, 3), (3, 4)]
    assert zone_groups(1, 1, 3, (5, 5)) == [(1, 4), (3, 5)]  # khu quán = nhà
    assert zone_groups(1, 2, 1, (5, 5)) == [(1, 4), (2, 5)]  # khu quán = chỗ làm
    assert zone_groups(1, 2, 2, (5, 5)) == [(1, 4), (2, 5)]  # nhà = chỗ làm
    assert zone_groups(4, 4, 4, (10,)) == [(4, 9)]
    with pytest.raises(ValueError):
        zone_groups(1, 2, 3, (5, 5))
    rng = np.random.default_rng(0)
    draws = Counter(cluster_sizes(3, rng) for _ in range(3000))
    assert set(draws) == set(R8_SIZES_3)
    assert all(abs(v / 3000 - 1 / 3) < 0.03 for v in draws.values())
    assert cluster_sizes(2, rng) == (5, 5) and cluster_sizes(1, rng) == (10,)


def test_popularity_units_are_a_fixed_decile_pair_partition():
    assert popularity_deciles(list(range(60))) == [r // 6 for r in range(60)]
    assert popularity_deciles([5] * 20) == [r // 2 for r in range(20)]  # ties: catalog order
    assert popularity_deciles([59 - i for i in range(60)]) == [(59 - i) // 6 for i in range(60)]
    assert popularity_units(list(range(600))) == [r // 120 for r in range(600)]
    assert popularity_units(list(range(60))) == [r // 12 for r in range(60)]
    assert popularity_units([599 - i for i in range(600)]) == [(599 - i) // 120
                                                               for i in range(600)]
    # last resort (pairs of < 10 merchants): neighbouring pairs merged bottom-up into units of
    # >= 10, a short top remainder joins the unit below -> still one shared partition
    assert popularity_units(list(range(24))) == [0] * 10 + [1] * 14
    assert popularity_units(list(range(12))) == [0] * 12
    with pytest.raises(ValueError, match="10"):
        popularity_units(list(range(9)))


# ---- export -------------------------------------------------------------------------------


def test_rows_match_requests_labels_and_history(food_sim, food_requests, food_jsonl_rows):
    config = food_sim.world.config
    by_user = {}
    for event in food_sim.events:
        by_user.setdefault(event.user_id, []).append(event)
    session_of = {e.event_id: e.session_id for e in food_sim.events}
    assert len(food_jsonl_rows) == len(food_requests)
    for row, r in zip(food_jsonl_rows, food_requests):
        inp, meta = row["input"], row["label_metadata"]
        assert row["sample_id"] == meta["request_id"] == r.request_id
        assert inp["user_id"] == r.session.user_id and meta["split"] == r.split
        assert [c["item_id"] for c in inp["candidates"]] == list(r.candidate_ids)
        assert [x["item_id"] for x in row["labels"]] == list(r.candidate_ids)
        assert [x["relevance"] for x in row["labels"]] == [
            int(c == r.positive_id) for c in r.candidate_ids]
        assert meta["candidate_source"] == CANDIDATE_POLICY
        assert meta["prediction_anchor"] == "session_start"
        assert meta["positive_action"] == "order"
        assert inp["context"] == {"hour": local_hour(r.cutoff_ts, config),
                                  "weekday": local_weekday(r.cutoff_ts, config)}
        assert inp["profile"]["as_of_time"] == inp["cutoff_time"]
        expected, group_truncated = last_history(by_user[r.session.user_id], r.cutoff_ts, 30)
        assert [e["event_id"] for e in inp["history"]] == [e.event_id for e in expected]
        assert inp["history_group_truncated"] is group_truncated
        assert all(session_of[e["event_id"]] != r.session.session_id for e in inp["history"])
        assert len(inp["history"]) <= 30


def test_history_event_shape(food_sim, food_jsonl_rows):
    events = {e.event_id: e for e in food_sim.events}
    config = food_sim.world.config
    seen = set()
    for row in food_jsonl_rows[:300]:
        for item in row["input"]["history"]:
            e = events[item["event_id"]]
            assert datetime.fromisoformat(item["event_time"]).utcoffset().total_seconds() == 25200
            assert int(datetime.fromisoformat(item["event_time"]).timestamp()) == e.ts
            assert item["service"] == e.service and item["event_type"] == e.event_type
            assert item["entity_id"] == e.entity_id
            public = {**e.context, **{k: v for k, v in e.action.items() if k != "type"}}
            assert item["attributes"] == public
            seen.add(e.service)
    assert seen == {"ride", "food", "ads"}
    assert config.utc_offset_hours == 7


def test_merchants_catalog_is_public_only(food_root, world):
    table = pq.read_table(food_root / "merchants.parquet")
    assert table.column_names == ["merchant_id", "merchant_name", "cuisine", "price_level",
                                  "geohash", "zone_type"]
    rows = table.to_pylist()
    assert [r["merchant_id"] for r in rows] == [m.merchant_id for m in world.merchants]
    for row, m in zip(rows, world.merchants):
        assert row["geohash"] == world.zones[m.zone_id].geohash
        assert row["price_level"] == m.price_level and row["cuisine"] == m.cuisine


def test_rows_pass_food_samples_validator(food_jsonl_rows):
    from src.downstream.food_samples import validate_food_samples

    validate_food_samples(food_jsonl_rows, CANDIDATE_POLICY)


def test_export_is_deterministic_and_never_mutates_events(food_sim, food_requests, food_root,
                                                          tmp_path):
    before = copy.deepcopy([(e.context, e.action) for e in food_sim.events])
    summary = export_food(food_sim, food_requests, tmp_path)
    assert [(e.context, e.action) for e in food_sim.events] == before
    for name in ("samples.jsonl", "merchants.parquet", "manifest.json"):
        assert sha256(tmp_path / "food" / name) == sha256(food_root / name)
    assert summary["requests"] == len(food_requests)
    assert summary["requests_by_split"] == dict(Counter(r.split for r in food_requests))
    with pytest.raises(FileExistsError):
        export_food(food_sim, food_requests, tmp_path)
