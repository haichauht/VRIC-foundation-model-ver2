import json
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import datetime
from functools import lru_cache
from itertools import pairwise

import numpy as np
import pyarrow.parquet as pq
import pytest

from src.baselines.contracts import BaselineDataset, eligible_history
from src.baselines.features import FeatureBuilder
from src.baselines.runner import load_dataset
from src.simulation.behavior import (
    AD_CLICK_DELAY_SECONDS,
    SESSION_GAP_SECONDS,
    food_logits,
    simulate,
)
from src.simulation.export_ads import build_ad_requests
from src.simulation.export_baseline import (
    AD_OUTCOME_MATURITY_SECONDS,
    IMPRESSION_EVENT_TYPE,
    export_baseline,
)
from src.simulation.export_food import build_food_requests
from src.simulation.export_oracle import export_oracle
from src.simulation.timeutil import day_start_ts, local_iso
from src.simulation.world import build_world


@lru_cache(maxsize=2)
def _built(config):
    rng = np.random.default_rng(config.seed)
    sim = simulate(build_world(config, rng), rng)
    return sim, build_food_requests(sim, rng), build_ad_requests(sim, rng)


@pytest.fixture(scope="module")
def built(tiny_config):
    return _built(tiny_config)


@pytest.fixture(scope="module")
def out(built, tmp_path_factory):
    root = tmp_path_factory.mktemp("bo_out")
    export_baseline(*built, root)
    export_oracle(*built, root)
    return root


@pytest.fixture(scope="module")
def baseline_root(out):
    return out / "baseline"


@pytest.fixture(scope="module")
def oracle_root(out):
    return out / "oracle"


def test_baseline_json_loads_with_runner(baseline_root, built):
    food = load_dataset(baseline_root / "food.json")
    ads = load_dataset(baseline_root / "ads.json")
    assert food.metadata.task == "food_recommendation" and ads.metadata.task == "ad_optimization"
    assert food.metadata.item_kind == "merchant" and ads.metadata.item_kind == "ad"
    assert food.metadata.dataset_name == "sim_v1"
    assert food.metadata.availability_policy == "handoff_timestamp"
    assert len(food.candidates) == 10 * len(food.queries) == 10 * len(built[1])
    assert len(ads.candidates) == len(ads.queries) == len(built[2])
    assert sum(label.target for label in food.labels) == len(food.queries)
    assert all(q.cutoff_time.utcoffset().total_seconds() == 7 * 3600 for q in food.queries)


def _week7_start(config) -> datetime:
    """First instant of the validation week (local), the B1 fit / frozen cutoff."""
    return datetime.fromisoformat(local_iso(day_start_ts(7 * (config.weeks - 2) + 1, config),
                                            config))


def _click_times(sim) -> dict[str, int]:
    return {e.session_id: e.ts for e in sim.events if e.event_type == "ad_click"}


def test_label_times_and_history_rules(baseline_root, built):
    sim, _, ad_requests = built
    config = sim.world.config
    food = load_dataset(baseline_root / "food.json")
    ads = load_dataset(baseline_root / "ads.json")
    # Ads labels become available when the outcome is observable: the click, or 27 s after the
    # exposure (one second after the latest possible click) when there was none.
    assert AD_OUTCOME_MATURITY_SECONDS == AD_CLICK_DELAY_SECONDS[1] + 1 == 27
    session_of = {e.event_id: e.session_id for e in sim.events if e.event_type == "ad_impression"}
    clicks = _click_times(sim)
    label_at = {lb.request_id: lb.available_at for lb in ads.labels}
    for r in ad_requests:
        click = clicks.get(session_of[r.impression.event_id])
        assert (click is not None) == r.impression.clicked
        mature = click if r.impression.clicked else r.impression.ts + 27
        assert label_at[r.request_id].isoformat() == local_iso(mature, config)
    for lb in food.labels:
        assert lb.available_at >= next(q for q in food.queries
                                       if q.request_id == lb.request_id).cutoff_time
    for ds in (food, ads):
        users = {q.user_id for q in ds.queries}
        assert {e.user_id for e in ds.history} <= users
        latest = {}
        for q in ds.queries:
            latest[q.user_id] = max(latest.get(q.user_id, q.cutoff_time), q.cutoff_time)
        assert all(e.event_time < latest[e.user_id] for e in ds.history)
        for e in ds.history:
            delay = (e.available_at - e.event_time).total_seconds()
            if e.event_type == IMPRESSION_EVENT_TYPE:
                low, high = AD_CLICK_DELAY_SECONDS
                assert (low <= delay <= high) if e.attributes["clicked"] else delay == 27
            else:
                assert delay == 0
        by_user = defaultdict(int)  # one history row per simulated event (clicks included)
        for e in sim.events:
            if e.user_id in latest and local_iso(e.ts, config) < latest[e.user_id].isoformat():
                by_user[e.user_id] += 1
        assert sum(by_user.values()) == len(ds.history)
        q = ds.queries[-1]
        assert all(e.event_time < q.cutoff_time for e in eligible_history(ds.history, q.cutoff_time))


def test_baseline_follows_b1_conventions(baseline_root, built):
    """docs/b1_features.md: Ads history = ``impression`` rows with ``clicked`` 0/1; candidate
    and history attributes named ``category`` / ``brand`` / ``price``."""
    sim = built[0]
    food = load_dataset(baseline_root / "food.json")
    ads = load_dataset(baseline_root / "ads.json")
    merchants = {m.merchant_id: m for m in sim.world.merchants}
    for c in food.candidates:
        m = merchants[c.candidate_id]
        assert c.attributes["category"] == c.attributes["cuisine"] == m.cuisine
        assert c.attributes["price"] == c.attributes["price_level"] == m.price_level
        assert "brand" not in c.attributes
    catalog = {a.ad_id: a for a in sim.world.ads}
    for c in ads.candidates:
        a = catalog[c.candidate_id]
        assert c.attributes["category"] == a.category
        assert c.attributes["brand"] == c.attributes["advertiser"] == a.advertiser
        assert "price" not in c.attributes  # ads have no price; B1 sees None
    session_of = {e.event_id: e.session_id for e in sim.events if e.event_type == "ad_impression"}
    clicks = _click_times(sim)
    for ds in (food, ads):
        types = Counter(e.event_type for e in ds.history)
        assert "ad_impression" not in types and types[IMPRESSION_EVENT_TYPE] > 0
        assert types["order"] > 0
        for e in ds.history:
            attrs = e.attributes
            if e.event_type == IMPRESSION_EVENT_TYPE:
                assert attrs["service"] == "ads"
                assert type(attrs["clicked"]) is int
                assert attrs["clicked"] == int(session_of[e.event_id] in clicks)
                assert attrs["category"] == catalog[e.entity_id].category
                assert attrs["brand"] == catalog[e.entity_id].advertiser
            elif e.event_type == "order":
                assert attrs["category"] == attrs["cuisine"]
                assert attrs["price"] == attrs["price_level"]
                assert not {"clicked", "brand"} & attrs.keys()
            else:
                assert not {"clicked", "category", "brand", "price"} & attrs.keys()


def test_ad_outcome_matures_before_the_users_next_exposure(built):
    """Clicks follow their impression by 2-26 s and a user's next exposure is >= 60 s later, so
    every earlier impression of a user is mature (B1 ``available_at < cutoff``) at their next
    Ads request."""
    sim = built[0]
    assert AD_CLICK_DELAY_SECONDS == (2, 26) and SESSION_GAP_SECONDS == 60
    assert AD_OUTCOME_MATURITY_SECONDS < SESSION_GAP_SECONDS
    exposure = {e.session_id: e.ts for e in sim.events if e.event_type == "ad_impression"}
    delays = {ts - exposure[sid] for sid, ts in _click_times(sim).items()}
    assert min(delays) >= 2 and max(delays) <= 26
    by_user = defaultdict(list)
    for imp in sim.ad_impressions:
        by_user[imp.user_id].append(imp.ts)
    gaps = [b - a for times in by_user.values() for a, b in pairwise(times)]
    assert gaps and min(gaps) >= SESSION_GAP_SECONDS


def test_export_refuses_impressions_closer_than_the_session_gap(built, tmp_path):
    sim, food_requests, ad_requests = built
    imps = list(sim.ad_impressions)
    k = next(i for i in range(1, len(imps)) if imps[i].user_id == imps[i - 1].user_id)
    imps[k] = replace(imps[k], ts=imps[k - 1].ts + 30)
    broken = replace(sim, ad_impressions=tuple(imps))
    with pytest.raises(ValueError, match="60"):
        export_baseline(broken, food_requests, ad_requests, tmp_path)
    assert not (tmp_path / "baseline").exists()


def _features(rows, key, test=lambda v: v is not None) -> int:
    return sum(bool(test(row.values[key])) for row in rows)


def test_b1_feature_builder_reads_history_and_candidate_attributes(baseline_root, built):
    """The real B1 FeatureBuilder (read-only import) on the exported files: history counts,
    CTRs, categories and prices are populated, as at the validation split's frozen cutoff."""
    frozen = _week7_start(built[0].world.config)
    builder = FeatureBuilder()
    ads = builder.build(load_dataset(baseline_root / "ads.json"), "validation",
                        frozen_cutoff=frozen)
    for key in ("user_count_30d", "item_count_30d", "user_item_count_30d", "user_clicks_30d",
                "item_clicks_30d"):
        assert _features(ads, key, lambda v: v > 0) > 0, key
    for key in ("user_ctr_30d", "item_ctr_30d", "user_item_ctr_30d", "user_category_affinity"):
        assert _features(ads, key) > 0, key
    for key in ("item_category", "item_brand"):
        assert _features(ads, key) == len(ads), key
    assert _features(ads, "item_price") == 0
    food = builder.build(load_dataset(baseline_root / "food.json"), "validation",
                         frozen_cutoff=frozen)
    for key in ("user_count_30d", "item_count_30d", "user_item_count_30d"):
        assert _features(food, key, lambda v: v > 0) > 0, key
    for key in ("item_category", "item_price"):
        assert _features(food, key) == len(food), key
    assert _features(food, "user_category_affinity") > len(food) // 2
    assert _features(food, "user_category_affinity", lambda v: v is not None and v > 0) > 0


def _user_subset(ds: BaselineDataset, n_users: int) -> BaselineDataset:
    users = set(sorted({q.user_id for q in ds.queries},
                       key=lambda u: int(u.rsplit(":", 1)[1]))[:n_users])
    queries = tuple(q for q in ds.queries if q.user_id in users)
    ids = {q.request_id for q in queries}
    return BaselineDataset(ds.metadata, queries,
                           tuple(c for c in ds.candidates if c.request_id in ids),
                           tuple(lb for lb in ds.labels if lb.request_id in ids),
                           tuple(e for e in ds.history if e.user_id in users))


@pytest.mark.parametrize("name", ["ads", "food"])
def test_b1_classifier_fits_at_the_validation_boundary(baseline_root, built, name):
    """B1 accepts ``training_cutoff`` = start of week 7: every train label (Ads: at its outcome
    maturity) is available before it. A user subset keeps per-query train features fast."""
    from src.baselines.lightgbm import B1Classifier

    cutoff = _week7_start(built[0].world.config)
    ds = _user_subset(load_dataset(baseline_root / f"{name}.json"), 20)
    model = B1Classifier(n_estimators=5, min_child_samples=5).fit(ds, training_cutoff=cutoff)
    table = model.predict(ds, "validation", "sim-test", frozen_cutoff=cutoff)
    validation = {q.request_id for q in ds.queries if q.split == "validation"}
    assert table.num_rows == sum(c.request_id in validation for c in ds.candidates)


def test_baseline_has_no_hidden_fields(baseline_root):
    text = (baseline_root / "food.json").read_text(encoding="utf-8")
    ads_text = (baseline_root / "ads.json").read_text(encoding="utf-8")
    for blob in (text, ads_text):
        assert "true_probability" not in blob and "persona" not in blob
        assert "NaN" not in blob


def test_baseline_deterministic(built, tmp_path):
    export_baseline(*built, tmp_path / "a")
    export_baseline(*built, tmp_path / "b")
    for name in ("food.json", "ads.json"):
        assert ((tmp_path / "a/baseline" / name).read_bytes()
                == (tmp_path / "b/baseline" / name).read_bytes())
    json.loads((tmp_path / "a/baseline/food.json").read_text(encoding="utf-8"))


def test_export_refuses_overwrite(built, tmp_path):
    export_baseline(*built, tmp_path)
    with pytest.raises(FileExistsError):
        export_baseline(*built, tmp_path)
    export_oracle(*built, tmp_path)
    with pytest.raises(FileExistsError):
        export_oracle(*built, tmp_path)


def test_oracle_probabilities(oracle_root, built):
    sim, food_requests, ad_requests = built
    food = pq.read_table(oracle_root / "food_oracle.parquet").to_pylist()
    sums = defaultdict(float)
    for row in food:
        sums[row["request_id"]] += row["true_probability"]
    assert len(sums) == len(food_requests) and len(food) == 10 * len(food_requests)
    assert all(abs(v - 1) < 1e-9 for v in sums.values())
    r = food_requests[0]
    logits = food_logits(sim.world, r.session, r.candidate_ids)
    expected = np.exp(logits - logits.max())
    expected /= expected.sum()
    got = {x["candidate_id"]: x["true_probability"] for x in food if x["request_id"] == r.request_id}
    assert [got[c] for c in r.candidate_ids] == pytest.approx(list(expected))
    ads = pq.read_table(oracle_root / "ads_oracle.parquet")
    assert ads.column_names == ["request_id", "true_click_probability"]
    probs = ads.column("true_click_probability").to_pylist()
    assert len(probs) == len(ad_requests) and all(0 < p < 1 for p in probs)
    assert probs[0] == ad_requests[0].impression.true_probability
