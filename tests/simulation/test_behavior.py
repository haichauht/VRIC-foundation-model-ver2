from datetime import datetime
from itertools import pairwise

import numpy as np
import pytest

from src.simulation.behavior import (
    AdImpression,
    Event,
    FoodSession,
    Intent,
    Simulation,
    click_probability,
    food_logits,
    simulate,
)
from src.simulation.timeutil import (
    day_start_ts,
    local_day,
    local_iso,
    naive_utc,
    split_of,
    week_of,
)
from src.simulation.world import build_world, haversine_km

HEALTHY = {"healthy/salad", "đồ chay"}
TRAVEL = {"du lịch", "khách sạn"}
RIDE_TYPES = ("search_destination", "get_quote", "book_ride", "complete_ride")
FOOD_TYPES = ("search", "view_merchant", "click_dish", "order")
ADS_TYPES = ("ad_impression", "ad_click")
RIDE_CONTEXT = {
    "pickup_geohash", "dropoff_geohash", "pickup_zone_type", "dropoff_zone_type",
    "vehicle_type", "driver_id",
}
FOOD_CONTEXT = {
    "merchant_id", "merchant_name", "cuisine", "price_level", "search_query", "dish_id",
    "dish_name",
}
ADS_CONTEXT = {"product", "industry_main", "industry_sub", "caption"}


def run(config):
    rng = np.random.default_rng(config.seed)
    return simulate(build_world(config, rng), rng)


@pytest.fixture(scope="module")
def sim(tiny_config):
    rng = np.random.default_rng(tiny_config.seed)
    return simulate(build_world(tiny_config, rng), rng)


def _lookups(sim):
    zones = sim.world.zones
    merchants = {m.merchant_id: m for m in sim.world.merchants}
    return (lambda zone_id: zones[zone_id]), (lambda merchant_id: merchants[merchant_id])


def zone_by_geohash_factory(sim):
    by_geohash = {}
    for z in sim.world.zones:
        by_geohash.setdefault(z.geohash, z)
    return lambda geohash: by_geohash[geohash]


def _selected(items, during):
    if during is None:
        return [x for x in items if not x.active_intents]
    return [x for x in items if during in x.active_intents]


def healthy_share(sim, during):
    merchants = {m.merchant_id: m for m in sim.world.merchants}
    sessions = _selected(sim.food_sessions, during)
    assert sessions, f"no food sessions for during={during!r}"
    return sum(merchants[s.merchant_id].cuisine in HEALTHY for s in sessions) / len(sessions)


def travel_ctr(sim, during):
    ads = {a.ad_id: a for a in sim.world.ads}
    imps = [i for i in _selected(sim.ad_impressions, during) if ads[i.ad_id].category in TRAVEL]
    assert imps, f"no travel impressions for during={during!r}"
    return sum(i.clicked for i in imps) / len(imps)


def test_simulation_is_deterministic(tiny_config):
    assert run(tiny_config).events == run(tiny_config).events


def test_density_and_ordering(sim):
    early = [e for e in sim.events if week_of(e.ts, sim.world.config) <= 6]
    per_user = len(early) / len(sim.world.users)
    assert 140 <= per_user <= 260
    keys = [(int(e.user_id.rsplit(":", 1)[1]), e.ts, e.session_id, e.action_index)
            for e in sim.events]
    assert keys == sorted(keys)


def test_food_orders_follow_current_location(sim):
    zone, merchant = _lookups(sim)
    near = sum(haversine_km(zone(s.location_zone), zone(merchant(s.merchant_id).zone_id)) <= 3.0
               for s in sim.food_sessions)
    assert near / len(sim.food_sessions) >= 0.6


def test_ctr_and_intent_effects(sim):
    imps = sim.ad_impressions
    assert 0.03 <= sum(i.clicked for i in imps) / len(imps) <= 0.07
    assert healthy_share(sim, during="diet") > healthy_share(sim, during=None) + 0.10
    assert travel_ctr(sim, during="trip") > travel_ctr(sim, during=None)


def test_trip_starts_with_airport_ride(sim):
    zone_by_geohash = zone_by_geohash_factory(sim)
    trips = [i for i in sim.intents if i.kind == "trip"]
    assert trips
    for intent in (i for i in sim.intents if i.kind == "trip"):
        first = next(e for e in sim.events
                     if e.user_id == intent.user_id and e.ts >= intent.start_ts
                     and e.event_type == "complete_ride")
        assert zone_by_geohash(first.context["dropoff_geohash"]).zone_type == "transport_hub"


def test_local_day_boundary(tiny_config):
    ts = day_start_ts(43, tiny_config) + 30 * 60          # 00:30 ngày 43 giờ +07:00
    assert split_of(ts, tiny_config) == "validation" and split_of(ts - 3600, tiny_config) == "train"


def test_time_helpers(tiny_config):
    day1 = day_start_ts(1, tiny_config)
    assert local_iso(day1, tiny_config) == "2026-01-05T00:00:00+07:00"
    assert naive_utc(day1) == datetime(2026, 1, 4, 17, 0, 0)
    assert local_day(day1, tiny_config) == 1 and local_day(day1 - 1, tiny_config) == 0
    assert week_of(day_start_ts(43, tiny_config), tiny_config) == 7
    assert split_of(day_start_ts(50, tiny_config), tiny_config) == "test"
    with pytest.raises(ValueError):
        split_of(day_start_ts(57, tiny_config), tiny_config)


def test_types_and_intents(sim):
    assert isinstance(sim, Simulation)
    assert all(isinstance(e, Event) for e in sim.events)
    assert all(isinstance(s, FoodSession) for s in sim.food_sessions)
    assert all(isinstance(i, AdImpression) for i in sim.ad_impressions)
    assert all(isinstance(i, Intent) for i in sim.intents)
    kinds = {i.kind for i in sim.intents}
    assert kinds <= {"diet", "trip", "job_change", "payday"}
    assert {"diet", "trip", "payday"} <= kinds
    for i in sim.intents:
        assert (i.end_ts is None) == (i.kind == "job_change")
        assert i.end_ts is None or i.end_ts > i.start_ts


def test_event_schema_has_spec_keys_only(sim):
    ids = [e.event_id for e in sim.events]
    assert len(set(ids)) == len(ids) and ids[0] == "sim:event:0"
    for e in sim.events:
        assert e.action["type"] in (e.event_type, "impression", "click")
        if e.service == "ride":
            assert e.event_type in RIDE_TYPES and set(e.context) <= RIDE_CONTEXT
            assert set(e.action) <= {"type", "fare", "distance_km", "wait_minutes"}
        elif e.service == "food":
            assert e.event_type in FOOD_TYPES and set(e.context) <= FOOD_CONTEXT
            assert ("search_query" in e.context) == (e.event_type == "search")
            assert ("dish_id" in e.context) == (e.event_type in ("click_dish", "order"))
            assert set(e.action) <= {"type", "order_amount", "delivery_distance_km"}
        else:
            assert e.service == "ads" and e.event_type in ADS_TYPES
            assert set(e.context) == ADS_CONTEXT and e.context["product"] == e.entity_id
        assert 0 <= e.action_index < e.session_size


def test_sessions_do_not_overlap_and_clicks_follow_impressions(sim):
    by_session = {}
    for e in sim.events:
        by_session.setdefault(e.session_id, []).append(e)
    spans = {}
    for evs in by_session.values():
        assert [e.action_index for e in evs] == list(range(len(evs)))
        assert all(e.session_size == len(evs) for e in evs)
        assert all(a.ts < b.ts for a, b in pairwise(evs))
        spans.setdefault(evs[0].user_id, []).append((evs[0].ts, evs[-1].ts, evs[0].session_index))
        if evs[0].event_type == "ad_impression" and len(evs) == 2:
            assert evs[1].event_type == "ad_click"
    for user_spans in spans.values():
        user_spans.sort()
        assert [s[2] for s in user_spans] == list(range(len(user_spans)))
        assert all(a[1] < b[0] for a, b in pairwise(user_spans))
    clicks = {e.session_id for e in sim.events if e.event_type == "ad_click"}
    imp_sessions = {e.event_id: e.session_id for e in sim.events if e.event_type == "ad_impression"}
    assert len(imp_sessions) == len(sim.ad_impressions)
    assert all((imp_sessions[i.event_id] in clicks) == i.clicked for i in sim.ad_impressions)


def test_location_is_latest_dropoff_of_the_local_day(sim):
    config = sim.world.config
    homes = {u.user_id: u.home_zone for u in sim.world.users}
    zones = sim.world.zones
    rides = {}
    for e in sim.events:
        if e.event_type == "complete_ride":
            rides.setdefault(e.user_id, []).append(e)

    def expected_geohash(user_id, ts):
        latest = None
        for r in rides.get(user_id, []):
            if r.ts < ts and local_day(r.ts, config) == local_day(ts, config):
                latest = r
        return latest.context["dropoff_geohash"] if latest else zones[homes[user_id]].geohash

    for s in sim.food_sessions:
        assert zones[s.location_zone].geohash == expected_geohash(s.user_id, s.start_ts)
    for imp in sim.ad_impressions[::7]:
        assert zones[imp.location_zone].geohash == expected_geohash(imp.user_id, imp.ts)


def test_food_sessions_match_order_events_and_some_are_abandoned(sim):
    orders = {e.session_id: e for e in sim.events if e.event_type == "order"}
    firsts = {e.session_id: e for e in sim.events if e.service == "food" and e.action_index == 0}
    assert len(orders) == len(sim.food_sessions) < len(firsts)
    for s in sim.food_sessions:
        order = orders[s.session_id]
        assert order.ts == s.order_ts and order.entity_id == s.merchant_id
        assert firsts[s.session_id].ts == s.start_ts < s.order_ts


def test_true_probabilities_are_consistent(sim):
    world = sim.world
    for imp in sim.ad_impressions[::25]:
        assert 0.0 < imp.true_probability < 1.0
        assert click_probability(world, imp) == pytest.approx(imp.true_probability, rel=1e-9)
    mean_p = float(np.mean([i.true_probability for i in sim.ad_impressions]))
    ctr = sum(i.clicked for i in sim.ad_impressions) / len(sim.ad_impressions)
    assert abs(mean_p - ctr) < 0.01
    all_ids = [m.merchant_id for m in world.merchants]
    chosen_p = []
    for s in sim.food_sessions:
        logits = food_logits(world, s, all_ids)
        assert logits.shape == (len(all_ids),) and np.all(np.isfinite(logits))
        p = np.exp(logits - logits.max())
        p /= p.sum()
        chosen_p.append(p[all_ids.index(s.merchant_id)])
    # the sampled merchant is far more likely under the true logits than a uniform pick
    assert float(np.mean(chosen_p)) > 3.0 / len(all_ids)
    subset = food_logits(world, sim.food_sessions[0], all_ids[:10])
    np.testing.assert_allclose(subset, food_logits(world, sim.food_sessions[0], all_ids)[:10])
