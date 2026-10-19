import dataclasses
import hashlib
import json
from functools import lru_cache

import numpy as np
import pyarrow.parquet as pq
import pytest

from src.simulation.behavior import simulate
from src.simulation.export_ads import AdRequest, build_ad_requests, export_ads
from src.simulation.timeutil import day_start_ts, split_of
from src.simulation.world import build_world

SPLITS = ("train", "validation", "test")
REQUIRED = {
    "profiles": {"user_id", "as_of_timestamp"},
    "ad_catalog": {"candidate_id", "first_observed_timestamp"},
    "events": {"event_id", "source_row_index", "user_id", "timestamp", "event_type", "service",
               "entity_id", "event_index"},
    "ad_opportunities": {"request_id", "source_row_index", "user_id", "cutoff_timestamp",
                         "candidate_id", "clicked", "split", "exposure_timestamp",
                         "candidate_timestamp", "profile_as_of_timestamp",
                         "history_end_timestamp", "history_start", "history_stop"},
}
FORBIDDEN_ATTRIBUTE_KEYS = {"clicked", "click", "session_size", "session_id", "action_count",
                            "n_actions", "label", "true_probability"}
IMPRESSION_ATTRIBUTE_KEYS = {"product", "industry_main", "industry_sub", "caption"}


@lru_cache(maxsize=4)
def _sim_and_requests(config):
    rng = np.random.default_rng(config.seed)
    sim = simulate(build_world(config, rng), rng)
    return sim, build_ad_requests(sim, rng)


@pytest.fixture(scope="module")
def built(tiny_config):
    return _sim_and_requests(tiny_config)


@pytest.fixture(scope="module")
def ads_root(built, tmp_path_factory):
    sim, requests = built
    out = tmp_path_factory.mktemp("ads_out")
    export_ads(sim, requests, out)
    return out / "ads"


def read_table(root, name):
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    return pq.read_table(root / manifest["files"][name]["path"]).to_pylist()


def read_events_by_user(root):
    by_user = {}
    for e in read_table(root, "events"):
        by_user.setdefault(e["user_id"], []).append(e)
    for events in by_user.values():
        events.sort(key=lambda e: e["event_index"])
    return by_user


def test_tables_have_reader_columns_and_hashes(ads_root):
    manifest = json.loads((ads_root / "manifest.json").read_text(encoding="utf-8"))
    for name, cols in REQUIRED.items():
        meta = manifest["files"][name]
        table = pq.read_table(ads_root / meta["path"])
        assert cols <= set(table.column_names) and table.num_rows == meta["row_count"]
        assert hashlib.sha256((ads_root / meta["path"]).read_bytes()).hexdigest() == meta["sha256"]
    assert manifest["source_dataset"] == "sim_v1" and manifest["negative_sampling"] == "none"
    assert manifest["candidate_policy"] == "sim-served-impression-single-candidate-v1"
    assert manifest["missing_click_policy"] == "reject"
    assert manifest["schema_version"] == 1 and manifest["timestamp_unit"] == "seconds"
    assert manifest["history_policy"] == {
        "timestamp_rule": "strictly < cutoff", "target_click_in_history": False,
        "equal_timestamp_events_excluded": True, "max_events": 30}


def test_history_pointers_strictly_before_exposure(ads_root):
    events, opps = read_events_by_user(ads_root), read_table(ads_root, "ad_opportunities")
    assert opps
    for o in opps:
        user_events = events[o["user_id"]]
        window = user_events[o["history_start"]:o["history_stop"]]
        assert o["history_stop"] - o["history_start"] <= 30
        assert all(e["timestamp"] < o["cutoff_timestamp"] for e in window)
        assert o["cutoff_timestamp"] == o["exposure_timestamp"]
        # maximal: the next event (if any) is not before the cutoff; window = latest 30
        if o["history_stop"] < len(user_events):
            assert user_events[o["history_stop"]]["timestamp"] >= o["cutoff_timestamp"]
        assert o["history_start"] == max(0, o["history_stop"] - 30)
        end = window[-1]["timestamp"] if window else None
        assert o["history_end_timestamp"] == end


def test_events_hold_all_services_and_are_indexed(ads_root, built):
    sim, _ = built
    rows = read_table(ads_root, "events")
    assert len(rows) == len(sim.events)
    assert {r["service"] for r in rows} == {"ride", "food", "ads"}
    assert [r["source_row_index"] for r in rows] == list(range(len(rows)))
    for events in read_events_by_user(ads_root).values():
        assert [e["event_index"] for e in events] == list(range(len(events)))
        assert [e["timestamp"] for e in events] == sorted(e["timestamp"] for e in events)


def test_opportunity_matches_its_impression_event(ads_root, built):
    _, requests = built
    events = {e["event_id"]: e for e in read_table(ads_root, "events")}
    opps = read_table(ads_root, "ad_opportunities")
    assert [o["request_id"] for o in opps] == [f"sim:ad_req:{n}" for n in range(len(opps))]
    for o, r in zip(opps, requests):
        imp = r.impression
        e = events[imp.event_id]
        assert (e["user_id"], e["timestamp"], e["entity_id"], e["source_row_index"]) == (
            o["user_id"], o["cutoff_timestamp"], o["candidate_id"], o["source_row_index"])
        assert e["event_type"] == "ad_impression"
        assert o["clicked"] == imp.clicked and o["source_row_index"] == r.source_row_index


def test_ctr_and_splits(ads_root, tiny_config):
    opps = read_table(ads_root, "ad_opportunities")
    assert 0.03 <= sum(o["clicked"] for o in opps) / len(opps) <= 0.07
    assert all(o["split"] == split_of(o["exposure_timestamp"], tiny_config) for o in opps)
    assert {o["split"] for o in opps} == set(SPLITS)


def test_profiles_static_and_catalog(ads_root, tiny_config):
    profiles = read_table(ads_root, "profiles")
    assert {p["as_of_timestamp"] for p in profiles} == {day_start_ts(1, tiny_config)}
    assert not any(column.endswith("_30d") for column in profiles[0])
    assert set(profiles[0]) == {"user_id", "as_of_timestamp", "age_group", "saved_home_geohash",
                                "saved_work_geohash"}
    assert len({p["user_id"] for p in profiles}) == len(profiles) == tiny_config.n_users
    catalog = read_table(ads_root, "ad_catalog")
    assert len(catalog) == tiny_config.n_ads
    assert {c["first_observed_timestamp"] for c in catalog} == {day_start_ts(1, tiny_config)}
    assert {"category", "advertiser", "title"} <= set(catalog[0])
    opps = read_table(ads_root, "ad_opportunities")
    assert {o["profile_as_of_timestamp"] for o in opps} == {day_start_ts(1, tiny_config)}
    assert {o["candidate_timestamp"] for o in opps} == {day_start_ts(1, tiny_config)}


def test_no_request_reveals_its_label(ads_root, built):
    """A clicked impression session has size 2 in the simulation; nothing exported may show it.

    Checked on the data a model would consume: the window of every request and the exported
    attributes of every impression event (identical key set for clicked and unclicked ones).
    """
    _, requests = built
    events = read_events_by_user(ads_root)
    opps = read_table(ads_root, "ad_opportunities")
    all_events = read_table(ads_root, "events")
    own = {r["event_id"]: r for r in all_events}
    clicked_n = 0
    for r, o in zip(requests, opps):
        window = events[o["user_id"]][o["history_start"]:o["history_stop"]]
        assert all(e["timestamp"] < o["exposure_timestamp"] for e in window)
        assert r.impression.event_id not in {e["event_id"] for e in window}
        if o["clicked"]:
            clicked_n += 1
            # the request's own click (later timestamp, same ad) is never in its window
            assert not any(e["event_type"] == "ad_click" and e["timestamp"] >= o["cutoff_timestamp"]
                           for e in window)
        attrs = json.loads(own[r.impression.event_id]["attributes_json"])
        assert set(attrs) == IMPRESSION_ATTRIBUTE_KEYS
    assert 0 < clicked_n < len(opps)
    assert not {"clicked", "click_timestamp", "session_id", "session_size"} & set(all_events[0])
    for e in all_events:
        assert not FORBIDDEN_ATTRIBUTE_KEYS & set(json.loads(e["attributes_json"]))
    imp_keys = {tuple(sorted(json.loads(e["attributes_json"])))
                for e in all_events if e["event_type"] == "ad_impression"}
    assert len(imp_keys) == 1


def test_inputs_not_mutated(built):
    sim, _ = built
    before = [(dict(e.context), dict(e.action)) for e in sim.events]
    rng = np.random.default_rng(3)
    build_ad_requests(sim, rng)
    assert [(e.context, e.action) for e in sim.events] == before


def test_deterministic_bytes(built, tmp_path):
    sim, requests = built
    export_ads(sim, requests, tmp_path / "a")
    export_ads(sim, requests, tmp_path / "b")
    for name in ("manifest.json", "events.parquet", "ad_opportunities.parquet",
                 "profiles.parquet", "ad_catalog.parquet"):
        assert (tmp_path / "a" / "ads" / name).read_bytes() == (
            tmp_path / "b" / "ads" / name).read_bytes()
    with pytest.raises(FileExistsError):
        export_ads(sim, requests, tmp_path / "a")


def test_request_limits_per_split(built, tiny_config):
    sim, all_requests = built
    assert all(isinstance(r, AdRequest) for r in all_requests)
    counts = {s: sum(r.split == s for r in all_requests) for s in SPLITS}
    assert len(all_requests) == len(sim.ad_impressions) and min(counts.values()) > 0

    def with_limits(limits):
        config = dataclasses.replace(tiny_config, ads_requests=limits)
        return dataclasses.replace(sim, world=dataclasses.replace(sim.world, config=config))

    limits = {"train": 50, "validation": 7, "test": 10**6}
    picked = build_ad_requests(with_limits(limits), np.random.default_rng(1))
    by_split = {s: sum(r.split == s for r in picked) for s in SPLITS}
    assert by_split == {"train": 50, "validation": 7, "test": counts["test"]}
    assert [r.request_id for r in picked] == [f"sim:ad_req:{n}" for n in range(len(picked))]
    for s in SPLITS:
        times = [r.impression.ts for r in picked if r.split == s]
        assert times == sorted(times)
    again = build_ad_requests(with_limits(limits), np.random.default_rng(1))
    assert [r.impression.event_id for r in again] == [r.impression.event_id for r in picked]
    with pytest.raises(ValueError):
        build_ad_requests(with_limits({"train": 5}), np.random.default_rng(1))
