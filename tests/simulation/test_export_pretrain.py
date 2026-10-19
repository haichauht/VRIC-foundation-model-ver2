import hashlib
import json
from dataclasses import replace
from datetime import datetime, timedelta
from functools import lru_cache

import numpy as np
import pyarrow.parquet as pq
import pytest

from src.simulation.behavior import Event, Simulation, simulate
from src.simulation.export_pretrain import export_pretrain
from src.simulation.history import (
    PROFILE_FIELDS,
    completed_sessions_before,
    last_history,
    profile_at,
)
from src.simulation.timeutil import SECONDS_PER_DAY, day_start_ts, local_day, naive_utc
from src.simulation.world import build_world

EVENT_COLUMNS = [
    "row_index", "event_id", "user_id", "source_record_id", "source_record_index",
    "action_index", "source_action_count", "source_timestamp", "event_time", "time_precision",
    "time_lower_bound", "time_upper_bound", "upper_bound_exclusive", "eligibility_time",
    "eligibility_inclusive", "scenario", "event_type", "context_json", "action_json",
]
SNAPSHOT_COLUMNS = [
    "sample_id", "user_id", "cutoff_time", "cutoff_inclusive", "split", "history_start",
    "history_stop", "history_event_ids", "history_order_known", "history_truncated",
    "history_group_truncated", "profile_json",
]
TRAIN_CUTOFFS = {datetime(2026, 1, 10, 16, 59, 59) + timedelta(days=3 * k) for k in range(10)}
VALIDATION_CUTOFFS = {datetime(2026, 2, 10, 16, 59, 59), datetime(2026, 2, 13, 16, 59, 59)}


# ---- helpers ------------------------------------------------------------------------------


@lru_cache(maxsize=4)
def _simulate(config) -> Simulation:
    rng = np.random.default_rng(config.seed)
    return simulate(build_world(config, rng), rng)


def session(session_id, times, *, index=0, service="ride", user_id="sim:user:0"):
    """Fake events of one session (one funnel step per timestamp)."""
    return [
        Event(f"{session_id}:{k}", user_id, session_id, index, k, len(times), ts, service,
              f"step{k}", "entity", {}, {"type": f"step{k}"})
        for k, ts in enumerate(times)
    ]


def simulation_with_single_user(config, first_event_day):
    sim = _simulate(config)
    user = sim.world.users[0]
    floor = day_start_ts(first_event_day, config)
    events = tuple(e for e in sim.events if e.user_id == user.user_id and e.ts >= floor)
    world = replace(sim.world, users=[user])
    return Simulation(world, events, (), (), ())


def _tables(root, kind):
    for path in sorted((root / kind).glob("*.parquet")):
        yield path, pq.read_table(path)


def max_event_upper_bound(root):
    return max(max(t.column("time_upper_bound").to_pylist(), default=datetime.min)
               for _, t in _tables(root, "events"))


def _snapshot_rows(root):
    for _, table in _tables(root, "snapshots"):
        yield from table.to_pylist()


def snapshot_cutoffs(root):
    out = {"train": set(), "validation": set(), "test": set()}
    for row in _snapshot_rows(root):
        out[row["split"]].add(row["cutoff_time"])
    return out


def history_lengths(root):
    return [len(row["history_event_ids"]) for row in _snapshot_rows(root)]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def sim(tiny_config):
    return _simulate(tiny_config)


@pytest.fixture(scope="module")
def exported(sim, tmp_path_factory):
    out = tmp_path_factory.mktemp("sim_export")
    summary = export_pretrain(sim, out)
    return out / "pretrain", summary


@pytest.fixture(scope="module")
def pretrain_root(exported):
    return exported[0]


# ---- tests from the brief -----------------------------------------------------------------


def test_pretrain_loads_with_foundation_loader(pretrain_root):
    from src.foundation_input.loader import iter_normalized_samples
    from src.foundation_input.validation import validate_sample
    train = list(iter_normalized_samples(pretrain_root, "train"))
    val = list(iter_normalized_samples(pretrain_root, "validation"))
    assert train and val and all(validate_sample(s) is s for s in train[:50])
    assert {s.dataset_name for s in train} == {"sim_v1"}
    assert not list(iter_normalized_samples(pretrain_root, "test"))


def test_no_event_on_or_after_week_7(pretrain_root, tiny_config):
    limit = naive_utc(day_start_ts(43, tiny_config))
    assert max_event_upper_bound(pretrain_root) < limit


def test_snapshot_cutoffs_and_history_rules(pretrain_root):
    cutoffs = snapshot_cutoffs(pretrain_root)
    assert cutoffs["train"] <= TRAIN_CUTOFFS
    assert cutoffs["validation"] <= VALIDATION_CUTOFFS
    assert all(1 <= n <= 30 for n in history_lengths(pretrain_root))
    # 80 users: every cutoff has at least one snapshot, and there is no test split
    assert cutoffs["train"] == TRAIN_CUTOFFS
    assert cutoffs["validation"] == VALIDATION_CUTOFFS
    assert not cutoffs["test"]


def test_in_progress_session_excluded():
    events = session("s1", [100, 200, 300]) + session("s2", [400, 500], index=1)
    kept = completed_sessions_before(events, cutoff_ts=450)
    assert {e.session_id for e in kept} == {"s1"}


def test_user_without_prior_events_gets_no_snapshot(tmp_path, tiny_config):
    sim = simulation_with_single_user(tiny_config, first_event_day=8)
    export_pretrain(sim, tmp_path)
    snaps = pq.read_table(tmp_path / "pretrain" / "snapshots" / "sim_user_0.parquet").to_pylist()
    assert min(s["cutoff_time"] for s in snaps) == datetime(2026, 1, 13, 16, 59, 59)


def test_manifest_hashes_match_files(pretrain_root):
    for shard in json.loads((pretrain_root / "manifest.json").read_text())["shards"]:
        assert sha256(pretrain_root / shard["events_file"]) == shard["events_sha256"]
        assert sha256(pretrain_root / shard["snapshots_file"]) == shard["snapshots_sha256"]


# ---- history helpers ----------------------------------------------------------------------


def test_completed_sessions_before_needs_every_action_and_the_whole_session():
    events = session("s1", [100, 200]) + session("s2", [300, 450], index=1)
    assert completed_sessions_before(events, 450) == events[:2]  # 450 is not < 450
    assert completed_sessions_before(events, 451) == events
    # a session cut short in the input (only 1 of its 2 actions present) is in progress
    assert completed_sessions_before(events[:3], 10_000) == events[:2]


def test_last_history_keeps_whole_sessions_and_reports_group_truncation():
    events = (session("a", [10, 11, 12, 13]) + session("b", [20, 21, 22], index=1)
              + session("c", [30, 31], index=2) + session("d", [40, 41], index=3))
    # everything fits
    hist, cut = last_history(events, 100, 30)
    assert hist == events and cut is False
    # 6 events: c+d = 4, adding b (3) would exceed 6 -> b dropped whole, flag set
    hist, cut = last_history(events, 100, 6)
    assert [e.session_id for e in hist] == ["c", "c", "d", "d"] and cut is True
    # 7 events: b+c+d = 7 lands on a session boundary -> nothing split, flag unset
    hist, cut = last_history(events, 100, 7)
    assert {e.session_id for e in hist} == {"b", "c", "d"} and len(hist) == 7 and cut is False
    # in-progress session d (41 >= 41) is excluded before selecting
    hist, cut = last_history(events, 41, 30)
    assert {e.session_id for e in hist} == {"a", "b", "c"} and cut is False
    assert last_history([], 100, 30) == ([], False)


def test_profile_at_fields_window_and_no_leak(sim, tiny_config):
    user = sim.world.users[0]
    events = [e for e in sim.events if e.user_id == user.user_id]
    cutoff = day_start_ts(34, tiny_config)
    profile = profile_at(events, user, cutoff, tiny_config, zones=sim.world.zones)
    assert tuple(profile) == PROFILE_FIELDS
    # only events before the cutoff matter
    before = [e for e in events if e.ts < cutoff]
    assert profile_at(before, user, cutoff, tiny_config, zones=sim.world.zones) == profile
    window = [e for e in before if e.ts >= cutoff - 30 * SECONDS_PER_DAY]
    rides = [e for e in window if e.event_type == "complete_ride"]
    orders = [e for e in window if e.event_type == "order"]
    assert profile["ride_count_30d"] == len(rides)
    assert profile["food_order_count_30d"] == len(orders)
    assert profile["ad_click_count_30d"] == sum(e.event_type == "ad_click" for e in window)
    assert profile["active_days_30d"] == len({local_day(e.ts, tiny_config) for e in window})
    spend = sum(e.action["fare"] for e in rides) + sum(e.action["order_amount"] for e in orders)
    assert profile["avg_monthly_spend"] == pytest.approx(spend)
    ride_s = len({e.session_id for e in window if e.service == "ride"})
    food_s = len({e.session_id for e in window if e.service == "food"})
    assert profile["most_used_service"] == ("ride" if ride_s >= food_s else "food")
    assert profile["age_group"] == user.age_group
    assert profile["saved_home_geohash"] == sim.world.zones[user.home_zone].geohash
    expected_work = (None if user.work_zone is None
                     else sim.world.zones[user.work_zone].geohash)
    assert profile["saved_work_geohash"] == expected_work
    empty = profile_at(events, user, day_start_ts(1, tiny_config), tiny_config,
                       zones=sim.world.zones)
    assert empty["most_used_service"] is None and empty["active_days_30d"] == 0


# ---- package layout -----------------------------------------------------------------------


def test_files_schema_and_manifest_totals(exported, sim, tiny_config):
    root, summary = exported
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    config = json.loads((root / "preprocessing_config.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "omnibehavior_pretraining_v2"
    assert manifest["status"] == "complete" and manifest["source_dataset"] == "sim_v1"
    assert manifest["seed"] == tiny_config.seed and manifest["generator_version"]
    assert manifest["config_sha256"] == sha256(root / "preprocessing_config.json")
    assert manifest["preview_sha256"] == sha256(root / "preview.json")
    assert config["max_history_events"] == 30
    assert config["splits"] == {
        "start_inclusive": "2026-01-04T17:00:00",
        "train_end_exclusive": "2026-02-06T17:00:00",
        "validation_end_exclusive": "2026-02-15T17:00:00",
        "test_end_exclusive": "2026-02-15T17:00:00",
    }
    shards = manifest["shards"]
    assert [s["user_id"] for s in shards] == [u.user_id for u in sim.world.users]
    assert shards[0]["events_file"] == "events/sim_user_0.parquet"
    assert shards[0]["snapshots_file"] == "snapshots/sim_user_0.parquet"
    limit = day_start_ts(43, tiny_config)
    assert summary["events"] == manifest["total_events"] == sum(
        e.ts < limit for e in sim.events)
    assert summary["users"] == manifest["total_users"] == len(sim.world.users)
    by_split = summary["snapshots_by_split"]
    assert by_split == manifest["snapshots_by_split"]
    assert by_split["train"] > 0 and by_split["validation"] > 0 and by_split["test"] == 0
    for shard in shards:
        events = pq.read_table(root / shard["events_file"])
        snaps = pq.read_table(root / shard["snapshots_file"])
        assert events.schema.names == EVENT_COLUMNS
        assert snaps.schema.names == SNAPSHOT_COLUMNS
        assert events.schema.metadata[b"schema_version"] == b"omnibehavior_pretraining_v2"
        assert events.num_rows == shard["event_rows"]
        assert snaps.num_rows == shard["snapshot_rows"] == sum(shard["snapshots_by_split"].values())
        assert shard["events_bytes"] == (root / shard["events_file"]).stat().st_size
    from src.foundation_input.manifest import load_manifest
    load_manifest(root, verify_checksums=True)


def test_event_rows_map_sessions_to_records(exported, sim, tiny_config):
    root, _ = exported
    user = sim.world.users[3]
    expected = sorted((e for e in sim.events if e.user_id == user.user_id
                       and e.ts < day_start_ts(43, tiny_config)),
                      key=lambda e: (e.ts, e.session_index, e.action_index))
    rows = pq.read_table(root / "events" / "sim_user_3.parquet").to_pylist()
    assert [r["event_id"] for r in rows] == [e.event_id for e in expected]
    scenario = {"ride": "Ride", "food": "Food", "ads": "Advertisement"}
    for i, (row, e) in enumerate(zip(rows, expected)):
        assert row["row_index"] == i and row["user_id"] == user.user_id
        assert row["source_record_id"] == e.session_id
        assert row["source_record_index"] == e.session_index
        assert row["action_index"] == e.action_index
        assert row["source_action_count"] == e.session_size
        assert row["scenario"] == scenario[e.service] and row["event_type"] == e.event_type
        assert row["event_time"] == naive_utc(e.ts).isoformat()
        assert row["time_lower_bound"] == row["time_upper_bound"] == naive_utc(e.ts)
        assert row["eligibility_time"] == naive_utc(e.ts)
        assert row["time_precision"] == "second" and row["eligibility_inclusive"] is True
        assert row["upper_bound_exclusive"] is False
        assert json.loads(row["context_json"]) == e.context
        assert json.loads(row["action_json"]) == e.action


def test_snapshots_hold_whole_sessions_and_honest_flags(pretrain_root, tiny_config):
    for path, snaps in _tables(pretrain_root, "snapshots"):
        events = pq.read_table(pretrain_root / "events" / path.name).to_pylist()
        for snap in snaps.to_pylist():
            start, stop = snap["history_start"], snap["history_stop"]
            hist = events[start:stop]
            assert [e["event_id"] for e in hist] == snap["history_event_ids"]
            assert all(e["eligibility_time"] <= snap["cutoff_time"] for e in hist)
            assert snap["cutoff_inclusive"] is True
            assert snap["sample_id"] == (
                f"{snap['user_id']}:{snap['cutoff_time'].isoformat()}:closed")
            counts = {}
            for e in hist:
                counts[e["source_record_id"]] = counts.get(e["source_record_id"], 0) + 1
            assert all(counts[e["source_record_id"]] == e["source_action_count"] for e in hist)
            # every event of the shard before the cutoff not in history is older than it
            later = [e for e in events[stop:] if e["eligibility_time"] <= snap["cutoff_time"]]
            assert not later
            assert snap["history_truncated"] == (start > 0 and stop - start >= 30)
            # Ruling R11: no group (session) is ever split -- whole sessions only -- so the flag
            # is always false and foundation_input keeps next-action targets for every snapshot.
            assert snap["history_group_truncated"] is False
            dropped_older = start > 0 and stop - start < 30
            if dropped_older:  # the dropped session really would not have fit
                prev = events[start - 1]
                assert stop - start + prev["source_action_count"] > 30
            profile = json.loads(snap["profile_json"])
            assert profile["as_of_time"] == snap["cutoff_time"].isoformat()
            assert set(profile) == {"as_of_time", *PROFILE_FIELDS}


def test_next_action_targets_for_every_snapshot_with_two_sessions(pretrain_root):
    """Ruling R11: ``history_group_truncated`` is false, so foundation_input (unchanged) builds a
    next-action target -- the latest session -- for every snapshot holding >= 2 sessions."""
    from itertools import islice

    from src.foundation_input.dataset import DatasetConfig, encode_sample
    from src.foundation_input.loader import iter_normalized_samples
    from src.foundation_input.tokenizer import TokenizerConfig
    from src.foundation_input.vocab import VocabularyConfig, fit_vocabulary_from_samples

    preprocessing = json.loads((pretrain_root / "preprocessing_config.json").read_text(
        encoding="utf-8"))
    tokenizer = TokenizerConfig(context_allowlist={
        scenario: tuple(keys) for scenario, keys in preprocessing["context_allowlist"].items()})
    samples = list(islice(iter_normalized_samples(pretrain_root, "train"), 200))
    artifacts = fit_vocabulary_from_samples(samples, VocabularyConfig(dataset_name="sim_v1"),
                                            tokenizer)
    config = DatasetConfig(max_tokens=4096, next_action_prediction=True)
    multi = 0
    for sample in samples:
        assert sample.history_group_truncated is False
        encoded = encode_sample(sample, artifacts, tokenizer, config)
        if len({e.source_record_id for e in sample.events}) >= 2:
            multi += 1
            groups = {r.token.event_group_id for r in encoded.next_action_records}
            assert groups == {sample.events[-1].source_record_id}, sample.sample_id
    assert multi > 150


def test_export_is_deterministic_and_refuses_to_overwrite(sim, tmp_path):
    export_pretrain(sim, tmp_path / "a")
    export_pretrain(sim, tmp_path / "b")
    for name in ("manifest.json", "preprocessing_config.json", "preview.json"):
        assert (tmp_path / "a" / "pretrain" / name).read_bytes() == (
            tmp_path / "b" / "pretrain" / name).read_bytes()
    with pytest.raises(FileExistsError):
        export_pretrain(sim, tmp_path / "a")
