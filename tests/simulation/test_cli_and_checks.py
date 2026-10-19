import hashlib
import json
import platform
import shutil
from datetime import datetime, timedelta

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from src.simulation import __main__ as cli
from src.simulation.__main__ import main
from src.simulation.calibrate import geohash_decode
from src.simulation.checks import CheckError, run_checks
from src.simulation.config import load_config
from src.simulation.export_pretrain import pretrain_end_day
from src.simulation.timeutil import day_start_ts, naive_utc
from src.simulation.world import geohash_encode

ROOT_NAME = "sim-smoke-v1"
PACKAGES = ("pretrain", "ads", "food", "oracle", "baseline")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_hashes(root):
    """SHA-256 of every exported file, ignoring data_report.json (it holds run timings)."""
    return {
        p.relative_to(root).as_posix(): sha256(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.name != "data_report.json"
    }


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _report(root):
    return _read(root / "data_report.json")


def _rewrite_jsonl(path, edit):
    """Apply ``edit(row) -> bool`` to rows until it returns True; rewrite the file in place."""
    lines = path.read_bytes().decode("utf-8").splitlines(keepends=True)
    for i, line in enumerate(lines):
        row = json.loads(line)
        if edit(row):
            lines[i] = json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            break
    else:
        raise AssertionError(f"no row of {path} could be edited")
    path.write_bytes("".join(lines).encode("utf-8"))


def inject_future_history_event(path):
    """Move the last history event of the first request with history to after its cutoff."""

    def edit(row):
        history = row["input"]["history"]
        if not history:
            return False
        cutoff = datetime.fromisoformat(row["input"]["cutoff_time"])
        history[-1]["event_time"] = (cutoff + timedelta(hours=1)).isoformat()
        return True

    _rewrite_jsonl(path, edit)


@pytest.fixture(scope="module")
def pristine(tmp_path_factory, tiny_config_yaml):
    """One CLI run on the tiny config; tests copy it before changing anything."""
    base = tmp_path_factory.mktemp("cli")
    config_file = base / "tiny.yaml"
    config_file.write_text(tiny_config_yaml, encoding="utf-8")
    assert main(["--config", str(config_file), "--out", str(base / "out")]) == 0
    return base / "out" / ROOT_NAME, config_file


def _copy(pristine, tmp_path, packages=PACKAGES):
    root, config_file = pristine
    target = tmp_path / ROOT_NAME
    for package in packages:
        shutil.copytree(root / package, target / package)
    return target, load_config(config_file)


@pytest.fixture
def copied(pristine, tmp_path):
    return _copy(pristine, tmp_path)


# ---- brief tests ------------------------------------------------------------------------------


def test_cli_end_to_end_tiny(tmp_path, tiny_config_file):
    assert main(["--config", str(tiny_config_file), "--out", str(tmp_path)]) == 0
    root = tmp_path / ROOT_NAME
    report = _report(root)
    assert {"pretrain", "food", "ads", "baseline", "oracle"} <= {p.name for p in root.iterdir()}
    d = report["difficulty"]
    assert d["food"]["oracle_ndcg_at_10"] > d["food"]["popularity_ndcg_at_10"]
    assert d["ads"]["oracle_auc"] > d["ads"]["popularity_auc"]
    assert d["food"]["candidate_only_ndcg_at_10"] <= d["food"]["popularity_ndcg_at_10"] + 0.05


def test_same_seed_same_bytes_different_seed_differs(tmp_path, tiny_config_file):
    a, b, c = (tmp_path / x for x in "abc")
    assert main(["--config", str(tiny_config_file), "--out", str(a)]) == 0
    assert main(["--config", str(tiny_config_file), "--out", str(b)]) == 0
    assert main(["--config", str(tiny_config_file), "--out", str(c), "--seed", "7"]) == 0
    ha, hb, hc = (tree_hashes(x / ROOT_NAME) for x in (a, b, c))
    assert ha == hb != hc
    assert set(ha) == set(hc)


def test_checks_catch_leak(tmp_path, tiny_config_file):
    main(["--config", str(tiny_config_file), "--out", str(tmp_path)])
    inject_future_history_event(tmp_path / ROOT_NAME / "food" / "samples.jsonl")
    with pytest.raises(ValueError, match="food_history_before_cutoff"):
        run_checks(tmp_path / ROOT_NAME, load_config(tiny_config_file))


# ---- config, CLI and data_report ---------------------------------------------------------------


def test_tiny_config_file_applies_overrides(tiny_config_file, tiny_config):
    assert load_config(tiny_config_file) == tiny_config


def test_data_report_contents(pristine):
    root, config_file = pristine
    config = load_config(config_file)
    report = _report(root)
    assert report["status"] == "complete" and report["failed_step"] is None
    assert report["dataset_version"] == ROOT_NAME and report["seed"] == config.seed
    datetime.strptime(report["generated_at_utc"], "%Y-%m-%dT%H:%M:%SZ")
    assert report["checks"]["status"] == "passed"
    assert all(c["status"] == "passed" for c in report["checks"]["results"].values())
    for name in ("food", "ads"):  # small files: load_dataset reads the whole file
        assert report["checks"]["results"][f"baseline_{name}_load_dataset"]["mode"] == "full_file"
    assert {"pretrain_manifest", "food_history_before_cutoff", "food_own_session_absent",
            "ads_history_before_exposure", "baseline_food_load_dataset",
            "oracle_food_sums_to_one"} <= set(report["checks"]["results"])
    # SHA-256 of every file except the report itself
    assert {k: v["sha256"] for k, v in report["files"].items()} == tree_hashes(root)
    assert report["counts"]["users"] == config.n_users
    assert sum(report["counts"]["food_requests_by_split"].values()) > 0
    assert set(report["personas"]) == {"office", "student", "night_shift", "home", "traveler"}
    assert set(report["intents"]) == {"diet", "trip", "job_change", "payday"}
    assert 0.0 < report["ctr"]["all_impressions"] < 0.2
    assert report["timings_seconds"]["simulate"] >= 0 and report["skipped"] == []
    food = report["difficulty"]["food"]
    assert food["experiment_grade"] is False  # 24 merchants: sparse catalog (ruling R9)
    for key in ("distance_to_home_ndcg_at_10", "home_or_work_by_hour_ndcg_at_10"):
        assert 0.0 < food[key] <= 1.0  # profile rankers are reported (ruling R7)
    assert report["difficulty"]["split"] == "test"
    assert report["rng_order"][:4] == ["build_world", "simulate", "build_food_requests",
                                       "build_ad_requests"]
    # Ruling R13: library/platform provenance lives only in data_report.json (not hashed).
    env = report["environment"]
    assert set(env) == {"python", "numpy", "pyarrow", "platform"}
    assert env["numpy"] == np.__version__ and env["pyarrow"] == pa.__version__
    assert env["python"] == platform.python_version() and env["platform"]
    assert all("environment" not in _read(root / p) for p in report["files"]
               if p.endswith("manifest.json"))


def test_skip_pretrain(tmp_path, tiny_config_file):
    assert main(["--config", str(tiny_config_file), "--out", str(tmp_path),
                 "--skip-pretrain"]) == 0
    root = tmp_path / ROOT_NAME
    assert not (root / "pretrain").exists()
    report = _report(root)
    assert report["skipped"] == ["pretrain"]
    assert report["checks"]["results"]["pretrain_manifest"]["status"] == "skipped"
    assert report["checks"]["status"] == "passed"


def test_refuses_non_empty_output(pristine, capsys):
    root, config_file = pristine
    before = tree_hashes(root)
    assert main(["--config", str(config_file), "--out", str(root.parent)]) == 2
    assert "not empty" in capsys.readouterr().err
    assert tree_hashes(root) == before


@pytest.fixture
def fast_exports(monkeypatch):
    """No-op exporters: these tests are about the CLI's failure handling, not the files."""
    for name in ("export_pretrain", "export_food", "export_ads", "export_baseline",
                 "export_oracle"):
        monkeypatch.setattr(cli, name, lambda *args, **kwargs: {})


def _check_error(*args, **kwargs):
    raise CheckError("simulator checks failed: food_split (1 violation(s): sim:food_req:0)",
                     {"status": "failed", "results": {"food_split": {"status": "failed"}},
                      "files": {}})


def _disk_error(*args, **kwargs):
    raise OSError("disk went away")


def _passing_checks(*args, **kwargs):
    return {"status": "passed", "results": {}, "files": {}, "tree_sha256": None}


@pytest.mark.parametrize("failure", ["check_error", "checker_crash", "difficulty_crash"])
def test_post_export_failure_exits_3_with_a_failed_report(tmp_path, tiny_config_file,
                                                          monkeypatch, fast_exports, failure):
    checks = {"check_error": _check_error, "checker_crash": _disk_error,
              "difficulty_crash": _passing_checks}[failure]
    monkeypatch.setattr(cli, "run_checks", checks)
    monkeypatch.setattr(cli, "difficulty_report", _disk_error)
    assert main(["--config", str(tiny_config_file), "--out", str(tmp_path)]) == 3
    report = _report(tmp_path / ROOT_NAME)
    assert report["status"] == "failed" and report["difficulty"] is None
    if failure == "difficulty_crash":
        assert report["failed_step"] == "difficulty" and report["checks"]["status"] == "passed"
        assert "OSError: disk went away" in report["error"]
    else:
        assert report["failed_step"] == "checks" and report["checks"]["status"] == "failed"
        assert ("food_split" if failure == "check_error" else "OSError") in report["error"]


@pytest.mark.parametrize("step", ["simulate", "export_ads"])
def test_generation_crash_leaves_an_incomplete_report(tmp_path, tiny_config_file, monkeypatch,
                                                      fast_exports, capsys, step):
    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(cli, step, boom)
    argv = ["--config", str(tiny_config_file), "--out", str(tmp_path)]
    assert main(argv) == 4
    report = _report(tmp_path / ROOT_NAME)
    assert report["status"] == "incomplete" and report["failed_step"] == step
    assert "RuntimeError: boom" in report["error"]
    assert report["checks"] is None and report["difficulty"] is None
    capsys.readouterr()
    assert main(argv) == 2  # never overwrites; says why the directory is there
    assert "incomplete" in capsys.readouterr().err


# ---- run_checks --------------------------------------------------------------------------------


def test_checks_pass_with_baseline_subset_loading(copied):
    """Above the size limit each Baseline file is streamed and a request subset is loaded."""
    root, config = copied
    result = run_checks(root, config, baseline_full_load_max_bytes=0)
    assert result["status"] == "passed"
    for name in ("food", "ads"):
        loaded = result["results"][f"baseline_{name}_load_dataset"]
        assert loaded["mode"] == "subset" and 0 < loaded["queries"] <= 200
    assert {k: v["sha256"] for k, v in result["files"].items()} == tree_hashes(root)


def test_only_selected_packages_are_checked(pristine, tmp_path):
    root, config = _copy(pristine, tmp_path, ("ads",))
    result = run_checks(root, config, only=("ads",))
    statuses = {name: r["status"] for name, r in result["results"].items()}
    assert result["status"] == "passed"
    assert statuses["ads_history_before_exposure"] == "passed"
    assert {statuses[n] for n in statuses if not n.startswith("ads_")} == {"skipped"}
    with pytest.raises(ValueError, match="unknown"):
        run_checks(root, config, only=("adz",))


def test_unreadable_package_fails_readable_and_marks_dependents_not_run(pristine, tmp_path):
    root, config = _copy(pristine, tmp_path, ("ads", "food", "oracle", "baseline"))
    (root / "food" / "samples.jsonl").unlink()
    with pytest.raises(CheckError, match="food_readable") as info:
        run_checks(root, config, only=("ads", "food", "oracle", "baseline"))
    results = info.value.report["results"]
    assert "FileNotFoundError" in results["food_readable"]["error"]
    assert results["food_manifest"]["status"] == "failed"  # samples.jsonl listed but missing
    for name in ("food_history_before_cutoff", "oracle_food_matches_candidates",
                 "baseline_food_structure", "baseline_food_load_dataset"):
        assert results[name]["status"] == "not_run", name
    for name in ("ads_history_before_exposure", "baseline_ads_load_dataset", "oracle_ads"):
        assert results[name]["status"] == "passed", name


# ---- injected violations: each check must catch its own ------------------------------------------


def _set_column(path, column, values, type_):
    table = pq.read_table(path)
    index = table.schema.get_field_index(column)
    pq.write_table(table.set_column(index, table.schema.field(index), pa.array(values, type_)),
                   path)


def _edit_json(path, edit, *, compact):
    doc = json.loads(path.read_text(encoding="utf-8"))
    edit(doc)
    text = (json.dumps(doc, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            if compact else json.dumps(doc, ensure_ascii=False, indent=2))
    path.write_bytes((text + "\n").encode("utf-8"))


def _epoch(iso):
    return int(datetime.fromisoformat(iso).timestamp())


# -- food


def _food_rows(root, edit):
    _rewrite_jsonl(root / "food" / "samples.jsonl", edit)


def _inject_food_distance_key(root, config):
    def edit(row):
        row["input"]["candidates"][0]["distance_km"] = 1.2
        return True

    _food_rows(root, edit)


def _inject_own_order_in_history(root, config):
    """The request's own ``order`` (its label) appears in the history, back-dated to before the
    cutoff: only the global event table reveals it."""
    events = pq.read_table(root / "ads" / "events.parquet",
                           columns=["event_id", "user_id", "timestamp", "event_type"])
    orders = {(e["user_id"], e["timestamp"]): e["event_id"]
              for e in events.to_pylist() if e["event_type"] == "order"}

    def edit(row):
        inp, history = row["input"], row["input"]["history"]
        if not history:
            return False
        positive = next(lb["item_id"] for lb in row["labels"] if lb["relevance"] == 1)
        order_id = orders[(inp["user_id"], _epoch(row["label_metadata"]["label_time"]))]
        history[-1].update(event_id=order_id, service="food", event_type="order",
                           entity_id=positive, attributes={"merchant_id": positive})
        return True

    _food_rows(root, edit)


def _inject_candidate_attribute_mismatch(root, config):
    def edit(row):
        cand = row["input"]["candidates"][0]
        cand["cuisine"] = "pizza" if cand["cuisine"] != "pizza" else "lẩu"
        return True

    _food_rows(root, edit)


def _inject_label_at_cutoff(root, config):
    def edit(row):
        row["label_metadata"]["label_time"] = row["input"]["cutoff_time"]
        return True

    _food_rows(root, edit)


def _inject_profile_after_cutoff(root, config):
    def edit(row):
        row["input"]["profile"]["as_of_time"] = row["label_metadata"]["label_time"]
        return True

    _food_rows(root, edit)


def _inject_food_wrong_split(root, config):
    def edit(row):
        row["label_metadata"]["split"] = "test"  # request 0 is in week 1
        return True

    _food_rows(root, edit)


def _inject_popularity_window(root, config):
    def edit(manifest):
        manifest["popularity_end_exclusive"] = "2026-03-02T00:00:00+07:00"  # includes weeks 7-8

    _edit_json(root / "food" / "manifest.json", edit, compact=False)


# -- ads


def _ads_path(root, name):
    return root / "ads" / f"{name}.parquet"


def _inject_ads_window_reaching_exposure(root, config):
    path = _ads_path(root, "ad_opportunities")
    stops = pq.read_table(path)["history_stop"].to_pylist()
    stops[next(k for k, s in enumerate(stops) if s > 0)] += 1  # now holds the impression itself
    _set_column(path, "history_stop", stops, pa.int64())


def _inject_ads_window_too_short(root, config):
    path = _ads_path(root, "ad_opportunities")
    table = pq.read_table(path)
    starts = table["history_start"].to_pylist()
    stops = table["history_stop"].to_pylist()
    starts[next(k for k, (a, b) in enumerate(zip(starts, stops)) if b > a)] += 1
    _set_column(path, "history_start", starts, pa.int64())


def _inject_ads_wrong_impression_row(root, config):
    path = _ads_path(root, "ad_opportunities")
    rows = pq.read_table(path)["source_row_index"].to_pylist()
    rows[0] += 1
    _set_column(path, "source_row_index", rows, pa.int64())


def _inject_ads_wrong_split(root, config):
    path = _ads_path(root, "ad_opportunities")
    splits = pq.read_table(path)["split"].to_pylist()
    splits[splits.index("train")] = "validation"
    _set_column(path, "split", splits, pa.string())


def _inject_ads_click_flag_attribute(root, config):
    path = _ads_path(root, "events")
    table = pq.read_table(path)
    attributes = table["attributes_json"].to_pylist()
    i = table["event_type"].to_pylist().index("ad_impression")
    attributes[i] = json.dumps({**json.loads(attributes[i]), "session_size": 2})
    _set_column(path, "attributes_json", attributes, pa.large_string())


def _inject_ads_click_label_flip(root, config):
    """The label of request 0 disagrees with the event log (its ad_click row or its absence)."""
    path = _ads_path(root, "ad_opportunities")
    clicked = pq.read_table(path)["clicked"].to_pylist()
    clicked[0] = not clicked[0]
    _set_column(path, "clicked", clicked, pa.bool_())


def _inject_ads_event_index_gap(root, config):
    path = _ads_path(root, "events")
    index = pq.read_table(path)["event_index"].to_pylist()
    index[next(k for k, v in enumerate(index) if v > 0)] = 999
    _set_column(path, "event_index", index, pa.int64())


# -- pretrain


def _pretrain_shard(root, kind, user):
    return root / "pretrain" / kind / f"sim_user_{user}.parquet"


def _inject_pretrain_stray_file(root, config):
    shutil.copy(_pretrain_shard(root, "events", 0), root / "pretrain" / "events" / "stray.parquet")


def _inject_pretrain_preview_tamper(root, config):
    path = root / "pretrain" / "preview.json"
    path.write_bytes(path.read_bytes() + b"\n")


def _inject_pretrain_event_in_week_7(root, config):
    shard = _pretrain_shard(root, "events", 1)
    table = pq.read_table(shard)
    limit = pa.scalar(naive_utc(day_start_ts(pretrain_end_day(config), config)),
                      pa.timestamp("us"))
    late = pc.if_else(pc.equal(pa.array(range(table.num_rows)), table.num_rows - 1),
                      limit, table["time_upper_bound"])
    _set_column(shard, "time_upper_bound", late, pa.timestamp("us"))


def _inject_pretrain_future_history_id(root, config):
    """The first snapshot of user 2 points at the user's last event (after its cutoff)."""
    last = pq.read_table(_pretrain_shard(root, "events", 2))["event_id"][-1]
    path = _pretrain_shard(root, "snapshots", 2)
    ids = pq.read_table(path)["history_event_ids"].to_pylist()
    ids[0] = [last.as_py()]
    _set_column(path, "history_event_ids", ids, pa.list_(pa.field("element", pa.string())))


def _inject_pretrain_hidden_profile_field(root, config):
    path = _pretrain_shard(root, "snapshots", 1)
    profiles = pq.read_table(path)["profile_json"].to_pylist()
    profiles[0] = json.dumps({**json.loads(profiles[0]), "persona": "office"})
    _set_column(path, "profile_json", profiles, pa.string())


def _inject_pretrain_group_truncated(root, config):
    """Ruling R11: the flag would turn off foundation_input's next-action target."""
    path = _pretrain_shard(root, "snapshots", 1)
    flags = pq.read_table(path)["history_group_truncated"].to_pylist()
    flags[0] = True
    _set_column(path, "history_group_truncated", flags, pa.bool_())


def _inject_pretrain_unparseable_actions(root, config):
    """User 0 is in the loader spot check; only the loader parses ``action_json``."""
    path = _pretrain_shard(root, "events", 0)
    rows = pq.read_metadata(path).num_rows
    _set_column(path, "action_json", ["{"] * rows, pa.large_string())


# -- oracle


def _oracle_path(root, name):
    return root / "oracle" / f"{name}.parquet"


def _inject_oracle_sum(root, config):
    path = _oracle_path(root, "food_oracle")
    probs = pq.read_table(path)["true_probability"].to_pylist()
    probs[max(range(10), key=lambda i: probs[i])] *= 0.5  # rows 0..9 = the first request
    _set_column(path, "true_probability", probs, pa.float64())


def _inject_oracle_foreign_candidate(root, config):
    path = _oracle_path(root, "food_oracle")
    ids = pq.read_table(path)["candidate_id"].to_pylist()
    ids[0] = next(f"sim:merchant:{m}" for m in range(config.n_merchants)
                  if f"sim:merchant:{m}" not in ids[:10])
    _set_column(path, "candidate_id", ids, pa.string())


def _inject_oracle_ads_probability_one(root, config):
    path = _oracle_path(root, "ads_oracle")
    probs = pq.read_table(path)["true_click_probability"].to_pylist()
    probs[0] = 1.0
    _set_column(path, "true_click_probability", probs, pa.float64())


# -- baseline


def _inject_baseline_future_history(root, config):
    def edit(doc):
        doc["history"][0]["event_time"] = doc["history"][0]["available_at"] = (
            "2026-12-31T00:00:00+07:00")

    _edit_json(root / "baseline" / "ads.json", edit, compact=True)


def _inject_baseline_flipped_label(root, config):
    def edit(doc):
        next(lb for lb in doc["labels"] if lb["target"] == 1)["target"] = 0

    _edit_json(root / "baseline" / "food.json", edit, compact=True)


def _inject_baseline_bad_schema_version(root, config):
    def edit(doc):
        doc["metadata"]["schema_version"] = 2  # BaselineDataset.validate rejects it

    _edit_json(root / "baseline" / "food.json", edit, compact=True)


def _inject_baseline_ads_label_before_outcome(root, config):
    """The pre-R12 rule: an Ads label available 1 s after the exposure, before its outcome."""
    def edit(doc):
        cutoff = {q["request_id"]: q["cutoff_time"] for q in doc["queries"]}
        label = doc["labels"][0]
        label["available_at"] = (datetime.fromisoformat(cutoff[label["request_id"]])
                                 + timedelta(seconds=1)).isoformat()

    _edit_json(root / "baseline" / "ads.json", edit, compact=True)


def _inject_baseline_fake_click(root, config):
    """An unclicked impression claims a click 10 s after it: self-consistent, but not in the
    event log (ads/events.parquet)."""
    def edit(doc):
        row = next(e for e in doc["history"]
                   if e["event_type"] == "impression" and e["attributes"]["clicked"] == 0)
        row["attributes"]["clicked"] = 1
        row["available_at"] = (datetime.fromisoformat(row["event_time"])
                               + timedelta(seconds=10)).isoformat()

    _edit_json(root / "baseline" / "ads.json", edit, compact=True)


def _inject_baseline_order_without_category(root, config):
    def edit(doc):
        del next(e for e in doc["history"] if e["event_type"] == "order")["attributes"]["category"]

    _edit_json(root / "baseline" / "food.json", edit, compact=True)


def _inject_baseline_candidate_category(root, config):
    def edit(doc):
        attributes = doc["candidates"][0]["attributes"]
        attributes["category"] = "pizza" if attributes["cuisine"] != "pizza" else "lẩu"

    _edit_json(root / "baseline" / "food.json", edit, compact=True)


FOOD = ("ads", "food")  # the Food checks use ads/events.parquet (own-session check)
ORACLE = ("ads", "food", "oracle")
BASELINE = ("ads", "food", "baseline")
# id -> (check that must fail, packages, injection, other checks it legitimately breaks, kwargs)
CASES = {
    "food_location_key": ("food_inputs_public", FOOD, _inject_food_distance_key,
                          {"food_manifest"}, {}),
    "food_own_order_in_history": ("food_own_session_absent", FOOD, _inject_own_order_in_history,
                                  {"food_manifest"}, {}),
    "food_candidate_attributes": ("food_candidates", FOOD, _inject_candidate_attribute_mismatch,
                                  {"food_manifest"}, {}),
    # the own session is located by label_time, so it cannot be found either
    "food_label_at_cutoff": ("food_label_after_cutoff", FOOD, _inject_label_at_cutoff,
                             {"food_manifest", "food_own_session_absent"}, {}),
    "food_profile_after_cutoff": ("food_profile_as_of_cutoff", FOOD,
                                  _inject_profile_after_cutoff, {"food_manifest"}, {}),
    "food_wrong_split": ("food_split", FOOD, _inject_food_wrong_split, {"food_manifest"}, {}),
    "food_popularity_window": ("food_manifest", FOOD, _inject_popularity_window, set(), {}),
    # history_stop + 1 also breaks the window policy (stop = events before exposure) and puts
    # the impression row inside the window
    "ads_window_reaches_exposure": ("ads_history_before_exposure", ("ads",),
                                    _inject_ads_window_reaching_exposure,
                                    {"ads_manifest", "ads_history_window_policy",
                                     "ads_impression_rows"}, {}),
    "ads_window_too_short": ("ads_history_window_policy", ("ads",), _inject_ads_window_too_short,
                             {"ads_manifest"}, {}),
    "ads_wrong_impression_row": ("ads_impression_rows", ("ads",),
                                 _inject_ads_wrong_impression_row, {"ads_manifest"}, {}),
    "ads_wrong_split": ("ads_split", ("ads",), _inject_ads_wrong_split, {"ads_manifest"}, {}),
    "ads_click_flag_attribute": ("ads_columns_public", ("ads",), _inject_ads_click_flag_attribute,
                                 {"ads_manifest"}, {}),
    "ads_event_index_gap": ("ads_events_order", ("ads",), _inject_ads_event_index_gap,
                            {"ads_manifest"}, {}),
    "ads_click_label_flip": ("ads_impression_rows", ("ads",), _inject_ads_click_label_flip,
                             {"ads_manifest"}, {}),
    "pretrain_stray_file": ("pretrain_manifest", ("pretrain",), _inject_pretrain_stray_file,
                            set(), {}),
    "pretrain_preview_sha": ("pretrain_manifest", ("pretrain",), _inject_pretrain_preview_tamper,
                             set(), {}),
    # user 1's last event is in no snapshot history (it follows the last snapshot cutoff)
    "pretrain_event_in_week_7": ("pretrain_events_before_week7", ("pretrain",),
                                 _inject_pretrain_event_in_week_7, {"pretrain_manifest"}, {}),
    "pretrain_future_history_id": ("pretrain_snapshot_history", ("pretrain",),
                                   _inject_pretrain_future_history_id, {"pretrain_manifest"}, {}),
    "pretrain_hidden_profile_field": ("pretrain_snapshots", ("pretrain",),
                                      _inject_pretrain_hidden_profile_field,
                                      {"pretrain_manifest"}, {}),
    "pretrain_unparseable_actions": ("pretrain_loader", ("pretrain",),
                                     _inject_pretrain_unparseable_actions,
                                     {"pretrain_manifest"}, {}),
    "pretrain_group_truncated": ("pretrain_snapshots", ("pretrain",),
                                 _inject_pretrain_group_truncated, {"pretrain_manifest"}, {}),
    "oracle_sum": ("oracle_food_sums_to_one", ORACLE, _inject_oracle_sum, set(), {}),
    "oracle_foreign_candidate": ("oracle_food_matches_candidates", ORACLE,
                                 _inject_oracle_foreign_candidate, set(), {}),
    "oracle_ads_probability_one": ("oracle_ads", ORACLE, _inject_oracle_ads_probability_one,
                                   set(), {}),
    "baseline_future_history": ("baseline_ads_history_before_cutoff", BASELINE,
                                _inject_baseline_future_history, set(),
                                {"baseline_full_load_max_bytes": 0}),
    "baseline_flipped_label": ("baseline_food_structure", BASELINE,
                               _inject_baseline_flipped_label, set(), {}),
    "baseline_bad_schema_version": ("baseline_food_load_dataset", BASELINE,
                                    _inject_baseline_bad_schema_version, set(), {}),
    "baseline_ads_label_before_outcome": ("baseline_ads_structure", BASELINE,
                                          _inject_baseline_ads_label_before_outcome, set(), {}),
    "baseline_fake_click": ("baseline_ads_history_before_cutoff", BASELINE,
                            _inject_baseline_fake_click, set(), {}),
    "baseline_order_without_category": ("baseline_food_history_before_cutoff", BASELINE,
                                        _inject_baseline_order_without_category, set(), {}),
    "baseline_candidate_category": ("baseline_food_structure", BASELINE,
                                    _inject_baseline_candidate_category, set(), {}),
}


@pytest.mark.parametrize("case", list(CASES))
def test_check_catches_injected_violation(pristine, tmp_path, case):
    """Each injection breaks one rule; exactly its check (plus documented collateral) fails."""
    target, packages, inject, collateral, kwargs = CASES[case]
    root, config = _copy(pristine, tmp_path, packages)
    inject(root, config)
    with pytest.raises(CheckError) as info:
        run_checks(root, config, only=packages, **kwargs)
    results = info.value.report["results"]
    failed = {name for name, r in results.items() if r["status"] == "failed"}
    assert failed == {target} | collateral
    assert target in str(info.value)
    assert results[target]["violations"] >= 1 and results[target]["ids"]


def test_geohash_decode_round_trips():
    for lat, lon in ((10.7769, 106.7009), (57.64911, 10.40744), (-33.86, 151.21)):
        gh = geohash_encode(lat, lon, 6)
        dlat, dlon = geohash_decode(gh)
        assert geohash_encode(dlat, dlon, 6) == gh
        assert abs(dlat - lat) < 0.003 and abs(dlon - lon) < 0.006
