"""Ads-owned bridge from a frozen handoff to foundation and Baseline APIs.

``data.format`` selects the handoff: ``taobao`` (default; observed Taobao display-ad
impressions, checkpoint bound to this Ads manifest) or ``sim_v1`` (``ads/`` of the
simulator, docs/simulation_data_contract.md "Ads": events of every service, history
through ``src.downstream.sim_events`` exactly as pretrain encodes it, checkpoint bound
by vocabulary namespace and simulator config hash).

This adapter reads Data artifacts without repairing labels, resampling exposures,
or refitting the foundation tokenizer. Missing click availability permits only an
explicit diagnostic evaluation; it never becomes an audited source timestamp. The
sim package has no availability column: it is derived from the event stream (the
impression's click, else exposure + 27 s), the simulator's Baseline convention.
"""

from __future__ import annotations

import json
from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from zoneinfo import ZoneInfo

import pyarrow.parquet as pq
import torch

from src.contracts.output import fingerprint, sha256_file
from src.downstream.candidate_features import (
    attribute_names,
    encode_attributes,
    fit_attribute_vocabularies,
)
from src.downstream.foundation_init import check_sim_checkpoint_binding
from src.downstream.sim_events import (
    SCENARIOS,
    SESSION_START_TYPES,
    SIM_DATASET_NAME,
    ads_window_events,
    sim_canonical_sample,
)
from src.foundation_input.dataset import DatasetConfig, encode_sample, foundation_collate
from src.foundation_input.tokenizer import TokenizerConfig
from src.foundation_input.types import CanonicalSample, EventRecord
from src.foundation_input.validation import validate_sample
from src.training.checkpoint import load_checkpoint

EXPOSURE_POLICY = "taobao-observed-impression-single-candidate-v1"
NAMESPACE = "taobao_display_ads"
SIM_EXPOSURE_POLICY = "sim-served-impression-single-candidate-v1"
# Contract "Baseline" (R12): an unclicked impression's outcome is known one second after the
# latest possible click (exposure + 2..26 s).
SIM_AD_OUTCOME_MATURITY_SECONDS = 27
_PROFILE_IDENTITY = frozenset({"user_id", "as_of_timestamp"})


@dataclass(frozen=True)
class AdFormat:
    """Identity rules of one ``data.format``."""

    name: str
    namespace: str  # manifest source_dataset == checkpoint vocabulary dataset_name
    exposure_policy: str
    user_prefix: str
    ad_prefix: str
    exclusive_split_ends: bool  # sim manifests: train_end/valid_end are exclusive

    @property
    def sim(self):
        return self.name == "sim_v1"


AD_FORMATS = {
    "taobao": AdFormat("taobao", NAMESPACE, EXPOSURE_POLICY, "taobao:user:", "taobao:ad:", False),
    "sim_v1": AdFormat("sim_v1", SIM_DATASET_NAME, SIM_EXPOSURE_POLICY, "sim:user:", "sim:ad:",
                       True),
}


def ad_format(config):
    """The identity rules of the configured ``data.format`` (default Taobao)."""
    from src.downstream.ad_optimization import ad_data_format

    return AD_FORMATS[ad_data_format(config)]


def _time(value):
    if type(value) is not int:
        raise ValueError("Ads timestamp must be integer epoch seconds")
    return datetime.fromtimestamp(value, timezone.utc)


def _tokenizer(vocabulary):
    contract = vocabulary.tokenizer_contract
    if not contract:
        raise ValueError("Ads checkpoint lacks frozen tokenizer contract")
    tokenizer = TokenizerConfig(
        context_allowlist={key: tuple(value) for key, value in contract["context_allowlist"].items()},
        max_text_chars=contract["max_text_chars"],
        include_action_content=contract["include_action_content"],
        numeric_fields=frozenset(tuple(value) for value in contract["numeric_fields"]),
        numeric_transform_policy={
            (key.rsplit(":", 3)[-3], key.rsplit(":", 3)[-1]): value
            for key, value in vocabulary.numeric_transform_policy.items()
        },
        time_gap_boundaries_seconds=tuple(contract["time_gap_boundaries_seconds"]),
    )
    vocabulary.validate_tokenizer_config(tokenizer)
    return tokenizer


def _check_checkpoint(payload, vocabulary, expected_version):
    preprocess = payload.get("preprocess_metadata", {})
    if preprocess.get("dataset_version") != expected_version:
        raise ValueError("Ads checkpoint dataset version mismatch")
    if vocabulary.dataset_name != NAMESPACE or preprocess.get("dataset_namespace") != NAMESPACE:
        raise ValueError("Ads checkpoint namespace mismatch; cross-dataset user IDs cannot be joined")
    if preprocess.get("vocabulary_sha256") != fingerprint(payload["vocabularies"]):
        raise ValueError("Ads checkpoint vocabulary fingerprint mismatch")
    _tokenizer(vocabulary)


def _check_sim_checkpoint(payload, vocabulary, expected_sim_config_sha256):
    """Spec §6: the sim world binds the checkpoint, not this Ads manifest's own hash."""
    check_sim_checkpoint_binding(vocabulary, payload.get("preprocess_metadata", {}),
                                 expected_sim_config_sha256, package="Ads")
    _tokenizer(vocabulary)


def load_ad_foundation(path, *, expected_version, expected_sim_config_sha256=None):
    """Load the Person 3 foundation implementation with Taobao provenance checks.

    With ``expected_sim_config_sha256`` (``data.format: sim_v1``) the checkpoint is bound to
    the simulated world instead; the input loader matches ``expected_version`` against the
    Ads manifest.
    """
    model, _, vocabulary, payload = load_checkpoint(path, device="cpu")
    if expected_sim_config_sha256 is None:
        _check_checkpoint(payload, vocabulary, expected_version)
    else:
        _check_sim_checkpoint(payload, vocabulary, expected_sim_config_sha256)
    model.eval()
    return model


def _ordered_by_user(grouped):
    for events in grouped.values():
        events.sort(key=lambda event: (event["timestamp"], event["source_row_index"]))
        if any(type(event["event_index"]) is not int or event["event_index"] != index
               for index, event in enumerate(events)):
            raise ValueError("Ads history event_index must be unique, ordered and contiguous")
    return grouped


def _index_events(tables):
    grouped = defaultdict(list)
    by_id = {}
    for event in tables["events"]:
        identifier = event["event_id"]
        if not isinstance(identifier, str) or not identifier or identifier in by_id:
            raise ValueError("Ads duplicate or invalid event_id")
        if event["event_type"] != "ad_impression" or event["service"] != "ads":
            raise ValueError("Ads history must contain observed ad_impression events")
        _time(event["timestamp"])
        by_id[identifier] = event
        grouped[event["user_id"]].append(event)
    return _ordered_by_user(grouped), by_id


def _index_sim_events(tables):
    """Sim events of every service: per-user histories and the table (row = source_row_index)."""
    grouped, identifiers = defaultdict(list), set()
    events = tables["events"]
    for position, event in enumerate(events):
        identifier = event["event_id"]
        if not isinstance(identifier, str) or not identifier or identifier in identifiers:
            raise ValueError("Ads duplicate or invalid event_id")
        identifiers.add(identifier)
        if type(event["source_row_index"]) is not int or event["source_row_index"] != position:
            raise ValueError("Ads sim events source_row_index must equal the row number")
        if event["service"] not in SCENARIOS:
            raise ValueError(f"event_id={identifier}: unknown service {event['service']!r}")
        _time(event["timestamp"])
        grouped[event["user_id"]].append(event)
    return _ordered_by_user(grouped), events


def _sim_impression(row, events):
    """The opportunity's own impression: ``events[source_row_index]``."""
    index = row["source_row_index"]
    event = events[index] if type(index) is int and 0 <= index < len(events) else None
    if event is None or (event["event_type"], event["service"]) != ("ad_impression", "ads"):
        raise ValueError("Ads event/opportunity exposure identity mismatch")
    return event


def _sim_label_available_at(row, impression, user_events):
    """When the request's outcome is observable: its click, else exposure + 27 s.

    The click is the user's next event, an ``ad_click`` of the same ad (an ad session has a
    second event exactly when clicked); the row's ``clicked`` must agree with the stream.
    """
    position = impression["event_index"] + 1
    following = user_events[position] if position < len(user_events) else None
    clicked = following is not None and (
        following["event_type"], following["service"], following["entity_id"]
    ) == ("ad_click", "ads", row["candidate_id"])
    if clicked != bool(row["clicked"]):
        raise ValueError(
            f"request_id={row['request_id']}: clicked disagrees with the event stream")
    if clicked:
        return following["timestamp"]
    return row["exposure_timestamp"] + SIM_AD_OUTCOME_MATURITY_SECONDS


def _check_sim_requests(rows, manifest):
    if len({row["source_row_index"] for row in rows}) != len(rows):
        raise ValueError("Ads sim opportunities must reference distinct impressions")
    counts = {split: sum(row["split"] == split for row in rows)
              for split in ("train", "validation", "test")}
    if manifest["request_counts"] != counts:
        raise ValueError(f"Ads manifest request_counts {manifest['request_counts']} differ "
                         f"from ad_opportunities {counts}")


def read_ad_handoff(config):
    """Validate checksums, temporal pointers and complete observed exposure identity.

    ``sim_v1`` opportunity rows gain the derived ``label_available_at_timestamp``.
    """
    from src.downstream.ad_optimization import is_sha256, validate_opportunities

    fmt = ad_format(config)
    root = Path(config["data"]["root"]).resolve()
    path = root / "manifest.json"
    if not path.is_file():
        raise ValueError("Ads handoff requires manifest.json")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    required = {"dataset_version", "schema_version", "generator_version", "seed", "timezone",
                "timestamp_unit", "files", "split", "source_dataset", "history_policy",
                "candidate_policy", "missing_click_policy", "negative_sampling", "row_count"}
    if fmt.sim:
        required |= {"sim_config_sha256", "request_counts"}
    if not required <= set(manifest) or manifest["schema_version"] != 1:
        raise ValueError("Ads manifest schema/required fields invalid")
    if manifest["timestamp_unit"] != "seconds":
        raise ValueError("Ads timestamp unit must be seconds")
    ZoneInfo(manifest["timezone"])
    if manifest["dataset_version"] != config["data"]["expected_version"]:
        raise ValueError("Ads data_version mismatch")
    if manifest["source_dataset"] != fmt.namespace:
        raise ValueError("Ads handoff namespace mismatch")
    policy = fmt.exposure_policy
    if manifest["candidate_policy"] != policy or config["exposure_policy"] != policy:
        raise ValueError("Ads unsupported exposure policy; do not invent candidates")
    if fmt.sim and not is_sha256(manifest["sim_config_sha256"]):
        raise ValueError("Ads sim_v1 manifest sim_config_sha256 must be a SHA-256")
    if manifest["negative_sampling"] != "none" or manifest["missing_click_policy"] != "reject":
        raise ValueError("Ads exposure or missing-click policy mismatch")
    history = manifest["history_policy"]
    if (history.get("timestamp_rule") != "strictly < cutoff"
            or history.get("target_click_in_history") is not False
            or history.get("equal_timestamp_events_excluded") is not True
            or type(history.get("max_events")) is not int or history["max_events"] < 1):
        raise ValueError("Ads unsafe history policy")
    tables = {}
    schemas = {
        "profiles": {"user_id", "as_of_timestamp"},
        "ad_catalog": {"candidate_id", "first_observed_timestamp"},
        "events": {"event_id", "source_row_index", "user_id", "timestamp", "event_type",
                   "service", "entity_id", "event_index"},
        "ad_opportunities": {"request_id", "source_row_index", "user_id", "cutoff_timestamp",
                             "candidate_id", "clicked", "split", "exposure_timestamp",
                             "candidate_timestamp", "profile_as_of_timestamp",
                             "history_end_timestamp", "history_start", "history_stop"},
    }
    if fmt.sim:  # events of every service keep their public context and numeric actions
        schemas["events"] = schemas["events"] | {"attributes_json"}
    for name, required_columns in schemas.items():
        metadata = manifest["files"].get(name)
        if not isinstance(metadata, dict):
            raise ValueError(f"{name}: missing Ads handoff file metadata")  # noqa: TRY004
        source = (root / metadata["path"]).resolve()
        if not source.is_relative_to(root) or metadata.get("format") != "parquet" or not source.is_file():
            raise ValueError(f"{name}: expected existing in-root Parquet handoff")
        if sha256_file(source) != metadata["sha256"]:
            raise ValueError(f"{name}: checksum mismatch")
        table = pq.read_table(source)
        if type(metadata.get("row_count")) is not int or table.num_rows != metadata["row_count"]:
            raise ValueError(f"{name}: row count mismatch")
        if not required_columns <= set(table.column_names):
            raise ValueError(f"{name}: missing required columns")
        # The sim profile also carries static fields (as of day 1) that pretrain tokenizes.
        if name == "profiles" and not fmt.sim and set(table.column_names) != required_columns:
            raise ValueError("Ads identity-only profiles exclude untimestamped static attributes")
        if name == "events" and set(table.column_names) & {"clicked", "clk", "nonclk", "click_timestamp"}:
            raise ValueError("Ads target click leaked into history features")
        tables[name] = table.to_pylist()
    rows = tables["ad_opportunities"]
    validate_opportunities(rows, manifest, config["data"]["expected_version"], policy,
                           exclusive_split_ends=fmt.exclusive_split_ends)
    if type(manifest["row_count"]) is not int or manifest["row_count"] != len(rows):
        raise ValueError("Ads manifest exposure row count mismatch")
    by_request = {row["request_id"]: row for row in rows}
    if len(by_request) != len(rows):
        raise ValueError("Ads single-candidate exposure policy requires unique requests")
    if fmt.sim:  # events of every service; a request names its impression by row
        events_by_user, events_by_row = _index_sim_events(tables)
        _check_sim_requests(rows, manifest)
    else:
        events_by_user, events_by_id = _index_events(tables)
        if set(by_request) != set(events_by_id):
            raise ValueError("Ads opportunity/event set must match exactly")
    profiles = {row["user_id"]: row["as_of_timestamp"] for row in tables["profiles"]}
    catalog = {row["candidate_id"]: row["first_observed_timestamp"] for row in tables["ad_catalog"]}
    if len(profiles) != len(tables["profiles"]) or len(catalog) != len(tables["ad_catalog"]):
        raise ValueError("Ads duplicate profile or catalog identity")
    users, candidates = {row["user_id"] for row in rows}, {row["candidate_id"] for row in rows}
    if fmt.sim:  # the sim package lists every user and every ad
        if not users <= set(profiles) or not candidates <= set(catalog):
            raise ValueError("Ads profile/catalog population must cover handoff opportunities")
    elif set(profiles) != users or set(catalog) != candidates:
        raise ValueError("Ads profile/catalog population must match handoff opportunities")
    event_times = {user: [event["timestamp"] for event in events] for user, events in events_by_user.items()}
    for row in rows:
        user, cutoff = row["user_id"], row["cutoff_timestamp"]
        if not user.startswith(fmt.user_prefix) or not row["candidate_id"].startswith(fmt.ad_prefix):
            raise ValueError(f"Ads canonical identities must use the {fmt.name} namespace")
        event = _sim_impression(row, events_by_row) if fmt.sim else events_by_id[row["request_id"]]
        if (event["user_id"], event["timestamp"], event["entity_id"], event["source_row_index"]) != (
                user, cutoff, row["candidate_id"], row["source_row_index"]):
            raise ValueError("Ads event/opportunity exposure identity mismatch")
        if row["exposure_timestamp"] != cutoff:
            raise ValueError("Ads exposure timestamp differs from cutoff")
        if row["profile_as_of_timestamp"] != profiles[user] or row["candidate_timestamp"] != catalog[row["candidate_id"]]:
            raise ValueError("Ads profile/catalog availability mismatch")
        _time(profiles[user])
        _time(catalog[row["candidate_id"]])
        stop = bisect_left(event_times[user], cutoff)
        start = max(0, stop - history["max_events"])
        if type(row["history_start"]) is not int or type(row["history_stop"]) is not int or (
                row["history_start"], row["history_stop"]) != (start, stop):
            raise ValueError("Ads history pointer mismatch; only strictly earlier events are allowed")
        expected_end = event_times[user][stop - 1] if stop else None
        if row["history_end_timestamp"] != expected_end:
            raise ValueError("Ads history_end_timestamp mismatch")
        if fmt.sim:
            available = _sim_label_available_at(row, event, events_by_user[user])
            if row.get("label_available_at_timestamp") not in (None, available):
                raise ValueError("Ads label availability disagrees with the event stream")
            row["label_available_at_timestamp"] = available
        if row.get("label_available_at_timestamp") is not None:
            _time(row["label_available_at_timestamp"])
            if row["label_available_at_timestamp"] < cutoff:
                raise ValueError("Ads label availability precedes exposure")
    return manifest, tables, sha256_file(path)


def _canonical(row, manifest, events_by_user):
    user = row["user_id"]
    history = events_by_user[user][row["history_start"]:row["history_stop"]]
    records = tuple(EventRecord(
        event_id=event["event_id"], user_id=user, source_record_id=event["event_id"],
        source_record_index=event["source_row_index"], action_index=0, source_action_count=1,
        event_time=_time(event["timestamp"]), time_precision="second",
        eligibility_time=_time(event["timestamp"]), eligibility_inclusive=True,
        scenario="Advertisement", event_type="ad_impression",
        context={"product": event["entity_id"]}, action={"type": "ad_impression"},
    ) for event in history)
    return validate_sample(CanonicalSample(
        sample_id=row["request_id"], dataset_name=manifest["source_dataset"], split=row["split"],
        user_id=user, cutoff_time=_time(row["cutoff_timestamp"]), cutoff_inclusive=False,
        history_order_known=True, history_truncated=row["history_start"] > 0,
        history_group_truncated=False,
        profile={"as_of_time": _time(row["profile_as_of_timestamp"]).isoformat()}, events=records,
        history_start=row["history_start"], history_stop=row["history_stop"],
    ))


def _sim_canonical(row, manifest, events_by_user, profiles):
    """A sim request as pretrain sees it: history through the shared sim event converter."""
    user = row["user_id"]
    window = events_by_user[user][row["history_start"]:row["history_stop"]]
    profile = {key: value for key, value in profiles[user].items() if key not in _PROFILE_IDENTITY}
    profile["as_of_time"] = _time(row["profile_as_of_timestamp"])
    return validate_sample(sim_canonical_sample(
        sample_id=row["request_id"], dataset_name=manifest["source_dataset"], split=row["split"],
        user_id=user, cutoff_time=_time(row["cutoff_timestamp"]), profile=profile,
        events=ads_window_events(window), history_truncated=row["history_start"] > 0,
        # A window cut mid-session does not open with a session-start event type.
        history_group_truncated=bool(window) and window[0]["event_type"] not in SESSION_START_TYPES,
    ))


def load_ad_inputs(config, foundation):
    """Batch every original exposure with the checkpoint's frozen preprocessing."""
    from src.downstream.ad_optimization import AdBatch, AdInputs

    fmt = ad_format(config)
    manifest, tables, manifest_sha = read_ad_handoff(config)
    _, model_config, vocabulary, checkpoint = load_checkpoint(
        config["checkpoint"]["path"], device="cpu", load_model=False,
    )
    if fmt.sim:
        _check_sim_checkpoint(checkpoint, vocabulary, manifest["sim_config_sha256"])
    else:
        _check_checkpoint(checkpoint, vocabulary, config["data"]["expected_version"])
    preprocess = checkpoint["preprocess_metadata"]
    if model_config != foundation.config:
        raise ValueError("Ads checkpoint/model configuration mismatch")
    if not fmt.sim:  # Taobao: the vocabulary was fitted on this very handoff
        if preprocess.get("source_manifest_sha256") != manifest_sha:
            raise ValueError("Ads handoff manifest differs from checkpoint preprocessing source")
        if vocabulary.source_fingerprint != manifest_sha:
            raise ValueError("Ads checkpoint vocabulary source fingerprint mismatch")
    rows = tables["ad_opportunities"]
    train_ids = sorted({row["candidate_id"] for row in rows if row["split"] == "train"})
    if not train_ids:
        raise ValueError("Ads requires training exposures")
    candidate_vocabulary = {candidate: index + 1 for index, candidate in enumerate(train_ids)}
    names = attribute_names(config.get("model", {}))
    catalog = attribute_vocabularies = None
    if names:
        if not fmt.sim:
            raise ValueError("model.candidate_attributes needs the sim_v1 ad catalog")
        catalog = {row["candidate_id"]: row for row in tables["ad_catalog"]}
        attribute_vocabularies = fit_attribute_vocabularies(catalog, names, train_ids)
    tokenizer = _tokenizer(vocabulary)
    sequence = preprocess["sequence"]
    dataset = DatasetConfig(max_tokens=sequence["max_tokens"], max_event_count=sequence["max_event_count"])
    if dataset.max_tokens != model_config.max_length:
        raise ValueError("Ads checkpoint sequence/model length mismatch")
    if fmt.sim:
        events_by_user, _ = _index_sim_events(tables)
        profiles = {row["user_id"]: row for row in tables["profiles"]}
        to_sample = partial(_sim_canonical, manifest=manifest, events_by_user=events_by_user,
                            profiles=profiles)
    else:
        events_by_user, _ = _index_events(tables)
        to_sample = partial(_canonical, manifest=manifest, events_by_user=events_by_user)
    batches = {split: [] for split in ("train", "validation", "test")}
    pending = {split: [] for split in batches}
    batch_size = config["training"].get("batch_size", 64)
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("Ads training.batch_size must be a positive integer")

    def emit(split):
        groups = pending[split]
        if not groups:
            return
        encoded, originals = zip(*groups)
        ids = torch.tensor([[candidate_vocabulary.get(row["candidate_id"], 0)] for row in originals], dtype=torch.long)
        request_rows = [[row] for row in originals]
        attributes = None if not names else encode_attributes(
            request_rows, catalog, names, attribute_vocabularies, 1)
        batches[split].append(AdBatch(foundation_collate(list(encoded)), ids, request_rows,
                                      candidate_attributes=attributes))
        groups.clear()

    for row in sorted(rows, key=lambda row: (row["cutoff_timestamp"], row["request_id"])):
        sample = to_sample(row)
        pending[row["split"]].append((encode_sample(sample, vocabulary, tokenizer, dataset), row))
        if len(pending[row["split"]]) >= batch_size:
            emit(row["split"])
    for split in batches:
        emit(split)
    return AdInputs(batches, rows, manifest, manifest_sha, fingerprint(preprocess),
                    len(candidate_vocabulary) + 1, model_config.d_model, candidate_vocabulary,
                    names, attribute_vocabularies)


def ad_readiness_report(config):
    """Report concrete data limitations without changing the handoff."""
    manifest, tables, digest = read_ad_handoff(config)
    rows = tables["ad_opportunities"]
    train_candidates = {row["candidate_id"] for row in rows if row["split"] == "train"}
    splits = {}
    for split in ("train", "validation", "test"):
        selected = [row for row in rows if row["split"] == split]
        positives = sum(row["clicked"] for row in selected)
        splits[split] = {
            "exposures": len(selected), "requests": len(selected), "clicks": positives,
            "class_prevalence": positives / len(selected) if selected else None,
            "unseen_candidate_exposures": sum(row["candidate_id"] not in train_candidates for row in selected),
            "empty_history_requests": sum(row["history_stop"] == 0 for row in selected),
            "missing_label_availability": sum(row.get("label_available_at_timestamp") is None for row in selected),
        }
    return {"status": "passed", "data_version": manifest["dataset_version"],
            "data_manifest_sha256": digest, "exposure_policy": manifest["candidate_policy"],
            "negative_sampling": "none", "splits": splits,
            "ranking_metrics_informative": False,
            "limitations": list(manifest.get("limitations", [])) + [
                "Only observed singleton exposures are supplied; evaluate CTR discrimination and calibration.",
                "Missing audited label availability permits only explicitly marked diagnostic evaluation.",
            ]}


def ad_baseline_dataset(config):
    """Map labels to the Baseline team's public data contract without metric logic."""
    from src.baselines.contracts import BaselineDataset, Candidate, DatasetMetadata, Label, Query

    manifest, tables, _ = read_ad_handoff(config)
    policy = config.get("evaluation", {}).get("label_availability_policy")
    if policy not in {"handoff_timestamp", "at_cutoff_diagnostic"}:
        raise ValueError("Ads evaluation requires explicit label_availability_policy")
    metadata = DatasetMetadata(
        dataset_name=manifest["source_dataset"], data_version=manifest["dataset_version"],
        task="ad_optimization", item_kind="ad", timezone=manifest["timezone"],
        candidate_policy=manifest["candidate_policy"], availability_policy=policy,
    )
    queries, candidates, labels = [], [], []
    for row in tables["ad_opportunities"]:
        request, candidate, cutoff = row["request_id"], row["candidate_id"], row["cutoff_timestamp"]
        queries.append(Query(request, row["user_id"], _time(cutoff), row["split"]))
        candidates.append(Candidate(request, candidate))
        available = row.get("label_available_at_timestamp") if policy == "handoff_timestamp" else cutoff
        if type(available) is not int:
            raise ValueError(f"request_id={request}: label_available_at_timestamp missing or invalid")
        labels.append(Label(request, candidate, int(row["clicked"]), _time(available)))
    dataset = BaselineDataset(metadata, tuple(queries), tuple(candidates), tuple(labels))
    dataset.validate()
    return dataset


def ad_baseline_evaluator(config):
    """Bind the public Baseline evaluator to an explicitly registered Ads handoff."""
    from src.evaluation.evaluator import DatasetRegistry, evaluate_predictions

    registry = DatasetRegistry()
    registry.register(ad_baseline_dataset(config))
    return partial(evaluate_predictions, registry=registry)
