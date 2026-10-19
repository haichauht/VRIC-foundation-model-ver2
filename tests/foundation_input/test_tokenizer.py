from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

import pytest

from src.foundation_input.tokenizer import TokenizerConfig, semantic_tokens, tokenize_sample
from src.foundation_input.types import CanonicalSample, EventRecord
from src.foundation_input.vocab import (
    VocabularyArtifacts,
    VocabularyConfig,
    fit_vocabulary,
    fit_vocabulary_from_samples,
)

from .fixtures.samples import sample_mapping


def _sample(profile: dict, events: tuple[EventRecord, ...]) -> CanonicalSample:
    raw = sample_mapping("omnibehavior_v2")
    return CanonicalSample(**{**raw, "profile": profile, "events": events})


def _event(**overrides) -> EventRecord:
    raw = sample_mapping("omnibehavior_v2")["events"][0]
    return EventRecord(**{**raw, **overrides})


def _vocabulary_config() -> VocabularyConfig:
    return VocabularyConfig(dataset_name="omnibehavior_v2")


def _score_sample(value: float | None) -> CanonicalSample:
    return _sample({"score": value}, (_event(),))


def test_profile_column_order_does_not_change_tabular_token_set() -> None:
    event = _event()
    first = _sample({"event_count_30d": 2, "active_days_30d": 1}, (event,))
    second = _sample({"active_days_30d": 1, "event_count_30d": 2}, (event,))

    first_tokens = {
        (token.key, token.value, token.modality)
        for token in semantic_tokens(first, TokenizerConfig())
        if token.source == "tabular"
    }
    second_tokens = {
        (token.key, token.value, token.modality)
        for token in semantic_tokens(second, TokenizerConfig())
        if token.source == "tabular"
    }
    assert first_tokens == second_tokens
    assert ("user_id", "user-001", "categorical") in first_tokens


def test_numeric_missing_and_list_text_are_encoded_deterministically() -> None:
    event = _event(
        action={"click_cnt": 0, "content": [{"role": "user", "content": "hello"}]}
    )
    config = TokenizerConfig(
        numeric_fields=frozenset({("tabular", "profile.score")}),
        include_action_content=True,
    )
    train = _sample({"score": 0.0}, (event,))
    artifacts = fit_vocabulary(
        semantic_tokens(train, config),
        VocabularyConfig(dataset_name="omnibehavior_v2", include_action_content=True),
        numeric_field_registry=config.numeric_fields,
    )

    transformed = _sample({"score": None}, (event,))
    records = tokenize_sample(transformed, artifacts, config)
    score = next(record for record in records if record.token.key == "profile.score")
    content = next(record for record in records if record.token.key == "action.content")

    assert score.numeric_missing is True
    assert score.numeric_value == 0.0
    assert content.token.value == "user: hello"


def test_action_content_can_be_disabled_without_dropping_other_action_fields() -> None:
    event = _event(action={"type": "click", "content": [{"role": "user", "content": "hello"}]})
    tokens = semantic_tokens(_sample({}, (event,)), TokenizerConfig())
    keys = {token.key for token in tokens}

    assert "action.content" not in keys
    assert "action.type" in keys


def test_event_tokens_follow_allowlist_and_keep_source_record_group() -> None:
    event = _event(
        context={"product_id": "p-1", "secret_outcome": "must-not-enter"},
        action={"type": "click", "click_cnt": 1},
    )
    tokens = semantic_tokens(_sample({}, (event,)), TokenizerConfig())
    keys = {token.key for token in tokens}

    assert "context.product_id" in keys
    assert "context.secret_outcome" not in keys
    assert all(
        token.event_group_id == "user-001:0" for token in tokens if token.source == "sequence"
    )


def test_tokenize_assigns_ids_and_time_gap_buckets() -> None:
    first = _event()
    second = _event(
        event_id="user-001:1:0",
        source_record_id="user-001:1",
        source_record_index=1,
        event_time=datetime(2025, 9, 19, 1, tzinfo=timezone.utc),
        eligibility_time=datetime(2025, 9, 19, 1, tzinfo=timezone.utc),
    )
    sample = _sample({"event_count_30d": 2}, (first, second))
    config = TokenizerConfig()
    semantic = semantic_tokens(sample, config)
    artifacts = fit_vocabulary(semantic, VocabularyConfig(dataset_name=sample.dataset_name))
    encoded = tokenize_sample(sample, artifacts, config)

    assert encoded
    assert all(record.value_id != 4 for record in encoded)
    event_gaps = [record.time_gap_id for record in encoded if record.token.source == "sequence"]
    assert event_gaps[0] == 0
    assert max(event_gaps) > 0


def test_interval_values_remain_categorical_strings() -> None:
    event = _event(action={"type": "purchase", "order_price_interval": "[29,36)"})
    tokens = semantic_tokens(_sample({}, (event,)), TokenizerConfig())
    interval = [token for token in tokens if token.key == "action.order_price_interval"]
    assert interval[0].value == "[29,36)"
    assert interval[0].modality == "categorical"


def test_unknown_history_order_does_not_create_factual_positions_or_gaps() -> None:
    first = _event()
    second = _event(
        event_id="user-001:1:0",
        source_record_id="user-001:1",
        source_record_index=1,
        event_time=datetime(2025, 9, 19, 1, tzinfo=timezone.utc),
        eligibility_time=datetime(2025, 9, 19, 1, tzinfo=timezone.utc),
    )
    sample = _sample({}, (first, second))
    sample = CanonicalSample(**{**sample.__dict__, "history_order_known": False})
    semantic = semantic_tokens(sample, TokenizerConfig())
    artifacts = fit_vocabulary(semantic, VocabularyConfig(dataset_name=sample.dataset_name))

    records = tokenize_sample(sample, artifacts, TokenizerConfig())

    sequence = [record for record in records if record.token.source == "sequence"]
    assert {record.position_id for record in sequence} == {0}
    assert {record.time_gap_id for record in sequence} == {0}


def test_artifact_rejects_mismatched_time_gap_boundaries() -> None:
    config = TokenizerConfig(time_gap_boundaries_seconds=(60.0, 300.0))
    artifacts = fit_vocabulary_from_samples([_score_sample(1.0)], _vocabulary_config(), config)

    with pytest.raises(ValueError, match="TokenizerConfig"):
        tokenize_sample(
            _score_sample(1.0), artifacts, TokenizerConfig(time_gap_boundaries_seconds=(60.0,))
        )


def test_artifact_rejects_mismatched_numeric_registry() -> None:
    config = TokenizerConfig(numeric_fields=frozenset({("tabular", "profile.score")}))
    artifacts = fit_vocabulary_from_samples([_score_sample(1.0)], _vocabulary_config(), config)

    with pytest.raises(ValueError, match="TokenizerConfig"):
        tokenize_sample(
            _score_sample(None), artifacts, TokenizerConfig(numeric_fields=frozenset())
        )


def test_vocabulary_config_policy_is_not_silently_overwritten() -> None:
    artifacts = fit_vocabulary_from_samples(
        [_score_sample(1.0)],
        replace(
            _vocabulary_config(),
            numeric_transform_policy={("tabular", "profile.score"): "log1p_zscore"},
        ),
        TokenizerConfig(numeric_fields=frozenset({("tabular", "profile.score")})),
    )

    assert artifacts.numeric_transform_policy[
        "omnibehavior_v2:tabular:numeric:profile.score"
    ] == "log1p_zscore"


def test_tokenizer_contract_round_trips(tmp_path) -> None:
    config = TokenizerConfig(
        context_allowlist={"Food": ("item_price",)},
        max_text_chars=64,
        numeric_fields=frozenset({("tabular", "profile.score")}),
        time_gap_boundaries_seconds=(60.0, 300.0),
    )
    artifacts = fit_vocabulary_from_samples([_score_sample(1.0)], _vocabulary_config(), config)
    path = tmp_path / "vocabulary.json"
    artifacts.save(path)

    assert VocabularyArtifacts.load(path).tokenizer_contract == artifacts.tokenizer_contract
