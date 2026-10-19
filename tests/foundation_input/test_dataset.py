from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.foundation_input.dataset import (
    DatasetConfig,
    FoundationPretrainDataset,
    _truncate_records,
    encode_sample,
    foundation_collate,
)
from src.foundation_input.loader import iter_normalized_samples
from src.foundation_input.tokenizer import TokenizerConfig, semantic_tokens
from src.foundation_input.types import (
    SPECIAL_TOKEN_IDS,
    CanonicalSample,
    EncodedSample,
    EventRecord,
    KeyValueToken,
    TokenRecord,
)
from src.foundation_input.vocab import VocabularyConfig, fit_vocabulary

from .fixtures.samples import sample_mapping
from .test_manifest import _write_dataset


class FakeTensorBackend:
    long = np.int64
    bool = np.bool_
    float32 = np.float32

    @staticmethod
    def tensor(values, *, dtype):
        return np.asarray(values, dtype=dtype)


def _record(value_id: int, *, source: str, group: str | None, position: int) -> TokenRecord:
    return TokenRecord(
        token=KeyValueToken(
            key="profile.x" if source == "tabular" else "event_type",
            value=value_id,
            modality="numeric" if source == "tabular" else "categorical",
            source=source,
            timestamp=None,
            event_group_id=group,
        ),
        value_id=value_id,
        key_id=value_id + 10,
        modality_id=1,
        source_id=1 if source == "tabular" else 2,
        position_id=position,
        time_gap_id=position,
    )


def _canonical_sample() -> CanonicalSample:
    raw = sample_mapping("omnibehavior_v2")
    raw["events"] = tuple(EventRecord(**event) for event in raw["events"])
    return CanonicalSample(**raw)


def _artifacts_for(sample: CanonicalSample):
    return fit_vocabulary(
        semantic_tokens(sample, TokenizerConfig()),
        VocabularyConfig(dataset_name=sample.dataset_name),
    )


def _too_large_group_sample() -> CanonicalSample:
    base = _canonical_sample()
    event = replace(base.events[0], action={"type": "click", "click_cnt": 1})
    return replace(base, events=(event,))


def _two_complete_event_sample() -> CanonicalSample:
    base = _canonical_sample()
    second = replace(
        base.events[0],
        event_id="user-001:1:0",
        source_record_id="user-001:1",
        source_record_index=1,
    )
    return replace(base, events=(base.events[0], second))


def test_collator_builds_layout_and_padding_without_changing_real_tokens() -> None:
    first = EncodedSample(
        sample_id="s1",
        dataset_name="omnibehavior_v2",
        user_id="u1",
        records=(
            _record(11, source="tabular", group=None, position=0),
            _record(21, source="sequence", group="e1", position=1),
        ),
    )
    second = EncodedSample(
        sample_id="s2",
        dataset_name="omnibehavior_v2",
        user_id="u2",
        records=(
            _record(12, source="tabular", group=None, position=0),
            _record(22, source="sequence", group="e2", position=1),
            _record(23, source="sequence", group="e3", position=2),
        ),
    )

    batch = foundation_collate([first, second], tensor_backend=FakeTensorBackend())

    assert batch.input_ids.shape == (2, 5)
    assert batch.input_ids[0, :4].tolist() == [1, 11, 2, 21]
    assert batch.input_ids[0, 4:].tolist() == [0]
    assert batch.input_ids[1, :5].tolist() == [1, 12, 2, 22, 23]
    assert batch.attention_mask.dtype == np.bool_
    assert batch.numeric_values.shape == batch.input_ids.shape
    assert batch.numeric_missing_mask.shape == batch.input_ids.shape
    assert batch.attention_mask[0].tolist() == [True, True, True, True, False]
    assert batch.targets["event_group_ids"][0].tolist() == [0, 0, 0, 1, 0]


def test_collator_keeps_tabular_before_sequence_tokens() -> None:
    sample = EncodedSample(
        sample_id="s1",
        dataset_name="omnibehavior_v2",
        user_id="u1",
        records=(
            _record(11, source="tabular", group=None, position=0),
            _record(12, source="tabular", group=None, position=0),
            _record(21, source="sequence", group="e1", position=1),
        ),
    )
    batch = foundation_collate([sample], tensor_backend=FakeTensorBackend())

    assert batch.input_ids[0].tolist() == [1, 11, 12, 2, 21]


def test_foundation_dataset_streams_encoded_samples(tmp_path) -> None:
    _write_dataset(tmp_path)
    canonical = next(iter_normalized_samples(tmp_path, split="train", validate=True))
    artifacts = fit_vocabulary(
        semantic_tokens(canonical, TokenizerConfig()),
        VocabularyConfig(dataset_name=canonical.dataset_name),
    )
    dataset = FoundationPretrainDataset(
        str(tmp_path),
        "train",
        artifacts,
        tokenizer_config=TokenizerConfig(),
        dataset_config=DatasetConfig(max_event_count=30, max_tokens=16),
    )

    encoded = dataset[0]
    assert encoded.sample_id == canonical.sample_id
    assert encoded.records


def test_next_action_target_is_excluded_from_input() -> None:
    first = _record(21, source="sequence", group="e1", position=1)
    second = _record(22, source="sequence", group="e2", position=2)
    sample = EncodedSample(
        sample_id="s1",
        dataset_name="omnibehavior_v2",
        user_id="u1",
        records=(first, second),
        next_action_records=(second,),
    )

    batch = foundation_collate([sample], tensor_backend=FakeTensorBackend())

    assert batch.input_ids[0].tolist() == [1, 2, 21]
    assert batch.targets["event_group_ids"][0].tolist() == [0, 0, 1]
    assert batch.targets["next_action_group_ids"].tolist() == [2]
    assert batch.targets["next_action_type_ids"].tolist() == [22]
    assert batch.targets["next_action_valid_mask"].tolist() == [True]


def test_foundation_dataset_does_not_keep_unbounded_sample_cache(tmp_path) -> None:
    _write_dataset(tmp_path)
    canonical = next(iter_normalized_samples(tmp_path, split="train", validate=True))
    artifacts = fit_vocabulary(
        semantic_tokens(canonical, TokenizerConfig()),
        VocabularyConfig(dataset_name=canonical.dataset_name),
    )
    dataset = FoundationPretrainDataset(
        str(tmp_path), "train", artifacts, tokenizer_config=TokenizerConfig()
    )

    list(dataset)

    assert not hasattr(dataset, "_cache")


def test_max_tokens_reserves_cls_and_sep() -> None:
    sample = EncodedSample(
        sample_id="s1",
        dataset_name="omnibehavior_v2",
        user_id="u1",
        records=(
            _record(11, source="tabular", group=None, position=0),
            _record(21, source="sequence", group="e1", position=1),
        ),
    )
    retained = _truncate_records(list(sample.records), DatasetConfig(max_tokens=4)).records
    assert [record.value_id for record in retained] == [11, 21]


def test_oov_user_ids_do_not_collapse_to_unk() -> None:
    known_sample = _canonical_sample()
    artifacts = fit_vocabulary(
        semantic_tokens(known_sample, TokenizerConfig()),
        VocabularyConfig(dataset_name=known_sample.dataset_name),
    )
    first = replace(known_sample, user_id="unseen-user-1")
    second = replace(known_sample, user_id="unseen-user-2")
    config = DatasetConfig(next_action_prediction=False)

    encoded = [
        encode_sample(sample, artifacts, TokenizerConfig(), config)
        for sample in (first, second)
    ]
    batch = foundation_collate(encoded, tensor_backend=FakeTensorBackend())

    assert all(sample.user_id_value_id is None for sample in encoded)
    assert all(value != SPECIAL_TOKEN_IDS["UNK"] for value in batch.user_ids.tolist())
    assert batch.user_ids[0] != batch.user_ids[1]


def test_next_action_requires_known_order_and_complete_groups() -> None:
    base = _canonical_sample()
    second_event = EventRecord(
        **{
            **sample_mapping("omnibehavior_v2")["events"][0],
            "event_id": "user-001:1:0",
            "source_record_id": "user-001:1",
            "source_record_index": 1,
            "event_time": sample_mapping("omnibehavior_v2")["cutoff_time"],
            "eligibility_time": sample_mapping("omnibehavior_v2")["cutoff_time"],
        }
    )
    base = replace(base, events=(base.events[0], second_event))
    artifacts = fit_vocabulary(
        semantic_tokens(base, TokenizerConfig()),
        VocabularyConfig(dataset_name=base.dataset_name),
    )
    config = DatasetConfig(next_action_prediction=True)

    valid = encode_sample(base, artifacts, TokenizerConfig(), config)
    unknown_order = encode_sample(
        replace(base, history_order_known=False), artifacts, TokenizerConfig(), config
    )
    incomplete_group = encode_sample(
        replace(base, history_group_truncated=True), artifacts, TokenizerConfig(), config
    )

    assert valid.next_action_records
    assert valid.next_action_records[0].token.event_group_id == "user-001:1"
    assert not unknown_order.next_action_records
    assert not incomplete_group.next_action_records


@pytest.mark.parametrize(
    "replacement", [{"history_order_known": False}, {"history_group_truncated": True}]
)
def test_invalid_next_action_is_masked_not_targeted(replacement) -> None:
    sample = replace(_two_complete_event_sample(), **replacement)
    encoded = encode_sample(
        sample, _artifacts_for(sample), TokenizerConfig(), DatasetConfig(next_action_prediction=True)
    )

    batch = foundation_collate([encoded], tensor_backend=FakeTensorBackend())

    assert encoded.next_action_records == ()
    assert batch.targets["next_action_type_ids"].tolist() == [0]
    assert batch.targets["next_action_valid_mask"].tolist() == [False]


def test_one_complete_group_is_masked_for_next_action() -> None:
    sample = _canonical_sample()
    encoded = encode_sample(
        sample, _artifacts_for(sample), TokenizerConfig(), DatasetConfig(next_action_prediction=True)
    )

    batch = foundation_collate([encoded], tensor_backend=FakeTensorBackend())

    assert encoded.next_action_records == ()
    assert batch.targets["next_action_type_ids"].tolist() == [0]
    assert batch.targets["next_action_valid_mask"].tolist() == [False]


def test_event_group_ids_are_independent_from_position_ids() -> None:
    sample = EncodedSample(
        sample_id="s1",
        dataset_name="omnibehavior_v2",
        user_id="u1",
        records=(
            _record(21, source="sequence", group="e1", position=0),
            _record(22, source="sequence", group="e2", position=0),
        ),
    )

    batch = foundation_collate([sample], tensor_backend=FakeTensorBackend())

    assert batch.position_ids[0].tolist() == [0, 0, 0, 0]
    assert batch.targets["event_group_ids"][0].tolist() == [0, 0, 1, 2]


def test_truncation_keeps_recent_groups_and_an_event_token() -> None:
    records = [
        _record(11, source="tabular", group=None, position=0),
        _record(21, source="sequence", group="old", position=1),
        _record(22, source="sequence", group="middle", position=2),
        _record(23, source="sequence", group="recent", position=3),
    ]

    retained = _truncate_records(records, DatasetConfig(max_event_count=2, max_tokens=6)).records
    retained_groups = [
        record.token.event_group_id
        for record in retained
        if record.token.source == "sequence"
    ]

    assert retained_groups == ["middle", "recent"]
    crowded = _truncate_records(
        [
            _record(11, source="tabular", group=None, position=0),
            _record(12, source="tabular", group=None, position=0),
            _record(13, source="tabular", group=None, position=0),
            _record(21, source="sequence", group="recent", position=1),
        ],
        DatasetConfig(max_tokens=4),
    )
    assert any(record.token.source == "sequence" for record in crowded.records)


def test_encode_sample_rejects_group_that_cannot_fit_without_splitting() -> None:
    sample = _too_large_group_sample()

    with pytest.raises(ValueError, match=rf"sample_id={sample.sample_id}.*complete event group"):
        encode_sample(sample, _artifacts_for(sample), TokenizerConfig(), DatasetConfig(max_tokens=4))


def test_truncation_does_not_return_a_group_prefix() -> None:
    records = [
        _record(11, source="tabular", group=None, position=0),
        _record(21, source="sequence", group="recent", position=1),
        _record(22, source="sequence", group="recent", position=1),
        _record(23, source="sequence", group="recent", position=1),
    ]

    retained = _truncate_records(records, DatasetConfig(max_tokens=4))

    assert retained.rejected_recent_group
    assert all(record.token.source != "sequence" for record in retained.records)


def test_truncation_prefers_complete_event_group_over_tabular_token() -> None:
    records = [
        _record(11, source="tabular", group=None, position=0),
        _record(21, source="sequence", group="recent", position=1),
        _record(22, source="sequence", group="recent", position=1),
        _record(23, source="sequence", group="recent", position=1),
    ]

    retained = _truncate_records(records, DatasetConfig(max_tokens=5))

    assert not retained.rejected_recent_group
    assert [record.value_id for record in retained.records] == [21, 22, 23]
    assert all(record.token.source == "sequence" for record in retained.records)
