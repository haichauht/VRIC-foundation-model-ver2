# Foundation Input Contract

## Dataset selection

The canonical foundation pretraining run uses `data/pretrain_data/OmniBehavior_data`, as specified by `data/README_pretrain_data.md`. The Meituan-compatible run uses `data/pretrain_data/meituan_data`, which is a Meituan TRD-derived dataset converted to the same `omnibehavior_pretraining_v2` storage schema.

Each process receives one `dataset_root` and one logical split. The adapter does not concatenate sources or join users across roots. Dataset identity is part of the vocabulary namespace.

## Canonical sample

`CanonicalSample` contains:

- `sample_id`, `dataset_name`, `split`, `user_id`;
- `cutoff_time` and `cutoff_inclusive`;
- `profile` parsed from `profile_json`;
- ordered `EventRecord` values reconstructed from `history_event_ids`;
- `history_order_known`, `history_truncated`, and `history_group_truncated`.
- `history_start`/`history_stop` are checked against the contiguous event-shard slice;
  the requested split and cutoff coordinate are checked against preprocessing split bounds.

An `EventRecord` preserves `event_id`, `source_record_id`, source/action indices, event and eligibility timestamps, timestamp precision, scenario, event type, and parsed context/action dictionaries.

## Semantic tokens

- Profile fields become unordered `source="tabular"` tokens with keys such as `profile.event_count_30d`.
- Event fields become ordered `source="sequence"` tokens with keys such as `event.scenario`, `event.event_type`, `context.product_id`, and `action.click_cnt`.
- Context fields are restricted to the dataset's `context_allowlist`.
- `action.content` is excluded by default with `TokenizerConfig.include_action_content=False`;
  enabling it is an explicit ablation and requires a separately fitted vocabulary.
- Tokens preserve `event_group_id=source_record_id` for coordinated masking.
- Numeric tokens carry a train-fitted normalized continuous value and an explicit
  missing flag; zero is a valid observed value and is not equivalent to missing.
- `as_of_time`, event IDs, source indices, eligibility flags, and other storage metadata are not business-value tokens.
- Date-only/uncertain timestamps remain uncertain; no timezone or sub-day precision is invented.

## Vocabulary and numeric values

Vocabulary, numeric boundaries, and normalization metadata are fit only from the selected dataset's logical `train` split through `fit_vocabulary_from_samples`. Validation and test are transform-only. The serialized vocabulary records its fit split, schema version, and optional source fingerprint. OOV values map to `UNK`.

`numeric_values` and `numeric_missing_mask` are required `FoundationBatch` v2
fields aligned with `input_ids`. Missing/non-finite numeric values use
`numeric_values=0.0` and `numeric_missing_mask=True`; padding and non-numeric
tokens use `0.0` and `False`. The normalization policy is per numeric field and
must be serialized with the vocabulary. The vocabulary also serializes the tokenizer
contract (allowlist, text limit, action-content policy, numeric registry, and time-gap
boundaries); transform-time tokenization must use that exact contract.

`order_price_interval` and other source buckets remain categorical values. They are never converted into exact monetary values.

## Batch layout

```text
[CLS] [tabular tokens...] [SEP] [event tokens...] [PAD...]
```

Special IDs are fixed: `PAD=0`, `CLS=1`, `SEP=2`, `MASK=3`, `UNK=4`.

`max_history_events=30` in the source data is a limit on raw action observations, not tokens or event groups. `max_tokens` is a separate token limit; adapter truncation preserves recent complete `source_record_id` groups. If the most recent group cannot fit, the sample is rejected with its `sample_id`; a group is never split. Source-provided `history_group_truncated` provenance is retained for next-action eligibility.

The collator returns `FoundationBatch` and includes `targets["event_group_ids"]` as provenance metadata. When `DatasetConfig.next_action_prediction=True`, the final retained event group is moved to an explicit target-only view (`next_action_records`), excluded from input, and exposed through `next_action_type_ids`, `next_action_group_ids`, and `next_action_valid_mask`. The mask is false when order is uncertain, the source group is incomplete, or fewer than two complete groups remain. For a false mask, target IDs are sentinel `0`, not a class. Member 3 must exclude those rows from next-action loss using `next_action_valid_mask`, while other pretraining objectives may still use the same row. This creates a training objective view; it does not claim that stored data contains labels or masks.
