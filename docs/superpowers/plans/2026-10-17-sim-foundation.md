# Foundation on Simulated Data (Sub-project 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make pretrain → embeddings → Food and Ads downstream run correctly on the simulated `sim-v1` packages, with a trained `long_term` embedding and a random-init control arm.

**Architecture:** Pretrain takes the context allowlist and provenance from the package; the encoder optionally adds the UserIdAdapter embedding to the CLS input (`long_term_injection: cls`). One new module `src/downstream/sim_events.py` turns public sim events (Food `input.history`, Ads `events.parquet`) into `EventRecord`s that tokenize exactly like the pretrain loader's. Food and Ads readers gain a `sim_v1` data format next to the existing Meituan / Taobao paths.

**Tech Stack:** Python 3.11 (`.venv`), PyTorch, pyarrow, pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-10-17-sim-foundation-design.md` (Vietnamese; binding).

## Global Constraints

- Branch `feat/sim-foundation`. Never push. Every commit message ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Never modify `data/**`, `src/data/**`, `src/simulation/**` (the simulated data is frozen); never commit anything under `experiments/`.
- Defaults keep old behaviour: `long_term_injection` defaults to `"none"`, Food `data.format` to `"meituan"`, Ads `data.format` to `"taobao"`; checkpoints saved before this work still load. Existing tests keep passing.
- Dataset name of the simulated data: `"sim_v1"`. Scenario map: `ride → "Ride"`, `food → "Food"`, `ads → "Advertisement"`.
- Numeric action fields of sim events: `fare, distance_km, wait_minutes, order_amount, delivery_distance_km`. Ads action `type`: `ad_impression → "impression"`, `ad_click → "click"`; every other event's action `type` = its `event_type`.
- Session-start event types: `search_destination`, `search`, `ad_impression`.
- Sim Ads label availability: click time if clicked, else exposure + 27 s. Sim split bounds (`train_end`, `valid_end`) are exclusive upper bounds.
- Lint/test gate: `.venv/Scripts/python -m pytest -q` and `.venv/Scripts/python -m ruff check src tests configs` (repo-wide `ruff check .` has pre-existing errors; do not fix them). Python 3.10-compatible syntax, line length 100.

## Review Focus

- A checkpoint whose `model_config` has no `long_term_injection` key (Person 3's) must load and encode exactly as before → Task 2 test `test_old_checkpoint_without_injection_key_loads`.
- An empty history (Food `input.history == []`, Ads window `history_start == history_stop`) must encode to a valid sample, not crash → Task 3 test `test_empty_history_encodes`.
- A vocab fitted with the default allowlist, reused (`--vocab_path`) on a sim package, must stop with a clear error instead of silently dropping Ride/Food context → Task 1 test `test_reused_vocab_missing_package_allowlist_is_rejected`.
- An Ads window that starts in the middle of a session must group its leading events into one group and still match pretrain tokens → Task 3 test `test_ads_window_starting_mid_session_matches_pretrain`.
- A sim checkpoint from a different simulator config (`sim_config_sha256` mismatch), or a Food candidate missing from `merchants.parquet`, must fail before any training → Task 4 tests `test_sim_checkpoint_provenance_mismatch_is_rejected`, `test_sim_candidate_not_in_merchants_is_rejected`; Task 5 test `test_ads_sim_checkpoint_provenance_mismatch_is_rejected`.

---

### Task 1: Pretrain reads the package allowlist and provenance

**Files:**
- Create: `tests/conftest.py` (shared fixture), `tests/training/test_pretrain_sim.py`, `configs/pretrain_sim.yaml`, `configs/pretrain_sim_kaggle.yaml`
- Modify: `src/training/pretrain.py` (vocab fit ~L402-418, checkpoint saves ~L620-645)

**Interfaces:**
- Produces (fixture, `tests/conftest.py`): session fixture `sim_tiny_packages` → `SimPackages(root: Path, pretrain: Path, food: Path, ads: Path)` (frozen dataclass). Built once per session from `replace(PRESETS["smoke"], n_users=80, n_merchants=24, n_ads=40, n_drivers=40)`: `rng = np.random.default_rng(config.seed)`; `build_world(config, rng)` → `simulate(world, rng)` → `build_food_requests(sim, rng)` → `build_ad_requests(sim, rng)` → `export_pretrain(sim, root)`, `export_food(sim, food_requests, root)`, `export_ads(sim, ad_requests, root)` into `tmp_path_factory.mktemp("sim")`.
- Produces (`src/training/pretrain.py`):
  - `tokenizer_config_for_manifest(manifest: DatasetManifest) -> TokenizerConfig` — `context_allowlist = {**DEFAULT_CONTEXT_ALLOWLIST, **package}` where `package = manifest.config.get("context_allowlist", {})` (values as tuples); no package allowlist → `TokenizerConfig()`.
  - `check_vocab_allowlist(vocabularies: VocabularyArtifacts, tokenizer_config: TokenizerConfig) -> None` — raises `ValueError` naming the scenario and missing keys when the vocab's `tokenizer_contract["context_allowlist"]` lacks any scenario/key of `tokenizer_config.context_allowlist`.
  - `pretrain_preprocess_metadata(manifest: DatasetManifest, vocabularies: VocabularyArtifacts, *, max_length: int, max_event_count: int) -> dict` — keys `dataset_name`, `source_schema_version`, `source_fingerprint` (as today's default), `source_manifest_sha256` (sha256 of `<data_dir>/manifest.json`), `sequence: {"max_tokens": max_length, "max_event_count": max_event_count}`, `dataset_provenance: {k: manifest.data[k] for k in ("dataset_version", "generator_version", "sim_config_sha256") if k in manifest.data}`. Passed as `preprocess_metadata=` to both `save_checkpoint` calls.

- [ ] **Step 1: Write the failing tests** in `tests/training/test_pretrain_sim.py`:

```python
def test_package_allowlist_reaches_the_vocab(sim_tiny_packages):
    manifest = load_manifest(sim_tiny_packages.pretrain, validate_shard_files=False)
    config = tokenizer_config_for_manifest(manifest)
    assert "pickup_geohash" in config.context_allowlist["Ride"]
    assert "merchant_id" in config.context_allowlist["Food"]
    assert config.context_allowlist["E-commerce"] == DEFAULT_CONTEXT_ALLOWLIST["E-commerce"]
    vocab = fit_vocabulary_from_samples(
        iter_normalized_samples(sim_tiny_packages.pretrain, split="train", validate=False),
        VocabularyConfig(dataset_name=manifest.dataset_name), config)
    keys = set(vocab.key_to_id)
    assert any(k.endswith("context.pickup_geohash") for k in keys)
    assert any(k.endswith("context.cuisine") for k in keys)

def test_manifest_without_allowlist_uses_defaults(tmp_path): ...
    # a DatasetManifest(root=tmp_path, data={}, config={}) → TokenizerConfig() equality

def test_reused_vocab_missing_package_allowlist_is_rejected(sim_tiny_packages):
    # vocab fitted with TokenizerConfig() on the sim package →
    # check_vocab_allowlist(vocab, tokenizer_config_for_manifest(manifest)) raises ValueError
    # whose message contains "Ride"

def test_preprocess_metadata_records_sim_provenance(sim_tiny_packages):
    md = pretrain_preprocess_metadata(manifest, vocab, max_length=512, max_event_count=30)
    assert md["dataset_provenance"]["sim_config_sha256"] == manifest.data["sim_config_sha256"]
    assert md["sequence"] == {"max_tokens": 512, "max_event_count": 30}
    assert md["source_manifest_sha256"] == sha256 of (sim_tiny_packages.pretrain / "manifest.json")
```

- [ ] **Step 2: Run** `.venv/Scripts/python -m pytest tests/training/test_pretrain_sim.py -q` — Expected: FAIL (ImportError on the new names).
- [ ] **Step 3: Implement** the three functions; in `pretrain_single_dataset` build `tokenizer_config = tokenizer_config_for_manifest(manifest)`, pass it to `fit_vocabulary_from_samples(stream, vocab_config, tokenizer_config)`, call `check_vocab_allowlist` after `VocabularyArtifacts.load(vocab_path)`, and pass `preprocess_metadata=pretrain_preprocess_metadata(...)` to both `save_checkpoint` calls.
- [ ] **Step 4: Add configs.** `configs/pretrain_sim.yaml`: copy of `configs/pretrain.yaml` with `dataset_version: "sim-smoke-v1"`, `data_dir: "experiments/sim/sim-smoke-v1/pretrain"`, `output_dir: "experiments/pretrain_sim"`, `device: "cpu"`, `model.long_term_injection: "cls"`, `sequence.max_length: 512`, `training.batch_size: 16`, `training.epochs: 1`, `training.mixed_precision: false`. `configs/pretrain_sim_kaggle.yaml`: same with `dataset_version: "sim-v1"`, `data_dir: "/kaggle/working/sim-v1/pretrain"`, `output_dir: "/kaggle/working/pretrain_sim"`, `device: "cuda"`, `training.batch_size: 64`, `training.epochs: 6`, `training.learning_rate: 0.0003`, `training.mixed_precision: true`. (Task 2 makes `model.long_term_injection` effective.)
- [ ] **Step 5: Run** the new tests, `tests/training -q`, and ruff — Expected: PASS.
- [ ] **Step 6: Commit** `feat(pretrain): package context allowlist, provenance metadata, sim configs`.

### Task 2: `long_term` receives gradient (`long_term_injection`)

**Files:**
- Modify: `src/models/encoder.py` (`FoundationModelConfig` L17-50, `FoundationModel.forward` ~L196-225), `src/models/objectives.py` (`PretrainObjectivesConfig`, `apply_mask` L41-80), `src/training/pretrain.py` (model and objectives config construction ~L426-455)
- Test: `tests/models/test_encoder.py`, `tests/training/test_objectives.py`, `tests/training/test_checkpoint.py`

**Interfaces:**
- Produces: `FoundationModelConfig.long_term_injection: str = "none"` (allowed `"none"`, `"cls"`; else `ValueError` in `__post_init__`). `PretrainObjectivesConfig.unmaskable_key_ids: tuple[int, ...] = ()`. Pretrain reads `model.long_term_injection` from YAML; when `"cls"`, sets `unmaskable_key_ids = (key id of the "user_id" token,)` — obtain it with `vocabularies.encode_key(...)` on the `user_id` `KeyValueToken` exactly as `semantic_tokens` emits it (`source="tabular"`).
- `forward` with `"cls"`: `token_features[:, 0, :] = token_features[:, 0, :] + self.user_adapter(batch.user_ids)` before the Transformer; `FoundationOutput.long_term` stays `self.user_adapter(batch.user_ids)`.

- [ ] **Step 1: Write the failing tests:**
  - `test_cls_injection_gives_user_adapter_gradient`: tiny model (`d_model=16, num_heads=2, num_layers=1`, the existing test batch builder in `tests/models/conftest.py`), `long_term_injection="cls"`; `loss = output.hidden_states[batch.attention_mask].sum()`; `loss.backward()`; assert `model.user_adapter` weight grad is not None and `.abs().sum() > 0`.
  - `test_no_injection_matches_previous_forward`: same seed, `"none"` → `hidden_states` and `short_term` equal (`torch.equal`) to a model built without passing the field.
  - `test_unknown_injection_mode_rejected`: `FoundationModelConfig(long_term_injection="sum")` raises `ValueError`.
  - `test_unmaskable_keys_are_never_masked` (objectives): a batch whose every valid token has the user_id key id except one; `mask_prob=1.0`; with `unmaskable_key_ids=(that id,)`, assert no token with that key id is masked, over 20 generator seeds.
  - `test_old_checkpoint_without_injection_key_loads` (checkpoint): save a checkpoint, delete `"long_term_injection"` from `payload["model_config"]`, `torch.save` it back, `load_checkpoint` → `model_config.long_term_injection == "none"` and the encoded output equals the original model's.
- [ ] **Step 2: Run** them — Expected: FAIL.
- [ ] **Step 3: Implement** the config field, the forward change, `apply_mask` excluding tokens whose `key_ids` are in `unmaskable_key_ids` from `candidate_mask`, and the pretrain wiring (`long_term_injection=str(model_cfg_dict.get("long_term_injection", "none"))`).
- [ ] **Step 4: Run** `tests/models tests/training -q` and ruff — Expected: PASS.
- [ ] **Step 5: Commit** `feat(model): optional CLS injection trains the long-term user embedding`.

### Task 3: Shared sim event converter (`src/downstream/sim_events.py`)

**Files:**
- Create: `src/downstream/sim_events.py`, `tests/downstream/test_sim_events.py`

**Interfaces:**
- Consumes: `sim_tiny_packages` (Task 1), `tokenizer_config_for_manifest` (Task 1), `EventRecord` / `CanonicalSample` (`src/foundation_input/types.py`), pretrain loader (`iter_normalized_samples`, or the parquet event rows through `src.foundation_input.loader._event_from_row`).
- Produces:
  - `@dataclass(frozen=True) class SimEvent: event_id: str; user_id: str; event_time: datetime` (timezone-aware) `; service: str; event_type: str; entity_id: str | None; attributes: dict`
  - `food_history_events(record: dict) -> list[SimEvent]` — from a Food `samples.jsonl` row's `input.history` (`event_time` ISO `+07:00`).
  - `ads_window_events(rows: Sequence[dict]) -> list[SimEvent]` — from `ads/events.parquet` rows (`timestamp` epoch seconds → aware UTC; `attributes_json` parsed).
  - `sim_event_records(events: Sequence[SimEvent]) -> tuple[EventRecord, ...]` — `scenario` by the service map; `context` = attributes minus the numeric action fields; `action` = the numeric action fields present plus `type` (Ads map / `event_type`); `event_time` and `eligibility_time` = naive UTC; `time_precision="second"`, `eligibility_inclusive=True`; groups: a new group starts at each session-start event type, leading events before the first start form one group; `source_record_id` = event_id of the group's first event, `source_record_index` = group index, `action_index` = index in group, `source_action_count` = group size; `user_id` = the event's user id (full `sim:user:N`).
  - `sim_canonical_sample(*, sample_id: str, dataset_name: str, split: str, user_id: str, cutoff_time: datetime, profile: dict, events: Sequence[SimEvent], history_truncated: bool, history_group_truncated: bool) -> CanonicalSample` — `cutoff_inclusive=False`, `history_order_known=True`, `cutoff_time` naive UTC, `profile` copied with `as_of_time` converted to naive-UTC ISO like the pretrain package.

- [ ] **Step 1: Write the failing tests:**
  - `test_scenario_context_action_split`: a `book_ride` SimEvent with attributes `{pickup_geohash, dropoff_geohash, pickup_zone_type, dropoff_zone_type, vehicle_type, driver_id, distance_km, fare, wait_minutes}` → scenario `"Ride"`, action keys `{type, distance_km, fare, wait_minutes}` with `type == "book_ride"`, context = the other six keys; an `ad_click` → scenario `"Advertisement"`, action `{"type": "click"}`.
  - `test_groups_start_at_session_start_types`: event types `[view_merchant, order, search, view_merchant, order, ad_impression, ad_click, search_destination]` → group sizes `[2, 3, 2, 1]`, `source_record_index` `[0,0,1,1,1,2,2,3]`.
  - `test_empty_history_encodes`: `sim_canonical_sample(..., events=[])` passes `validate_sample` and `encode_sample` with a vocab fitted on the tiny pretrain package (records contain CLS / tabular tokens only).
  - `test_food_history_matches_pretrain_tokens`: for every train-split Food sample of `sim_tiny_packages.food` (all cutoffs are before day 43, so their history events exist in `pretrain/events`): build the sample from `food_history_events`; build the reference from the pretrain events with the same `event_id`s (same order); wrap both with the same `user_id`, `profile={}`; `encode_sample` with one vocab fitted on the pretrain package with `tokenizer_config_for_manifest`; assert the tuples `(value_id, key_id, modality_id, source_id, position_id, time_gap_id, numeric_value, numeric_missing)` of all records are equal. At least 50 samples compared.
  - `test_ads_window_starting_mid_session_matches_pretrain`: same parity over train-split Ads requests (`events.parquet` rows `[history_start, history_stop)` of the request's user, `ads_window_events`), restricted to windows whose events all precede day 43; assert at least one compared window starts with a non-session-start event.
- [ ] **Step 2: Run** `.venv/Scripts/python -m pytest tests/downstream/test_sim_events.py -q` — Expected: FAIL (module missing).
- [ ] **Step 3: Implement** `src/downstream/sim_events.py`.
- [ ] **Step 4: Run** the tests and ruff — Expected: PASS. If parity fails, the converter is wrong; never loosen the comparison.
- [ ] **Step 5: Commit** `feat(downstream): sim event converter with token parity to pretrain`.

### Task 4: Food reads the sim package; random-init control arm

**Files:**
- Create: `src/downstream/foundation_init.py`, `configs/food_recommendation_sim.yaml`, `tests/downstream/test_food_sim.py`
- Modify: `src/downstream/food_samples.py`, `src/downstream/food_inputs.py` (`load_food_foundation`, `resolve_food_checkpoint_metadata`, `food_baseline_dataset`, `load_food_inputs`), `src/downstream/food_recommendation.py` (config validation)

**Interfaces:**
- Consumes: Task 3 (`food_history_events`, `sim_canonical_sample`), Task 1 metadata (`preprocess_metadata["dataset_provenance"]`).
- Produces:
  - `reinitialize_foundation(model: FoundationModel, seed: int) -> FoundationModel` (`src/downstream/foundation_init.py`) — new `FoundationModel(model.config)` built under `torch.manual_seed(seed)`, same device, `eval()` mode; used by Food and Ads when config `foundation.init == "random"` (default `"pretrained"`; the run records the arm in its metrics/artifacts metadata).
  - Food config `data.format: "meituan" | "sim_v1"` (default `"meituan"`, existing keys unchanged). `sim_v1` keys: `data.root` (the `food/` directory), `data.expected_version` (= manifest `dataset_version`); `catalog` and `candidate_policy` are omitted (taken from `food/manifest.json` and `food/merchants.parquet`); `checkpoint.handoff` absent.
  - `load_merchant_catalog(path: Path, expected_sha256: str) -> dict[str, dict]` keyed by `merchant_id`.
  - `sim_to_canonical_sample(record: dict, dataset_name: str) -> CanonicalSample` — via Task 3, reading `history_truncated` / `history_group_truncated` from the record, full ids.
  - `sim_candidate_rows(record: dict, catalog: dict) -> list[dict]` — same row shape as `candidate_rows`; unknown `item_id` → the same error type `candidate_rows` uses, naming the request and item.
  - Checkpoint binding for `sim_v1`: `vocabulary.dataset_name == "sim_v1"` and `preprocess_metadata["dataset_provenance"]["sim_config_sha256"] == food manifest "sim_config_sha256"`; otherwise the existing checkpoint-error path (same exit code as other checkpoint mismatches).

- [ ] **Step 1: Write the failing tests** (use `sim_tiny_packages`; build a tiny checkpoint with `pretrain_preprocess_metadata` and `save_checkpoint` from a `FoundationModel` sized by a vocab fitted on the tiny pretrain package):
  - `test_sim_food_inputs_build`: `load_food_inputs(config, foundation)` on the tiny package returns inputs for all three splits with request counts equal to `food/manifest.json` `request_counts`, and every candidate id comes from `merchants.parquet`.
  - `test_sim_user_ids_are_not_stripped`: the encoded `user_id` token of a sample is not UNK (the full `sim:user:N` is in the vocab).
  - `test_sim_checkpoint_provenance_mismatch_is_rejected`: checkpoint metadata with a different `sim_config_sha256` → the checkpoint error before any batch is built.
  - `test_sim_candidate_not_in_merchants_is_rejected`: a record whose candidate `item_id` is `"sim:merchant:999999"` → error naming that id.
  - `test_random_init_changes_weights_not_config`: `reinitialize_foundation(model, seed=7)` → same `config`, different `state_dict` values, identical across two calls with seed 7.
  - `test_meituan_path_unchanged`: the existing Meituan tests still pass (no new test body; run `tests/downstream -q`).
- [ ] **Step 2: Run** `tests/downstream/test_food_sim.py -q` — Expected: FAIL.
- [ ] **Step 3: Implement**; `configs/food_recommendation_sim.yaml` = `configs/food_recommendation.yaml` with the `sim_v1` data block (`root: experiments/sim/sim-v1/food`, `expected_version: sim-v1`), `checkpoint.path: experiments/pretrain_sim/checkpoints/best_checkpoint.pt` (no handoff), `foundation: {init: pretrained}`, `sequence: {max_tokens: 512, max_event_count: 30}`. `food_baseline_dataset` must also work for `sim_v1` (evaluation stays on).
- [ ] **Step 4: Run** `tests/downstream -q` and ruff — Expected: PASS.
- [ ] **Step 5: Commit** `feat(food): sim_v1 data format and random-init foundation arm`.

### Task 5: Ads downstream merged and generalised to sim

**Files:**
- Merge: `origin/feat/ad-optimization` (adds `src/downstream/ad_inputs.py`, `src/downstream/ad_optimization.py`, `configs/ad_optimization.yaml`, `docs/ad_optimization_handoff.md`, `tests/downstream/test_ad_inputs.py`, `tests/downstream/test_ad_optimization.py`)
- Create: `configs/ad_optimization_sim.yaml`, `tests/downstream/test_ad_sim.py`
- Modify: `src/downstream/ad_inputs.py` (constants L28-29, `_check_checkpoint` L57, `_index_events` L76, `read_ad_handoff` L96-200, `_canonical` L203, `load_ad_inputs` L224-276, `ad_baseline_dataset` L303), `src/downstream/ad_optimization.py` (`validate_opportunities` L223, `_validate_config` L417)

**Interfaces:**
- Consumes: Task 3 (`ads_window_events`, `sim_canonical_sample`), Task 4 (`reinitialize_foundation`), Task 1 metadata.
- Produces: Ads config `data.format: "taobao" | "sim_v1"` (default `"taobao"`; every Taobao check unchanged). For `sim_v1`: namespace = manifest `source_dataset` (`"sim_v1"`), exposure policy = manifest `candidate_policy`, id prefixes `sim:user:` / `sim:ad:`, split bounds exclusive, events of every type indexed by `source_row_index` (the opportunity's impression = `events[source_row_index]`, which must be the user's `ad_impression` of `candidate_id` at `exposure_timestamp`), profile columns beyond `{user_id, as_of_timestamp}` allowed and passed into the sample profile, history via Task 3, label availability = click time (the user's next event, `ad_click` of the same ad) or exposure + 27 s, checkpoint binding by `dataset_name == "sim_v1"` and `dataset_provenance.sim_config_sha256` (not the Ads manifest sha). `foundation.init` as in Task 4.

- [ ] **Step 1: Merge** `git merge --no-ff origin/feat/ad-optimization -m "merge: Ads downstream (feat/ad-optimization)"`; run `tests/downstream -q` — Expected: PASS (merge-tree was clean).
- [ ] **Step 2: Write the failing tests** in `tests/downstream/test_ad_sim.py` (tiny package and tiny checkpoint as in Task 4):
  - `test_sim_ads_inputs_build`: inputs for all splits, counts equal manifest `request_counts`, labels equal `ad_opportunities.clicked`.
  - `test_sim_label_availability`: for a clicked request, availability = its click event timestamp; for a non-clicked one, `exposure_timestamp + 27`.
  - `test_sim_split_bounds_are_exclusive`: an opportunity with `cutoff_timestamp == manifest split.train_end` must have split `validation` and validates.
  - `test_ads_sim_checkpoint_provenance_mismatch_is_rejected`.
  - `test_taobao_defaults_unchanged`: config without `data.format` behaves as before (existing Taobao tests pass).
- [ ] **Step 3: Run** — Expected: FAIL. **Step 4: Implement**; `configs/ad_optimization_sim.yaml` = Ads config with `data: {format: sim_v1, root: experiments/sim/sim-v1/ads, expected_version: sim-v1}`, `checkpoint.path: experiments/pretrain_sim/checkpoints/best_checkpoint.pt`, `exposure_policy: sim-served-impression-single-candidate-v1`, `evaluate: true`, `evaluation.label_availability_policy: handoff_timestamp`, `foundation: {init: pretrained}`.
- [ ] **Step 5: Run** `tests/downstream -q` and ruff — Expected: PASS.
- [ ] **Step 6: Commit** `feat(ads): sim_v1 data format, generic events and provenance binding`.

### Task 6: End-to-end smoke, sequence-length check, team notes

**Files:**
- Create: `docs/sim_foundation_team_notes.md`
- Modify: `docs/simulation_data_contract.md` (hand-off section: the gaps are now closed, pointers to the new configs)

- [ ] **Step 1: Generate smoke data:** `.venv/Scripts/python -m src.simulation --preset smoke --out experiments/sim` (skip if `experiments/sim/sim-smoke-v1/data_report.json` exists with `tree_sha256` `4ed1499e6a36e6f2f048b3afe79a7d4b1f8e5370441776a03c891c25275afb04`; otherwise move the old directory aside first). Expected: exit 0.
- [ ] **Step 2: Measure sequence length** on 2,000 train samples of `sim-smoke-v1/pretrain` with the package tokenizer config: distribution of token counts before truncation (p50/p90/p99/max) and the share over 512. Expected: share over 512 ≤ 1% (if higher, record it and raise `max_length` in both sim pretrain configs to the p99 rounded up to a multiple of 64).
- [ ] **Step 3: Pretrain smoke:** `.venv/Scripts/python -m src.training.pretrain --config configs/pretrain_sim.yaml` (1 epoch, CPU). Expected: `experiments/pretrain_sim/checkpoints/best_checkpoint.pt`; a log line or check showing the `user_adapter` weights changed from their initial values (compare `long_term` of two users before/after, or the checkpoint's `user_adapter` weights vs a fresh init with the same seed).
- [ ] **Step 4: Downstream smoke:** copies of `configs/food_recommendation_sim.yaml` and `configs/ad_optimization_sim.yaml` in the session scratchpad pointing at `experiments/sim/sim-smoke-v1/{food,ads}` and `expected_version: sim-smoke-v1`, `training.epochs: 1`; run each with `foundation.init: pretrained` and `random`. Expected: exit 0 and metric files for all four runs (values not judged — 1 epoch on smoke).
- [ ] **Step 5: Write `docs/sim_foundation_team_notes.md`** (Vietnamese, for the team): per owner — tokenizer/pretrain (Người 1/3): package allowlist, `check_vocab_allowlist`, new `preprocess_metadata` keys, sim configs, sequence-length numbers from Step 2, and the three issues found on `origin/feat/pretrain-food-recommendation` (cosine horizon hard-coded to 100k steps, gradient-accumulation tail never runs, automatic `vocab.json` reuse conflicting with the allowlist check); encoder/objectives (Người 2): `long_term_injection`, `unmaskable_key_ids`, defaults, `short_term` now also sees user information through attention; Ads (Người 4): `data.format`, what moved from constants to config; Food: `sim_v1`; known limits (Ads profile is static only; Food/Ads candidates use ids only). Include the exact commands of Steps 3–4.
- [ ] **Step 6: Run** the full suite and ruff — Expected: all pass. **Commit** `docs: sim foundation team notes and e2e smoke results`.
