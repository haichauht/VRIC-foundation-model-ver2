# Implementation spec — Rail-hailing User Foundation Model

> Đọc toàn bộ file này trước khi code. Đây là hợp đồng kỹ thuật của team
> foundation. Nếu một yêu cầu trong task/PR mâu thuẫn với file này, dừng ở
> contract tương ứng và mở issue; không tự thay đổi interface của team khác.

## 1. Mục tiêu, phạm vi và ranh giới team

Ta xây dựng một MVP dựa trên ý tưởng trong *User foundation models for Grab*:

1. joint pretraining trên profile dạng bảng và chuỗi hành vi;
2. adapters theo modality đưa categorical/numeric/text/geo về cùng `d_model`;
3. hai user embedding:
   - `long_term`: stable embedding từ User-ID adapter;
   - `short_term`: recent-intent embedding từ chuỗi event qua Transformer;
4. Food Recommendation và Ad Optimization dùng các embedding này.

MVP phải chạy CPU trên smoke dataset simulated. Không cần Ray, distributed
training, pretrained LLM, image/audio adapter, hay scale vocabulary hàng trăm
triệu ID. Thiết kế không được làm các mở rộng đó bất khả thi sau này.

### Ownership giữa ba team

| Team | Được làm | Không được làm |
|---|---|---|
| Data | simulate/ETL/version/split/candidates/exposures | Foundation, downstream, baseline/evaluation |
| Baseline & evaluation | baselines, metrics, evaluator, experiment comparison | Data simulator, foundation, downstream model |
| Team foundation (chúng ta) | input adapter, foundation model, pretraining, Food/Ads downstream | Tự sinh/sửa data, baseline, metric/evaluator |

Quy tắc cứng:

- Không đổi `src/data/**`, `data/**`, `src/baselines/**`, `src/evaluation/**`.
- Không copy metric/baseline vào downstream module.
- Không commit data thật, dataset generated lớn, checkpoint, output experiment,
  secret. Các artifact nằm ở path gitignored do config chỉ định.
- Không dùng event/label/candidate/exposure sau prediction cutoff.

## 2. Trạng thái repository và kiến trúc đích

Repository hiện mới là skeleton. Các phần do team foundation thêm vào:

```text
src/
  foundation_input/             # Người 1 — đọc handoff, tokenize, batch
  models/
    adapters.py                 # Người 2
    encoder.py                  # Người 2
    pooling.py                  # Người 2
    objectives.py               # Người 3
  contracts/
    output.py                   # Người 3 — Output Contract v1 validator/writer
  training/
    pretrain.py                 # Người 3
    checkpoint.py               # Người 3
  downstream/
    food_recommendation.py      # Người 3
    ad_optimization.py          # Người 4
configs/
  pretrain.yaml                 # Người 3
  food_recommendation.yaml      # Người 3
  ad_optimization.yaml          # Người 4
tests/
  foundation_input/             # Người 1
  models/                       # Người 2
  training/                     # Người 3
  downstream/                   # Người 3 và 4, file khác nhau
```

`src/data/**` là namespace của Data team; team foundation tạo
`src/foundation_input/**` thay vì sửa preprocessor/dataset của họ.

## 3. Handoff contract — phải có trước khi train thật

### 3.1 Data team bàn giao

Data version đã freeze nằm ngoài Git, ví dụ `data/versions/sim-v1/`, có cấu
trúc logic sau (format Parquet/CSV/JSONL do Data team chọn nhưng phải ghi trong
manifest):

```text
sim-v1/
  manifest.json
  profiles.{parquet|csv}
  events.{parquet|csv}
  splits/{train,validation,test}_users_or_requests.*
  food_candidates.*
  ad_opportunities.*
```

`manifest.json` bắt buộc có:

```json
{
  "dataset_version": "sim-v1",
  "schema_version": 1,
  "generator_version": "...",
  "seed": 42,
  "timezone": "Asia/Ho_Chi_Minh",
  "timestamp_unit": "seconds",
  "files": {"events": {"path": "...", "sha256": "..."}},
  "split": {"kind": "temporal", "train_end": 0, "valid_end": 0},
  "profile_as_of_rule": "...",
  "candidate_policy": "...",
  "missing_value_policy": "..."
}
```

Schema tối thiểu dưới đây là contract generic cho data/downstream simulation, không phải schema trực tiếp của pretrain v2 hiện tại. Pretrain v2 dùng `snapshots/*.parquet` và `events/*.parquet` với `context_json`/`action_json`; input adapter phải theo `docs/foundation_input_contract.md` và `data/pretrain_data/*/preprocessing_config.json`.

| Data | Cột bắt buộc | Cột/ý nghĩa cần thêm cho task |
|---|---|---|
| Event | `user_id`, `timestamp`, `event_type`, `service`, `entity_id`, `value` | `merchant_id`, `food_item_id`, `query_text`, `amount`, `lat`, `lon`, `ad_id`, `campaign_id` khi áp dụng |
| Profile | `user_id`, `as_of_timestamp` | thuộc tính categorical/numeric dài hạn |
| Food request | `request_id`, `user_id`, `cutoff_timestamp`, `candidate_id`, `is_positive`, `split` | candidate là food item hoặc merchant, mọi candidate phải tồn tại trước cutoff |
| Ad opportunity | `request_id`, `user_id`, `cutoff_timestamp`, `candidate_id`, `clicked`, `split` | candidate là ad/campaign; `clicked` 0/1; request có thể không có click |

Data team chịu trách nhiệm chứng minh temporal split đúng, candidate/exposure
không đến từ tương lai và profile mang đúng trạng thái `as_of`. Team foundation
chỉ validate, không âm thầm “sửa” input.

### 3.2 Baseline & evaluation team bàn giao

Họ sở hữu `src/baselines/**` và `src/evaluation/**`. Team foundation bàn giao
output theo **Output Contract v1** ở mục 4.6; Baseline team cung cấp API:

```python
evaluate_predictions(
    predictions_path: str,
    task: Literal["food_recommendation", "ad_optimization"],
    data_version: str,
    split: Literal["validation", "test"],
) -> dict[str, float]
```

Metrics do họ triển khai:

- Food: Recall@10, NDCG@10, MRR.
- Ads: ROC-AUC, PR-AUC, log loss, Brier score; NDCG@K/Recall@K chỉ trên
  request có ít nhất một click.

Nếu prediction-table contract không đáp ứng, mở issue có cả Data và Baseline
team. Không thêm một evaluator riêng trong `src/downstream`.

## 4. Contract chung trong team foundation

### 4.1 Token và batch

```python
# src/foundation_input/types.py
@dataclass(frozen=True)
class KeyValueToken:
    key: str                 # column name hoặc event_type
    value: Any               # raw attribute/entity/text/numeric/geo
    modality: str            # categorical | numeric | text | missing | geo (geo optional)
    source: str              # tabular | sequence
    timestamp: datetime | None  # None cho tabular; giữ date-only/uncertain precision của source

@dataclass
class FoundationBatch:
    input_ids: torch.LongTensor       # [B, L]
    key_ids: torch.LongTensor         # [B, L]
    modality_ids: torch.LongTensor    # [B, L]
    source_ids: torch.LongTensor      # [B, L]
    position_ids: torch.LongTensor    # [B, L]
    time_gap_ids: torch.LongTensor    # [B, L]
    attention_mask: torch.BoolTensor  # [B, L]; True=real token, False=pad
    numeric_values: torch.FloatTensor  # [B, L]; train-normalized continuous values
    numeric_missing_mask: torch.BoolTensor  # [B, L]; True only for missing numeric fields
    user_ids: torch.LongTensor        # [B]
    targets: dict[str, torch.Tensor]  # labels/mask metadata; includes validity masks
```

Encoding rules:

- Tabular: `key=<column>`, `value=<profile value>`, `source="tabular"`.
  Đây là unordered set: không gán position theo thứ tự cột.
- Sequence: `key=<event_type>`, `value=<event entity/value>`,
  `source="sequence"`; gán ordered position và time-gap bucket.
- Context/action JSON phải đi qua allowlist theo scenario; list payload như
  OmniBehavior `action.content` (role/content messages) mặc định bị loại khỏi
  MVP bằng `TokenizerConfig.include_action_content=False`. Nếu bật cho ablation,
  phải flatten deterministic theo role và thứ tự message, sau đó giới hạn 256 ký
  tự bằng head+tail truncation; không được implicit `str(list)`.
- Layout luôn là `[CLS] [tabular...] [SEP] [event sequence...] [PAD...]`.
- Special IDs phải cố định: `PAD=0`, `CLS=1`, `SEP=2`, `MASK=3`, `UNK=4`.
- Vocab, scaler, bucket boundary và taxonomy chỉ fit trên train. Val/test chỉ
  transform, unknown -> `UNK`; tuyệt đối không `fit` lại.

### 4.2 Model interface

```python
# src/models/encoder.py
@dataclass
class FoundationOutput:
    hidden_states: torch.Tensor       # [B, L, D]
    long_term: torch.Tensor           # [B, D]
    short_term: torch.Tensor          # [B, D]
    padding_mask: torch.BoolTensor    # [B, L], True = valid token

class FoundationModel(nn.Module):
    def forward(
        self, batch: FoundationBatch, *, attention_mode: str
    ) -> FoundationOutput: ...

    def encode(self, batch: FoundationBatch) -> FoundationOutput: ...

    def embed_users(self, batch: FoundationBatch) -> dict[str, torch.Tensor]:
        # Returns {"long_term": [B,D], "short_term": [B,D]}
        ...
```

Downstream chỉ gọi `embed_users()`. Không dùng `hidden_states`, private adapter
weights, token positions hoặc forward hooks từ downstream code.

### 4.3 Architecture quyết định trước

1. Adapter categorical: embedding lookup.
2. Adapter numeric: normalized float + missing flag qua MLP/projection; bucket ID vẫn giữ cho masked-value objective.
3. Adapter text: local deterministic tokenizer/hash + embedding; không tải
   model từ Internet.
4. Adapter geo: chỉ bật khi selected pretrain manifest thực sự có latitude/longitude; current OmniBehavior/Meituan v2 không có geo business fields được allowlist.
5. Mọi adapter qua alignment projection về `d_model`.
6. Transformer có hai mode:
   - `bidirectional`: masked-value reconstruction;
   - `next_action`: event chỉ attend tabular và các event quá khứ; tabular
     không mang ý nghĩa thứ tự.
7. `long_term` = row User-ID adapter. `short_term` = pool token sequence hợp
   lệ; bỏ tabular/PAD, fallback CLS nếu không có event.

### 4.4 Pretraining objective

Tổng loss:

```text
L = w_mask * L_masked_value + w_key * L_next_key + w_value * L_next_value
```

- Chỉ mask token hợp lệ; không mask `PAD`, `CLS`, `SEP`.
- `L_masked_value`: reconstruct value đúng modality.
- `L_next_key`: CE dự đoán `event.event_type` tiếp theo; không dùng `action.type` như success label.
- `L_next_value`: head đúng modality của event tiếp theo.
- Categorical/ID: CE. MVP ID scalable qua hierarchical head `group -> local`.
- Numeric bucket IDs vẫn dùng CE như categorical cho masked-value objective; full numeric dùng MSE trên `numeric_values` với `numeric_missing_mask`; geo chỉ có loss khi geo contract được bật. Text hash/token IDs dùng CE.
- Loss component nào không có target trong batch phải log rõ `count=0`, không
  tạo NaN hoặc silently bỏ qua toàn batch.
- Next-action loss phải dùng `targets["next_action_valid_mask"]`; giá trị target `0`
  ở sample không đủ điều kiện là sentinel, không được coi là class thật. Sample mask false
  vẫn có thể đóng góp cho các objective khác.

### 4.5 Checkpoint contract

```python
{
  "format_version": 1,
  "model_config": {...},
  "vocabularies": {...},
  "preprocess_metadata": {...},
  "model_state_dict": {...},
  "training_metadata": {
    "dataset_version": "...", "seed": 42, "epoch": 0,
    "validation_loss": 0.0
  }
}
```

Không lưu `Dataset` object, absolute path, secret, hoặc artifact lớn trong Git.
Load checkpoint phải fail rõ nếu `format_version`, vocab hoặc preprocessing
metadata không tương thích với data version được yêu cầu.

### 4.6 Output Contract v1 — bắt buộc cho mọi run

Contract này áp dụng cho checkpoint, embedding API và prediction files. Mọi
command train/inference phải validate output **trước** khi báo thành công hoặc
gọi evaluator. Không được tự chữa output sau validation fail.

#### A. Foundation model output trong Python

`FoundationOutput` và giá trị từ `embed_users()` phải thỏa tất cả:

| Field | Dtype/shape | Quy tắc |
|---|---|---|
| `hidden_states` | `torch.float32`, `[B, L, D]` | finite tại tất cả token valid |
| `long_term` | `torch.float32`, `[B, D]` | finite; row `i` khớp `user_ids[i]` |
| `short_term` | `torch.float32`, `[B, D]` | finite; không aggregate tabular/PAD |
| `padding_mask` | `torch.bool`, `[B, L]` | `True` nghĩa token valid, đúng với `FoundationBatch.attention_mask` |

- Bốn tensor đều cùng batch order; các float tensor cùng device với model.
- `embed_users()` trả đúng hai key: `long_term`, `short_term`, không thêm
  hidden state. Trong `model.eval()` và input như nhau, output phải
  deterministic trong tolerance `1e-6` trên cùng device.
- Nếu shape, dtype, finite check hoặc user alignment sai: raise `ValueError`
  chứa batch size, field và expected/actual; không trả embedding một phần.

#### B. Artifact layout

Mọi run ghi vào thư mục độc lập, không ghi đè run cũ:

```text
<artifact_dir>/<task>/<run_id>/
  run_manifest.json
  validation_report.json
  predictions.<split>.parquet
  train_metrics.jsonl                 # chỉ train runs
  checkpoint_ref.json                 # Food/Ads
  foundation_checkpoint-v1.pt         # chỉ pretraining
```

- `<task>` chính xác là `foundation_pretrain`, `food_recommendation` hoặc
  `ad_optimization`.
- `<run_id>` là UUID v4 hoặc định danh unique có `seed` + UTC timestamp;
  không dùng `latest` làm output path.
- `artifact_dir` bắt buộc lấy từ YAML config. File `.parquet` là format chuẩn;
  JSONL/CSV chỉ cho debug và không được gửi evaluator.
- Ghi file theo atomic write: temporary file cùng volume -> validate -> rename.
  Run fail không được để lại file prediction mang tên chính thức.

#### C. Prediction table schema

Mỗi dòng biểu diễn **một candidate của một request**. Schema Parquet bắt buộc:

| Cột | Type | Ràng buộc |
|---|---|---|
| `schema_version` | `int32` | luôn `1` |
| `task` | UTF-8 string | đúng `food_recommendation` hoặc `ad_optimization` |
| `split` | UTF-8 string | đúng `validation` hoặc `test` |
| `data_version` | UTF-8 string | bằng `manifest.dataset_version` |
| `run_id` | UTF-8 string | bằng tên artifact directory |
| `request_id` | UTF-8 string | non-empty, canonical từ Data handoff |
| `user_id` | UTF-8 string | non-empty, canonical từ Data handoff |
| `cutoff_timestamp` | `int64` | UTC epoch seconds, bằng request input |
| `candidate_id` | UTF-8 string | non-empty, thuộc candidate/exposure set của request |
| `score` | `float64` | finite; semantics theo task ở dưới |

Có thể thêm cột optional `raw_logit: float64`, nhưng evaluator chỉ dùng `score`.
Không được đưa `label`, `clicked`, `is_positive`, feature, embedding hoặc cột
không có trong prediction table vào file gửi evaluator.

**Semantics của `score`:**

- Food: real-valued ranking score; score lớn hơn nghĩa là item được ưu tiên
  hơn. Không yêu cầu là probability.
- Ads: calibrated click probability, luôn trong đoạn đóng `[0.0, 1.0]`; score
  lớn hơn nghĩa là click likelihood cao hơn. Raw logit, nếu cần debug, nằm ở
  `raw_logit`, không thay thế `score`.

**Uniqueness và ordering:**

- Primary key `(task, split, run_id, request_id, candidate_id)` phải unique.
- Table phải chứa **đúng** candidate set Data team bàn giao cho mỗi request:
  không thiếu, không thừa, không duplicate.
- Write order deterministic: `request_id` ascending, sau đó `score`
  descending, cuối cùng `candidate_id` ascending để break tie. Evaluator vẫn
  phải rank theo `score`, không dựa vào vị trí file.
- Không có null ở bất kỳ cột bắt buộc nào; mọi score phải pass `isfinite`.

#### D. Run manifest và validation report

`run_manifest.json` bắt buộc có các key sau:

```json
{
  "contract_version": 1,
  "run_id": "...",
  "task": "food_recommendation",
  "split": "validation",
  "data_version": "sim-v1",
  "data_manifest_sha256": "...",
  "preprocess_fingerprint": "...",
  "config_sha256": "...",
  "code_revision": "git SHA or unknown",
  "seed": 42,
  "created_at_utc": "RFC-3339 timestamp",
  "foundation": {
    "checkpoint_path": "...",
    "checkpoint_sha256": "...",
    "format_version": 1,
    "freeze_foundation": true
  },
  "candidate_or_exposure_policy": "...",
  "row_count": 0,
  "request_count": 0,
  "prediction_sha256": "..."
}
```

- Pretraining run thay `foundation` bằng `model_config`, `vocabulary_sha256`,
  `best_validation_loss`, `checkpoint_sha256`.
- `validation_report.json` ghi `status: "passed"`, row/request count,
  duplicate/missing/extra candidate count, NaN/Inf count, timestamp mismatch
  count, score range và validation timestamp. Không ghi raw user data.
- `train_metrics.jsonl` gồm một JSON object/line với tối thiểu `epoch`,
  `step`, `split`, `total_loss`, từng loss component, target counts, `seed` và
  UTC timestamp. Không được ghi `NaN`/`Infinity` JSON values.

#### E. Failure protocol và exit codes

| Exit code | Lý do | Hành động |
|---:|---|---|
| 0 | output pass toàn bộ validation | cho phép gọi evaluator |
| 2 | config/contract version sai | dừng trước train/inference |
| 3 | data manifest/schema/checksum/split sai | trả về Data team |
| 4 | checkpoint/preprocess incompatibility | sửa đúng checkpoint/config |
| 5 | prediction schema/duplicate/missing/extra/NaN/score-range sai | sửa downstream exporter |
| 6 | evaluator API/runtime fail sau khi output đã valid | trả về Baseline team kèm artifact manifest |

Lỗi phải nêu `run_id`, task, split, field/rule vi phạm và tối đa 20 examples
an toàn (request IDs, không log raw profile/event). Không được fallback sang
score 0, drop row lỗi, tự clip Ads score ngoài `[0,1]`, hoặc evaluate output
partial mà không có phê duyệt bằng issue.

## 5. Phân công chi tiết cho bốn người

### Người 1 — Foundation input adapter và tokenization

**Branch:** `feat/foundation-input-adapter`  
**Sở hữu file:** `src/foundation_input/**`, `tests/foundation_input/**`,
`docs/foundation_input_contract.md`.  
**Không chạm:** `src/data/**`, `data/**`, `src/models/**`, `src/training/**`,
`src/downstream/**`, `src/baselines/**`, `src/evaluation/**`, configs.

### Deliverables

1. `manifest.py`: load + validate required fields/checksum/schema version.
2. `tokenizer.py`: profile/event -> `KeyValueToken`, special token layout,
   source/modality/key IDs, time-gap bucket.
3. `vocab.py`: fit/load/transform train-only vocabulary, normalizer, taxonomy,
   serialize metadata.
4. `dataset.py`: `FoundationPretrainDataset` and collator output đúng
   `FoundationBatch` contract.
5. `validation.py`: temporal/as-of/candidate safety validation; message lỗi
   chỉ rõ file, field, row/request và expected vs actual.

### Tests bắt buộc

- manifest thiếu field hoặc checksum sai -> error có ngữ cảnh;
- event out-of-order -> reject/sort theo contract đã chốt;
- profile sau cutoff -> reject;
- OOV val/test -> `UNK`, không tăng vocab;
- padding không đổi token thật;
- đổi thứ tự cột tabular không đổi set token;
- cross-service sample vẫn nằm đúng thứ tự temporal.

### Người 2 — Modality adapters, Transformer encoder, dual embeddings

**Branch:** `feat/foundation-encoder`  
**Sở hữu file:** `src/models/adapters.py`, `src/models/encoder.py`,
`src/models/pooling.py`, `tests/models/**`.  
**Không chạm:** input adapter, `objectives.py`, training, downstream, data,
baseline/evaluation, config.

### Deliverables

1. `FoundationModelConfig` dataclass: `d_model`, heads, layers, dropout,
   vocabulary sizes, max length, dimensions adapter.
2. Adapter registry cho categorical/numeric/text/missing và alignment projection; geo là optional extension gated by data.
3. User-ID/key/source embeddings; position/time-gap chỉ áp dụng sequence.
4. Attention-mask builder cho `bidirectional` và `next_action`; document rõ
   quy ước mask của PyTorch để tránh đảo `True/False`.
5. `FoundationModel`, `FoundationOutput`, `encode`, `embed_users`, pooling.

### Tests bắt buộc

- CPU forward shape đúng trên fixture FoundationBatch;
- mọi output finite; pad value không ảnh hưởng token valid;
- routing từng modality đúng adapter;
- tabular column permutation invariant trong `eval()`;
- event tương lai bị block ở `next_action`;
- short-term không pool tabular/PAD; zero-event user fallback hợp lệ.

### Người 3 — Pretraining, checkpoint và Food Recommendation

**Branch:** `feat/pretrain-food-recommendation`  
**Sở hữu file:** `src/models/objectives.py`, `src/training/**`,
`src/contracts/output.py`,
`configs/pretrain.yaml`, `src/downstream/food_recommendation.py`,
`configs/food_recommendation.yaml`, `tests/training/**`,
`tests/downstream/test_food_recommendation.py`, runtime ML dependency trong
`pyproject.toml`.  
**Không chạm:** input adapter/encoder internals, Data/Baseline team files,
Ad task config/module/tests.

### Foundation v2 startup gate (required before pretraining)

Member 3 must not start production pretraining from default IDs or a recount of
the vocabulary. Every run must:

1. Record `vocabulary_artifact_path`, SHA-256 checksum, `dataset_name`,
   `fit_split`, schema version, and source fingerprint in run-manifest and
   checkpoint metadata.
2. Construct `FoundationModelConfig` from `artifacts.model_cardinalities()` and
   pass **exactly** `artifacts.source_to_id["sequence"]` as
   `sequence_source_id` and `artifacts.modality_to_id["numeric"]` as
   `numeric_modality_id`. Neither ID may be hard-coded: both belong to the
   selected vocabulary artifact.
3. Use the same artifact's tokenizer contract with `FoundationPretrainDataset`.
   Any tokenizer-contract mismatch or `FoundationBatch`/model dtype, shape, or
   finite-value validation error aborts startup before the first optimizer step.
4. Pass `targets["next_action_valid_mask"]` into next-action loss. Target `0`
   in a false-mask row is a sentinel, never a class. Log
   `next_action_valid_count / batch_size` per batch and aggregate it per epoch;
   false-mask rows may still contribute to other objectives.

Only after this gate passes may Member 3 train/checkpoint the foundation model
and pass `long_term` and `short_term` to Food Recommendation. The Baseline team
continues to own baselines, evaluation, and metrics; downstream code must not
reimplement them.

### Deliverables — pretraining

1. `PretrainingObjective` nhận `(FoundationOutput, FoundationBatch)` và trả
   `total_loss`, từng named loss, count mỗi target. Không sửa encoder để nhét
   loss vào model.
2. Deterministic masker dựa trên seed, mask rate config, special token safety.
3. Modality heads + hierarchical categorical ID head.
4. `python -m src.training.pretrain --config configs/pretrain.yaml`:
   set seed, validate data version, train/valid, log JSONL, optional grad clip,
   save best checkpoint ngoài Git.
5. Checkpoint loader/exporter đúng contract, test round-trip.
6. Cài public `src.contracts.output`: validate `FoundationOutput`, atomic
   prediction writer, validation report và exit-code mapping đúng Output
   Contract v1. Người 4 chỉ import/call module này, không tạo writer riêng.

### Deliverables — Food Recommendation

```python
class FoodRanker(nn.Module):
    def forward(
        self,
        long_term: torch.Tensor,     # [B, D]
        short_term: torch.Tensor,    # [B, D]
        food_candidate_ids: torch.LongTensor,  # [B, C]
    ) -> torch.Tensor:               # scores [B, C]
        ...
```

- Use concatenated/fused long + short embedding, candidate embedding và scorer.
- Train loss: listwise CE over one-positive candidate set (hoặc documented
  multi-positive variant nếu Data contract cung cấp); không tự tạo candidates.
- `freeze_foundation: true` mặc định. Fine-tune chỉ qua config explicit.
- Đọc Food request/candidates từ handoff Data team, temporal train/valid/test.
- Xuất prediction table đúng contract Baseline team; gọi evaluator của họ,
  không cài Recall/NDCG/MRR/baseline trong module này.

### Tests bắt buộc

- masked loss không mask special/pad và finite từng modality;
- hierarchical head output/loss shape đúng;
- smoke train 2 epoch CPU -> checkpoint load/encode lại được;
- Food scores `[B,C]`, positive candidate train path hợp lệ;
- candidate hoặc label tương lai -> validation fail;
- prediction table pass Output Contract v1: schema/dtype, candidate-set
  exactness, primary key, ordering, score finite và manifest SHA.

### Người 4 — Ad Optimization

**Branch:** `feat/ad-optimization`  
**Sở hữu file:** `src/downstream/ad_optimization.py`,
`configs/ad_optimization.yaml`, `tests/downstream/test_ad_optimization.py`.  
**Không chạm:** input/model/pretrain, Food module, Data team files,
`src/baselines/**`, `src/evaluation/**`, `pyproject.toml`.

### Deliverables

```python
class AdRanker(nn.Module):
    def forward(
        self,
        long_term: torch.Tensor,     # [B, D]
        short_term: torch.Tensor,    # [B, D]
        ad_candidate_ids: torch.LongTensor,  # [B, C]
    ) -> torch.Tensor:               # logits/scores [B, C]
        ...
```

1. Fuse long + short embedding với ad/campaign candidate embedding.
2. Train binary CTR loss theo exposure/click labels do Data team bàn giao;
   request không click là valid. Không resample negatives hoặc rewrite exposure
   policy trừ khi policy được ghi trong config và Data/Baseline team phê duyệt.
3. Load checkpoint only qua `embed_users`; `freeze_foundation: true` mặc định;
   log checkpoint hash nếu fine-tune.
4. Export one prediction row mỗi ad candidate, gọi evaluator public của
   Baseline team. Bắt buộc dùng writer/validator `src.contracts.output` của
   Người 3; không implement AUC/logloss/Brier/NDCG hoặc ad baseline.
5. Record: data version, split cutoff, seed, exposure/candidate policy, class
   prevalence, missing-click policy, checkpoint hash, artifact path.

### Tests bắt buộc

- `[B,C]` score shape và finite CTR loss;
- request không có click vẫn train/evaluate path hợp lệ;
- duplicate candidate bị reject/handled đúng contract;
- future click/exposure/candidate bị validation fail;
- table có đủ cột evaluator cần;
- Ads score ngoài `[0,1]`, duplicate/missing/extra candidate, NaN/Inf, sai
  `data_version` hoặc run manifest thiếu key -> Output Contract validator fail;
- integration test với evaluator mock bám public API, không mock metric logic.

## 6. Config tối thiểu

```yaml
# configs/pretrain.yaml
seed: 42
data:
  root: /external/data/versions/sim-v1
  expected_version: sim-v1
sequence:
  max_length: 128
model:
  d_model: 128
  num_layers: 4
  num_heads: 4
training:
  batch_size: 32
  epochs: 10
  learning_rate: 0.0001
  artifact_dir: /external/artifacts/pretrain
objective:
  mask_rate: 0.15
  masked_value_weight: 1.0
  next_key_weight: 1.0
  next_value_weight: 1.0
```

Food/Ads config bắt buộc có `data.root`, `data.expected_version`,
`checkpoint.path`, `freeze_foundation`, `seed`, split name, artifact dir và
task-specific candidate/exposure policy identifier. Artifact/data paths không
được hard-code trong Python.

## 7. Trình tự làm song song và merge

### Gate 0 — trước khi code train

1. Data team freeze `sim-v1` + manifest + smoke subset.
2. Baseline team freeze prediction-table/evaluator API.
3. Cả team review file này. Người 1 merge một PR contract nhỏ với types và
   validation stub; không chờ full implementation.

### Phát triển song song

| Người | Làm ngay, không chờ ai | Chỉ cần chờ khi |
|---|---|---|
| 1 | handoff validator/tokenizer/collator | cần data version thật để integration test |
| 2 | adapters/encoder/mask với fixture batch | cần batch thật để smoke integration |
| 3 | objectives/trainer với FakeEncoder; FoodRanker với FakeFoundation | train thật cần 1 + 2; food thật cần checkpoint |
| 4 | AdRanker với FakeFoundation/evaluator mock | train thật cần checkpoint của 3 |

Ai train gì:

- Người 3 train **foundation model** và xuất checkpoint.
- Người 3 train FoodRanker trên checkpoint đó.
- Người 4 train AdRanker trên checkpoint đó.
- Người 1/2 hỗ trợ debug data/model; không sở hữu production train loop.

### Merge queue

1. Người 1 input adapter và Người 2 encoder có thể review/merge độc lập.
2. Người 3 merge objective/training sau khi API Người 1/2 ổn; checkpoint
   contract là dependency của downstream train thật.
3. Food PR của Người 3 và Ads PR của Người 4 merge độc lập: không chung file.
4. PR integration cuối chỉ chạy checkpoint thật, hai task và evaluator của
   Baseline team. Lỗi interface trả đúng branch owner; không hot-fix chéo.

Mỗi PR phải:

```bash
python -m pytest
python -m ruff check .
```

Không reformat toàn repo. Chỉ rebase `main` ngay trước review; một PR chỉ
đụng file mà owner được giao.

## 8. Definition of Done toàn hệ thống

```bash
python -m pytest
python -m ruff check .
python -m src.training.pretrain --config configs/pretrain.yaml
python -m src.downstream.food_recommendation --config configs/food_recommendation.yaml
python -m src.downstream.ad_optimization --config configs/ad_optimization.yaml
# Chạy evaluator command do Baseline team bàn giao trên hai prediction table.
```

Chỉ nhận MVP khi tất cả đúng:

- manifest/data validation pass và test chứng minh không temporal leakage;
- foundation checkpoint load lại được, metadata đầy đủ, embedding finite;
- Food table được evaluator chung chấm Recall@10/NDCG@10/MRR trên cùng
  candidate set với food baseline;
- Ads table được evaluator chung chấm ROC-AUC/PR-AUC/log loss/Brier và ranking
  metrics trên đúng exposure policy với ad baseline;
- mỗi experiment record có data/preprocess version, split, seed, checkpoint,
  candidate/exposure policy, negative/missing-click handling;
- pretrain-v2 test xác nhận các scenario hiện có (Advertisement, Live Streaming,
  Video Browsing, E-commerce, Customer Service, Search Behavior) không leak
  event/profile sau cutoff; ride/location chỉ được test khi một future dataset
  contract thực sự cung cấp các field đó.

## 9. Backlog, không chặn MVP

1. CPU tokenization và accelerator workers tách riêng; thêm Ray sau benchmark.
2. Pretrained text/image/audio adapters và entity embedding merchant/driver.
3. Taxonomy thật, sampled softmax/retrieval cho ID vocabulary lớn.
4. Fine-tune, calibration/fairness/privacy audit, drift monitoring và
   embedding registry kiểu “embeddings as a product”.
