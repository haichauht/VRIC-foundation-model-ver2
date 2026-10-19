# Ghi chú cho team — foundation chạy trên dữ liệu mô phỏng (sim-v1)

Branch `feat/sim-foundation` (dự án con 2). Spec: `docs/superpowers/specs/2026-10-17-sim-foundation-design.md`;
hợp đồng dữ liệu: `docs/simulation_data_contract.md`. **Tải dữ liệu `sim-v1`: `docs/sim_v1_data.md`** (`scripts/download_sim_v1.py`). Ghi chú này liệt kê thay đổi theo người phụ trách
file, kèm lý do, cờ cấu hình, mặc định tương thích, kết quả smoke end-to-end và giới hạn đã biết.

Tóm tắt: mọi thay đổi đều **mặc định tắt**, hoặc chỉ có tác dụng với gói/config sim mới. Đường Meituan
(Food), Taobao (Ads) và checkpoint Người 3 chạy như cũ (test hồi quy Meituan Food, Taobao Ads, nạp checkpoint
cũ và toàn bộ suite đều pass).

| Ai | File | Thay đổi | Cờ / mặc định |
|---|---|---|---|
| Người 1/3 | `src/training/pretrain.py` | allowlist context theo gói, chặn vocab thiếu khóa, metadata nguồn gốc, chọn `data_dir`; nhóm lr riêng cho `user_adapter` | allowlist tự bật khi gói có `context_allowlist`, gói khác như cũ; `training.user_adapter_lr_multiplier: 1.0` |
| Người 2 | `src/models/encoder.py`, `src/models/adapters.py`, `src/models/objectives.py` | `long_term` được học qua CLS; token `user_id` không bị mask; std khởi tạo user embedding | `long_term_injection: "none"`, `unmaskable_key_ids: ()`, `user_embedding_init_std: None` |
| Người 4 | `src/downstream/ad_inputs.py`, `src/downstream/ad_optimization.py` | `data.format: sim_v1`, `foundation.init` | `data.format: taobao`, `foundation.init: pretrained` |
| Food | `src/downstream/food_*.py` | `data.format: sim_v1`, `foundation.init` | `data.format: meituan`, `foundation.init: pretrained` |
| chung | `src/downstream/sim_events.py`, `src/downstream/foundation_init.py` (mới) | chuyển event sim giống hệt pretrain; nhánh đối chứng `random`; ràng buộc checkpoint–gói sim | — |

## 1. Tokenizer / pretrain (Người 1, Người 3)

### Allowlist context theo gói dữ liệu

Trước đây `DEFAULT_CONTEXT_ALLOWLIST` (`src/foundation_input/tokenizer.py`) không có scenario `Ride` và
`Food`, nên context của hai dịch vụ này (`pickup_geohash`, `merchant_id`, `cuisine`, ...) bị bỏ âm thầm.
Giờ trong `src/training/pretrain.py`:

- `tokenizer_config_for_manifest(manifest)` đọc `context_allowlist` trong `preprocessing_config.json` của
  gói (qua `manifest.config`) và trả `TokenizerConfig(context_allowlist = mặc định ∪ allowlist gói)`;
  trùng scenario thì allowlist của gói thắng. Gói không có `context_allowlist` (Meituan, OmniBehavior)
  nhận `TokenizerConfig()` mặc định như cũ.
- Config này được truyền vào `fit_vocabulary_from_samples(..., tokenizer_config)`. `tokenizer_contract`
  của vocab và checkpoint tự ghi allowlist này, nên `FoundationPretrainDataset` và downstream (dựng
  tokenizer từ contract) dùng đúng allowlist mà không phải sửa gì.
- `check_vocab_allowlist(vocabularies, tokenizer_config)`: khi nạp vocab có sẵn (`vocab_path` trong config
  hoặc `--vocab_path`), nếu allowlist của vocab thiếu khóa nào gói cho phép thì dừng với `ValueError`
  ("Vocabulary was fitted without context keys the package allows ... Refit the vocabulary against this
  package."), không âm thầm bỏ context.

### Metadata mới trong checkpoint (`preprocess_metadata`)

`pretrain_preprocess_metadata(...)` tạo `preprocess_metadata` của checkpoint. Ngoài các khóa cũ
(`dataset_name`, `source_schema_version`, `source_fingerprint`) có thêm:

- `source_manifest_sha256`: sha256 của `manifest.json` gói pretrain;
- `sequence`: `{"max_tokens": max_length, "max_event_count": ...}` — downstream đọc khóa này để mã hóa
  history với đúng ngân sách token của pretrain;
- `dataset_provenance`: `dataset_version`, `generator_version`, `sim_config_sha256` (chỉ các khóa có trong
  manifest; gói Meituan/OmniBehavior cho `{}`). Food và Ads `sim_v1` dùng `sim_config_sha256` để kiểm tra
  checkpoint và gói downstream đến từ cùng một thế giới mô phỏng.

Định dạng checkpoint (`format_version`) không đổi, nhưng checkpoint lưu từ branch này có thêm hai khóa trong
`model_config` (`long_term_injection`, `user_embedding_init_std`): code không có `FoundationModelConfig` của branch
này (ví dụ `origin/feat/pretrain-food-recommendation`) sẽ gặp `TypeError` khi nạp chúng. Checkpoint cũ vẫn nạp được
vào code mới (hai khóa lấy giá trị mặc định).

### Chọn dữ liệu: `resolve_targets`

`resolve_targets(config, overrides)`: `data_dir` của config chỉ được dùng khi không có `dataset` nào được
yêu cầu (CLI `--dataset` hoặc `dataset` trong config); `--data_dir` trên CLI luôn thắng. Vì vậy các config
sim đặt `dataset: null`.

### Config sim

- `configs/pretrain_sim.yaml`: CPU local, `data_dir: experiments/sim/sim-smoke-v1/pretrain`,
  `output_dir: experiments/pretrain_sim`, 1 epoch, batch 16, lr 1e-4, `max_length: 512`,
  `model.long_term_injection: "cls"`, `model.user_embedding_init_std: 0.02`,
  `training.user_adapter_lr_multiplier: 10.0` (mục 2).
- `configs/food_recommendation_sim.yaml` và `configs/ad_optimization_sim.yaml` đọc `experiments/sim/sim-smoke-v1/{food,ads}`
  (`expected_version: sim-smoke-v1`) và checkpoint `experiments/pretrain_sim/checkpoints/best_checkpoint.pt`: đi cặp với
  `pretrain_sim.yaml` (cùng gói `sim-smoke-v1`, cùng `output_dir`). Bản Kaggle: `*_sim_kaggle.yaml` (mục "Chạy trên Kaggle").
  Test `tests/downstream/test_sim_config_pairing.py` giữ hai bộ ba này khớp nhau.
- `configs/pretrain_sim_kaggle.yaml`: GPU Kaggle, `data_dir: /kaggle/working/sim-v1/pretrain`,
  `output_dir: /kaggle/working/pretrain_sim`, 6 epoch, batch 64, lr 3e-4, mixed precision,
  `d_model` 128, 4 layer, `max_length: 512`, `long_term_injection: "cls"`, cùng hai núm `user_adapter`
  (hệ số 10 chọn theo đo trên smoke; với lr 3e-4 và 6 epoch, hàng user sẽ di chuyển nhiều hơn smoke).

Lưu ý: `training.warmup_steps` có trong config (như `configs/pretrain.yaml`) nhưng `pretrain.py` của
branch này không dùng: không có scheduler, lr cố định. Giới hạn số mẫu là khóa **cấp cao nhất**
`max_train_samples` / `max_val_samples` (không nằm trong `training:`). Downstream Food/Ads mã hóa history
theo `preprocess_metadata.sequence` của checkpoint (Ads còn yêu cầu `max_tokens == model_config.max_length`).

### Độ dài chuỗi (đo trên `sim-smoke-v1`)

Đo trên 2.000 sample train đầu tiên (theo thứ tự manifest) của `experiments/sim/sim-smoke-v1/pretrain`,
với tokenizer của gói (`tokenizer_config_for_manifest`). Độ dài = số token ngữ nghĩa (`semantic_tokens`)
+ 2 (CLS/SEP mà `_truncate_records` chừa chỗ), **trước** khi cắt:

| p50 | p90 | p99 | max | trung bình | > 512 | > 256 |
|---|---|---|---|---|---|---|
| 264 | 290 | 310 | 317 | 251,3 | **0%** (0/2.000) | 61,6% (1.231/2.000) |

History mỗi snapshot: p50 28 event, tối đa 30 (`max_history_events`). Kết luận: `max_length: 512` đủ (không
sample nào bị cắt), giữ nguyên trong cả hai config sim. Với `max_length: 256` cũ, 61,6% sample sẽ mất các
phiên cũ nhất.

### Ba lỗi trên `origin/feat/pretrain-food-recommendation`

Branch của Người 3 (commit `3598b574` "optimize pretraining pipeline") có ích về tốc độ (prefetch batch,
scheduler warmup + cosine, gradient accumulation) và không đụng allowlist hay `long_term`. Branch này
**chưa** được gộp; khi gộp, phần sửa ở đây trong `pretrain.py` nhỏ và tập trung nên chỉ xung đột vài dòng
(chủ yếu đoạn nạp/fit vocab — branch đó gọi `fit_vocabulary_from_samples(train_samples_stream,
vocab_config)` **không** truyền `tokenizer_config`, tức là vẫn bỏ context Ride/Food). Ba lỗi cần sửa trước
khi dùng (số dòng theo `src/training/pretrain.py` của commit đó):

1. **Cosine cố định 100k bước** (dòng 642): `_lr_lambda` tính
   `progress = (step - warmup_steps) / (100_000 - warmup_steps)` bất kể số bước thật. `sim-smoke-v1`
   chỉ có 312 bước/epoch (batch 16); `sim-v1` (20.000 user, ≈ 10 snapshot train/user, batch 64, 6 epoch)
   ước tính ≈ 19k bước — LR cuối chỉ giảm còn ≈ 0,9 × đỉnh, tức gần như không có cosine decay. Sửa: tính
   tổng bước = `epochs × ceil(số snapshot train / (batch_size × gradient_accumulation_steps))` từ manifest
   (`snapshots_by_split`), hoặc cho config `training.total_steps`.
2. **Phần đuôi gradient accumulation không bao giờ chạy** (dòng 713 và 724): điều kiện
   `(step_in_epoch + 1) % grad_accum_steps == 0 or step_in_epoch == epoch_batches - 1` được kiểm tra
   **trước** `epoch_batches += 1`, nên lúc đó `epoch_batches == step_in_epoch` và vế thứ hai là
   `x == x - 1`, luôn sai. Khi số batch của epoch không chia hết cho `gradient_accumulation_steps`,
   gradient của các batch cuối được cộng dồn nhưng không bao giờ `optimizer.step()` (bị `zero_grad()` ở
   bước 0 của epoch sau; ở epoch cuối thì mất hẳn). Với mặc định `gradient_accumulation_steps: 1` không
   ảnh hưởng. Sửa: sau vòng lặp batch, nếu `epoch_batches % grad_accum_steps != 0` thì unscale/clip/step
   một lần (dataset streaming không biết trước số batch).
3. **Tự dùng lại `output_dir/vocab.json`** (dòng 529–533): khi không có `vocab_path` và không phải
   `--smoke`, nếu `output_dir/vocab.json` đã tồn tại thì nạp nó thay vì fit lại. Xung đột với §2 của spec
   (allowlist theo gói): một `vocab.json` còn lại từ lần chạy trước (fit trước khi có allowlist gói, hoặc
   fit trên gói/phiên bản khác, ví dụ `sim-smoke-v1` rồi `sim-v1` cùng `output_dir`) bị dùng lại âm thầm.
   Trên branch này, vocab nạp sẵn luôn đi qua `check_vocab_allowlist` và chỉ được nạp khi chỉ định rõ
   `vocab_path`. Khi gộp: bỏ việc tự dùng lại, hoặc ít nhất cho đường tự dùng lại cũng gọi
   `check_vocab_allowlist` và kiểm tra `dataset_name`/`source_fingerprint` của vocab với gói đang train.

## 2. Encoder / objectives (Người 2)

- `FoundationModelConfig.long_term_injection: "none" | "cls"`, mặc định `"none"`; giá trị khác bị
  `__post_init__` từ chối. Được lưu trong `model_config` của checkpoint; checkpoint cũ không có khóa này
  nạp với `"none"` (checkpoint Người 3 vẫn nạp được, output giống hệt trước khi sửa).
- `"cls"`: trong `FoundationModel.forward`, trước Transformer, `user_adapter(user_ids)` được cộng vào đặc
  trưng token CLS (vị trí 0). Mọi token attend tới CLS, nên loss masked và next-action đều truyền gradient
  về `UserIdAdapter`. Trước đây không loss nào dùng `long_term`, nên nó giữ nguyên giá trị khởi tạo ngẫu
  nhiên. Output `long_term` vẫn là `user_adapter(user_ids)`; API `embed_users` không đổi.
- **`short_term` giờ cũng thấy thông tin user** khi bật `"cls"`: `short_term` vẫn là mean-pool các token
  event, nhưng các token đó attend tới CLS (đã cộng `long_term`), nên mang một phần thông tin ID người
  dùng. Ai so sánh `short_term` giữa hai checkpoint `none`/`cls` cần biết điều này.
- `PretrainObjectivesConfig.unmaskable_key_ids: tuple[int, ...] = ()`: `MaskingPolicy` không bao giờ
  chọn token có `key_id` trong danh sách. `pretrain.py` đặt `(user_id_key_id(...),)` khi
  `long_term_injection == "cls"`, ngược lại `()`. Lý do: nếu token `user_id` bị mask, model chỉ việc học
  dịch lại ID từ CLS — lối tắt phí tín hiệu học.
- User ID ánh xạ vào `num_user_buckets` (50.000) bucket qua id trong vocab (`id % 50.000`).
- Test `test_cls_injection_gives_user_adapter_gradient` đã sửa (commit `77a53bfd`): loss cũ là tổng
  hidden state, mà mỗi hidden state ra khỏi LayerNorm nên tổng đó không phụ thuộc input — gradient chỉ là
  nhiễu số (≤ 6e-7) và bằng đúng 0 với 11/200 seed (test chập chờn). Giờ test cố định seed và chiếu hidden
  state lên một hướng cố định.

### Hai núm để `long_term` thật sự được học (quyết định R5)

Có gradient (T2) chưa đủ: spec §1 mục tiêu 2 cần `long_term` **được học**. Lần smoke đầu (khởi tạo
`nn.Embedding` mặc định N(0, 1), chuẩn hàng ≈ 11,4) cho thấy sau 1 epoch hàng user gần như không đổi: Adam
bước ≈ lr mỗi phần tử bất kể độ lớn gradient, mà mỗi hàng user chỉ được cập nhật khi user đó có trong
batch (≈ 10 snapshot train/user/epoch). Vì vậy thêm hai núm, mặc định giữ nguyên hành vi cũ:

- `FoundationModelConfig.user_embedding_init_std: float | None = None` (`src/models/encoder.py`, truyền vào
  `UserIdAdapter(init_std=...)` trong `src/models/adapters.py`). `None` = khởi tạo N(0, 1) y như trước (không
  tốn thêm RNG, nên mọi trọng số khởi tạo với cùng seed giống hệt trước; checkpoint cũ và config mặc định
  không đổi); một số dương = N(0, std). Lưu trong `model_config` của checkpoint, nên nhánh `random` của
  downstream dựng lại với cùng std. YAML: `model.user_embedding_init_std`.
- `training.user_adapter_lr_multiplier` (mặc định 1.0, `src/training/pretrain.py`,
  `optimizer_param_groups`): tham số `user_adapter` nằm trong nhóm AdamW riêng (`name: "user_adapter"`; nhóm
  `user_adapter_no_decay` nếu có) với lr × hệ số; cách chia decay / no-decay giữ nguyên (`bias`,
  `LayerNorm.weight`). Nếu gộp scheduler của Người 3 (`LambdaLR`), tỷ lệ được giữ vì `LambdaLR` nhân lr
  ban đầu của từng nhóm. Hệ số được ghi vào `experiment_record.json` (`user_adapter_lr_multiplier`) và tên
  nhóm/lr nằm trong `optimizer_state_dict` của checkpoint.
- Cả hai config sim đặt `model.user_embedding_init_std: 0.02` và `training.user_adapter_lr_multiplier: 10.0`.

Đo trên `sim-smoke-v1`, 1 epoch (312 bước, lr 1e-4), cùng cách đo ở mục 6 (so với
`FoundationModel(model_config)` dựng lại dưới `torch.manual_seed(42)`); tiêu chí chấp nhận: trung vị thay
đổi tương đối (‖Δ‖/‖w₀‖) của 500 hàng user ≥ 20%:

| Khởi tạo | Hệ số lr | Trung vị | Trung bình | Nhỏ nhất | cos(khởi tạo, sau train) user 5 / 6 | Đạt? |
|---|---|---|---|---|---|---|
| N(0, 1) (lần đầu) | 1 | 0,115% | 0,114% | 0,036% | 0,9999998 / 0,9999998 | không |
| N(0, 0,02) | 1 | 5,75% | 5,59% | 0,79% | 0,9995 / 0,9993 | không |
| N(0, 0,02) | **10** | **57,4%** | 55,9% | 7,9% | 0,946 / 0,932 | **có** |

Các hàng không có user (49.500 hàng) không nhận gradient, chỉ co theo weight decay của nhóm `user_adapter`:
trọng số sau train = khởi tạo × (1 − lr·hệ số·wd)^312, tức **× 0,99969** với hệ số 1 (lr·wd = 1e-6/bước; (1 − 1e-6)^312 = 0,999688)
và **× 0,99688** với hệ số 10 (1e-5/bước) — số đo với hệ số 10 khớp tới 5 chữ số thập phân, xác nhận khởi tạo dựng
lại trùng khởi tạo của lần chạy.

Lưu ý: hai lần chạy hệ số 1 và 10 cho loss train/validation **giống nhau tới 4 chữ số** (8,0203 / 5,8195):
với std 0,02 phần `long_term` cộng vào CLS (chuẩn ≈ 0,23 lúc khởi tạo) rất nhỏ so với đặc trưng CLS
(≈ 8,7, đo ở checkpoint lần đầu), nên sau 1 epoch nó chưa ảnh hưởng đáng kể tới loss. "Hàng user di
chuyển" mới là điều kiện cần; việc `long_term` có mang thông tin hữu ích hay không phải đo ở downstream
của dự án con 3 (`sim-v1`, 6 epoch).

### Nhánh `random` dùng seed của pretrain (quyết định R6)

Nhánh `random` của downstream khởi tạo lại dưới `torch.manual_seed(seed)` với `seed` của config downstream.
Các config sim đặt `seed: 42` giống pretrain, nên nhánh `random` **cố ý** tái tạo đúng trọng số khởi tạo của
chính lần pretrain đó (kể cả `user_adapter` với cùng std): đối chứng là "cùng mạng đó, chưa pretrain". Đổi
`seed` downstream sẽ đổi cả đối chứng lẫn khởi tạo head, nên giữ nguyên khi so sánh hai nhánh.

## 3. Ads (Người 4)

- Branch `origin/feat/ad-optimization` đã được gộp vào đây (merge sạch).
- `data.format: taobao | sim_v1`, mặc định `taobao` (đường cũ giữ nguyên). Các hằng số Taobao trước đây
  hard-code giờ gom vào bảng `AD_FORMATS` (`AdFormat` trong `src/downstream/ad_inputs.py`), chọn theo
  `data.format`: `namespace` (== `source_dataset` của manifest == `dataset_name` của vocab), chính sách
  exposure, tiền tố ID người dùng/quảng cáo (`taobao:user:`/`taobao:ad:` ↔ `sim:user:`/`sim:ad:`) và quy
  tắc cận split (Taobao `<=`; sim: `train_end`/`valid_end` là cận trên **không bao gồm**).
- Đường `sim_v1`: nhận event mọi dịch vụ (Ride, Food, Ads) làm history; impression của request tra qua
  `source_row_index` (phải bằng số dòng trong `events.parquet`); nhận cột profile ngoài
  `{user_id, as_of_timestamp}`; history đi qua `src/downstream/sim_events.py`; thời điểm nhãn có được =
  giờ click nếu click, ngược lại hiển thị + 27 s (`SIM_AD_OUTCOME_MATURITY_SECONDS`, quy ước Baseline R12),
  tính từ `events.parquet`.
- Ràng buộc checkpoint: Taobao giữ ràng buộc cũ (sha256 manifest Ads). `sim_v1`: vocab `dataset_name ==
  "sim_v1"` và `preprocess_metadata.dataset_provenance.sim_config_sha256` == `sim_config_sha256` của
  manifest Ads (`check_sim_checkpoint_binding`); `checkpoint.expected_version` phải bỏ trống.
- `foundation.init: pretrained | random` (xem mục 5). Ads cho phép `random` cùng
  `training.cache_foundation_embeddings: true`.
- Config mới `configs/ad_optimization_sim.yaml` (`exposure_policy:
  sim-served-impression-single-candidate-v1`, `missing_click_policy: reject`).
- Tài liệu cách dùng `sim_v1` / `foundation.init` cho Ads: docstring đầu `ad_inputs.py`,
  `ad_optimization.py` và ghi chú này.

## 4. Food (nhóm mình)

- `data.format: meituan | sim_v1`, mặc định `meituan`. Config mới `configs/food_recommendation_sim.yaml`
  (`data.root`, `data.expected_version`; không có `candidate_policy`, `catalog`, `checkpoint.handoff` —
  ba khóa này bị từ chối với `sim_v1`).
- `sim_v1` đọc `food/manifest.json` (đường dẫn + sha256 của `samples.jsonl` và `merchants.parquet`),
  catalog là `merchants.parquet` (ứng viên phải nằm trong catalog), `candidate_policy` lấy từ manifest,
  `history_truncated`/`history_group_truncated` đọc từ sample, user ID giữ dạng đầy đủ `sim:user:N`,
  history đi qua `sim_events`.
- Ràng buộc checkpoint như Ads (`dataset_name == "sim_v1"` + `sim_config_sha256`).
- Sửa hồi quy (commit `eb122e45`, siết lại ở `ad3b3a08`): từ khi `FoundationModelConfig` có thêm trường,
  record Person 3 (`experiments/person3_checkpoint/experiment_record.json`, `model_config` không có các
  khóa mới) bị `_validate_person3_payload` từ chối ("Food handoff model configuration mismatch"), tức là
  `configs/food_recommendation.yaml` không nạp được checkpoint Người 3. Giờ record chỉ được thiếu các
  trường thêm trên branch này (`long_term_injection`, `user_embedding_init_std`), và chỉ khi checkpoint
  giữ giá trị mặc định của chúng; thiếu bất kỳ trường cũ nào (kể cả đang ở mặc định) vẫn bị từ chối.
- `foundation.init: pretrained | random`. Với `random`, Food từ chối cache embedding tính sẵn
  (`FoodServices.frozen_embeddings`, chỉ dùng khi gọi từ code); cache trong lượt chạy
  (`training.cache_foundation_embeddings: true`) vẫn dùng được với `random` qua CLI — smoke chạy cả hai
  cách cho nhánh `random` và ra metric giống hệt nhau (foundation đóng băng ở chế độ `eval`).
- Tài liệu cách dùng `sim_v1` cho Food: docstring đầu `food_inputs.py` và ghi chú này.

## 5. Phần dùng chung

- `src/downstream/sim_events.py`: chuyển event công khai của sim thành `EventRecord` **giống hệt từng
  token** với loader pretrain (test parity T3 trong `tests/downstream/test_sim_events.py`): scenario
  `ride/food/ads → Ride/Food/Advertisement`; tách `attributes` thành context (theo allowlist) và action
  (`fare, distance_km, wait_minutes, order_amount, delivery_distance_km` + `type`; Ads
  `ad_impression → impression`, `ad_click → click`); thời gian đổi sang UTC naive; phiên dựng lại tại mỗi
  bước đầu phễu (`search_destination`, `search`, `ad_impression`).
- `src/downstream/foundation_init.py`: `reinitialize_foundation(model, seed)` — nhánh `random` giữ vocab và
  `model_config` của checkpoint, khởi tạo lại trọng số dưới `torch.manual_seed(seed)` (không làm xáo RNG
  của downstream), vẫn đóng băng như nhánh `pretrained`; `check_sim_checkpoint_binding(...)`.

## Chạy trên Kaggle

Bộ ba Kaggle: `configs/pretrain_sim_kaggle.yaml`, `configs/food_recommendation_sim_kaggle.yaml`,
`configs/ad_optimization_sim_kaggle.yaml` (không trộn với bộ ba local `sim-smoke-v1`).

1. Notebook đính kèm output của kernel `hunhthhichu/vric-sim-generate` làm input; dữ liệu nằm ở
   `/kaggle/input/vric-sim-generate/` (`sim-v1/{food,ads,baseline,oracle}/...`, `sim-v1-pretrain.tar`,
   `sim-v1/data_report.json`).
2. Giải nén gói pretrain vào `/kaggle/working` (config pretrain đọc `/kaggle/working/sim-v1/pretrain`):
   `tar -xf /kaggle/input/vric-sim-generate/sim-v1-pretrain.tar -C /kaggle/working`
3. Ba lệnh, theo thứ tự:

```
python -m src.training.pretrain --config configs/pretrain_sim_kaggle.yaml
python -m src.downstream.food_recommendation --config configs/food_recommendation_sim_kaggle.yaml
python -m src.downstream.ad_optimization --config configs/ad_optimization_sim_kaggle.yaml
```

Hai config downstream đọc `/kaggle/input/vric-sim-generate/sim-v1/{food,ads}` (`expected_version: sim-v1`) và
checkpoint `/kaggle/working/pretrain_sim/checkpoints/best_checkpoint.pt`; `cpu_threads: 4` cho cả Food và Ads.

Gợi ý tài nguyên (**ước lượng ngoại suy từ smoke**, chưa đo trên `sim-v1`): downstream chạy trên CPU (Ads gọi
`foundation.cpu()`), chỉ pretrain dùng GPU. Nhánh Ads ≈ 2 giờ với `cpu_threads: 1` cho 260k request (dùng 4 để
rút ngắn); nhánh Food ≈ 20–25 phút. RAM đỉnh của Ads ≈ 10 GB (≈ 5 GB dict event + ≈ 4 GB batch đã mã hóa): vừa
Kaggle (~30 GB RAM, 12 giờ), không vừa laptop 16 GB.

## 6. Smoke end-to-end (T4): lệnh và kết quả

Chạy ngày 2026-10-17 trên laptop Windows, CPU 8 luồng, branch `feat/sim-foundation`. Mọi output nằm dưới
`experiments/` (không commit).

**Bước 1 — dữ liệu.** `sim-smoke-v1` cũ (sinh trước khi hiệu chỉnh simulator, `tree_sha256` khác) được
đổi tên thành `experiments/sim/sim-smoke-v1-precalib`, rồi sinh lại (≈ 40 s, `status: complete`, mọi kiểm
tra `passed`):

```
.venv/Scripts/python -m src.simulation --preset smoke --out experiments/sim
```

`data_report.json`: `tree_sha256 = 4ed1499e6a36e6f2f048b3afe79a7d4b1f8e5370441776a03c891c25275afb04`,
`sim_config_sha256 = cd9945ba559f8397ececd2d23770dce6e27642ec7a0de3ed2dc24f1fbc5efa78`. Gói pretrain:
500 user, 4.988 snapshot train, 1.000 validation.

**Bước 2 — độ dài chuỗi:** xem mục 1 (0% sample vượt 512 token).

**Bước 3 — pretrain** (1 epoch, CPU, toàn bộ snapshot, không giới hạn số mẫu):

```
.venv/Scripts/python -m src.training.pretrain --config configs/pretrain_sim.yaml
```

Lần chạy hiện tại dùng config đã commit, có hai núm ở mục 2 (`user_embedding_init_std: 0.02`,
`user_adapter_lr_multiplier: 10.0`):

- Exit 0, 399 s tổng (vòng train 384 s, 312 bước batch 16 ≈ 1,23 s/bước; fit vocab ≈ 8 s).
- Loss train trung bình 8,020 (bước 20: 11,09 → bước 300: 5,33); validation 5,819 (masked 5,216, gồm
  categorical 4,104 + numeric 1,112; next-action 1,208; độ chính xác categorical bị mask 22,7%; 63 batch).
  Lần chạy đầu (khởi tạo N(0, 1), không hệ số lr): train 8,148, validation 5,771.
- `experiments/pretrain_sim/checkpoints/best_checkpoint.pt` (sha256 `33cd3ca3bb30…`), `model_config`
  có `long_term_injection: "cls"`, `user_embedding_init_std: 0.02`, `max_length: 512`;
  `experiment_record.json` có `user_adapter_lr_multiplier: 10.0`; `preprocess_metadata` có `sequence =
  {max_tokens: 512, max_event_count: 30}` và `dataset_provenance` (`sim-smoke-v1`, `src.simulation 1.0`,
  `sim_config_sha256` như trên). `tokenizer_contract.context_allowlist` có `Ride`, `Food`,
  `Advertisement`; vocab có khóa `context.pickup_geohash`, `context.merchant_id`, `context.cuisine`, ...
- `UserIdAdapter` được học: so trọng số trong checkpoint với `FoundationModel(model_config)` dựng lại dưới
  `torch.manual_seed(42)` (đúng khởi tạo của lần chạy — 49.500 hàng không có user khớp khởi tạo × 0,99688
  ≈ (1 − 1e-4·10·0,01)^312, hệ số decay của lr × 10; lần đầu với hệ số 1 là × 0,99969), thay đổi tương đối (‖Δ‖/‖w₀‖) của 500 hàng user: trung vị **57,4%**, trung bình
  55,9%, nhỏ nhất 7,9% (tiêu chí R5: trung vị ≥ 20%). Bảng so sánh ba cấu hình ở mục 2.

**Bước 4 — downstream.** Bốn bản sao của `configs/food_recommendation_sim.yaml` và
`configs/ad_optimization_sim.yaml` (để ngoài repo), chỉ đổi:

```yaml
data: {format: sim_v1, root: experiments/sim/sim-smoke-v1/food, expected_version: sim-smoke-v1}  # Ads: .../ads
training: {epochs: 1, ...}           # các khóa training khác giữ nguyên
foundation: {init: pretrained}       # hoặc random
artifact_dir: experiments/sim_e2e/<task>_<init>
```

Nhánh Food `random` chạy với `training.cache_foundation_embeddings: false`; ở lần chạy đầu, chạy thêm một
lần với `true` cho kết quả giống hệt từng chữ số (mục 4). Lệnh:

```
.venv/Scripts/python -m src.downstream.food_recommendation --config <bản sao Food>.yaml
.venv/Scripts/python -m src.downstream.ad_optimization --config <bản sao Ads>.yaml
```

Kết quả (split test của `sim-smoke-v1`; Food 1.268 request × 10 ứng viên, Ads 2.797 impression, 162 click).
Lần chạy hiện tại dùng checkpoint của config đã commit (sha256 `33cd3ca3bb30…`, hai núm `user_adapter`):

| Lượt chạy | Exit | Thời gian* | Metric |
|---|---|---|---|
| Food `pretrained` | 0 | 153 s | NDCG@10 0,5541; MRR 0,4175; Recall@1 0,2043 |
| Food `random` | 0 | 158 s | NDCG@10 0,5462; MRR 0,4069; Recall@1 0,1869 |
| Ads `pretrained` | 0 | 651 s | ROC-AUC 0,5491; PR-AUC 0,0660; log loss 0,2210 |
| Ads `random` | 0 | 558 s | ROC-AUC 0,4845; PR-AUC 0,0558; log loss 0,2215 |

\* chuỗi Food và chuỗi Ads chạy song song với nhau (và một phần với bộ test); config Ads đặt
`training.cpu_threads: 1` (Food: 4).

Lần chạy đầu (checkpoint `ce4f1bbac210…`, khởi tạo N(0, 1), không hệ số lr; mọi lượt exit 0): Food
`pretrained` / `random` NDCG@10 0,5449 / 0,5625 (MRR 0,4054 / 0,4284); Ads ROC-AUC 0,5150 / 0,4789.

Mốc tham chiếu của `smoke` (`data_report.json`, `difficulty`): Food phổ biến 0,495, oracle 0,958 (catalog
60 quán, chỉ để chẩn đoán); Ads AUC phổ biến 0,613, oracle 0,784. Metric Food là `diagnostic`
(`observation_complete_policy: assume_complete_diagnostic`). File output mỗi lượt: `predictions.test.parquet`,
`run_manifest.json` (ghi `foundation.init`), `train_metrics.jsonl`, `validation_report.json`, model head
(`food_ranker-v1.pt` / `ad_ranker-v1.pt`); Ads có thêm `evaluation.json`, metric Food in ra stdout.

**Không đánh giá giá trị**: 1 epoch pretrain + 1 epoch head trên thế giới 500 user. Ở lần chạy hiện tại
`pretrained` nhỉnh hơn `random` ở cả Food và Ads, còn ở lần đầu Food ngược lại; mọi AUC Ads vẫn gần 0,5,
dưới mốc phổ biến 0,613. Các chênh lệch này không nói gì về pretraining; chúng chỉ cho thấy cả hai nhánh
chạy hết pipeline và ra kết quả khác nhau.

## 7. Giới hạn đã biết

- **Rò rỉ thông tin sau cutoff ở split train của downstream** (quyết định R8). Snapshot train của pretrain chạy
  tới ngày 33, nên với một request train ở ngày 10, hàng `user_adapter` (qua `long_term`, và yếu hơn là qua
  encoder) đã được huấn luyện trên event của chính user đó ở ngày 10–33, gồm cả event nhãn của request. Head
  downstream fit trên đặc trưng train bị rò này, làm lệch so sánh pretrained-vs-random. Validation (tuần 7) và
  test (tuần 8) **sạch**: pretrain dừng trước ngày 43 nên metric test báo cáo không bị thổi phồng. Việc lọc
  thuộc thiết kế thí nghiệm của dự án con 3: ngoài lần chạy dùng toàn bộ train, chạy thêm một lần
  sensitivity chỉ fit head trên request có cutoff ≥ ngày local 34 (bắt đầu 2026-02-07 00:00 +07:00, ngày đầu
  tiên sau ngày cutoff 33 của snapshot train cuối cùng) và báo cáo cả hai.
- **Profile Ads chỉ có phần tĩnh** (`age_group`, `saved_home_geohash`, `saved_work_geohash`; gói Ads cố ý
  không có trường 30 ngày), nên so với pretrain, sample Ads thiếu vài token profile
  (`avg_monthly_spend`, `most_used_service`, `ride_count_30d`, `food_order_count_30d`,
  `ad_click_count_30d`, `active_days_30d`). Food đưa đủ profile như pretrain.
- **Ứng viên Food/Ads chỉ dùng ID** (embedding `candidate_id` học từ train, 0 = OOV); thuộc tính quán/quảng
  cáo (cuisine, giá, ngành hàng, ...) chưa vào head.
- `short_term` nhận một phần thông tin user qua attention khi `long_term_injection: "cls"` (mục 2).
- Kết quả smoke (1 epoch trên `sim-smoke-v1`, 60 quán) chỉ chứng minh pipeline chạy, **không** dùng để kết
  luận pretraining có ích; thí nghiệm thật thuộc dự án con 3 (`sim-v1`, `configs/pretrain_sim_kaggle.yaml`).

## Thay đổi trong dự án con 3 (branch `feat/sim-experiments`)

| Ai | File | Thay đổi | Ảnh hưởng |
|---|---|---|---|
| Food (chúng ta), Ads (Người 4) | `src/downstream/head_training.py` (mới), `ad_optimization.py`, `food_recommendation.py` | Head dùng `AdamW` thay `Adam(weight_decay=...)`: L2 gắn vào gradient làm embedding ứng viên hiếm gặp sụp về 0 (Ads: 0,0002 lần ban đầu) | Kết quả downstream đổi: Ads `random` 0,522 → 0,626, Food `epoch3` 0,745 → 0,726 (`docs/sim_pretrain_v1_results.md` mục 6) |
| Baseline | `src/baselines/features.py` | `FeatureBuilder` đánh chỉ mục history một lần (tuyến tính) thay vì quét lại cho từng ứng viên | Giá trị đặc trưng **không đổi** (test so với thuật toán cũ); B1 trên `sim-v1` full từ hết RAM / quá giờ → 6–7 phút |
