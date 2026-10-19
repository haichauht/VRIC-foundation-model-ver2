# Hợp đồng dữ liệu simulator (`src/simulation`)

Tài liệu mô tả các gói dữ liệu do simulator siêu app (ride + food + ads) xuất ra. Thiết kế:
`docs/superpowers/specs/2026-10-15-simulator-design.md` (§6). Mỗi lần chạy ghi vào
`<out>/<dataset_version>/` (ngoài Git), `dataset_version` = `sim-smoke-v1` hoặc `sim-v1`.

Quy ước chung:

- Namespace `sim_v1`. ID: `sim:user:<n>`, `sim:merchant:<n>`, `sim:dish:<n>`, `sim:ad:<n>`,
  `sim:driver:<n>`, `sim:event:<n>`, phiên `sim:session:<user n>:<k>`, request Food
  `sim:food_req:<n>`.
- Giờ địa phương cố định `+07:00`; ngày 1 = thứ Hai 2026-01-05 00:00 `+07:00`; ngày 43 = đầu tuần 7.
- Cùng `seed` và cùng cấu hình → cùng byte cho mọi file đầu ra (không ghi thời điểm chạy vào file),
  trừ `data_report.json` (có thời điểm, thời gian chạy và phiên bản thư viện; mục "data_report" ở cuối).
- Cách chạy, thứ tự dùng `rng` và kiểm tra sau xuất: mục "Chạy" ở cuối tài liệu.

## Pretrain (`pretrain/`) — schema `omnibehavior_pretraining_v2`

Code: `src/simulation/export_pretrain.py` (`export_pretrain(sim, out_dir)`), quy tắc history và
profile: `src/simulation/history.py`. Đọc bằng `src/foundation_input` hiện có, không cần sửa:

```python
from src.foundation_input.loader import iter_normalized_samples

for sample in iter_normalized_samples("<out>/sim-v1/pretrain", "train"):  # hoặc "validation"
    ...  # sample.dataset_name == "sim_v1"
```

Bàn giao cho dự án con 2 (đã xử lý trên branch `feat/sim-foundation`; chi tiết, lệnh chạy và kết
quả smoke ở `docs/sim_foundation_team_notes.md`):

- Allowlist mặc định của tokenizer (`DEFAULT_CONTEXT_ALLOWLIST`, `src/foundation_input/tokenizer.py`)
  **không có** scenario `Ride` / `Food`. **Đã đóng:** `src/training/pretrain.py`
  (`tokenizer_config_for_manifest`) hợp nhất `context_allowlist` của `preprocessing_config.json`
  (mục dưới) vào allowlist mặc định trước khi fit vocab, nên context Ride/Food được giữ; vocab nạp
  sẵn qua `vocab_path` mà thiếu khóa của gói thì bị từ chối (`check_vocab_allowlist`). Config:
  `configs/pretrain_sim.yaml` (CPU local, `sim-smoke-v1`), `configs/pretrain_sim_kaggle.yaml`
  (GPU Kaggle, `sim-v1`); downstream `configs/food_recommendation_sim.yaml` và
  `configs/ad_optimization_sim.yaml` (`data.format: sim_v1`).
- `history_group_truncated` luôn `false` (quyết định R11), nên target next-action của
  `foundation_input` (`dataset.encode_sample`) dùng được cho mọi snapshot.

### Thư mục và tên file

```
pretrain/
  manifest.json               ghi sau cùng; danh sách shard + số dòng + SHA-256
  preprocessing_config.json   max_history_events, ranh giới split, allowlist context
  preview.json                3 mẫu dựng lại (đọc nhanh bằng mắt)
  events/<user>.parquet       mỗi dòng = một bước phễu (action) của user
  snapshots/<user>.parquet    mỗi dòng = một mẫu (user, cutoff)
```

- Tên file shard = `user_id` với `:` thay bằng `_`: `sim:user:0` → `sim_user_0.parquet`
  (cả `events/` và `snapshots/`). Mọi user của world đều có hai file, kể cả khi rỗng.
- Parquet nén `zstd` mức 3; schema Arrow mang metadata `schema_version`.
- `export_pretrain` từ chối ghi đè (`FileExistsError`) nếu `pretrain/` đã có nội dung.

### Phạm vi thời gian và split

Mọi cột thời gian là **UTC không kèm múi giờ** (naive UTC), cùng hệ với `cutoff_time` và
`preprocessing_config.splits`. Chỉ xuất event có thời điểm trước đầu ngày 43
(`2026-02-15T17:00:00` naive UTC), tức tuần 1–6.

| Split | Ngày cutoff (lúc 23:59:59 `+07:00`) | `cutoff_time` (naive UTC) |
|---|---|---|
| `train` | 6, 9, 12, …, 33 | `2026-01-10T16:59:59`, …, `2026-02-06T16:59:59` |
| `validation` | 37, 40 | `2026-02-10T16:59:59`, `2026-02-13T16:59:59` |
| `test` | không có | — |

`preprocessing_config.splits` = `start_inclusive` `2026-01-04T17:00:00` (đầu ngày 1),
`train_end_exclusive` `2026-02-06T17:00:00` (đầu ngày 34), `validation_end_exclusive` và
`test_end_exclusive` `2026-02-15T17:00:00` (đầu ngày 43). Tổng quát theo `SimConfig`:
ngày kết thúc = `7·(weeks−2)+1`, train kết thúc trước 3 chu kỳ snapshot cuối
(`snapshot_every_days`), cutoff train từ ngày `2·snapshot_every_days`.

### `events/<user>.parquet`

Mỗi phiên (ride, food hoặc ads) là một bản ghi nguồn (`source_record_id`); mỗi bước phễu là một
action trong bản ghi đó, theo thứ tự. Dòng sắp theo thời gian (`ts`, chỉ số phiên, `action_index`);
các phiên của một user không chồng lấn nên mọi phiên nằm liền nhau.

| Cột | Kiểu Arrow | Ý nghĩa | Ví dụ |
|---|---|---|---|
| `row_index` | `int32` | Vị trí dòng trong file (0..n−1) | `25` |
| `event_id` | `string` | ID event toàn cục | `"sim:event:25"` |
| `user_id` | `string` | User | `"sim:user:0"` |
| `source_record_id` | `string` | ID phiên | `"sim:session:0:7"` |
| `source_record_index` | `int32` | Số thứ tự phiên của user (tăng theo thời gian) | `7` |
| `action_index` | `int32` | Bước trong phiên, từ 0 | `3` |
| `source_action_count` | `int32` | Số bước thật của phiên | `4` |
| `source_timestamp` | `string` | Giờ địa phương gốc, có offset | `"2026-01-21T20:17:47+07:00"` |
| `event_time` | `string` | ISO naive UTC | `"2026-01-21T13:17:47"` |
| `time_precision` | `string` | Luôn `"second"` | `"second"` |
| `time_lower_bound` | `timestamp[us]` | = thời điểm event | `2026-01-21 13:17:47` |
| `time_upper_bound` | `timestamp[us]` | = thời điểm event | `2026-01-21 13:17:47` |
| `upper_bound_exclusive` | `bool` | Luôn `false` | `false` |
| `eligibility_time` | `timestamp[us]` | = thời điểm event | `2026-01-21 13:17:47` |
| `eligibility_inclusive` | `bool` | Luôn `true` | `true` |
| `scenario` | `string` | `Ride`, `Food`, `Advertisement` (service `ride`/`food`/`ads`) | `"Food"` |
| `event_type` | `string` | Bước phễu (bảng dưới) | `"order"` |
| `context_json` | `large_string` | JSON ngữ cảnh (bảng dưới) | xem ví dụ |
| `action_json` | `large_string` | JSON hành động (bảng dưới) | `{"type":"order","order_amount":68000.0,"delivery_distance_km":5.59}` |

`context_json` / `action_json` theo scenario (JSON gọn, UTF-8, giữ thứ tự khóa):

| Scenario | `event_type` (thứ tự phễu) | Khóa `context_json` | `action_json` |
|---|---|---|---|
| `Ride` | `search_destination` → `get_quote` → `book_ride` → `complete_ride` | `pickup_geohash, dropoff_geohash, pickup_zone_type, dropoff_zone_type, vehicle_type`; thêm `driver_id` từ `book_ride` | `type, distance_km`; thêm `fare` từ `get_quote`; thêm `wait_minutes` từ `book_ride` |
| `Food` | `search` → `view_merchant` (1–3) → `click_dish` (1–2) → `order` | `search`: chỉ `search_query`; `view_merchant`: `merchant_id, merchant_name, cuisine, price_level`; `click_dish`/`order`: thêm `dish_id, dish_name` | `{"type": <event_type>}`; `order` thêm `order_amount, delivery_distance_km` |
| `Advertisement` | `ad_impression` → `ad_click` (nếu click) | `product` (= ad_id), `industry_main` (ngành hàng), `industry_sub` (nhà quảng cáo), `caption` (tiêu đề) | `{"type": "impression"}` / `{"type": "click"}` |

Phiên bỏ ngang chỉ có các bước đầu (vd. ride chỉ `search_destination`), `source_action_count`
luôn bằng số bước thật của phiên. Tiền tệ VND, khoảng cách km, chờ tài xế phút.

### `snapshots/<user>.parquet`

| Cột | Kiểu Arrow | Ý nghĩa | Ví dụ |
|---|---|---|---|
| `sample_id` | `string` | `<user_id>:<cutoff_time ISO>:closed` | `"sim:user:0:2026-01-31T16:59:59:closed"` |
| `user_id` | `string` | User | `"sim:user:0"` |
| `cutoff_time` | `timestamp[us]` | 23:59:59 địa phương của ngày cutoff, naive UTC | `2026-01-31 16:59:59` |
| `cutoff_inclusive` | `bool` | Luôn `true`: event lúc đúng `cutoff_time` được phép | `true` |
| `split` | `string` | `train` / `validation` | `"train"` |
| `history_start` | `int32` | Dòng đầu của history trong file events | `4` |
| `history_stop` | `int32` | Dòng sau dòng cuối (nửa mở) | `33` |
| `history_event_ids` | `list<string>` | = `event_id` của dòng `[start, stop)` | `["sim:event:4", …]` |
| `history_order_known` | `bool` | Luôn `true` | `true` |
| `history_truncated` | `bool` | Quy tắc của loader: `start > 0` và `stop − start ≥ 30` | `false` |
| `history_group_truncated` | `bool` | Luôn `false` (R11): history chỉ gồm trọn phiên, không nhóm nào bị cắt đôi; `true` sẽ tắt target next-action của loader | `false` |
| `profile_json` | `string` | JSON profile (dưới) | xem ví dụ |

Quy tắc history (`history.last_history`):

- Chỉ gồm event của phiên mà **mọi** bước có thời điểm ≤ `cutoff_time` (tức trước đầu ngày kế
  tiếp); phiên đang dở tại cutoff bị loại (`completed_sessions_before`).
- Lấy tối đa `max_history_events = 30` event gần nhất, **giữ trọn phiên**: nếu 30 event gần nhất
  bắt đầu giữa một phiên thì bỏ cả phiên đó (khi đó `stop − start < 30`). Gói pretrain vẫn ghi
  `history_group_truncated = false` vì không phiên nào bị cắt đôi (R11; gói Food giữ cờ của
  `last_history`, mục Food).
- History là một đoạn liền `[history_start, history_stop)` của file events.
- Snapshot có history rỗng (user chưa có phiên hoàn tất nào) thì **không** được ghi; vì vậy mọi
  snapshot có 1–30 event.

### `profile_json`

Tính từ event trước cutoff **chỉ của user đó** (`history.profile_at`); cửa sổ 30 ngày =
`(cutoff_time − 30 ngày, cutoff_time]` = 30 ngày địa phương kết thúc ở ngày cutoff, bị cắt ở ngày 1
(không ngoại suy). Nhóm người, khẩu vị, sở thích quảng cáo, mức dùng app và ý định là ẩn, không
bao giờ xuất.

| Khóa | Kiểu | Ý nghĩa | Ví dụ |
|---|---|---|---|
| `as_of_time` | string ISO | = `cutoff_time` (naive UTC); tokenizer bỏ qua | `"2026-01-31T16:59:59"` |
| `age_group` | string | `18-24`, `25-34`, `35-44`, `45-54`, `55+` | `"35-44"` |
| `saved_home_geohash` | string | Geohash 6 ký tự của khu nhà | `"w3gvhx"` |
| `saved_work_geohash` | string / null | Geohash khu làm/học **ban đầu** (null nếu không có); đổi chỗ làm không cập nhật trường này | `"w3gugf"` |
| `avg_monthly_spend` | float | Tổng cước `complete_ride` + `order_amount` của `order` trong cửa sổ (VND) | `981000.0` |
| `most_used_service` | string / null | `ride` hoặc `food`, dịch vụ có nhiều phiên hơn trong cửa sổ (mọi độ sâu phễu; hòa → `ride`; ads không tính); null nếu không có | `"ride"` |
| `ride_count_30d` | int | Số `complete_ride` | `5` |
| `food_order_count_30d` | int | Số `order` | `1` |
| `ad_click_count_30d` | int | Số `ad_click` | `1` |
| `active_days_30d` | int | Số ngày địa phương có ít nhất một event | `8` |

### `manifest.json`

| Khóa | Ý nghĩa |
|---|---|
| `schema_version` | `"omnibehavior_pretraining_v2"` |
| `status` | `"complete"` (manifest ghi sau cùng) |
| `source_dataset` | `"sim_v1"` → `DatasetManifest.dataset_name == "sim_v1"` |
| `dataset_version`, `generator_version`, `seed` | nguồn gốc lần chạy |
| `sim_config_sha256` | SHA-256 của `SimConfig` (`config.config_sha256`) |
| `config_sha256` | SHA-256 của `preprocessing_config.json` (loader kiểm tra) |
| `preview_sha256`, `pyarrow_version` | |
| `total_users`, `total_events`, `total_snapshots`, `snapshots_by_split` | tổng |
| `shards[]` | theo thứ tự user: `user_id, events_file, snapshots_file, event_rows, snapshot_rows, snapshots_by_split {train, validation, test}, events_bytes, snapshots_bytes, events_sha256, snapshots_sha256` |

### `preprocessing_config.json`

`schema_version`, `max_history_events = 30`, `splits` (trên), `context_allowlist` theo scenario
(đúng các khóa ở bảng context), `profile_fields`, `snapshot_days`, `events_end_exclusive`,
`source_utc_offset_hours = 7`, `timezone = null`, nén `zstd`/3, cùng mô tả văn bản của cửa sổ
profile, cách chọn history và nhịp snapshot.

### Ví dụ một dòng

`events/sim_user_0.parquet` (preset `smoke`), bước `order` của một phiên Food 4 bước:

```json
{"row_index": 25, "event_id": "sim:event:25", "user_id": "sim:user:0",
 "source_record_id": "sim:session:0:7", "source_record_index": 7, "action_index": 3,
 "source_action_count": 4, "source_timestamp": "2026-01-21T20:17:47+07:00",
 "event_time": "2026-01-21T13:17:47", "time_precision": "second",
 "time_lower_bound": "2026-01-21 13:17:47", "time_upper_bound": "2026-01-21 13:17:47",
 "upper_bound_exclusive": false, "eligibility_time": "2026-01-21 13:17:47",
 "eligibility_inclusive": true, "scenario": "Food", "event_type": "order",
 "context_json": "{\"merchant_id\":\"sim:merchant:35\",\"merchant_name\":\"Cơm Gà Bình 35\",\"cuisine\":\"cơm văn phòng\",\"price_level\":1,\"dish_id\":\"sim:dish:285\",\"dish_name\":\"Cơm cá kho\"}",
 "action_json": "{\"type\":\"order\",\"order_amount\":68000.0,\"delivery_distance_km\":5.59}"}
```

`snapshots/sim_user_0.parquet`, một snapshot đã bỏ nguyên phiên cũ nhất (một chuyến xe 4 bước):

```json
{"sample_id": "sim:user:0:2026-01-31T16:59:59:closed", "user_id": "sim:user:0",
 "cutoff_time": "2026-01-31 16:59:59", "cutoff_inclusive": true, "split": "train",
 "history_start": 4, "history_stop": 33,
 "history_event_ids": ["sim:event:4", "sim:event:5", "…29 id…"],
 "history_order_known": true, "history_truncated": false, "history_group_truncated": false,
 "profile_json": "{\"as_of_time\":\"2026-01-31T16:59:59\",\"age_group\":\"35-44\",\"saved_home_geohash\":\"w3gvhx\",\"saved_work_geohash\":\"w3gugf\",\"avg_monthly_spend\":981000.0,\"most_used_service\":\"ride\",\"ride_count_30d\":5,\"food_order_count_30d\":1,\"ad_click_count_30d\":1,\"active_days_30d\":8}"}
```

Quy mô preset `smoke` (500 user): 97.474 event, 4.988 snapshot train + 1.000 validation,
~12 MB trên đĩa (events 8,7 MB, snapshots 3,0 MB).

## Food (`food/`) — gợi ý quán

Code: `src/simulation/export_food.py` — `build_food_requests(sim, rng) -> list[FoodRequest]`
(chọn request và ứng viên) rồi `export_food(sim, requests, out_dir)` (ghi file). Oracle (gói
`oracle/`) dùng đúng `FoodRequest.candidate_ids`: xác suất thật = softmax của
`behavior.food_logits(world, session, candidate_ids)`.

### Thư mục và tên file

```
food/
  samples.jsonl       mỗi dòng = một request (mở rộng định dạng samples.jsonl của Meituan)
  merchants.parquet   catalog quán, chỉ thuộc tính công khai
  manifest.json       ghi sau cùng; nguồn gốc, SHA-256 + số dòng từng file, split, chính sách ứng viên
```

- `samples.jsonl`: UTF-8, JSON gọn (không khoảng trắng, không escape tiếng Việt), mỗi dòng kết
  thúc bằng `\n`, thứ tự khóa cố định như bảng dưới; các dòng theo thứ tự `request_id`.
- `merchants.parquet` nén `zstd` mức 3, metadata Arrow `schema_version = "1"`.
- `export_food` từ chối ghi đè (`FileExistsError`) nếu `food/` đã có nội dung.

### Request, mốc dự đoán và split

- Mỗi request = một phiên Food **có `order`** (`Simulation.food_sessions`). Mốc dự đoán là đầu
  phiên: `cutoff_time` = 1 giây trước event đầu tiên (`search`) của phiên. Mọi event của phiên đó
  (kể cả các `view_merchant` quán khác) đều **không** có trong history; positive = quán được đặt,
  `label_time` = thời điểm `order` (luôn sau `cutoff_time`).
- Split theo tuần địa phương của `cutoff_time`: `train` = tuần 1–6 (trước
  `2026-02-16T00:00:00+07:00`), `validation` = tuần 7 (trước `2026-02-23T00:00:00+07:00`),
  `test` = tuần 8. Phiên Food luôn kết thúc trong ngày nên `label_time` cùng split.
- Số request: `config.food_requests = None` (`smoke`) → lấy tất cả; `full` → lấy ngẫu nhiên không
  lặp 60.000 / 10.000 / 10.000 mỗi split bằng `rng` (split có ít hơn thì lấy hết). Nếu khác `None`,
  `food_requests` phải có đúng ba khóa `train`, `validation`, `test`.
- `request_id` = `sample_id` = `sim:food_req:<n>`, `n` từ 0 theo thứ tự thời gian của `cutoff_time`.

### Ứng viên — `candidate_source = "sim-popularity-matched-home-work-v1"`

Đúng 10 quán khác nhau: quán được đặt + 9 quán khác, chọn như sau (spec §5).

1. **Đơn vị phổ biến (phân hoạch cố định, chung cho mọi quán).** Đếm số `order` của mỗi quán
   trong tuần 1–6 (`order` trước `2026-02-16T00:00:00+07:00`, mọi user). Xếp hạng thứ tự (số đơn
   tăng dần, hòa → thứ tự catalog), thập phân vị = `hạng · 10 // số quán`, đơn vị = **cặp thập phân
   vị** {0,1}, {2,3}, {4,5}, {6,7}, {8,9}. Tập chọn = đơn vị của positive trừ positive, nên mọi
   ứng viên cùng hoặc kề thập phân vị và thuộc cùng một cặp. Không dùng cửa sổ quanh từng quán (cửa
   sổ như vậy để lộ positive: một ứng viên mà cửa sổ của chính nó khác dải của tập thì không thể là
   positive). **Phương án cuối (last resort)**, chỉ khi có cặp chứa < 10 quán (catalog < 50 quán):
   gộp các cặp kề nhau từ dưới lên thành đơn vị ≥ 10 quán, phần dư ở đầu trên nhập vào đơn vị ngay
   dưới (vẫn là một phân hoạch chung); catalog < 10 quán → `ValueError`. `smoke` (60 quán, 12 quán
   một cặp) và `full` (600 quán) không bao giờ cần phương án cuối.
2. **Cụm theo khu, kích thước hoán đổi được (quyết định R8).** Neo = khu của quán được đặt, khu
   nhà, khu làm việc/học (chỗ làm đã lưu, tức `saved_work_geohash`; người không có chỗ làm: khu TTTM
   gần nhà nhất). Mỗi neo **khác nhau** có một cụm; quán được đặt tính vào cụm của khu nó.
   - 3 neo khác nhau: kích thước cụm (4, 3, 3) gán cho (khu quán được đặt, nhà, chỗ làm) theo một
     hoán vị ngẫu nhiên đều rút bằng `rng` → cụm của quán được đặt có 3 âm (xác suất 1/3) hoặc 2 âm,
     hai cụm kia 3/4 âm; mỗi cụm có xác suất 1/3 là cụm 4.
   - 2 neo khác nhau (khu quán = nhà hoặc = chỗ làm, hoặc nhà = chỗ làm): (5, 5) → cụm của quán
     được đặt có 4 âm, cụm kia 5 âm.
   - 1 neo (cả ba trùng): một cụm 9 âm.
3. **Rút âm ngẫu nhiên theo khoảng cách (quyết định R10).** τ = `SimConfig.food_candidate_tau_km`
   (mặc định 2,0 km, hiệu chỉnh ở Task 8 — trước đó 1,5 km; xem "Độ khó"). Mỗi cụm có một
   **tâm**: khu nhà / chỗ làm (TTTM) cho cụm của chúng; cụm của quán được đặt có **tâm ẩn** rút
   trên cả 50 khu (kể cả khu của quán) với xác suất ∝ exp(−d(khu, khu quán được đặt)/τ). Âm của
   mỗi cụm (theo thứ tự neo trên) rút **không hoàn lại** từ cặp thập phân vị (trừ quán được đặt
   và quán đã chọn), lần lượt từng quán với xác suất ∝ exp(−d(khu của quán, tâm cụm)/τ) (`export_food.decay_sample`, cuộc đua mũ: k khóa nhỏ nhất
   `log E + d/τ`, `E ~ Exp(1)`). d = haversine giữa tâm khu. Vì vậy quán được đặt không nằm đúng tâm
   của một lần điền tất định và không âm nào là phần "tràn" đoán trước được; không có quy tắc "âm
   nằm trong khu neo" — chỉ đúng về trung bình (xem dưới).
4. **Thứ tự** 10 ứng viên được xáo bằng `rng`; `labels` theo đúng thứ tự đó.

`rng` được dùng theo thứ tự: lấy mẫu từng split (chỉ khi có `food_requests`), rồi với mỗi request
hoán vị kích thước cụm (chỉ khi có 3 neo khác nhau), tâm ẩn của cụm quán được đặt, âm của từng cụm
(thứ tự neo) và một hoán vị xáo ứng viên → cùng `rng` cho cùng kết quả. `manifest.candidate_stats`
ghi `tau_km`, tỷ lệ dùng phương án cuối, số request theo số neo khác nhau và — **chỉ để báo cáo** —
số âm nằm trong một khu neo và trong bán kính 3 km của một khu neo.

Đã đo gì (NDCG@10 trên request validation + test; "chỉ ứng viên" = id ứng viên + catalog công khai
+ số đơn tuần 1–6, không history, không profile). Kiểm tra lối tắt áp dụng ở **mật độ catalog
`full`** (600 quán; quyết định R9); số đo trên catalog `smoke` chỉ để tham khảo, không dùng làm số
thí nghiệm. Đo lại ở Task 8 với giá trị mặc định đã hiệu chỉnh (τ = 2,0 km, `NoiseConfig` ở mục
"Độ khó"), seed 42, bằng chính các ranker của `tests/simulation/test_export_food.py`
(`CandidateOnlyRankers`); hai ranker phát lại chạy trên 1/3 số request (phổ biến trên cùng mẫu:
`full` 0,510, `smoke` 0,494).

| Ranker | preset `full` (600 quán, 20.000 user; 20.000 req) | `smoke` (60 quán; 2.372 req) |
|---|---|---|
| phổ biến (số đơn tuần 1–6) | 0,509 | 0,493 |
| chỉ ứng viên: khớp cặp; ≤ 3 quán gần nhất trong tập; mật độ ≤ 1,5 / ≤ 3 km / mềm | 0,509; 0,473; 0,482 / 0,487 / 0,488 | 0,493; 0,509; 0,448 / 0,448 / 0,451 |
| chỉ ứng viên, biết quy tắc: độ "bão hòa" theo thứ tự điền; trung tâm nhất (TB 9 / 3 gần nhất) | 0,480; 0,473 / 0,490 | 0,534; 0,454 / 0,456 |
| chỉ ứng viên, biết quy tắc: phát lại điền tất định (đếm / nhị phân); likelihood mô hình R10 | 0,510 / 0,510; 0,491 | 0,511 / 0,516; 0,513 |
| chiều ngược: ít trung tâm nhất; ít khả dĩ nhất (R10) | 0,427; 0,412 | 0,453; 0,404 |
| profile (hợp lệ, R7): gần khu nhà đã lưu | 0,665 | 0,795 |
| oracle (`food_logits` thật) | 0,898 | 0,957 |

Ở `full`, mọi ranker chỉ-ứng-viên trên đều ≤ phổ biến (giới hạn spec §8: + 0,05); trên `smoke`
cao nhất là độ bão hòa (+ 0,041). Đây là các ranker **đã thử**, không phải chứng minh không có lối
tắt nào khác: một ranker **học** chỉ từ ứng viên (LightGBM lambdarank; đặc trưng: loại món, mức
giá, loại khu, khu, log số đơn, hạng phổ biến trong tập, hình học của tập — số ứng viên khác trong
1 / 3 km, khoảng cách trung bình/trung vị tới 9 ứng viên kia, số ứng viên cùng ô geohash; học trên
split train, đo trên test) đạt **0,551 = phổ biến + 0,041** ở `full` (chỉ phần hình học + phổ
biến: + 0,030; chỉ thuộc tính quán + phổ biến: + 0,004) — trong giới hạn nhưng biên mỏng. Phần vượt đến từ cấu
trúc cụm (khu nhà / chỗ làm có loại khu đặc trưng, cụm của quán được đặt có tâm ẩn) và giảm khi τ
tăng: cùng mô hình đạt + 0,058 với τ = 1,5 km và hệ số cũ, + 0,071 với τ = 1,5 km và khoảng cách
1,0 (bảng hiệu chỉnh ở mục "Độ khó"). Ranker theo profile (nhà/chỗ làm đã lưu) cao hơn phổ biến là
tín hiệu dài hạn hợp lệ (R7).

Đầu vào **không** chứa vị trí hiện tại hay khoảng cách từ vị trí hiện tại (vị trí hiện tại chỉ suy
ra được từ history, vd. điểm trả của chuyến xe gần nhất); candidate chỉ có 4 khóa dưới.

### `samples.jsonl`

| Khóa | Kiểu | Ý nghĩa | Ví dụ |
|---|---|---|---|
| `sample_id` | string | = `label_metadata.request_id` | `"sim:food_req:73"` |
| `input.user_id` | string | User | `"sim:user:373"` |
| `input.cutoff_time` | string ISO `+07:00` | Mốc dự đoán; history chỉ có event **trước hẳn** mốc này | `"2026-01-05T15:05:54+07:00"` |
| `input.profile` | object | Profile như gói pretrain (`history.profile_at`, cửa sổ 30 ngày trước `cutoff_time`, chỉ event `< cutoff_time`) + `as_of_time` = `cutoff_time` (ISO `+07:00`) | xem ví dụ |
| `input.history` | list | Tối đa 30 event mọi dịch vụ, cũ → mới, giữ trọn phiên, chỉ phiên đã hoàn tất trước `cutoff_time` (`history.last_history`); có thể rỗng | xem dưới |
| `input.history_truncated` | bool | Còn event cũ hơn (của phiên hoàn tất) bị bỏ vì giới hạn 30 | `false` |
| `input.history_group_truncated` | bool | Đã bỏ nguyên phiên cũ nhất để không cắt đôi (như gói pretrain) | `false` |
| `input.context` | object | `{hour, weekday}` giờ địa phương của `cutoff_time`; `hour` 0–23, `weekday` 0 = thứ Hai | `{"hour": 15, "weekday": 0}` |
| `input.candidates` | list (10) | Mỗi phần tử đúng 4 khóa `{item_id, cuisine, price_level, geohash}`: `item_id` = `merchant_id`, `price_level` 0/1/2 (thấp/vừa/cao), `geohash` 6 ký tự của khu quán | `{"item_id": "sim:merchant:22", "cuisine": "pizza", "price_level": 0, "geohash": "w3guuc"}` |
| `labels` | list (10) | `{item_id, relevance}` cùng thứ tự `candidates`; đúng một `relevance = 1` (quán được đặt) | `{"item_id": "sim:merchant:22", "relevance": 1}` |
| `label_metadata.request_id` | string | = `sample_id` | `"sim:food_req:73"` |
| `label_metadata.split` | string | `train` / `validation` / `test` | `"train"` |
| `label_metadata.label_time` | string ISO `+07:00` | Thời điểm `order` (> `cutoff_time`) | `"2026-01-05T15:08:04+07:00"` |
| `label_metadata.candidate_source` | string | `"sim-popularity-matched-home-work-v1"` | |
| `label_metadata.observation_complete` | bool | Luôn `true` (simulator quan sát đủ phiên) | `true` |
| `label_metadata.prediction_anchor` | string | Luôn `"session_start"` | |
| `label_metadata.positive_action` | string | Luôn `"order"` | |

Mỗi event trong `input.history`: `{event_id, event_time, service, event_type, entity_id,
attributes}` — `event_time` ISO `+07:00`, `service` ∈ `ride`/`food`/`ads`, `event_type` và
`entity_id` như event gốc (bảng phễu ở mục Pretrain; `entity_id`: ride `search_destination`/
`get_quote` = geohash điểm trả, `book_ride`/`complete_ride` = `driver_id`; food `search` = câu tìm,
`view_merchant`/`order` = `merchant_id`, `click_dish` = `dish_id`; ads = `ad_id`). `attributes` = các
khóa context công khai (đúng allowlist `context_json` của gói pretrain) + trường số của action
(`fare, distance_km, wait_minutes, order_amount, delivery_distance_km`; bỏ `type`). Không có thuộc
tính ẩn nào (nhóm người, khẩu vị, ý định, vị trí hiện tại, xác suất thật).

Đọc được bằng `src.downstream.food_samples.validate_food_samples(rows, candidate_policy)` hiện có
(đã kiểm thử): khóa bắt buộc, `labels` khớp `candidates`, history `< cutoff_time`,
`label_time > cutoff_time`, split nối tiếp theo thời gian, timestamp có offset.

### `merchants.parquet`

Mỗi dòng một quán, theo thứ tự catalog (`sim:merchant:0..n−1`).

| Cột | Kiểu Arrow | Ý nghĩa | Ví dụ |
|---|---|---|---|
| `merchant_id` | `string` | ID quán | `"sim:merchant:0"` |
| `merchant_name` | `string` | Tên quán | `"Green Bowl Nam 0"` |
| `cuisine` | `string` | Loại món (12 loại, `config.CUISINES`) | `"healthy/salad"` |
| `price_level` | `int32` | 0/1/2 = thấp/vừa/cao | `1` |
| `geohash` | `string` | Geohash 6 ký tự của khu quán (mỗi khu một geohash) | `"w3gvsu"` |
| `zone_type` | `string` | Loại khu: `residential, office, university, nightlife, mall, transport_hub` | `"office"` |

Độ phổ biến nền (`base_popularity`) là tham số ẩn của công thức chọn quán, **không** xuất.

### `manifest.json`

| Khóa | Ý nghĩa |
|---|---|
| `dataset_version`, `generator_version`, `seed`, `sim_config_sha256`, `pyarrow_version` | nguồn gốc lần chạy |
| `schema_version` | `1` |
| `source_dataset` | `"sim_v1"` |
| `timezone`, `utc_offset` | `"Asia/Ho_Chi_Minh"`, `"+07:00"` |
| `item_kind` | `"merchant"` |
| `files.samples`, `files.merchants` | `{path, sha256, row_count, bytes}` (`path` tương đối trong `food/`) |
| `split` | `{kind: "temporal", basis: "cutoff_time", train_end: "2026-02-16T00:00:00+07:00", valid_end: "2026-02-23T00:00:00+07:00"}` (cận trên, không bao gồm) |
| `request_counts` | `{train, validation, test}` |
| `candidate_policy`, `candidates_per_request` | `"sim-popularity-matched-home-work-v1"`, `10` |
| `candidate_stats` | `popularity_unit = "decile_pair"`, `tau_km`, `last_resort_requests` (đơn vị là cặp đã gộp), `requests_by_anchor_count` {số neo khác nhau → số request}, `merged_anchor_requests` (< 3 neo khác nhau), `negatives`, `negatives_in_an_anchor_zone`, `negatives_within_3km_of_an_anchor` (hai số cuối chỉ để báo cáo) |
| `popularity_end_exclusive` | cận trên khoảng đếm đơn cho độ phổ biến (= đầu tuần 7) |
| `max_history_events`, `prediction_anchor`, `positive_action` | `30`, `"session_start"`, `"order"` |

### Ví dụ một dòng (rút gọn)

Preset `smoke`; history thật có 10 event (chuyến xe 4 bước, 1 ad, phiên Food 5 bước), ở đây giữ 2;
10 ứng viên/nhãn, ở đây giữ 3 (vị trí 1, 2 và 10 — positive):

```json
{"sample_id": "sim:food_req:73",
 "input": {"user_id": "sim:user:373", "cutoff_time": "2026-01-05T15:05:54+07:00",
  "profile": {"as_of_time": "2026-01-05T15:05:54+07:00", "age_group": "25-34",
   "saved_home_geohash": "w3guf5", "saved_work_geohash": "w3guuq", "avg_monthly_spend": 133000.0,
   "most_used_service": "ride", "ride_count_30d": 1, "food_order_count_30d": 1,
   "ad_click_count_30d": 0, "active_days_30d": 1},
  "history": [
   {"event_id": "sim:event:98247", "event_time": "2026-01-05T08:47:51+07:00", "service": "ride",
    "event_type": "complete_ride", "entity_id": "sim:driver:291",
    "attributes": {"pickup_geohash": "w3guf5", "dropoff_geohash": "w3guuq",
     "pickup_zone_type": "residential", "dropoff_zone_type": "university", "vehicle_type": "bike",
     "driver_id": "sim:driver:291", "fare": 86000.0, "distance_km": 14.14, "wait_minutes": 6.8}},
   {"event_id": "sim:event:98253", "event_time": "2026-01-05T12:51:42+07:00", "service": "food",
    "event_type": "order", "entity_id": "sim:merchant:49",
    "attributes": {"merchant_id": "sim:merchant:49", "merchant_name": "Gà Sốt Thảo 49",
     "cuisine": "gà rán", "price_level": 0, "dish_id": "sim:dish:393",
     "dish_name": "Cánh gà chiên mắm", "order_amount": 47000.0, "delivery_distance_km": 3.6}}],
  "history_truncated": false, "history_group_truncated": false,
  "context": {"hour": 15, "weekday": 0},
  "candidates": [
   {"item_id": "sim:merchant:19", "cuisine": "lẩu", "price_level": 1, "geohash": "w3guuc"},
   {"item_id": "sim:merchant:57", "cuisine": "gà rán", "price_level": 1, "geohash": "w3guu6"},
   {"item_id": "sim:merchant:22", "cuisine": "pizza", "price_level": 0, "geohash": "w3guuc"}]},
 "labels": [{"item_id": "sim:merchant:19", "relevance": 0},
  {"item_id": "sim:merchant:57", "relevance": 0}, {"item_id": "sim:merchant:22", "relevance": 1}],
 "label_metadata": {"request_id": "sim:food_req:73", "split": "train",
  "label_time": "2026-01-05T15:08:04+07:00",
  "candidate_source": "sim-popularity-matched-home-work-v1", "observation_complete": true,
  "prediction_anchor": "session_start", "positive_action": "order"}}
```

Quy mô preset `smoke` (500 user, 60 quán): 9.214 request (train 6.842 / validation 1.104 /
test 1.268), `samples.jsonl` ~97 MB (~10,6 KB/dòng, trung bình 27 event history), xuất ~5 giây.
Không request nào cần phương án cuối; 53% request có neo trùng (47% có 3 neo khác nhau). Chỉ để
báo cáo: 6,7% âm nằm trong một khu neo, 19% trong 3 km của một khu neo (mỗi cặp chỉ 12 quán trên
50 khu). Ở mật độ `full` (600 quán, đo trên 2.000 user): 87% neo trùng, 30% âm trong một khu neo,
64% trong 3 km của một khu neo. `full` (80.000 request) ước ~0,85 GB `samples.jsonl`.

## Ads (`ads/`) — click prediction trên quảng cáo đã hiển thị

Code: `src/simulation/export_ads.py` — `build_ad_requests(sim, rng) -> list[AdRequest]` rồi
`export_ads(sim, requests, out_dir)`. Oracle dùng `AdImpression.true_probability` của chính
impression đó. Cột khớp tập cột reader Ads (`feat/ad-optimization`, `src/downstream/ad_inputs.py`)
yêu cầu; gói này **không** uốn theo hằng số Taobao. Dự án con 2 đã tổng quát hóa reader: quy tắc
định danh chọn theo `data.format` (`taobao` mặc định, hoặc `sim_v1`: event mọi loại, impression tra
qua `source_row_index`, cận split không bao gồm, checkpoint ràng buộc theo `sim_config_sha256`) —
xem `docs/sim_foundation_team_notes.md`.

### Thư mục và tên file

```
ads/
  profiles.parquet          trường tĩnh, mỗi user một dòng (đủ mọi user)
  ad_catalog.parquet        catalog quảng cáo (đủ mọi quảng cáo)
  events.parquet            event của MỌI dịch vụ (ride, food, ads), mọi user
  ad_opportunities.parquet  mỗi dòng một request = một impression
  manifest.json             ghi sau cùng; nguồn gốc, SHA-256 + số dòng từng file
```

Parquet nén `zstd` mức 3, metadata Arrow `schema_version = "1"`; timestamp là **epoch giây**
(`int64`). `export_ads` từ chối ghi đè (`FileExistsError`) nếu `ads/` đã có nội dung.

### Request, nhãn và split

- Mỗi request = một `ad_impression` (`Simulation.ad_impressions`), **một** ứng viên (quảng cáo đã
  được hiển thị), nhãn `clicked` rút từ xác suất thật khi mô phỏng. **Không** lấy mẫu âm lại.
- `cutoff_timestamp = exposure_timestamp` = thời điểm impression. Split theo tuần địa phương của
  `exposure_timestamp` (như Food: tuần 1–6 train, 7 validation, 8 test).
- Số request: `config.ads_requests = None` (`smoke`) → lấy tất cả; `full` → lấy ngẫu nhiên không lặp
  200.000 / 30.000 / 30.000 mỗi split bằng `rng` (split có ít hơn thì lấy hết); phải có đúng ba
  khóa `train`, `validation`, `test`. Thứ tự request: theo split (train, validation, test), trong
  split theo thời gian; `request_id` = `sim:ad_req:<n>`, `n` từ 0.

### `ad_opportunities.parquet`

| Cột | Kiểu | Ý nghĩa |
|---|---|---|
| `request_id` | string | `sim:ad_req:<n>` |
| `source_row_index` | int64 | số dòng của event impression trong `events.parquet` |
| `user_id`, `candidate_id` | string | user, quảng cáo `sim:ad:<n>` |
| `cutoff_timestamp`, `exposure_timestamp` | int64 | thời điểm hiển thị (bằng nhau) |
| `clicked` | bool | nhãn |
| `split` | string | `train` / `validation` / `test` |
| `candidate_timestamp` | int64 | = `first_observed_timestamp` của quảng cáo (đầu ngày 1) |
| `profile_as_of_timestamp` | int64 | = `as_of_timestamp` của profile (đầu ngày 1) |
| `history_end_timestamp` | int64, null | timestamp event cuối trong cửa sổ; null nếu rỗng |
| `history_start`, `history_stop` | int64 | cửa sổ `events[user][history_start:history_stop]` |

### `events.parquet`

Cột: `event_id, source_row_index, user_id, timestamp, event_type, service, entity_id, event_index,
attributes_json`. Mọi event của mọi user, user theo thứ tự catalog, trong user theo
`(timestamp, session_index, action_index)`; `source_row_index` là số dòng toàn cục (0..), `event_index`
là số thứ tự trong user (0..n−1, liên tục). `service` ∈ `ride, food, ads`; `event_type` là event
của phễu (`ad_impression`, `ad_click`, ...). `attributes_json` là JSON gọn chứa **chỉ** trường
ngữ cảnh công khai (danh sách cho phép như gói pretrain) và các trường số của hành động.

### Cửa sổ history (không rò nhãn)

- `history_stop` = số event của user có `timestamp < cutoff_timestamp` (**nhỏ hơn hẳn**, loại cả
  event cùng giây); `history_start = max(0, history_stop − 30)`. Tối đa 30 event, mọi dịch vụ.
- Impression của chính request (timestamp = cutoff) và click của nó (timestamp sau cutoff) không
  bao giờ nằm trong cửa sổ.
- Trong mô phỏng, phiên `ad_impression` có hai event đúng khi được click, nên `attributes_json`
  và mọi cột không chứa kích thước phiên, số action, hay cờ click; `events` không có cột
  `clicked`/`session_id`/`session_size`. Có test kiểm tra điều này.

### `profiles.parquet`, `ad_catalog.parquet`

- `profiles`: `user_id, as_of_timestamp, age_group, saved_home_geohash, saved_work_geohash`
  (null nếu không có chỗ làm). Chỉ trường **tĩnh**; `as_of_timestamp` = đầu ngày 1. Không có trường
  tổng hợp 30 ngày (reader giữ một dòng profile mỗi user nên không dùng được giá trị theo cutoff;
  tránh dùng thông tin sau cutoff — spec §4.6).
- `ad_catalog`: `candidate_id, first_observed_timestamp` (= đầu ngày 1), `category, advertiser,
  title`. Mục tiêu nhắm (`target_zone`) và xác suất thật là thông tin ẩn, không xuất.

### `manifest.json`

| Khóa | Ý nghĩa |
|---|---|
| `dataset_version`, `generator_version`, `seed`, `sim_config_sha256`, `pyarrow_version` | nguồn gốc lần chạy |
| `schema_version` | `1`; `timestamp_unit` = `"seconds"`; `timezone`, `utc_offset` = `"Asia/Ho_Chi_Minh"`, `"+07:00"` |
| `files.<profiles\|ad_catalog\|events\|ad_opportunities>` | `{path, format: "parquet", sha256, row_count, bytes}` |
| `split` | `{kind: "temporal", basis: "exposure_timestamp", train_end, valid_end}` (epoch giây, cận trên không bao gồm) |
| `request_counts` | `{train, validation, test}` |
| `source_dataset` | `"sim_v1"` |
| `history_policy` | `{timestamp_rule: "strictly < cutoff", target_click_in_history: false, equal_timestamp_events_excluded: true, max_events: 30}` |
| `candidate_policy` | `"sim-served-impression-single-candidate-v1"` |
| `missing_click_policy`, `negative_sampling` | `"reject"`, `"none"` |
| `row_count` | số dòng `ad_opportunities` |

## Baseline (`baseline/food.json`, `baseline/ads.json`)

Định dạng của `src.baselines.runner.load_dataset`, qua `BaselineDataset.validate()`. Ghi tất định
(`sort_keys`, UTF-8, không NaN), stream từng phần tử; khóa cấp trên theo thứ tự chữ cái.

- `metadata`: `dataset_name="sim_v1"`, `data_version` (`<generator_version>+<12 ký tự đầu sha cấu hình>`),
  `task` = `food_recommendation` | `ad_optimization`, `item_kind` = `merchant` | `ad`,
  `timezone="Asia/Ho_Chi_Minh"`, `candidate_policy` (như gói Food/Ads), `availability_policy="handoff_timestamp"`.
- `queries`: `request_id, user_id, cutoff_time` (ISO `+07:00`), `split`, `context{hour, weekday}`.
Theo quy ước B1 (`docs/b1_features.md`, `src.baselines.features.FeatureBuilder`; quyết định R12):

- `candidates`: Food 10 dòng/request (`attributes`: `cuisine, price_level, geohash` + `category` =
  cuisine, `price` = price_level); Ads 1 dòng/request (`category, advertiser, title` + `brand` =
  advertiser; không có `price`).
- `labels`: đúng một dòng mỗi candidate, `target` 0/1; `available_at` = `label_time` (giờ đặt đơn) với Food;
  với Ads = lúc kết quả impression quan sát được: giờ click nếu click (hiển thị + 2–26 s), ngược lại
  hiển thị + 27 s (`AD_OUTCOME_MATURITY_SECONDS`); luôn >= `cutoff_time`.
- `history`: Interaction (`event_id, user_id, entity_id, event_type, event_time, available_at, attributes`) của
  mọi event có `ts` **nghiêm ngặt trước** cutoff lớn nhất của user có request trong file đó;
  `attributes` = thuộc tính công khai + `service`. Event `ad_impression` được ghi là
  `event_type = "impression"` với `attributes.clicked` 0/1, `category` (= `industry_main`), `brand`
  (= `industry_sub`) và `available_at` theo đúng quy tắc của nhãn Ads; dòng `ad_click` giữ nguyên;
  `order` thêm `category` / `price`; mọi dòng khác `available_at = event_time`. Hai lần hiển thị quảng
  cáo của một user luôn cách nhau ≥ 60 s (`SESSION_GAP_SECONDS`, exporter kiểm tra), nên mọi
  impression trước đó đã có kết quả khi tới request Ads kế tiếp. Runner tự cắt theo cutoff từng query
  (`eligible_history`: `event_time` và `available_at` đều trước cutoff). Không có trạng thái ẩn hay xác
  suất thật.
- Quy mô phía người dùng (dự án con 3): `load_dataset` dùng bộ nhớ đỉnh ≈ 5,8 × kích thước file
  (~12 GB cho `ads.json` ~2 GB của `full`), và `FeatureBuilder` quét lại history ở mỗi cutoff train;
  với `full` nên chạy Baseline trên một tập con user. Simulator `full` (20.000 user) dùng đỉnh
  RAM ≈ 9 GB (ước lượng của review).

## Oracle (`oracle/`)

Chỉ dùng để chấm điểm/kiểm định, **không bao giờ là đầu vào model**.

- `food_oracle.parquet`: `request_id, candidate_id, true_probability` = softmax của `food_logits` (công thức
  §4.5 tại trạng thái ẩn của phiên) trên đúng 10 ứng viên; tổng mỗi request = 1.
- `ads_oracle.parquet`: `request_id, true_click_probability` = xác suất click thật đã lưu của lần hiển thị (0 < p < 1).

## Chạy (`python -m src.simulation`)

Code: `src/simulation/__main__.py` (`main(argv) -> int`), kiểm tra sau xuất `src/simulation/checks.py`
(`run_checks`), độ khó `src/simulation/calibrate.py` (`difficulty_report`).

```
python -m src.simulation --preset smoke --out experiments/sim
python -m src.simulation --config configs/simulation/full.yaml --out DIR [--seed N] [--skip-pretrain]
```

- `--preset {smoke,full}` hoặc `--config PATH` (YAML có `preset:` + khóa ghi đè, vd.
  `{preset: smoke, n_users: 80}`; `noise:` ghi đè `NoiseConfig`), bắt buộc chọn một. `--seed N` thay
  `config.seed` (giữ nguyên `dataset_version`). `--skip-pretrain` bỏ gói `pretrain/` (chạy hiệu chỉnh
  trên `full` cho nhanh); các gói khác không đổi một byte nào.
- Ghi vào `<out>/<dataset_version>/`: `pretrain/`, `food/`, `ads/`, `baseline/`, `oracle/`, rồi
  `data_report.json`; in đường dẫn report ra stdout, tiến độ từng bước ra stderr.
- `data_report.json` luôn được ghi khi thư mục đầu ra đã tạo, với `status`: `complete`; `failed`
  (một kiểm tra thất bại, hoặc bộ kiểm tra hay `difficulty_report` bị lỗi — `failed_step`, `error`
  cho biết bước nào; `difficulty = null`); `incomplete` (mô phỏng hoặc xuất bị lỗi giữa chừng — các
  gói chỉ có một phần, **không được dùng**; `checks = null`). CLI không bao giờ ghi đè thư mục
  không rỗng: lần chạy sau thoát `2` và nêu `status` trong report cũ; xóa thư mục để chạy lại.
- Mã thoát: `0` `complete`; `2` sai tham số hoặc `<out>/<dataset_version>` đã có nội dung (không ghi
  đè gì); `3` `failed`; `4` `incomplete`.
- **Thứ tự dùng `rng` (cố định, thuộc hợp đồng):** một `numpy.random.default_rng(config.seed)` duy
  nhất được dùng lần lượt bởi `build_world` → `simulate` → `build_food_requests` →
  `build_ad_requests`. Các bước xuất, kiểm tra và đo độ khó không rút số ngẫu nhiên nào. Đổi thứ tự
  này là đổi mọi byte đầu ra. Cùng seed + cấu hình → cùng byte mọi file trừ `data_report.json` (có test;
  hai lần chạy `smoke` cho cùng `tree_sha256`).
- Kiểm tra khác máy: chạy `--preset smoke` trên máy đích (vd. Kaggle) và so `tree_sha256` với một lần
  chạy `smoke` ở máy dev; với `full`, so SHA của `food/samples.jsonl` và `baseline/*.json` trong
  `files`. Phiên bản thư viện nằm ở `data_report.json` → `environment` (R13), không ở manifest có hash.
- Sau khi xuất, CLI giải phóng mô phỏng trong bộ nhớ; kiểm tra và độ khó **chỉ đọc lại file đã xuất**.

Preset `smoke` trên máy dev (Windows, Python 3.11): exit 0, 61–68 s; 237 MB (1.015 file):

| Bước | Giây |
|---|---|
| build_world / simulate / build_food_requests / build_ad_requests | 0,04 / 2,6 / 0,7 / 0,2 |
| export_pretrain / food / ads / baseline / oracle | 6,9 / 9,7 / 2,3 / 10,2 / 0,4 |
| checks / difficulty | 25,6 / 2,3 |

Dung lượng `smoke`: pretrain 12,0 MB, food 97,3 MB, ads 3,3 MB, baseline 131,7 MB (food.json
72,9 MB, ads.json 58,8 MB), oracle 0,9 MB. Ước lượng `full` cho các kiểm tra (đo trên `smoke`,
nhân theo số dòng): ~4–5 phút, phần lớn là stream hai file Baseline (~13 µs/phần tử, ~12 triệu phần tử).

Preset `full` với `--skip-pretrain` (đo ở Task 8, cùng máy, giá trị mặc định đã hiệu chỉnh): exit 0,
`total_seconds` 849 s (~14 phút); 4,87 GB (13 file):

| Bước | Giây |
|---|---|
| build_world / simulate / build_food_requests / build_ad_requests | 1,2 / 127,5 / 11,3 / 8,5 |
| export_food / ads / baseline / oracle | 91,3 / 104,5 / 293,5 / 4,1 |
| summary / checks / difficulty | 0,8 / 190,8 / 15,7 |

Dung lượng `full` (byte thập phân): food 852,7 MB (`samples.jsonl` 852,7 MB), ads 107,2 MB
(`events.parquet` 102,2 MB), baseline 3,90 GB (food.json 1,84 GB, ads.json 2,06 GB), oracle 9,8 MB.
5.457.523 event, 379.565 phiên Food có `order`, 914.724 impression.

## `data_report.json`

Ghi sau cùng vào `<out>/<dataset_version>/data_report.json` (UTF-8, JSON chặt: NaN → `null`). Không
tham gia so sánh byte (có thời điểm chạy).

| Khóa | Ý nghĩa |
|---|---|
| `status`, `failed_step`, `error` | `complete` / `failed` / `incomplete` (mục "Chạy"); bước lỗi (vd. `checks`, `difficulty`, `export_ads`) và thông điệp lỗi, `null` khi `complete` |
| `dataset_version`, `generator_version`, `seed`, `sim_config_sha256`, `config` | nguồn gốc; `config` = toàn bộ `SimConfig` (kể cả `noise`) |
| `generated_at_utc` | `YYYY-MM-DDTHH:MM:SSZ` |
| `environment` | `{python, numpy, pyarrow, platform}` — nguồn gốc môi trường (R13); chỉ ghi ở đây, không vào manifest có hash |
| `command` | `{preset, config, seed_override, skip_pretrain}` |
| `rng_order` | `["build_world", "simulate", "build_food_requests", "build_ad_requests"]` |
| `packages`, `skipped` | gói đã ghi; `["pretrain"]` khi `--skip-pretrain` |
| `counts` | `users, merchants, ads, drivers, events` (cả 8 tuần), `events_before_week7`, `food_sessions_with_order`, `ad_impressions`, `food_requests_by_split`, `ads_requests_by_split`, `pretrain` (`{users, events, snapshots_by_split}` hoặc `null`) |
| `ctr` | `all_impressions` (mọi impression mô phỏng), `requests_by_split` |
| `personas`, `intents` | số user theo nhóm người; mỗi loại ý định `{episodes, users}` (trạng thái ẩn — chỉ để báo cáo, không phải đầu vào model) |
| `timings_seconds`, `total_seconds` | thời gian từng bước và tổng |
| `checks` | `{status: "passed"\|"failed", results: {tên: {status, description, violations, ids?, …}}, error?}`; `null` khi `incomplete`; `results = null` khi bộ kiểm tra bị lỗi |
| `difficulty` | xem dưới; `null` khi `status` khác `complete` |
| `tree_sha256`, `files` | `files` = `{đường dẫn tương đối: {sha256, bytes}}` của **mọi** file trừ `data_report.json`; `tree_sha256` = SHA-256 của các dòng `path\tsha256\n` theo thứ tự đường dẫn |

### Kiểm tra (`checks.results`)

`run_checks(out_dir, config)` đọc lại **file đã xuất** (không dùng đối tượng trong bộ nhớ); tùy chọn
`only=("ads", "food", …)` chỉ kiểm một số gói (gói khác `skipped`; dùng cho test, CLI luôn kiểm hết).
Mỗi kiểm tra có test tiêm một vi phạm thật vào bản sao gói và đòi đúng kiểm tra đó thất bại. Mỗi kiểm
tra có `status` = `passed` / `failed` / `skipped` (pretrain khi `--skip-pretrain`) / `not_run` (file
của gói không đọc được — khi đó `<gói>_readable` thất bại kèm lỗi). Có lỗi → `CheckError`
(`ValueError`) nêu mọi kiểm tra thất bại và tối đa 20 ID mỗi kiểm tra; `error.report` là toàn bộ kết
quả. SHA-256 của mọi file được tính một lần rồi so với manifest.

| Gói | Kiểm tra |
|---|---|
| pretrain | `pretrain_manifest` (manifest `complete`, nguồn gốc khớp cấu hình, SHA `preprocessing_config.json` qua `load_manifest`, SHA/kích thước **mọi** shard + `preview.json`, không file lạ); `pretrain_events_before_week7` (mọi event: `time_*`/`eligibility_time` < đầu ngày 43); `pretrain_snapshots` (mọi snapshot: cutoff < ngày 43, đúng khoảng split, `profile_json` đúng `as_of_time` + trường §4.6, `as_of_time` = cutoff, `history_group_truncated` = `false` — R11); `pretrain_snapshot_history` (mọi id history là event cùng user, thời điểm ≤ cutoff, 1–30 id; một phép join Arrow); `pretrain_loader` (kiểm tra mẫu: mọi mẫu của 20 user cách đều qua `iter_normalized_samples` + `validate_sample` — loader mất ~3,5 ms/mẫu nên không chạy hết `full`) |
| ads | `ads_manifest`; `ads_columns_public` (đúng tập cột 4 bảng; khóa `attributes_json` ⊆ context công khai + trường số của action — không cờ click, kích thước phiên, số action); `ads_events_order`; `ads_history_before_exposure` (cửa sổ chỉ gồm event **trước hẳn** lúc hiển thị, không chứa impression/click của chính request); `ads_history_window_policy` (`history_start = max(0, stop − 30)`, `stop` = mọi event trước lúc hiển thị, `history_end_timestamp`, cutoff = exposure); `ads_impression_rows` (kể cả nhãn: `clicked` = kết quả trong log — dòng kế tiếp của user là `ad_click` cùng quảng cáo sau 2–26 s); `ads_split` |
| food | `food_manifest` (nguồn gốc, chính sách, SHA/số dòng/bytes, split, `popularity_end_exclusive` = đầu tuần 7 — mốc mà độ khó dùng); `food_format`; `food_candidates` (10 quán khác nhau, thuộc tính khớp catalog, nhãn đúng thứ tự, đúng một positive); `food_inputs_public` (đúng tập khóa công khai ở `input`, `context`, `profile`, `candidates`, event history; thuộc tính history theo allowlist — không vị trí hiện tại, khoảng cách hay trạng thái ẩn; `distance_km`/`delivery_distance_km` trong history là trường của event quá khứ; cột `merchants.parquet`); `food_history_before_cutoff` (≤ 30 event, mọi `event_time` < `cutoff_time`); `food_own_session_absent` (phiên của request tìm trong `ads/events.parquet`: event Food của user trong `(cutoff, label_time]` bắt đầu bằng `search` lúc `cutoff + 1 s`, kết thúc bằng `order` quán positive lúc `label_time`, không id nào nằm trong history); `food_label_after_cutoff`; `food_profile_as_of_cutoff`; `food_split` |
| oracle | `oracle_food_sums_to_one` (hữu hạn, trong [0, 1], tổng mỗi request = 1 ± 1e-9); `oracle_food_matches_candidates`; `oracle_ads` (đúng thứ tự request Ads, 0 < p < 1) |
| baseline | Mỗi file stream **một lần** (không đọc cả file thành một chuỗi): `baseline_<food\|ads>_structure` (đủ 5 mục, metadata; queries/candidates/labels trùng gói Food/Ads, mỗi candidate đúng một label, `target` đúng, `available_at` ≥ cutoff); `baseline_<…>_history_before_cutoff` (chỉ user có request, `event_time` < cutoff muộn nhất của user, `available_at = event_time` trừ impression — ghi là `impression` với `clicked` và `available_at` khớp log, thuộc tính công khai + `service` + tên B1 (`category`/`price`/`brand`), id không trùng); `baseline_<…>_load_dataset` (`src.baselines.runner.load_dataset` đọc cả file khi ≤ 256 MiB — `mode = "full_file"`, `smoke`; file lớn hơn — `full`, ~1,8–2,1 GB mỗi file — `mode = "subset"`: 200 request cách đều + candidate, label, history của user đó, ghi bằng chính writer của exporter rồi `load_dataset`; `load_dataset` từ chối → kiểm tra này thất bại kèm lỗi, file còn lại vẫn được kiểm) |

### Độ khó (`difficulty`)

`difficulty_report(out_dir)` đo trên split **test**, chỉ từ file đã xuất; `targets`/`within_targets`
nhắc lại mục tiêu spec §5 (dành cho `full`).

- `food`: NDCG@10 trung bình (`src.evaluation.metrics.ranking_metrics`, hòa điểm xếp theo id):
  - `popularity_ndcg_at_10`: số `order` của quán trước đầu tuần 7 (đếm từ `ads/events.parquet`, cận
    `food/manifest.json: popularity_end_exclusive` — đúng số dùng khi chọn ứng viên).
  - `candidate_only_ndcg_at_10`: `log(rate(cuisine, price_level)) + log1p(phổ biến)`, `rate` = tỷ lệ
    positive của cặp `(cuisine, price_level)` trên ứng viên split train, làm mịn về tỷ lệ chung với 20
    ứng viên giả. Chỉ ứng viên + catalog + phổ biến (không history, không profile): rào lối tắt §8.
  - `oracle_ndcg_at_10`: `oracle/food_oracle.parquet`.
  - Ranker profile (quyết định R7: báo cáo, không giới hạn): `distance_to_home_ndcg_at_10` (khoảng cách
    haversine giữa tâm ô geohash của ứng viên và `saved_home_geohash`, gần hơn điểm cao hơn);
    `home_or_work_by_hour_ndcg_at_10` (như trên nhưng tới `saved_work_geohash` khi thứ Hai–Sáu, 9 ≤
    giờ < 17 và user có chỗ làm; còn lại tới nhà).
  - `oracle_minus_popularity`, `requests`, `merchants`; `experiment_grade = merchants ≥ 300` — catalog
    thưa (`smoke`, 60 quán) chỉ là số chẩn đoán, kèm `note` (quyết định R9).
- `ads`: ROC AUC (`src.evaluation.metrics.advertising_metrics`): `popularity_auc` = CTR train theo quảng
  cáo, làm mịn `(click + 20·CTR_train) / (impression + 20)`; `oracle_auc` =
  `oracle/ads_oracle.parquet`; kèm `requests`, `test_ctr`, `train_ctr`.

### Hiệu chỉnh (Task 8) và giá trị đạt được

Chỉ chỉnh hệ số `NoiseConfig` và `food_candidate_tau_km` (không đổi mô hình, cách lấy mẫu hay
exporter); mục tiêu spec §5 đo trên `full`. Giá trị mặc định đã chọn (test
`test_noise_defaults_are_calibrated` ghim chúng; đổi một giá trị là bảng dưới hết hiệu lực):

| Tham số | Trước | Sau | Lý do |
|---|---|---|---|
| `food_cuisine_weight` / `food_price_weight` / `food_popularity_weight` / `food_logit_noise_sd` | 2,0 / 0,6 / 1,0 / 0,5 | không đổi | — |
| `food_distance_weight` | 2,0 | **1,25** | 2,0 dồn 99,5% đơn `full` vào ≤ 3 km quanh vị trí hiện tại (90,0% ≤ 1 km, TB 0,17 km); 1,25 → 95,0% ≤ 3 km, 73,6% ≤ 1 km, TB 0,58 km |
| `ad_interest_weight` / `ad_zone_weight` / `ad_cross_service_weight` | 3,0 / 0,8 / 1,2 | không đổi | — |
| `ad_intent_weight` | 1,5 | **1,0** | đợt "tiêu mạnh tay" (ngày 25–28) rơi vào tuần 8 đẩy CTR test lên 5,57% > 5,5% |
| `ad_logit_noise_sd` | 0,5 | **1,25** | trên cùng một mô phỏng, AUC kỳ vọng (không nhiễu nhãn) của "CTR theo quảng cáo" 0,634 → 0,615 (xa trần 0,65 hơn), oracle 0,797 → 0,778 |
| `food_candidate_tau_km` | 1,5 | **2,0** | ranker học chỉ từ ứng viên (mục Food) giảm từ + 0,058 xuống + 0,041 trên phổ biến; 2,0 km là τ lớn nhất còn giữ hai bảo đảm "âm gần khu quán được đặt / gần khu neo" (test: τ = 2,5 cho 1,86 âm gần khu quán được đặt < 2,0 và 45% âm ≤ 3 km quanh neo < 50%) |

Preset `smoke` giữ `food_distance_weight = 2,0` (`config.SMOKE_FOOD_DISTANCE_WEIGHT`): catalog
60 quán (~1 quán/khu) quá thưa, với 1,25 lựa chọn quán trôi khỏi vị trí hiện tại (thế giới test 24
quán dẫn xuất từ `smoke`: 62% → 53% đơn ≤ 3 km). Mọi tham số khác của `smoke` dùng giá trị mặc định.

Giá trị đạt được (seed 42, split test). `full`: `python -m src.simulation --preset full --out
experiments/sim --skip-pretrain` (10.000 request Food, 30.000 request Ads; `status = complete`, mọi
kiểm tra `passed`); `smoke`: `--preset smoke` (1.268 / 2.797 request):

| Chỉ số | Mục tiêu spec §5 (`full`) | `full` | `smoke` |
|---|---|---|---|
| Food NDCG@10 phổ biến | [0,40; 0,55] | **0,509** | 0,495 |
| Food chỉ ứng viên (`candidate_only`) | ≤ phổ biến + 0,05 | **0,513** (+ 0,003) | 0,493 |
| Food oracle | — | 0,899 | 0,958 |
| Food oracle − phổ biến | ≥ 0,20 | **0,390** | 0,463 |
| Food profile: gần nhà / nhà hoặc chỗ làm theo giờ (R7, chỉ báo cáo) | — | 0,663 / 0,636 | 0,800 / 0,767 |
| Food `experiment_grade` | — | `true` | `false` (60 quán) |
| Ads AUC phổ biến (CTR theo quảng cáo) | [0,55; 0,65] | **0,638** | 0,613 |
| Ads AUC oracle | ≥ 0,75 | **0,788** | 0,784 |
| CTR: mọi impression / request train / validation / test | ≈ 5% ([4,5%; 5,5%]) | 5,09% / 5,07% / 5,02% / 5,27% | 5,14% / 5,01% / 5,31% / 5,79% |

`within_targets` = `true` cho cả Food và Ads của `full`. AUC Ads thực đo phụ thuộc nhãn test đã
rút: lấy kỳ vọng theo xác suất oracle (không nhiễu nhãn) thì `full` có AUC phổ biến 0,621 và oracle
0,781; độ lệch chuẩn do nhãn test ~0,006 / ~0,004 — 0,638 nằm ở phía cao của dao động đó (seed
khác có thể cho số thấp hơn, giữa khoảng mục tiêu).

Mốc "không cần history" cho dự án con 3 (review cuối, `full`): một GBDT học trên profile + phổ biến +
context đạt **0,740** NDCG@10 Food (gần nhà 0,663; profile + loại trừ theo tập ứng viên 0,758; oracle
0,899). Lợi ích của foundation trên Food nên được đánh giá so với ≈ 0,74, không phải với phổ biến 0,509.
