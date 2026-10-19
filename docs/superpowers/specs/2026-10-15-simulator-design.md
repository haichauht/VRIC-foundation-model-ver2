# Simulator dữ liệu siêu app (ride + food + ads) — Design spec

- Ngày: 2026-10-15
- Branch: `feat/sim-data` (tách từ `feat/food-recommendation`)
- Dự án con 1/3. Dự án con 2: sửa chỗ đọc dữ liệu (tokenizer thêm scenario, reader Food nhiều dịch vụ, reader Ads tổng quát) và cho `long_term` nhận gradient. Dự án con 3: pretrain trên Kaggle + Food + Ads + so sánh.
- Hợp đồng team: `docs/parallel_vibe_implementation_plan.md` (spec giao simulator cho Data team; bản này đặt code ở `src/simulation/`, không sửa `src/data/**` hay `data/**`, và phải báo Data team).

## 1. Mục tiêu

Sinh một "thế giới" người dùng siêu app có quy luật cài sẵn và biết xác suất thật, đủ để
**chứng minh foundation model có ích**: pretrain trên cả ba dịch vụ phải giúp Food
Recommendation và Ad Optimization tốt hơn các cách không dùng foundation, và tách đo được
phần đóng góp của tín hiệu dài hạn (`long_term`), ngắn hạn (`short_term`) và xuyên dịch vụ.

Ngoài phạm vi dự án con này: sửa reader/tokenizer/encoder, chạy pretrain, train downstream.

## 2. Bám theo Grab

Theo *User foundation models for Grab* (engineering.grab.com/user-foundation-models-for-grab):

| Grab | Simulator |
|---|---|
| Profile bảng: tuổi, địa điểm đã lưu, chi tiêu trung bình tháng, dịch vụ dùng nhiều nhất | profile có các trường tương ứng (mục 4.6) |
| Clickstream phễu xem → click → cân nhắc → giao dịch, có khoảng thời gian giữa event | phễu Food và Ride (mục 4.4) |
| Token `key:value`, ví dụ `view_merchant : merchant_id_114` | mỗi event có `event_type` và một thực thể chính |
| Giao dịch: giá đơn, cước, khoảng cách giao, thời gian chờ tài xế | trường số trong event |
| Text: câu tìm kiếm, tên quán, mô tả món | câu tìm kiếm và tên quán/món có nghĩa |
| Vị trí: tọa độ đón/trả, địa chỉ giao, sân bay, TTTM | khu có lat/lon + geohash |
| Thực thể `user_id`, `merchant_id`, `driver_id` | có cả ba, cộng `ad_id` |
| Next-action dự đoán loại hành động và giá trị | dữ liệu có đủ cả hai |
| Downstream: gợi ý đồ ăn, tối ưu quảng cáo | Food gợi ý **quán**; Ads dự đoán click |

## 3. Thông số

| Thông số | `smoke` | `full` |
|---|---|---|
| User | 500 | 20.000 |
| Thời gian | 8 tuần từ 2026-01-05 00:00 `+07:00` (thứ Hai = ngày 1; ngày 43 = đầu tuần 7) | như smoke |
| Khu | 50 | 50 |
| Quán / món | 60 / ~480 | 600 / ~4.800 |
| Quảng cáo / ngành hàng | 150 / 15 | 1.500 / 15 |
| Tài xế | 300 | 3.000 |
| Mật độ event (tuần 1–6) | ~200/user | ~200/user |
| Food request train / val / test | lấy tất cả | 60.000 / 10.000 / 10.000 |
| Ads request train / val / test | lấy tất cả | 200.000 / 30.000 / 30.000 |

Split downstream: train = tuần 1–6, validation = tuần 7, test = tuần 8 (theo thời điểm request).
Pretrain chỉ dùng event của tuần 1–6.

Mọi tham số nằm trong `SimConfig` (mục 7); hai preset trên là giá trị mặc định, trừ một ghi đè:
`smoke` giữ `food_distance_weight = 2,0` (`config.SMOKE_FOOD_DISTANCE_WEIGHT`; catalog 60 quán quá
thưa), `full` dùng giá trị đã hiệu chỉnh 1,25 (`docs/simulation_data_contract.md`, mục "Độ khó").
Cùng `seed` và cùng cấu hình → cùng byte cho mọi file đầu ra.

## 4. Mô hình thế giới

### 4.1 Thành phố
50 khu, mỗi khu có loại, tâm lat/lon (trong hộp ~20×20 km quanh 10.7769, 106.7009) và
geohash độ chính xác 6: dân cư 20, văn phòng 12, trường học 4, ăn chơi về đêm 6, trung tâm
thương mại 5, sân bay/ga 3. Khoảng cách giữa khu tính theo haversine.

### 4.2 Catalog
- Quán: khu, loại món (12 loại: cơm văn phòng, phở/bún, healthy/salad, gà rán, pizza, lẩu,
  đồ Nhật, đồ Hàn, trà sữa, cà phê/bánh, đồ chay, ăn vặt đêm), mức giá (thấp/vừa/cao), tên có nghĩa,
  độ phổ biến nền. ~8 món/quán, mỗi món có tên, mô tả ngắn, giá.
- Quảng cáo: 15 ngành hàng (đồ ăn, thể thao, du lịch, khách sạn, thời trang, làm đẹp, điện tử,
  tài chính, giáo dục, game, mẹ & bé, nhà cửa, xe cộ, sức khỏe, giải trí), nhà quảng cáo,
  tiêu đề, khu nhắm tới (tùy chọn).
- Tài xế: loại xe (xe máy/ô tô/cao cấp), khu hoạt động chính.

### 4.3 User: "tính cách ẩn" ổn định (tín hiệu dài hạn)
- Nhóm người (tỷ lệ): văn phòng 40%, sinh viên 20%, làm ca đêm 10%, ở nhà/hưu trí 15%,
  hay đi công tác 15%.
- Khu nhà; khu làm việc/học (trừ ở nhà/hưu trí); lịch sinh hoạt theo nhóm (giờ đi/về, ngày làm).
- Mức dùng app (Gamma theo nhóm), nhóm tuổi.
- Khẩu vị: Dirichlet trên 12 loại món, tiên nghiệm theo nhóm người.
- Mức chi tiêu: thấp/vừa/cao, tương quan với nhóm người.
- Sở thích quảng cáo: vector trên 15 ngành hàng = phần theo nhóm người + phần theo khẩu vị
  (vd. healthy → thể thao/sức khỏe) + nhiễu riêng.

### 4.4 Ý định ngắn hạn (tín hiệu ngắn hạn)
Mỗi ý định có thời điểm bật, thời lượng và tác động; được ghi lại cho oracle và kiểm thử.

| Ý định | Tần suất | Thời lượng | Tác động |
|---|---|---|---|
| Ăn kiêng | ~5%/user/tuần | 5–10 ngày | khẩu vị dồn về healthy/chay; click thể thao/sức khỏe ↑ |
| Công tác | ~4%/user/tuần (nhóm hay đi công tác ×4) | 3–7 ngày | chuyến xe ra sân bay ở đầu và cuối; ít/không đặt đồ ăn; click du lịch/khách sạn ↑ |
| Đổi chỗ làm | ~1%/user/tuần | vĩnh viễn | khu làm việc đổi; chuyến xe và khu đặt món đổi theo |
| Tiêu mạnh tay | ngày 25–28 hằng tháng | 2–4 ngày | mức chi tiêu tạm tăng một bậc |

### 4.5 Ba dịch vụ và cách nối
Mỗi ngày mô phỏng theo lịch sinh hoạt; "vị trí hiện tại" của user là điểm trả của chuyến xe
gần nhất trong ngày (mặc định ở nhà).

- **Ride** (`service = ride`): phễu `search_destination` → `get_quote` → `book_ride` →
  `complete_ride` (có tỷ lệ bỏ ngang). Loại chuyến: đi làm sáng/về tối (ngày làm), đi chơi tối
  và cuối tuần (khu ăn chơi/TTTM), sân bay khi công tác. Trường: khu/tọa độ/geohash đón và trả,
  loại xe (theo mức chi tiêu), cước, khoảng cách, thời gian chờ tài xế, `driver_id`.
- **Food** (`service = food`): phiên đặt món vào giờ ăn: `search` (câu tìm theo khẩu vị/ý định)
  → `view_merchant` (vài quán) → `click_dish` → `order`. Quán được chọn theo xác suất ∝
  exp(khẩu vị·loại món + hệ số giá·mức chi tiêu − hệ số khoảng cách·dist(vị trí hiện tại, quán)
  + độ phổ biến nền + nhiễu). **Vị trí hiện tại lấy từ chuyến xe** là mắt xích xuyên dịch vụ chính.
  Trường: `merchant_id`, món, giá đơn, khoảng cách giao.
- **Ads** (`service = ads`): `ad_impression` trong lúc dùng app, `ad_click` khi click.
  Quảng cáo hiển thị chọn bởi chính sách nhắm mục tiêu có nhiễu (50% theo nhóm người thô,
  50% ngẫu nhiên), độc lập với nhãn. Xác suất click = sigmoid(nền + sở thích ngành hàng
  + ý định ngắn hạn + khớp khu đang ở + event gần đây ở dịch vụ khác, vd. vừa đi sân bay → du lịch
  ↑, vừa đặt healthy → thể thao ↑). Hiệu chỉnh hệ số nền để CTR tổng thể ≈ 5%.

### 4.6 Profile (tính tại thời điểm cutoff, chỉ từ event trước cutoff)
`age_group`, `saved_home_geohash`, `saved_work_geohash` (nếu có), `avg_monthly_spend`
(ước từ 30 ngày), `most_used_service`, `ride_count_30d`, `food_order_count_30d`,
`ad_click_count_30d`, `active_days_30d`. Nhóm người, khẩu vị, sở thích, ý định là **ẩn**,
không bao giờ xuất ra đầu vào.

### 4.7 Oracle
Với mỗi request Food, simulator ghi xác suất thật của 10 quán ứng viên (theo công thức 4.5 tại
cutoff). Với mỗi request Ads, ghi xác suất click thật. Chỉ xuất vào gói oracle (mục 6.5).

## 5. Chống lối tắt và độ khó

- **Food:** mỗi request = một phiên đặt món có `order`. Mốc dự đoán (`cutoff`) = ngay trước
  event đầu tiên của phiên; mọi event của phiên đó **không** vào history. 10 quán ứng viên:
  quán được đặt + 9 quán chưa đặt trong phiên, cùng **cặp thập phân vị độ phổ biến** cố định
  {0,1}, {2,3}, …, {8,9} với quán được đặt (đếm số đơn tuần 1–6; quyết định R6), chia thành cụm
  quanh ba khu neo: **khu của quán được đặt**, **nhà**, **chỗ làm** (người không có chỗ làm: khu
  TTTM gần nhà nhất). Kích thước cụm **hoán đổi được** (R8), quán được đặt tính vào cụm của khu nó:
  ba khu neo khác nhau → kích thước (4, 3, 3) gán cho (khu quán được đặt, nhà, chỗ làm) theo một
  hoán vị ngẫu nhiên đều; hai khu trùng nhau → (5, 5); cả ba trùng → một cụm 9 âm. Ứng viên âm rút
  **ngẫu nhiên theo khoảng cách** (R10, τ = `food_candidate_tau_km` = 2,0 km, đã hiệu chỉnh —
  trước đó 1,5 km): mỗi cụm có một tâm —
  chính khu nhà / chỗ làm cho cụm của chúng; cụm của quán được đặt có **tâm ẩn** rút trên mọi khu
  với xác suất ∝ exp(−d(khu, khu quán được đặt)/τ); âm của mỗi cụm rút không hoàn lại từ cặp thập
  phân vị (trừ quán được đặt và quán đã chọn) với xác suất ∝ exp(−d(khu của quán, tâm)/τ). Nhờ vậy
  quán được đặt không nằm đúng tâm một lần điền tất định, không âm nào là phần "tràn" đoán trước
  được, cụm chứa nó không phải lúc nào cũng lớn nhất, và về trung bình khu của quán được đặt có
  nhiều âm ở gần hơn hẳn rút đều (không có "quán lạc khu"). Không có hai ứng viên cùng quán.
- Đầu vào không chứa vị trí hiện tại hay khoảng cách từ vị trí hiện tại (nếu có thì model khỏi cần
  nhìn chuyến xe); vị trí hiện tại chỉ suy ra được từ history.
- **Ads:** mỗi request = một `ad_impression`, một ứng viên, nhãn `clicked` từ xác suất thật;
  history chỉ gồm event có thời điểm **nhỏ hơn hẳn** thời điểm hiển thị; không resample âm.
- **Mục tiêu độ khó** (đo trên `full` bằng script hiệu chỉnh, mục 8): Food — NDCG@10 của "phổ biến"
  trong [0,40; 0,55] và oracle − phổ biến ≥ 0,20; Ads — AUC của "CTR theo quảng cáo" trong
  [0,55; 0,65] và AUC oracle ≥ 0,75. Hệ số nhiễu trong `SimConfig` được chỉnh để đạt các khoảng này;
  giá trị đạt được ghi vào `data_report.json`.
- Nhãn không bao giờ vào profile hay history; profile tính chỉ từ event trước cutoff.

## 6. Đầu ra

Một lần chạy ghi vào `<out>/<dataset_version>/` (ngoài Git), `dataset_version` = `sim-smoke-v1`
hoặc `sim-v1`. Namespace `sim_v1`; ID: `sim:user:<n>`, `sim:merchant:<n>`, `sim:dish:<n>`,
`sim:ad:<n>`, `sim:driver:<n>`, `sim:event:<n>`, `sim:food_req:<n>`, `sim:ad_req:<n>`.

### 6.1 Gói pretrain `pretrain/` — schema `omnibehavior_pretraining_v2`
Đọc được bằng `src/foundation_input` hiện có (`DatasetManifest`, `iter_normalized_samples`).
- `events/<user>.parquet` với các cột như `data/README_pretrain_data.md` §1.1:
  `row_index, event_id, user_id, source_record_id, source_record_index, action_index,
  source_action_count, source_timestamp, event_time, time_precision="second", time_lower_bound,
  time_upper_bound, upper_bound_exclusive=False, eligibility_time, eligibility_inclusive=True,
  scenario, event_type, context_json, action_json`. Thời gian dạng **UTC không kèm múi giờ** (naive UTC),
  nhất quán với `cutoff_time` và `preprocessing_config.splits`. Một phiên (ride hoặc food) là một
  `source_record_id`; các bước phễu là các action trong cùng record theo thứ tự.
- `scenario`: `Ride`, `Food`, `Advertisement`. `context_json` theo scenario:
  - Ride: `pickup_geohash, dropoff_geohash, pickup_zone_type, dropoff_zone_type, vehicle_type, driver_id`.
  - Food: `merchant_id, merchant_name, cuisine, price_level, search_query` (chỉ ở bước `search`), `dish_id`, `dish_name` (ở `click_dish`/`order`).
  - Advertisement: `product` (= ad_id), `industry_main` (ngành hàng), `industry_sub` (nhà quảng cáo), `caption` (tiêu đề).
- `action_json`: `{"type": <event_type>}` cộng trường số: Ride `fare, distance_km, wait_minutes`; Food `order_amount, delivery_distance_km` (ở `order`); Ads `{"type": "impression"|"click"}`.
- `snapshots/<user>.parquet` như README §1.2: cutoff mỗi 3 ngày, ngày 6, 9, …, 33 (lúc 23:59:59 giờ +07:00, quy về naive UTC) cho `split=train`; ngày 37 và 40 cho `split=validation`; không có `test`. History = tối đa 30 event gần nhất trước cutoff (`max_history_events=30`), giữ trọn record (không cắt đôi một phiên; nếu phải cắt thì bỏ cả record và đặt `history_group_truncated` đúng sự thật). `profile_json` theo 4.6.
- `manifest.json`: `schema_version="omnibehavior_pretraining_v2"`, `status="complete"`,
  `source_dataset="sim_v1"`, `generator_version`, `seed`, `config_sha256`, tổng số, và danh sách shard
  (`user_id, events_file, snapshots_file, event_rows, snapshot_rows, snapshots_by_split, *_bytes, *_sha256`)
  như bản Meituan.
- `preprocessing_config.json`: `max_history_events=30`, `splits` = {`start_inclusive` = đầu tuần 1,
  `train_end_exclusive` = đầu ngày 34, `validation_end_exclusive` = đầu tuần 7, `test_end_exclusive` = đầu tuần 7}.

### 6.2 Gói Food `food/`
- `samples.jsonl`, mỗi dòng một request, mở rộng định dạng `samples.jsonl` của Meituan:
  `sample_id`, `input` {`user_id`, `cutoff_time` (ISO `+07:00`), `profile` (4.6, có `as_of_time`),
  `history` (tối đa 30 event mọi dịch vụ, mỗi event: `event_id, event_time, service, event_type,
  entity_id, attributes{…}`), `context` {`hour, weekday`}, `candidates` [10 × {`item_id`
  = merchant_id, `cuisine, price_level, geohash`}]}, `labels`
  [{`item_id, relevance`}], `label_metadata` {`request_id, split, label_time` (thời điểm `order`),
  `candidate_source="sim-popularity-matched-home-work-v1"`, `observation_complete=true`,
  `prediction_anchor="session_start"`, `positive_action="order"`}.
- `merchants.parquet` (catalog quán, mọi thuộc tính công khai).
- `manifest.json` (`dataset_version, schema_version=1, generator_version, seed, timezone="Asia/Ho_Chi_Minh",
  files{path, sha256, row_count}, split{kind="temporal", train_end, valid_end}, candidate_policy`).

### 6.3 Gói Ads `ads/` — theo schema reader Ads (`feat/ad-optimization`)
- `profiles.parquet` (`user_id, as_of_timestamp` + trường 4.6), `ad_catalog.parquet`
  (`candidate_id, first_observed_timestamp` + ngành hàng, nhà quảng cáo, tiêu đề),
  `events.parquet` (`event_id, source_row_index, user_id, timestamp, event_type, service, entity_id,
  event_index` + `attributes_json`; mọi dịch vụ), `ad_opportunities.parquet` (`request_id,
  source_row_index, user_id, cutoff_timestamp, candidate_id, clicked, split, exposure_timestamp,
  candidate_timestamp, profile_as_of_timestamp, history_end_timestamp, history_start, history_stop`).
  Timestamp là epoch giây.
- `manifest.json`: `dataset_version, schema_version=1, generator_version, seed, timezone, timestamp_unit="seconds",
  files, split, source_dataset="sim_v1", history_policy={timestamp_rule="strictly < cutoff",
  target_click_in_history=false, equal_timestamp_events_excluded=true, max_events=30},
  candidate_policy="sim-served-impression-single-candidate-v1", missing_click_policy="reject",
  negative_sampling="none", row_count`.
- Reader Ads hiện hard-code namespace/prefix/policy của Taobao; tổng quát hóa thuộc dự án con 2.

### 6.4 Gói Baseline `baseline/food.json`, `baseline/ads.json`
Định dạng `src/baselines/runner.load_dataset`: {`metadata` (DatasetMetadata), `queries`, `candidates`,
`labels` (có `available_at`), `history` (Interaction: `event_id, user_id, entity_id, event_type,
event_time, available_at, attributes`)}. Phải qua `BaselineDataset.validate()`.

### 6.5 Gói oracle `oracle/`
`food_oracle.parquet` (`request_id, candidate_id, true_probability`), `ads_oracle.parquet`
(`request_id, true_click_probability`). Không bao giờ là đầu vào model.

### 6.6 `data_report.json`
Số user/event/snapshot/request theo split, CTR, phân bố nhóm người và ý định, kết quả các kiểm tra
mục 8, chỉ số độ khó (phổ biến, oracle), thời gian sinh, SHA của mọi file.

## 7. Cấu trúc code

`src/simulation/`:
`config.py` (`SimConfig` + preset `smoke`/`full`), `world.py` (khu, catalog, user, tính cách ẩn),
`behavior.py` (sinh event theo ngày, ý định, phễu; xác suất thật), `export_pretrain.py`,
`export_food.py`, `export_ads.py`, `export_baseline.py`, `export_oracle.py`, `checks.py`
(kiểm tra sau xuất + `data_report.json`), `calibrate.py` (đo độ khó), `__main__.py`
(`python -m src.simulation --preset smoke|full --out <dir> [--seed N]`).
Kèm `configs/simulation/{smoke,full}.yaml` và `docs/simulation_data_contract.md` (định dạng chính xác
ba gói, là đầu vào cho dự án con 2). Chỉ dùng numpy, pyarrow, pyyaml (đã có trong dự án).

## 8. Kiểm thử

Test chạy trên world nhỏ dẫn xuất từ preset `smoke` (mỗi test vài giây); riêng kiểm tra lối tắt vị
trí chạy ở **mật độ catalog `full`** (quyết định R9, xem dưới), không phải trên catalog `smoke`.

- Tái lập: hai lần chạy cùng seed → SHA-256 giống hệt cho mọi file; seed khác → khác.
- Quy luật có thật: quán được đặt nằm trong bán kính của vị trí hiện tại (suy từ ride) ở đa số request;
  CTR trong [3%, 7%]; trong ý định ăn kiêng tỷ lệ healthy/chay tăng rõ; trong công tác click du lịch tăng.
- Không lộ dữ liệu: gói pretrain không có event ≥ đầu tuần 7; mọi history trước cutoff; event của phiên
  đang đoán không có trong history Food; nhãn không có trong profile; profile chỉ từ event trước cutoff.
- Đúng định dạng: gói pretrain đọc được bằng `iter_normalized_samples` và từng mẫu qua `validate_sample`;
  `baseline/*.json` đọc được bằng `load_dataset`; gói Food/Ads khớp `docs/simulation_data_contract.md`
  (cột, kiểu, SHA trong manifest).
- Ứng viên Food: đúng 10, không trùng quán, cùng một cặp thập phân vị phổ biến (R6), kích thước
  cụm theo R8 (phân phối hoán đổi được); âm rút ngẫu nhiên quanh tâm cụm (R10) nên vị trí chỉ kiểm
  về trung bình: âm nằm gần khu neo và gần khu của quán được đặt nhiều hơn hẳn rút đều; đầu vào
  không có vị trí hiện tại/khoảng cách từ vị trí hiện tại.
- Không lối tắt vị trí — kiểm ở **mật độ catalog `full`** (600 quán, 50 khu, ít user cho nhanh;
  quyết định R9), không phải trên catalog `smoke`: mọi ranker chỉ dùng ứng viên + catalog công khai
  + số đơn tuần 1–6 (không history, không profile), kể cả ranker biết quy tắc lấy mẫu (mật độ, độ
  trung tâm, độ "bão hòa" theo thứ tự điền, phát lại quy tắc điền tất định, likelihood theo mô hình
  R10, cả chiều ngược), không vượt "phổ biến" quá 0,05 NDCG@10. Ranker dựa trên nhà/chỗ làm đã lưu
  trong profile là tín hiệu dài hạn hợp lệ, không bị giới hạn (R7).
- Độ khó (smoke, ngưỡng lỏng): oracle > phổ biến trên cả Food và Ads. Các số độ khó Food đo trên
  `smoke` (60 quán, 12 quán một cặp trên 50 khu) **không** dùng làm số thí nghiệm (R9).

## 9. Chạy thật (thuộc dự án con 3, ghi lại để thống nhất)

- `smoke` sinh trên máy. `full` sinh ngay trên Kaggle (cùng seed → cùng dữ liệu, không upload 40k file).
- Ngân sách pretrain < 10 giờ: chạy thử 2k user × 1 epoch để đo thời gian/epoch và RAM, chọn số epoch
  (~6) sao cho pretrain ≤ 8 giờ; checkpoint mỗi epoch.

## 10. Rủi ro

- RAM khi fit vocab trên ~200k mẫu (ước 13–19 GB, Kaggle ~29 GB): đo ở bước chạy thử.
- Dữ liệu giả chỉ chứng minh model học được quy luật đã cài; kết quả không thay thế đánh giá trên dữ liệu thật.
- Mục tiêu độ khó có thể cần vài vòng hiệu chỉnh hệ số nhiễu.
