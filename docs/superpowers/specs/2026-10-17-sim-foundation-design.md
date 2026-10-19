# Dự án con 2 — Foundation chạy được trên dữ liệu mô phỏng (sim-v1)

Ngày: 2026-10-17. Branch: `feat/sim-foundation` (từ `feat/sim-data`, commit `3a20411f`).
Dữ liệu: gói `sim-v1` đã sinh trên Kaggle (kernel private `hunhthhichu/vric-sim-generate`),
hợp đồng dữ liệu `docs/simulation_data_contract.md`. Dự án con 1 (simulator) đã xong; dự án con 3
(pretrain + thí nghiệm trên Kaggle) dùng kết quả của dự án này.

## 1. Mục tiêu và tiêu chí thành công

Mục tiêu: pipeline foundation hiện có (pretrain → embedding → downstream Food và Ads) chạy đúng
trên sim-v1, để dự án con 3 chỉ còn việc chạy thí nghiệm. "Đúng" nghĩa là:

1. Pretrain dùng được context của **cả ba** dịch vụ (Ride, Food, Advertisement). Hiện tại context
   Ride/Food bị bỏ âm thầm vì allowlist mặc định của tokenizer không có hai scenario này.
2. `long_term` (UserIdAdapter) **được học** trong pretrain. Hiện tại không loss nào dùng nó, nên nó
   giữ nguyên giá trị khởi tạo ngẫu nhiên.
3. Downstream Food và Ads đọc gói sim và mã hóa history **giống hệt từng token** với cách pretrain
   mã hóa cùng các event đó. Đây là điều kiện để embedding đã pretrain có ý nghĩa ở downstream.
4. Có nhánh đối chứng "encoder khởi tạo ngẫu nhiên" chỉ bằng một dòng cấu hình. Dự án con 3 cần
   nhánh này để chứng minh **pretraining** có ích, chứ không chỉ kiến trúc.
5. Đường Meituan (Food) và Taobao (Ads) hiện có vẫn chạy như cũ.

Tiêu chí kiểm được:

- (T1) Test: vocab fit trên gói `smoke` chứa token context Ride (`pickup_geohash`, ...) và Food
  (`merchant_id`, `cuisine`, ...).
- (T2) Test: sau một bước tối ưu với `long_term_injection: cls`, gradient của trọng số
  UserIdAdapter khác 0. Với `none`, output của model giống hệt trước khi sửa: test hiện có pass,
  checkpoint Người 3 vẫn nạp được.
- (T3) Test parity: cùng một user và cùng tập event, token history do reader Food và reader Ads tạo
  ra **bằng** token do loader pretrain tạo ra từ `pretrain/events/*.parquet`.
- (T4) Smoke end-to-end ở máy local (CPU, vài phút): pretrain 1 epoch trên `sim-smoke-v1` → Food và
  Ads downstream trên `sim-smoke-v1` ra file metric, chạy được cả nhánh `pretrained` và `random`.
- (T5) Toàn bộ test hiện có pass; `ruff check src tests configs` sạch.

Ngoài phạm vi:

- Chạy pretrain full và thí nghiệm trên Kaggle (dự án con 3).
- Sinh lại dữ liệu. Simulator không đổi.
- Gộp phần tối ưu tốc độ của `origin/feat/pretrain-food-recommendation`. Lý do ở §7.

## 2. Pretrain: allowlist theo gói dữ liệu

- `pretrain.py` đọc `context_allowlist` trong `preprocessing_config.json` của gói (manifest đã nạp
  sẵn file này). Nó tạo `TokenizerConfig(context_allowlist = mặc định ∪ allowlist của gói)`; khi
  trùng scenario thì dùng allowlist của gói. Config này được truyền vào `fit_vocabulary_from_samples`.
  `tokenizer_contract` trong vocab và checkpoint sẽ tự ghi allowlist này, nên downstream (vốn dựng
  tokenizer từ contract) dùng đúng allowlist mà không phải sửa gì.
- Có chặn sai: khi dùng vocab fit sẵn (`--vocab_path`), nếu allowlist của vocab không bao trùm
  allowlist của gói thì dừng với lỗi rõ ràng, không âm thầm bỏ context.
- Gói không có `context_allowlist` (Meituan, OmniBehavior) thì dùng mặc định như cũ.
- Nếu manifest của gói có `sim_config_sha256` và `dataset_version`, pretrain chép hai giá trị này vào
  `preprocess_metadata` của checkpoint; downstream dựa vào chúng để kiểm tra ràng buộc (§5, §6).
- Độ dài chuỗi: mỗi phiên là một nhóm event. Một phiên ride khoảng 40 token, history 30 event
  khoảng 300 token, nên `max_length = 256` hiện tại cắt mất phần lớn history. Config sim dùng
  `max_length = 512`. Plan sẽ đo phân bố độ dài trên `smoke` và ghi tỷ lệ mẫu bị cắt vào
  `docs/sim_foundation_team_notes.md`.
- Config mới: `configs/pretrain_sim.yaml` (local, smoke) và `configs/pretrain_sim_kaggle.yaml`
  (GPU, cho dự án con 3; thông số ban đầu: 6 epoch, `d_model` 128, 4 layer).

## 3. `long_term` được học (theo Grab)

Grab đưa embedding ID người dùng vào **đầu vào** Transformer, nên mọi loss pretrain đều truyền
gradient về nó. Thiết kế ở đây làm theo cách đó:

- `FoundationModelConfig.long_term_injection: "none" | "cls"`, mặc định `"none"` (giữ hành vi cũ và
  tương thích checkpoint cũ). Giá trị này được lưu trong `model_config` của checkpoint.
- `"cls"`: trong `FoundationModel.forward`, trước Transformer, cộng `user_adapter(user_ids)` vào
  đặc trưng token CLS (vị trí 0). Mọi token attend tới CLS, nên loss masked và next-action đều
  truyền gradient về UserIdAdapter. Output `long_term` vẫn là `user_adapter(user_ids)`, API
  downstream không đổi. `short_term` vẫn là mean-pool token event; nó có nhận một phần thông tin
  user qua attention, điều này được ghi rõ trong ghi chú cho team.
- Chống lối tắt: khi `"cls"` bật, token bảng `user_id` không bao giờ bị mask. Nếu bị mask, model
  chỉ việc học dịch lại ID từ CLS, phí tín hiệu học.
- Không đổi định dạng checkpoint. User ID ánh xạ vào 50.000 bucket qua id trong vocab; 20.000 user
  sim nên không va chạm.

Đã cân nhắc nhưng không chọn: ghép `cat(short_term, long_term)` vào head next-action. Target
next-action chỉ là bước đầu của phiên kế tiếp (gần như "dịch vụ kế tiếp", 3–6 lớp), nên tín hiệu quá
nghèo cho một embedding dài hạn.

## 4. Bộ chuyển event sim dùng chung (`src/downstream/sim_events.py`, mới)

Một module duy nhất chuyển event công khai của sim thành `EventRecord` cho `foundation_input`.
Food dùng nó cho `input.history`, Ads dùng cho `events.parquet`, nên token **khớp với pretrain**:

- `scenario` = `{"ride": "Ride", "food": "Food", "ads": "Advertisement"}[service]`; `event_type` giữ
  nguyên.
- `attributes` được tách lại thành:
  - context = các khóa nằm trong allowlist context của scenario;
  - action = các trường số (`fare, distance_km, wait_minutes, order_amount, delivery_distance_km`)
    cộng `type`. `type` là `event_type`, riêng Advertisement thì `ad_impression → "impression"`,
    `ad_click → "click"` (đúng như `action_json` của gói pretrain).
- Thời gian: Food (`+07:00`) và Ads (epoch giây) đều đổi sang UTC naive như loader pretrain.
- Nhóm phiên: gói downstream không có session id. Phiên được dựng lại theo quy tắc chính xác của
  simulator: mỗi phiên bắt đầu bằng đúng một bước đầu phễu (`search_destination`, `search`,
  `ad_impression`) và bước này không xuất hiện ở giữa phiên. Vì vậy mở nhóm mới tại mỗi bước đầu
  phễu; các event trước bước đầu phễu đầu tiên của cửa sổ (cửa sổ Ads có thể cắt giữa phiên) gom
  thành một nhóm. Test parity (T3) là bằng chứng cho quy tắc này.
- User ID giữ nguyên dạng đầy đủ `sim:user:N`, khớp token `user_id` trong vocab pretrain. Reader
  Food Meituan hiện cắt còn `N`; đường sim không cắt.
- Profile: Food đưa nguyên các trường profile của sample, giống loader pretrain. Ads chỉ có profile
  tĩnh (`age_group`, `saved_*_geohash`; gói Ads cố ý không có trường 30 ngày), so với pretrain thiếu
  vài token profile. Đây là giới hạn đã biết, ghi trong ghi chú cho team.

## 5. Food: đường đọc gói sim

- Config `data.format: meituan | sim_v1`, mặc định `meituan` (đường cũ giữ nguyên). File mới
  `configs/food_recommendation_sim.yaml`.
- `sim_v1`:
  - đọc `food/manifest.json` (đường dẫn, sha256 của `samples.jsonl`) và `food/merchants.parquet`
    (catalog thay cho `pois.txt`);
  - ứng viên được kiểm tra `item_id ∈ merchants`;
  - `candidate_policy` lấy từ manifest;
  - `history_truncated` và `history_group_truncated` đọc từ sample;
  - history đi qua `sim_events`.
- Ràng buộc checkpoint–dữ liệu: `vocabulary.dataset_name == "sim_v1"`, và checkpoint phải ghi
  `sim_config_sha256` của gói pretrain (lấy từ manifest pretrain vào `preprocess_metadata`), bằng
  giá trị trong manifest Food. Như vậy cả hai chắc chắn đến từ cùng một thế giới mô phỏng.
- Head và metric (FoodRanker, NDCG@10, MRR) không đổi.

## 6. Ads: tổng quát hóa reader

- Gộp `origin/feat/ad-optimization` vào branch này. Merge thử sạch, không đụng file chung.
- `ad_inputs.py`: các hằng số Taobao chuyển thành cấu hình, mặc định giữ giá trị Taobao:
  - `namespace`, `exposure_policy`, tiền tố ID người dùng và quảng cáo;
  - quy tắc cận split: Taobao `<=`; sim là cận trên **không bao gồm**, theo manifest sim.
- Đường sim:
  - nhận event mọi loại, tra impression qua `source_row_index`;
  - nhận cột profile ngoài `{user_id, as_of_timestamp}`;
  - history đi qua `sim_events`;
  - thời điểm nhãn có được = giờ click nếu click, ngược lại hiển thị + 27 s, tính từ
    `events.parquet` theo đúng quy ước Baseline R12. Gói Ads không có cột này và không cần sinh
    lại dữ liệu.
- Ràng buộc checkpoint: Taobao giữ ràng buộc hiện có theo sha256 của manifest Ads. Sim dùng ràng
  buộc như Food (`dataset_name` + `sim_config_sha256`).
- File mới `configs/ad_optimization_sim.yaml`.

## 7. Nhánh đối chứng và quan hệ với branch của team

- Food và Ads có `foundation.init: pretrained | random`. `random` giữ vocab và `model_config` của
  checkpoint nhưng khởi tạo lại trọng số với seed cố định, vẫn đóng băng như nhánh `pretrained`.
  Đây là đối chứng cho câu hỏi "pretraining có giúp không".
- `origin/feat/pretrain-food-recommendation` (Người 3, cập nhật 2026-10-17) không đụng allowlist
  hay `long_term`. Branch này có ích về tốc độ (prefetch, scheduler) nhưng có ba lỗi:
  - cosine cố định 100k bước;
  - phần đuôi gradient accumulation không bao giờ chạy;
  - tự dùng lại `vocab.json` cũ, xung đột với §2.

  Không gộp lúc này. Phần sửa ở đây trong `pretrain.py` nhỏ và tập trung, nên team gộp sau sẽ chỉ
  xung đột vài dòng. Các lỗi trên được ghi vào ghi chú cho team.
- Ghi chú cho team: `docs/sim_foundation_team_notes.md`. Ghi chú liệt kê thay đổi theo người phụ
  trách file (tokenizer/pretrain: Người 1/3; encoder/objectives: Người 2; reader Ads: Người 4),
  kèm lý do, cờ cấu hình, mặc định tương thích và giới hạn đã biết.

## 8. Kiểm thử

- Đơn vị:
  - allowlist hợp nhất và chặn vocab sai (§2);
  - gradient của `long_term`, bất biến khi `none`, `user_id` không bị mask (§3);
  - chuyển event: scenario, tách context/action, `type` Ads, thời gian, dựng nhóm phiên (§4);
  - reader Food sim và Ads sim: ràng buộc checkpoint, kiểm tra ứng viên, split, nhãn (§5–6).
- Parity (T3): dựng thế giới nhỏ từ preset `smoke` bằng `src.simulation` trong test, xuất đủ ba gói,
  so token history của Food và Ads với token loader pretrain cho cùng event (event trước đầu tuần 7).
- End-to-end (T4): một test chậm (đánh dấu) hoặc lệnh trong plan chạy pretrain → Food → Ads trên
  `smoke`.
- Hồi quy: test Meituan Food, test Taobao Ads (từ branch Ads), toàn bộ suite.
