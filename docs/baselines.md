# Chạy B0/B1 trong workspace

B2–B4 dùng cùng CLI; xem [b2_b4_baselines.md](b2_b4_baselines.md) cho sequence
contract, checkpoint fixture và evaluator pretraining.

Chạy từ repository root bằng Python của .venv:

```powershell
.venv\Scripts\python.exe -m src.baselines validate --config configs/baselines/sim_food_b0.json
.venv\Scripts\python.exe -m src.baselines run --config configs/baselines/sim_food_b0.json
```

`validate` kiểm tra config cơ bản và handoff; chưa fit model hoặc kiểm tra toàn bộ
điều kiện training B1. `run` thực hiện fit → predict → validate Parquet → evaluate
→ report. Lỗi trả exit code 2; success in JSON có artifact_dir. Chưa có lệnh
train/predict tách riêng, tuning search, refit hoặc bootstrap report tổng hợp.

Config JSON cho `sim-v1` nằm trong configs/baselines: sim_food_b0, sim_food_b1, sim_ads_b0,
sim_ads_b1 (đúng thiết lập của lần chạy B0/B1 chính thức: B1 train tuần 1–6, early stopping tuần 7,
đánh giá tuần 8 = split test). Relative paths tính từ thư mục config; các config đọc
`data_sim/sim-v1/baseline/{food,ads}.json` (tải theo docs/sim_v1_data.md). sim_food_b1f và
sim_ads_b1f là B1 + đặc trưng foundation (khóa `foundation_features`; embedding xuất bằng
`python -m src.downstream.user_embeddings`, xem docs/sim_foundation_team_notes.md).

Input canonical JSON có metadata, queries, candidates, labels, history (optional).
Tên field theo dataclasses ở baseline_input_contract.md; datetime lưu ISO với
offset. Ví dụ Query: request_id/user_id/cutoff_time/split/context. Label chứa
request_id/candidate_id/target/available_at. Interaction chứa event_id/user_id/
entity_id/event_type/event_time/available_at/attributes. JSON không thay thế ETL:
Data team hoặc exporter handoff phải cung cấp nội dung, không suy availability.

B0 dùng fit_cutoff. B1 dùng fit_cutoff làm training cutoff, prediction_cutoff làm
history state boundary cho evaluation; history_start optional phải là coverage
assertion đã audit. B1 train/validation labels phải đủ contract. ks và
zero_positive_policy áp dụng evaluator; Ad PR-AUC là average precision.

Artifact chỉ ghi bên dưới workspace experiments/<UUID>/artifacts:

```text
model.json hoặc model/{model.txt,encoder.json,metadata.json}
predictions.<split>.parquet
config.json
dataset_metadata.json
run_manifest.json
validation_report.json
metrics.json
report.md
```

UUID tránh ghi đè run khác. Manifest baseline-run-v1 là extension cho baseline,
foundation=null, code_revision=unknown: runner không đọc/ghi Git. Checksum bao gồm
input JSON, config và model/prediction đã tạo. Chưa dùng manifest/checksum Data
team nhiều-file hoặc xác minh source immutability trong lúc chạy. File JSON và
prediction dùng atomic write; model directory hiện chưa atomic. Run lỗi sau khi
tạo output có failure.json; không gọi evaluator trước khi prediction validated.
Validation report là tóm tắt pass, chưa xuất từng violation count/coverage.

Single-class Ad: ROC-AUC/AP undefined ghi null trong metrics JSON và undefined
trong report. Elapsed_seconds là tổng thời gian fit/predict/evaluate; chưa đo peak
memory hoặc tách inference latency. Chưa report mean/std nhiều seed hay CI.
Registry được nạp từ handoff mỗi run, không suy dataset theo prediction filename.

Tests runner tạo fixture tổng hợp trong pytest temporary dir và artifact smoke
nhỏ trong workspace experiments theo UUID. Không sửa data thật, không commit/push,
không kết nối hoặc thao tác repo lưu trữ. Các tests chưa chứng minh B1 chất lượng
trên Food/Ad thực tế; xem b1_lightgbm.md về fairness/feature coverage trước benchmark.
