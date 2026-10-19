# B2–B4 và evaluator foundation

> Chưa có config B2–B4 cho `sim-v1`; các lệnh dưới đây là mẫu, cần tự tạo config.

Baseline team sở hữu implementation trong `src/baselines/sequence.py` và
`src/evaluation/`; không sửa ETL, encoder hoặc tokenizer của foundation team.

## Chạy

CLI hiện có hỗ trợ `model_kind` B2/B3/B4:

```powershell
.venv\Scripts\python.exe -m src.baselines validate --config configs/baselines/food_b2.json
.venv\Scripts\python.exe -m src.baselines run --config configs/baselines/food_b2.json
```

Templates chưa có dữ liệu ở dataset_path. Data team bàn giao canonical JSON
như B0/B1. Optional `sequence_profiles` là object keyed bởi request_id; mỗi
entry có `values`, `as_of`, `available_at` (ISO timestamp có offset). Cả hai
timestamp phải strict trước history cutoff. Không có entry nghĩa là profile
missing; không lấy profile trong file IDE rồi suy provenance.

History được lọc theo user, event_time và available_at, giữ thứ tự event_time /
event_id. Chỉ entity_id và event_type được đưa vào sequence; outcome/labels
không được đưa vào tokenizer. Train dùng prefix riêng mỗi query; evaluation
đóng băng ở prediction_cutoff. Không tự sửa dữ liệu hoặc tạo negative.

B2 fit vocabulary train, dùng FoundationModel public. B3/B4 giữ vocabulary
checkpoint. Cả ba nối long_term + short_term + candidate embedding qua MLP
một hidden layer, BCE binary. Candidate vocabulary fit train; item mới dùng
unknown embedding. Food chọn checkpoint NDCG@10, Ads chọn LogLoss validation.
B3 chỉ cập nhật head/candidate embeddings; B4 backbone LR bằng 0,1 head LR.

## Checkpoint và giới hạn

B3/B4 hiện chỉ nhận checkpoint thử có `fixture_only=true`, format_version=1,
model_config, model_state_dict, vocabulary_file và vocabulary_sha256. Vocabulary
phải có tokenizer contract và namespace phù hợp. Không coi checkpoint random
là pretrained; config cũng phải bật fixture_only. Test tạo bundle trong tmpdir.

Production bị từ chối đến khi foundation team bàn giao public checkpoint loader,
vocabulary/preprocessing metadata và provenance pretraining cutoff. Không thay
interface hoặc viết loader production thay team foundation.

Save/load downstream dùng native tensor state với weights_only, vocabulary JSON,
metadata/checksum. Model directory chưa atomic; prediction dùng atomic writer
hiện có. CPU, batch_size=1; runtime/parameter count/inference per-request được
ghi vào manifest. Peak memory là RSS high-water của toàn process (gồm native
tensors), không phải allocation riêng run; dùng process mới để so sánh công bằng.
Nếu OS không cung cấp phép đo thì ghi null. Chưa tối ưu cho dataset lớn.

## Evaluator pretraining

`evaluate_pretraining_predictions(components, vocabulary_policy=...,
candidate_policy=..., k=10)` trả report JSON-compatible, tách masked/next_key/
next_value. Component gồm kind, targets và boolean valid_mask; masked bổ sung
masked_mask và special_mask. Numeric nhận predictions trong cùng không gian
chuẩn hóa. Categorical nhận probabilities [N,C] normalized từ head thực và
candidate_ids tùy chọn. Hierarchical head phải xuất joint probabilities đúng.

Vocab lớn có thể xuất target_probabilities và ranked_ids [N,K]. Không có ranking
thì accuracy/recall unavailable; MRR chỉ có khi ranking đầy đủ. CE/PPL không
được tính từ sampled distribution rồi gọi là full-vocabulary metric. Target
count=0 trả null/undefined; special/PAD cần được producer đánh dấu special_mask.
Evaluator không xây objective, sampler hoặc pretraining trainer.

`summarize_seeds` tổng hợp >=3 seeds; `paired_user_bootstrap` nhận per-request
metrics ghép đúng request order theo user, trả confidence interval của delta.
Tuning tối đa 10 configs/task trên validation; test chạy sau chọn config.

## Kiểm thử và handoff

Tests mô phỏng Food/Ads: B2/B3/B4 train/predict/reload, frozen/update backbone,
future invariance, profile cutoff, metrics tính tay và invalid probabilities.
Kết quả chỉ xác nhận pipeline, không phải benchmark. Khi data thật đến, chạy
B0/B1 nhỏ trước, kiểm tra split/candidate/history coverage, rồi train B2 từ đầu
và B3/B4 từ checkpoint pretrained thật. E5/E6/E7 là bước sau benchmark cơ bản.
