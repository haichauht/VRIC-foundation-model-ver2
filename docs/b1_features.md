# Feature builder B1 — bước 4

Implementation: `src.baselines.features.FeatureBuilder`,
`src.baselines.feature_schema.FeatureEncoder`. LightGBM classifier đã triển khai;
xem `docs/b1_lightgbm.md`.

```python
from src.baselines.features import FeatureBuilder
from src.baselines.feature_schema import FeatureEncoder

builder = FeatureBuilder()
train_rows = builder.build(dataset, "train", history_start=audited_history_start)
encoder = FeatureEncoder().fit(train_rows)
train_matrix = encoder.transform(train_rows)
valid_rows = builder.build(dataset, "validation", frozen_cutoff=validation_start,
                           history_start=audited_history_start)
valid_matrix = encoder.transform(valid_rows)
encoder.save("feature_schema.json")
```

FeatureRow giữ request/candidate/split để nối target bên ngoài; các ID và target
không nằm trong values/matrix. Training dùng prefix riêng cho từng query; frozen
cutoff bị từ chối cho train. Validation/test có thể dùng state đóng băng ở đầu
split để khớp protocol B0; nếu bỏ frozen_cutoff sẽ dùng per-query history, đây là
policy khác cần ghi rõ trong experiment config, không trộn trong so sánh.

History eligibility strict event_time < cutoff và available_at < cutoff. Cửa sổ
count là [cutoff-days, cutoff). Recency tính từ query time đến event cuối trong
history đủ điều kiện; do đó recency vẫn tăng khi state đóng băng. Timestamp ties
được loại trước khi feature được tạo, không nhìn nhãn cùng thời điểm.

Food chỉ nhận order thành công đã được Data team chuẩn hóa. Feature:
user/item/user-item counts 7/30 ngày; user và user-item recency; repeat ratio;
category affinity. Ad nhận event_type=impression với attributes.clicked integer
0/1; available_at là thời điểm outcome impression đủ điều kiện quan sát, không
phải thời điểm impression được hiển thị. Chỉ impression mature mới được đếm,
CTR thiếu history là None. Ad bổ sung clicks và CTR 7/30 ngày cho user/item/pair.
Feature builder không tính impression logs từ candidate/negative rows hoặc labels
của query; Data team phải bàn giao mature history riêng.

Context allowlist: meal_period, placement, zone. Candidate allowlist: category,
brand, price. Các field khác như labels, audit, request IDs không được đưa vào
matrix. Hour/weekday tính theo timezone dataset. Data team xác nhận metadata đã
sẵn có tại query; hiện contract chưa có attribute-level as-of validator.
Profile demographics, user-category CTR và item recency là các mở rộng sau;
không suy chúng từ history đang thiếu field.

history_start là assertion của Data team về lịch sử đầy đủ từ thời điểm đó, không
phải min(event_time). history_complete_7d/30d chỉ true khi history_start đủ sớm;
không truyền assertion thì coverage là 0. Counts vẫn phản ánh dữ liệu quan sát,
không được diễn giải là tổng đầy đủ khi cờ 0. Lịch sử toàn phần dùng cho recency,
repeat ratio/category affinity; cửa sổ count dùng các mốc nêu trên.

Encoder fit chỉ chấp nhận non-empty train rows và cố định schema/column order.
Category codes: 0=missing, 1=unknown, >=2=train vocabulary, giữ khác nhau giữa
giá trị string và số. Numeric missing/non-finite dùng 0 + missing indicator;
zero quan sát được có indicator 0. Không scale numeric cho tree baseline.
categorical_indices phải được truyền cho LightGBM để codes không trở thành
thứ tự numerical tùy ý. Save/load JSON lưu columns/vocabularies/version.

Builder hiện dùng lịch sử trong bộ nhớ, cache theo cutoff rồi lọc user/item;
phù hợp fixtures và integration nhỏ. Chưa tối ưu sweep/index cho hàng triệu query;
trước benchmark lớn cần đo thời gian và thêm index/incremental aggregation khi
cần, giữ nguyên các tests point-in-time. Artifact writer/feature cache đầy đủ
sẽ nối ở runner/CLI; hiện encoder save dùng file trực tiếp.
