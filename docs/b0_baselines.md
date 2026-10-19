# B0 Food và Advertising

Implementation: `src.baselines.baseline.B0Scorer`. Không phụ thuộc foundation
checkpoint. Các variant Food: random, popularity, meal_popularity, recent.
Các variant Ad: global_ctr, ad_ctr, category_ctr.

```python
from src.baselines.baseline import B0Scorer
from src.evaluation.predictions import write_predictions

model = B0Scorer("popularity").fit(dataset, cutoff=split_start)
table = model.predict(dataset, "test", run_id="b0-popularity-001")
write_predictions("predictions.test.parquet", table, dataset, "test")
model.save("b0.json")
restored = B0Scorer.load("b0.json")
```

State đóng băng tại fit cutoff. Scoring query trước cutoff bị từ chối; fit state
mới cho mỗi split theo protocol. predict không cập nhật history hoặc đọc target
để tính score. Dataset có cùng namespace/task/item_kind mới được predict; data
version có thể khác để hỗ trợ handoff train/eval, caller kiểm tra provenance.

Food chỉ đếm Interaction.event_type="order" từ history thực, với event_time và
available_at đều < fit cutoff. Nguồn Meituan dùng order_merchant hoặc action khác
phải có mapping semantics rõ từ adapter Data team. Không đếm candidate rows,
view/click hay order thất bại. Data team chịu trách nhiệm loại order thất bại
hoặc ánh xạ loại event khác; scorer không suy thành công từ tên observation.

Popularity dùng tổng order/item. Recent dùng [cutoff-window_days, cutoff), mặc
định 7 ngày, lựa chọn window trên validation. Meal popularity dùng count theo
meal_period trong Interaction.attributes và Query.context; khi item không có
count ở meal đó thì fallback global count. Đây là fallback trực tiếp, chưa có
interpolation/smoothing meal. Item chưa gặp score 0. Meal encoding phải nhất quán.
Random dùng SHA256(seed, namespace, request, item), score [0,1), không phụ thuộc
thứ tự candidate, batch, hash process hoặc seed global Python.

Ad đếm từng Label của train query như một impression, chỉ khi query cutoff và
label.available_at đều < fit cutoff. Contract yêu cầu query/candidate đại diện
impression thật; không dùng mẫu âm giả để fit CTR. Test/validation label luôn
bị bỏ qua. Không cần raw interaction log cho ba variant CTR hiện tại.

Global CTR = clicks/impressions. Category CTR có smoothing về global prior;
ad CTR smoothing về category prior nếu có category, còn lại về global prior.
alpha mặc định 10, phải >0. Chưa có history đủ điều kiện thì dùng empty_prior
(mặc định 0.5, đây là fallback cấu hình, không phải CTR ước lượng từ dữ liệu).
Không có category thì không cập nhật category counts. Ad/categorical value mới
fallback prior. Feature category phải được Data team xác nhận đã biết tại cutoff.

Save/load dùng JSON artifact format_version=1, không pickle. Artifact chứa
aggregate state và config, không chứa raw profiles/history. API save hiện ghi
file trực tiếp; versioned run directory, atomic model writer, fingerprints và
manifest sẽ bổ sung ở runner/CLI. Chỉ load artifact do pipeline này tạo, validator
artifact đầy đủ chưa triển khai. Không dùng path checkpoint foundation giả.

Kiểm thử có đối chiếu count/CTR tính tay, smoothing, item mới, delayed label,
future invariance, query trước cutoff, namespace khác, reorder/batch và reload;
Food/Ad predictions được ghi Parquet và chạy evaluator public trên fixture.
Chưa benchmark dữ liệu thật hoặc tuning các variant.
