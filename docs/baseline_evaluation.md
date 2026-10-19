# Adapter và evaluator B0/B1 — bước 2

`src.baselines.adapters.adapt_records` ánh xạ Food/Ad JSON sang BaselineDataset.
Caller cung cấp DatasetMetadata, split và mapping `(request_id, candidate_id)`
→ datetime nhãn sẵn có. Không tự suy nhãn, availability hoặc timezone.
Food yêu cầu observation_complete=true và candidate_source khớp metadata.
Profile/recent_events chưa được chuyển thành business feature; bước feature
builder sẽ nhận history/profile có audit riêng. Adapter không sửa ETL/dataset.

Public API evaluator:

```python
from src.evaluation.evaluator import DatasetRegistry, evaluate_predictions

registry = DatasetRegistry()
registry.register(dataset)  # BaselineDataset đã được Data team bàn giao/audit
metrics = evaluate_predictions(
    "predictions.test.parquet", dataset.metadata.task,
    dataset.metadata.data_version, "test", registry=registry,
    ks=(5, 10, 20), zero_positive_policy="include",
)
```

Có thể đăng ký vào DEFAULT_REGISTRY để dùng đúng API bốn positional arguments
của handoff. Registry là bộ nhớ của process, chưa có manifest resolver từ disk;
CLI/config ở bước 6 sẽ nạp dataset và đăng ký rõ ràng. Không chọn dataset theo
tên prediction file. Không đăng ký đè một version đã có.

`src.evaluation.predictions.PREDICTION_SCHEMA` mô tả Arrow schema Output Contract
v1. `write_predictions` validate trước khi ghi, sort theo request/score/item,
ghi temporary cùng volume, đọc lại validate rồi atomic replace. Validator kiểm
tra dtype, required fields, run_id nhất quán, version, split, user, cutoff và
candidate set chính xác; chỉ cho thêm raw_logit float64 hữu hạn. Cutoff phải
chính xác đến giây vì Output Contract v1 dùng epoch seconds; không âm thầm làm
tròn microseconds. Caller phải quản lý run directory/manifest/checksum ở bước CLI.

Food: macro theo request, tie-break candidate_id ascending, Recall/NDCG/HitRate@K
và MRR. Mặc định nhóm không positive được tính 0; cấu hình exclude loại nhóm
nhưng vẫn báo zero_positive_request_count và evaluated_request_count. Không
đổi policy sau khi xem test. Nếu exclude hết nhóm thì báo lỗi.

Ad: ROC-AUC, PR-AUC định nghĩa average precision, LogLoss, Brier. Khi chỉ có một
lớp, discrimination_defined=0 và ROC-AUC/PR-AUC là NaN; LogLoss/Brier vẫn hợp lệ.
Exporter JSON ở bước CLI phải đổi giá trị undefined thành null với trạng thái,
không ghi NaN thành JSON chuẩn. Metrics trực tiếp là helper nội bộ; production
gọi evaluator để validate prediction và nhãn trước.

Môi trường hiện tại: .venv tạo với system-site-packages để dùng pyarrow,
scikit-learn, pytest/torch hiện có; Ruff được cài trong .venv. Đây chưa phải
environment lock độc lập. Cài mới dùng extras dev/evaluation trong pyproject;
LightGBM đã triển khai; cài extra baseline để sử dụng B1.
