# B1 LightGBM — bước 5

`src.baselines.lightgbm.B1Classifier`: binary objective cho Food relevance/order
và Ad CTR, mỗi task một model. Food rank bằng P(order), Ad xuất P(click).

```python
from src.baselines.lightgbm import B1Classifier

model = B1Classifier(seed=42).fit(
    dataset, training_cutoff=validation_start, history_start=audited_history_start,
)
predictions = model.predict(dataset, "test", "b1-run-001",
                            frozen_cutoff=test_start, history_start=audited_history_start)
model.save("experiments/b1-run-001/artifacts/model")
restored = B1Classifier.load("experiments/b1-run-001/artifacts/model")
```

fit yêu cầu train và validation theo thời gian, hai lớp train, nhãn train sẵn có
strict trước training_cutoff. Train features dùng prefix; validation dùng history
đóng băng tại training_cutoff; encoder fit train. Early stopping dùng validation
binary logloss, không dùng test. Chọn config Food cần evaluator ranking validation
ở runner/tuning; logloss early stopping chưa thay thế selection theo NDCG.

Categorical được truyền rõ cho LightGBM. Model deterministic với seed/num_threads=1;
không dùng class reweight hoặc negative sampling ngầm. Không clip probability.
Save gồm model.txt native LightGBM, encoder.json và metadata.json; load kiểm tra
feature names. Không ghi đè directory có sẵn. Atomic artifact/checksum và CLI
nằm ở bước runner; chưa có refit train+validation hoặc tuning search.

Môi trường .venv có LightGBM 4.7.0, dependency baseline khai báo >=4,<5.
Implementation dùng native train/early_stopping callbacks, không dùng eval_set
đã deprecated ở sklearn API phiên bản mới.

## Dữ liệu training và so sánh

Các lần train hiện tại chỉ nằm trong tests trên fixtures tổng hợp, có tín hiệu
giả để kiểm tra model học, save/load, xuất prediction và evaluator. Không có kết
quả thực nghiệm từ data thật; model test không phải artifact benchmark.

Meituan trong data/downstream1_meituan_data và raw data/meituan_data là nguồn
Food dự kiến sau handoff chuẩn. data/taobao_data là nguồn Ad sau pipeline bàn giao.
Pretraining OmniBehavior/Meituan không được tự dùng thay nhãn downstream.
Không sửa data/ETL; Data team bàn giao history, cutoff, labels, candidates,
split/version và feature provenance đủ contract.

So sánh foundation/B1 dùng cùng downstream train/validation/test, candidate,
target, metric, sampling và policy lịch sử; B1 được feature engineering/tuning
đầy đủ trên validation, không cố giảm chất lượng để foundation thắng. Foundation
có pretraining là khác biệt cố ý cần ghi riêng và phải kiểm soát leakage.

B1 hiện là core chưa phải baseline đã tối ưu: feature profile/demographic, rating,
user-category CTR, metadata point-in-time bổ sung và tuning/refit/calibration cần
đánh giá theo nguồn thực tế. History thiếu coverage hoặc feature nghèo có thể
làm B1 yếu; thiếu dữ liệu không được che bằng missing flags rồi kết luận foundation
tốt hơn baseline feature-based đầy đủ. Cần report coverage, ablation và hạn chế.
Fixing leakage có thể làm metric thấp hơn bản dùng future nhưng là baseline hợp lệ.

Test đạt chỉ chứng minh pipeline vận hành, không chứng minh chất lượng thực tế
hoặc Ad probability đã calibrated; calibration nếu cần chỉ fit validation.
