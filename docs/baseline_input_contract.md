# Baseline Input Contract v1

Implementation: `src/baselines/contracts.py`. Đây là contract nội bộ B0/B1,
không thay thế schema ETL hoặc FoundationBatch. Adapter chỉ ánh xạ và validate,
không sửa candidate, nhãn hay split. ID có ý nghĩa trong namespace dataset.

DatasetMetadata chứa dataset_name/data_version/schema_version=1, task,
item_kind, timezone IANA, candidate_policy và availability_policy có version.
Food hỗ trợ merchant hoặc food, không trộn hai loại trong một dataset. Ad dùng ad.

Query chứa request_id duy nhất, user_id, cutoff_time có timezone, split
train/validation/test và context đã biết khi dự đoán. Candidate chứa request_id,
candidate_id và attributes. Khóa request/candidate duy nhất; query có ít nhất
một candidate. Label được giữ riêng, target integer 0/1, available_at không trước
cutoff; mỗi candidate có đúng một label. Đây là handoff đã có nhãn cho train/eval;
inference chưa có nhãn sẽ cần contract riêng, không tạo label giả.

Interaction chứa event_id duy nhất, user_id, entity_id, event_type, event_time,
available_at và attributes. Cả hai mốc có timezone; availability không trước event.
History cho query chỉ lấy event_time và available_at nhỏ hơn cutoff (strict).
History toàn dataset có thể chứa event tương lai; scorer phải dùng eligibility
filter cho mỗi cutoff. Không coi candidate rows là interactions.

Mapping JSON dự kiến: Food dùng label_metadata.request_id làm request_id,
input.cutoff_time làm cutoff, candidates.item_id làm candidate_id và relevance
làm target; sample_id giữ provenance. Ad dùng impression_id làm request_id,
prediction_time làm cutoff, ad_id làm candidate_id và clicked làm target.
Các ví dụ bên ngoài chưa có đủ available_at cho mọi label/event: adapter phải
nhận policy/provenance do Data team cung cấp, không suy đoán từ example.

Validator v1 kiểm tra cấu trúc, khóa, target và thời gian từng record. Việc chứng
minh chronological split, label maturity trước training cutoff, metadata as-of,
candidate exposure và coverage lịch sử vẫn cần manifest/audit của Data team;
validator này chưa thay thế các kiểm tra tích hợp đó. Các dictionary feature là
payload raw; feature builder phải dùng allowlist để loại labels/metadata.

Output theo Output Contract v1 ở docs/parallel_vibe_implementation_plan.md mục
4.6: Food score finite, Ad score là probability [0,1]. B0/B1 không cần checkpoint
foundation. Timestamp thiếu offset hoặc chỉ ngày không được tự gán timezone/giờ.
Missing giữ khác 0; cửa sổ 7/30 ngày cần history đủ coverage hoặc aggregate
point-in-time được bàn giao. History 30 events không chứng minh coverage này.
