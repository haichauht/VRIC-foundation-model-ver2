# Data schema

## Event-level record

| Field | Required | Description |
|---|---:|---|
| `user_id` | Yes | Định danh user đã được chuẩn hóa |
| `timestamp` | Yes | Thời điểm event |
| `event_type` | Yes | Loại hành vi, ví dụ `search`, `quote`, `book` |
| `service` | Yes | Domain/service, ví dụ `ride`, `food`, `ads` |
| `entity_id` | Yes | Item, route, merchant hoặc campaign |
| `value` | Recommended | Giá trị hoặc metadata định lượng |

Pipeline phải bảo toàn thứ tự event theo user và không để thông tin tương lai lọt
vào train features.
