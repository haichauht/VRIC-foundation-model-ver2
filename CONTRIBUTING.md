# Contributing

## Branches

Tạo branch theo task, không dùng branch dài hạn theo nhóm:

```text
feat/<short-description>
fix/<short-description>
docs/<short-description>
exp/<short-description>
```

## Pull Request

- Không push trực tiếp vào `main`.
- Mỗi thay đổi lớn cần Pull Request và ít nhất một reviewer.
- Mô tả mục tiêu, cách kiểm tra và ảnh hưởng đến data/model/evaluation.
- Không commit dataset lớn, checkpoint, secret hoặc output experiment.

## Trước khi mở PR

```bash
python -m pytest
python -m ruff check .
```

## Commit

Dùng dạng ngắn gọn theo nhóm thay đổi, ví dụ:

```text
feat: add event sequence dataset
fix: prevent temporal data leakage
docs: define event schema
```
