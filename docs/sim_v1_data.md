# Dữ liệu mô phỏng `sim-v1` — mô tả và cách tải

Bộ dữ liệu mô phỏng siêu app (Ride + Food + Ads, chung một tập user, kiểu Grab) dùng để chứng
minh foundation model có ích cho Food recommendation và Ads. Dữ liệu là **giả lập**, không chứa
dữ liệu người dùng thật, không ràng buộc giấy phép như Meituan/Taobao.

- Sinh bởi `src/simulation` (branch `feat/sim-foundation`, commit `3a20411f`), preset `full`,
  seed 42, `sim_config_sha256 = 631010161f2f…b300`, `dataset_version = sim-v1`.
- Mô tả chi tiết từng file và từng cột: [`docs/simulation_data_contract.md`](simulation_data_contract.md).
- Cách dùng với pretrain / Food / Ads: [`docs/sim_foundation_team_notes.md`](sim_foundation_team_notes.md).

## 1. Có gì trong bộ dữ liệu

20.000 user, 8 tuần (thứ Hai 2026-01-05 → 2026-03-01, giờ `+07:00`), 5.457.523 event,
379.565 phiên Food có đơn, 914.724 lần hiển thị quảng cáo (CTR 5,09%). Split theo thời gian:
tuần 1–6 train, tuần 7 validation, tuần 8 test.

| Thư mục | Dung lượng | Dùng cho | Nội dung chính |
|---|---|---|---|
| `pretrain/` | 475 MB (40.003 file; tải về dạng `sim-v1-pretrain.tar` 542 MiB) | Pretrain foundation (Người 1/3) | Schema `omnibehavior_pretraining_v2`: `events/<user>.parquet` (4.086.468 event, chỉ 6 tuần đầu), `snapshots/<user>.parquet` (199.518 train + 39.998 validation), `manifest.json`, `preprocessing_config.json` (allowlist context Ride/Food/Advertisement) |
| `food/` | 813 MB | Food recommendation | `samples.jsonl` (80.000 request: 60k/10k/10k; mỗi request 10 quán ứng viên, đúng 1 quán được đặt), `merchants.parquet` (600 quán), `manifest.json` |
| `ads/` | 102 MB | Ads click prediction (Người 4) | `events.parquet` (mọi event mọi dịch vụ), `ad_opportunities.parquet` (260.000 request: 200k/30k/30k), `profiles.parquet`, `ad_catalog.parquet`, `manifest.json` |
| `baseline/` | 3,9 GB | Baseline B0–B4 (không foundation) | `food.json`, `ads.json` theo định dạng `src.baselines.runner.load_dataset` (quy ước B1) |
| `oracle/` | 9 MB | Trần so sánh | Xác suất thật của mô phỏng cho từng request Food/Ads |
| `data_report.json` | 6 MB | Kiểm tra | Kết quả 35 kiểm tra (đều đạt), độ khó, SHA-256 + kích thước của **mọi** file |

Độ khó đo trên split test (để biết mô hình "tốt" là bao nhiêu):

| Chỉ số | Giá trị |
|---|---|
| Food NDCG@10: phổ biến / chỉ dùng ứng viên / oracle | 0,509 / 0,513 / 0,899 |
| Food NDCG@10: GBDT chỉ dùng profile + phổ biến + context (mốc "không cần history") | ≈ 0,74 |
| Ads AUC: phổ biến (CTR theo quảng cáo) / oracle | 0,638 / 0,788 |

## 2. Quyền truy cập

Dữ liệu là output của kernel Kaggle **public** `hunhthhichu/vric-sim-generate`
(https://www.kaggle.com/code/hunhthhichu/vric-sim-generate) — không cần xin quyền. Tải bằng
Kaggle CLI (cách B, C) chỉ cần một tài khoản Kaggle bất kỳ và API token của chính bạn.
Đừng chạy lại kernel này: version mới sẽ có `data_report.json` khác SHA (file ghi thời gian chạy)
và script kiểm tra sẽ báo lệch, dù dữ liệu không đổi.

## 3. Cách tải

### Cách A — dùng trực tiếp trong notebook Kaggle (không tải về máy)

Trong notebook: **Add Input** → mục **Notebooks** → tìm `vric-sim-generate`. Dữ liệu nằm ở
`/kaggle/input/vric-sim-generate/`:

```bash
# gói pretrain cần giải nén một lần
tar -xf /kaggle/input/vric-sim-generate/sim-v1-pretrain.tar -C /kaggle/working
python -m src.training.pretrain --config configs/pretrain_sim_kaggle.yaml \
    --data_dir /kaggle/working/sim-v1/pretrain
```

`configs/food_recommendation_sim_kaggle.yaml` và `configs/ad_optimization_sim_kaggle.yaml` đã trỏ
sẵn tới `/kaggle/input/vric-sim-generate/sim-v1/{food,ads}`.

### Cách B — tải về máy bằng script (khuyên dùng)

Cần Kaggle CLI và API token (Kaggle → Settings → API → *Create New Token*, đặt file
`kaggle.json` vào `~/.kaggle/` — Windows: `C:\Users\<tên>\.kaggle\`):

```bash
pip install kaggle
python scripts/download_sim_v1.py --out data_sim                        # tất cả, ≈ 5,3 GB
python scripts/download_sim_v1.py --out data_sim --parts food ads pretrain   # chỉ phần cần (≈ 1,4 GB)
```

Script tải đúng các phần được chọn, giải nén `sim-v1-pretrain.tar` thành `data_sim/sim-v1/pretrain/`
rồi **kiểm tra SHA-256 của từng file** so với `data_report.json` (chính `data_report.json` được
ghim SHA trong script). Kết thúc bằng `OK: ... verified` là dữ liệu đúng từng byte. Kiểm tra lại
một bản đã có: thêm `--verify-only`. `--parts` nhận `pretrain food ads baseline oracle`.

Kết quả:

```
data_sim/sim-v1/
  data_report.json
  pretrain/   food/   ads/   baseline/   oracle/
```

Trỏ config vào đó, ví dụ `--data_dir data_sim/sim-v1/pretrain` cho pretrain, `data.root:
data_sim/sim-v1/food` cho Food (`expected_version: sim-v1`).

### Cách C — tải tay bằng Kaggle CLI

```bash
kaggle kernels output hunhthhichu/vric-sim-generate -p data_sim --file-pattern "^sim-v1/food/"
```

Mẫu cho từng phần: `^sim-v1/food/`, `^sim-v1/ads/`, `^sim-v1/baseline/`, `^sim-v1/oracle/`,
`^sim-v1-pretrain\.tar$`, `^sim-v1/data_report\.json$`. Không có `--file-pattern` là tải hết.
Sau đó nên chạy `python scripts/download_sim_v1.py --out data_sim --parts <phần đã tải> --verify-only`.

### Cách D — tự sinh lại (không cần Kaggle)

Simulator tất định: cùng code + seed → cùng dữ liệu.

```bash
git clone https://github.com/haichauht/VRIC-foundation-model-ver2.git && cd VRIC-foundation-model-ver2
python -m src.simulation --preset full --out experiments/sim     # ≈ 15–20 phút, RAM đỉnh ≈ 9 GB
```

So với bản Kaggle: `food/samples.jsonl` (`f41178f8e9e89ac2e1284dc9b9519610ea2b0a2d020e67efed40da5b24035e02`)
và `baseline/*.json` phải khớp từng byte; file parquet chỉ khác byte khi khác phiên bản pyarrow
(Kaggle dùng pyarrow 23), nội dung giống nhau. Vì vậy `--verify-only` sẽ báo lệch ở parquet với
bản tự sinh — dùng bản Kaggle khi cần khớp tuyệt đối.

## 4. Lưu ý khi dùng

- Mỗi gói có `manifest.json` ghi nguồn gốc và SHA; các reader trong repo kiểm tra chúng và từ chối
  dữ liệu lệch (exit code ≠ 0).
- Checkpoint pretrain trên dữ liệu này phải đi cùng gói Food/Ads của **cùng** `sim_config_sha256`
  (reader Food/Ads kiểm tra), nên đừng trộn với bộ `sim-smoke-v1`.
- `baseline/*.json` lớn (2 GB mỗi file, `load_dataset` cần ≈ 12 GB RAM cho `ads.json`); với máy
  yếu, chạy Baseline trên một tập con user.
- Không commit dữ liệu vào Git; để ngoài repo hoặc trong `experiments/` (đã gitignore).
- Checkpoint foundation pretrain trên `sim-v1` (xong 2026-10-17, best = epoch 3) nằm ở output của
  kernel private `hunhthhichu/vric-sim-pretrain` (`pretrain_sim/checkpoints/best_checkpoint.pt`,
  `vocab.json`, `metrics.jsonl`); cần được thêm làm collaborator của kernel đó để tải. Kết quả và
  phân tích: [sim_pretrain_v1_results.md](sim_pretrain_v1_results.md).
