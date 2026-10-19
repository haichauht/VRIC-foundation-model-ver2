# VRIC Foundation Model — ver2 (dữ liệu mô phỏng `sim-v1`)

User foundation model kiểu Grab ("User foundation models for Grab"), huấn luyện trên dữ liệu mô phỏng
một siêu app **Ride + Food + Ads** với chung một tập user, rồi dùng cho hai bài toán downstream:
**gợi ý quán ăn (Food)** và **dự đoán click quảng cáo (Ads)**.

Bản ver2 chỉ giữ code và tài liệu cho bộ dữ liệu mô phỏng. Dữ liệu cũ (Meituan, Taobao, Omni), code xử
lý dữ liệu cũ, config/tài liệu/script GCP cũ đã bỏ. Một số module dùng chung (`src/downstream`,
`src/foundation_input`) vẫn còn nhánh định dạng cũ bên trong nhưng không còn được dùng; mọi config
trong repo đều là `sim_v1`.

## Kết quả chính (tập test = tuần 8 của `sim-v1`)

| | Food NDCG@10 | Ads ROC AUC |
|---|---|---|
| B0 (phổ biến / CTR quảng cáo) | 0,508 | 0,641 |
| B1 (LightGBM + đặc trưng) | 0,747 | 0,700 |
| Foundation đóng băng + head hiện tại | 0,726 | 0,628 |
| **B1 + embedding foundation** | **0,789** | **0,710** |
| B1 + embedding encoder ngẫu nhiên (đối chứng) | 0,747 | 0,702 |

Pretraining giúp khi embedding được dùng đúng cách; encoder ngẫu nhiên thì không. Chi tiết, chẩn đoán
và các giới hạn: [docs/sim_pretrain_v1_results.md](docs/sim_pretrain_v1_results.md).

## Cài đặt

```bash
python -m venv .venv
python -m pip install -e ".[dev,baseline,sequence]"
python -m pytest
```

## Dữ liệu

Bộ `sim-v1`: 20.000 user, 8 tuần, sinh trên Kaggle (kernel public `hunhthhichu/vric-sim-generate`).
Cách tải và mô tả đầy đủ: [docs/sim_v1_data.md](docs/sim_v1_data.md).

```bash
python scripts/download_sim_v1.py --out data_sim --parts food ads pretrain   # ≈ 1,4 GB
```

Hoặc tự sinh lại (tất định, cùng seed → cùng dữ liệu):

```bash
python -m src.simulation --preset smoke --out experiments/sim   # bộ nhỏ để thử
python -m src.simulation --preset full  --out experiments/sim   # bộ đầy đủ, ≈ 15–20 phút, RAM ≈ 9 GB
```

## Chạy pipeline

Config trong `configs/` có hai bộ: `*_sim.yaml` (máy local, trỏ tới `experiments/sim/sim-smoke-v1`) và
`*_sim_kaggle.yaml` (Kaggle, đọc output kernel `vric-sim-generate`). Muốn chạy bộ full ở máy local thì
sao chép config và sửa `data.root` / `data_dir`.

```bash
# 1. Pretrain foundation
python -m src.training.pretrain --config configs/pretrain_sim.yaml

# 2. Downstream (foundation đóng băng + head); foundation.init: pretrained | random
python -m src.downstream.food_recommendation --config configs/food_recommendation_sim.yaml
python -m src.downstream.ad_optimization --config configs/ad_optimization_sim.yaml

# 3. Baseline B0/B1 (đọc data_sim/sim-v1/baseline/*.json)
python -m src.baselines run --config configs/baselines/sim_food_b1.json
python -m src.baselines run --config configs/baselines/sim_ads_b1.json

# 4. B1 + embedding foundation: xuất embedding theo request rồi chạy B1 có foundation_features
python -m src.downstream.user_embeddings --task food --config configs/food_recommendation_sim.yaml     --out experiments/embeddings --device auto
python -m src.baselines run --config configs/baselines/sim_food_b1f.json
```

Head Food/Ads trong config sim dùng thêm thuộc tính catalog của ứng viên (`model.candidate_attributes`);
`training.encoder_device: auto` chạy encoder trên GPU khi có.

Hướng dẫn chạy trên Kaggle và các lưu ý: [docs/sim_foundation_team_notes.md](docs/sim_foundation_team_notes.md).

## Cấu trúc

```text
configs/                 # pretrain, downstream Food/Ads, baseline, preset simulator
docs/                    # hợp đồng dữ liệu, ghi chú team, kết quả; docs/superpowers = spec/plan
scripts/download_sim_v1.py
src/simulation/          # simulator siêu app (world, hành vi, export các gói dữ liệu)
src/foundation_input/    # tokenizer, vocab, loader cho dữ liệu pretrain
src/models/              # encoder foundation, adapter, pooling
src/training/            # pretrain, checkpoint
src/downstream/          # Food recommendation, Ads CTR, head
src/baselines/           # B0, B1 (LightGBM), B2–B4 (sequence)
src/evaluation/          # metric, evaluator
tests/
```

## Tài liệu

- [docs/sim_v1_data.md](docs/sim_v1_data.md) — mô tả dữ liệu và cách tải
- [docs/simulation_data_contract.md](docs/simulation_data_contract.md) — hợp đồng các gói dữ liệu simulator
- [docs/sim_foundation_team_notes.md](docs/sim_foundation_team_notes.md) — thay đổi theo từng phần, lệnh chạy, giới hạn
- [docs/sim_pretrain_v1_results.md](docs/sim_pretrain_v1_results.md) — kết quả pretrain, downstream, baseline, chẩn đoán
- [docs/foundation_input_contract.md](docs/foundation_input_contract.md), [docs/baseline_input_contract.md](docs/baseline_input_contract.md) — hợp đồng đầu vào
- [docs/baselines.md](docs/baselines.md), [docs/b1_features.md](docs/b1_features.md) — baseline

## Quy tắc

Không commit dữ liệu, checkpoint hay output thí nghiệm (`data_sim/`, `experiments/` đã gitignore).
Xem [CONTRIBUTING.md](CONTRIBUTING.md) trước khi tạo branch hoặc Pull Request.
