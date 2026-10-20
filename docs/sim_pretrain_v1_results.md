# Pretrain foundation trên `sim-v1` — kết quả và phân tích (2026-10-17)

Lần pretrain đầy đủ đầu tiên trên dữ liệu mô phỏng (dự án con 3). Dữ liệu: `docs/sim_v1_data.md`;
thay đổi code: `docs/sim_foundation_team_notes.md`.

## 1. Cấu hình lần chạy

| Mục | Giá trị |
|---|---|
| Kernel | Kaggle private `hunhthhichu/vric-sim-pretrain` (version 1), GPU Tesla T4 (Kaggle cấp 2, code dùng 1), torch 2.11 + CUDA 12.8 |
| Code | `feat/sim-foundation` commit `62bccc46`, config `configs/pretrain_sim_kaggle.yaml` |
| Dữ liệu | `sim-v1/pretrain`: 20.000 user, 199.518 snapshot train, 39.998 snapshot validation (ngày 37 và 40) |
| Model | d_model 128, 4 tầng, 4 head, max_length 512, `long_term_injection: cls`, `user_embedding_init_std: 0.02` |
| Huấn luyện | batch 64, lr 3e-4 (user_adapter ×10), weight decay 0,01, AMP, 6 epoch (kernel tự chọn sau khi chạy thử), không scheduler |
| Mục tiêu | masked (che 15% token; cross-entropy + MSE) + 0,5 × next-action (đoán loại phiên kế tiếp từ `short_term`) |
| Vocab | 38 loại key, 20.003 giá trị (dải id theo từng key; lớn nhất là `user_id`), allowlist có Ride/Food/Advertisement |

Thời gian: fit vocab 8 phút, chạy thử 40 s (ước 1.600 s/epoch), 6 epoch × ~1.420 s (24 phút),
tổng 2 giờ 32 phút. Tốc độ ổn định ~2,9 step/s (~190 mẫu/s), 3.118 step/epoch, 18.708 step.
Log sạch: không lỗi, không NaN, không mẫu bị bỏ; chỉ có cảnh báo vô hại (`tarfile` filter trong
script kernel, `nbconvert` của Kaggle).

## 2. Kết quả theo epoch

Train = trung bình các batch được log trong epoch (loss tổng: trung bình cả epoch).

| Epoch | Loss train | Loss val | Masked train / val | Next-action train / val | Độ chính xác token che (val) |
|---|---|---|---|---|---|
| 1 | 2,569 | 1,487 | 2,09 / 1,114 | 0,89 / 0,747 | 79,7% |
| 2 | 1,304 | 1,127 | 0,96 / 0,781 | 0,67 / **0,692** | 84,6% |
| **3** | 1,050 | **0,978** | 0,74 / 0,616 | 0,63 / 0,725 | 86,8% |
| 4 | 0,894 | 0,989 | 0,62 / 0,553 | 0,54 / 0,872 | 88,0% |
| 5 | 0,805 | 1,009 | 0,57 / 0,518 | 0,47 / 0,982 | 88,7% |
| 6 | 0,744 | 1,061 | 0,53 / **0,489** | 0,43 / 1,143 | **89,2%** |

- **Masked học thật:** train và val cùng giảm tới epoch 6, val luôn thấp hơn train (dropout lúc
  train; trung bình train gồm cả đầu epoch).
- **Next-action overfit từ epoch 3:** train giảm 0,63 → 0,43 nhưng val tăng 0,73 → 1,14, cuối cùng
  tệ hơn đoán theo tần suất (~1,0 nats, mục 4).
- Checkpoint tốt nhất theo val loss tổng: **epoch 3**.

## 3. Checkpoint

| Mục | Giá trị |
|---|---|
| File | output kernel `vric-sim-pretrain`: `pretrain_sim/checkpoints/best_checkpoint.pt` (315 MB) |
| SHA-256 | `be6c3a42293d192870f08603900fec5d52b0364b3861d2b6cbe15e7be2cfdb05` (epoch 3, step 9.354) |
| Nguồn gốc | `dataset_version sim-v1`, `sim_config_sha256 631010161f2f…b300` (khớp gói Food/Ads `sim-v1`) |
| Khác | `latest_checkpoint.pt` (epoch 6), `vocab.json`, `metrics.jsonl`, `experiment_record.json` |

`long_term` đã học: so với trọng số khởi tạo (dựng lại bằng seed 42), vector của 19.998 user đổi
**7,3 lần** độ dài ban đầu (trung vị; p10 5,6, p90 9,3; độ dài 0,23 → 1,66). Các hàng không có
user chỉ co 24% do weight decay — đúng tính toán (1 − 3e-3 × 0,01)^9354.

## 4. Vì sao next-action trên val tệ dần — đang debug

Target next-action của một snapshot = loại bước đầu của **phiên cuối cùng** trong history (phiên
đó bị bỏ khỏi input): `ad_impression` (quảng cáo), `search_destination` (đặt xe), `search` (đồ ăn).

Bằng chứng đã có (đo trên gói `sim-smoke-v1`, cùng cấu trúc, 1.000 user):

| | Train | Val |
|---|---|---|
| Phân bố target (QC / xe / đồ ăn) | 56% / 24% / 20% | 54% / 29% / 18% |
| Entropy = loss của đoán theo tần suất | 0,99 | 1,00 |
| Phiên target đã nằm trong **input** của một snapshot train khác | **78,9%** | **0%** |

- Không phải lệch phân bố: hai tập có cùng phân bố target.
- Có **đường học thuộc chỉ ở train**: snapshot cách nhau 3 ngày nhưng cửa sổ 30 event dài ~8 ngày,
  nên đáp án của snapshot này là input của snapshot sau; đáp án của val (ngày 37, 40) chưa bao giờ
  xuất hiện trong dữ liệu train.

**Giả thuyết:** train càng lâu, model càng học thuộc chuỗi event của từng user (qua `long_term`
và/hoặc qua chính nội dung chuỗi) → đoán đúng target train đã thấy, nhưng tự tin sai trên target
val mới → val tăng, cuối cùng tệ hơn đoán theo tần suất.

**Đang kiểm chứng** (tái hiện trên smoke, CPU, cùng siêu tham số Kaggle, 6 epoch): mỗi epoch đo
loss next-action riêng trên (a) train có đáp án đã lộ, (b) train chưa lộ, (c) val; chạy hai cấu
hình `long_term_injection: cls` và `none`.
- Nếu (a) ≪ (b) ≈ (c) và khoảng cách tăng theo epoch → xác nhận học thuộc.
- So `cls` với `none` → học thuộc đi qua `long_term` hay qua nội dung chuỗi.

Kết quả (smoke, batch 16, 6 epoch = 1.860 step; loss next-action):

| Epoch | `cls`: đã lộ / chưa lộ / val | `none`: đã lộ / chưa lộ / val |
|---|---|---|
| 1 | 1,00 / 1,09 / 1,05 | 1,00 / 1,09 / 1,04 |
| 3 | 0,95 / 1,02 / 0,99 | 0,93 / 1,02 / 0,98 |
| 6 | 0,79 / 0,85 / 0,89 | 0,85 / 0,90 / 0,93 |

- **Không tái hiện được overfit**: val giảm đều tới epoch 6 ở cả hai cấu hình. Smoke train quá ít —
  masked cuối cùng 1,10 ≈ mức epoch 1 của bản full (1,11) — nên chưa tới giai đoạn overfit.
- Khoảng cách "đã lộ < chưa lộ" có ngay từ epoch 1 và **không** rộng ra → do hai nhóm vốn khác
  nhau (nhóm "chưa lộ" chủ yếu là snapshot train cuối của mỗi user, ngày 33), không phải học thuộc.
  Giả thuyết "học thuộc đáp án lộ trong input" **yếu đi**; giả thuyết học thuộc theo từng user khi
  train lâu **chưa kiểm được**.
- `cls` không gây hại ở giai đoạn này (val 0,89 so với 0,93 của `none`).

Bước kiểm tiếp (chỉ khi downstream cho thấy overfit gây hại): tái hiện đúng quy mô trên Kaggle GPU
(tập con ~5k user của `sim-v1`, cùng phép đo, `cls` và `none`).

Ghi nhận phụ (chưa kết luận, không giải thích được khoảng cách train/val):
- Dataset đọc snapshot theo thứ tự user, **không xáo trộn**, nên mỗi batch là vài user liên tiếp
  và thứ tự giống nhau mọi epoch.
- Profile tại cutoff tính cả phiên target (vd. `ride_count_30d` đã gồm chuyến cần đoán) — rò rỉ
  nhẹ, như nhau ở train và val.

## 5. Downstream (đóng băng foundation, test = tuần 8)

Kernel CPU `hunhthhichu/vric-sim-downstream-food` / `-ads` (code `62bccc46`, config
`*_sim_kaggle.yaml`). Ba nhánh: `epoch3` (best), `epoch6` (latest), `random` (cùng config + vocab,
trọng số khởi tạo lại bằng seed của pretrain — R6).

**Food** (10.000 request test; mỗi nhánh ~26 phút):

| Nhánh | NDCG@10 | MRR | HitRate@1 | HitRate@3 |
|---|---|---|---|---|
| `epoch3` | 0,745 | 0,664 | 49,4% | 78,9% |
| `epoch6` | 0,747 | 0,665 | 49,7% | 79,3% |
| `random` | 0,562 | 0,428 | 22,1% | 50,6% |

Mốc: phổ biến 0,509 · gần nhà 0,663 · GBDT profile + phổ biến + context ≈ 0,740 · oracle 0,899.

- Pretraining giúp rõ: **+0,18 NDCG@10** so với `random`, HitRate@1 gấp đôi.
- Overfit next-action **không hại** Food: `epoch6` ≈ `epoch3`.
- So với mô hình đặc trưng mạnh (≈ 0,740) thì foundation mới ngang — cần chạy Baseline B1 trên cùng
  tập test để so chính thức.

**Ads** (30.000 request test, CTR 5,27%; mỗi nhánh ~1 giờ 58 phút, trong đó ~1 giờ 44 phút là mã
hóa embedding trên CPU, train head ~3 phút):

| Nhánh | ROC AUC | PR AUC | Log loss | Brier |
|---|---|---|---|---|
| `epoch3` | 0,617 | 0,110 | 0,2001 | 0,0490 |
| `epoch6` | 0,613 | 0,109 | 0,2001 | 0,0490 |
| `random` | 0,522 | 0,056 | 0,2064 | 0,0499 |

Mốc: phổ biến (CTR theo quảng cáo) AUC 0,638 · oracle 0,788 · PR AUC ngẫu nhiên = CTR 0,053.

- Pretraining giúp rõ: **+0,095 AUC**, PR AUC gấp đôi so với `random`.
- Overfit next-action **không hại** đáng kể: `epoch6` − `epoch3` = −0,004 (trong dao động ~0,006).
- **Nhưng vẫn thua mốc phổ biến (0,617 < 0,638)**: ở nhánh `random`, loss train đứng yên 0,2006 =
  entropy của CTR 5,07% suốt 10 epoch → head Ads (`AdRanker`) không học được cả CTR riêng của từng
  quảng cáo. Đây là vấn đề của head/cấu hình Ads, cần debug riêng.

## 6. Lỗi head downstream và kết quả sau khi sửa (`feat/sim-experiments`, `bfda9320`)

**Lỗi:** head Food và Ads dùng `torch.optim.Adam(weight_decay=1e-4)` — L2 *gắn vào gradient*.
Hàng embedding của ứng viên không có trong batch vẫn nhận gradient `wd × p`, và Adam chuẩn hóa nó
thành bước kéo ≈ lr về 0. Đo trong head đã train (embedding khởi tạo N(0,1), độ dài ~2,83 Ads /
~4,0 Food): Ads `random` **0,0005**, Ads `epoch3` 0,16, Food `epoch3` 1,11, Food `random` 0,73.
Ads có 1.500 quảng cáo, mỗi cái chỉ ở ~4% số batch → embedding quảng cáo **sụp về 0**, head không
học được CTR theo quảng cáo (nhánh `random` đứng ở loss = entropy của CTR).

**Sửa:** `src/downstream/head_training.head_optimizer` = `AdamW` (decay tách khỏi gradient), dùng
chung cho Food và Ads; test `tests/downstream/test_head_training.py` (đỏ với `Adam`, xanh với `AdamW`).

**Chạy lại** (Kaggle CPU; `epoch3` và `random`):

| | Trước (`Adam`) | Sau (`AdamW`) |
|---|---|---|
| Độ dài embedding ứng viên | Ads 0,0005–0,16 · Food 0,7–1,1 | Ads 3,1 · Food 3,9 (giữ nguyên) |
| Ads ROC AUC `epoch3` / `random` | 0,617 / 0,522 | **0,628 / 0,626** |
| Food NDCG@10 `epoch3` / `random` | 0,745 / 0,562 | **0,726 / 0,534** |

**Đính chính:** "+0,095 AUC nhờ pretraining" ở mục 5 là **giả** — do nhánh `random` hỏng head. Khi
head đúng, pretrained chỉ hơn `random` **+0,002 AUC**, và cả hai vẫn dưới mốc phổ biến 0,638:
với head hiện tại, embedding foundation gần như không giúp Ads. Food vẫn **+0,19 NDCG@10**
(pretrained so với `random`), nhưng cả hai nhánh giảm ~0,02: với Food, việc `Adam` bóp embedding
quán hóa ra là regularization có ích — cần tăng weight decay của `AdamW` (chọn theo validation).

## 7. Baseline chính thức B0/B1 trên `sim-v1` (bộ full, cùng tập test, cùng evaluator)

Kernel CPU `hunhthhichu/vric-sim-baselines-{food,ads}` (version 2). B1 train tuần 1–6, early stopping
tuần 7 (`fit_cutoff` 2026-02-16), đánh giá tuần 8 với history đóng băng ở 2026-02-23; B0 Food =
phổ biến 7 ngày, B0 Ads = `ad_ctr`.

| | Food NDCG@10 (10.000) | Ads ROC AUC (30.000) |
|---|---|---|
| B0 | 0,508 | 0,641 |
| **B1 (LightGBM + đặc trưng)** | **0,747** (MRR 0,666, HitRate@1 50,9%) | **0,700** (PR AUC 0,128, log loss 0,1925) |
| Foundation `epoch3` — head `Adam` / `AdamW` | 0,745 / 0,726 | 0,617 / 0,628 |
| `random` — head `Adam` / `AdamW` | 0,562 / 0,534 | 0,522 / 0,626 |

B1 lần đầu không chạy được trên bộ full: `FeatureBuilder` quét lại toàn bộ history cho **từng
ứng viên** và cache một bản history theo từng cutoff → Food bị giết ở 30 GB sau 7,7 giờ, Ads quá
11 giờ. Đã viết lại (`ad5da5f9`): đánh chỉ mục history một lần theo user và theo item; giá trị đặc
trưng **giống hệt** (test `tests/baselines/test_features_equivalence.py` giữ thuật toán cũ làm
chuẩn). Smoke: Food 39 phút → 17 giây, Ads 1.121 giây → 11 giây, metric trùng từng chữ số;
full: 6–7 phút mỗi nhiệm vụ.

## 8. Kết luận và bước tiếp theo

- Food: foundation (đóng băng + head) **ngang B1** (0,745 so với 0,747) và hơn encoder ngẫu nhiên
  +0,19 NDCG@10.
- Ads: foundation **thua B1 rõ** (0,628 so với 0,700), thua cả B0, gần như không hơn ngẫu nhiên.
- Foundation đứng một mình chưa vượt baseline đặc trưng mạnh. B1 có sẵn đặc trưng trùng cơ chế mô
  phỏng (CTR user × quảng cáo, ái lực user × loại món); head hiện tại ([long_term, short_term,
  embedding ứng viên] → MLP) không học được tương tác user × ứng viên.
- Overfit next-action không hại downstream.

Thứ tự đề xuất:
1. **B1 + embedding foundation** (thêm `long_term`/`short_term` làm đặc trưng cho LightGBM) — cách
   Grab đánh giá: embedding có cải thiện mô hình đang có không.
2. Head có tương tác user × ứng viên (dot-product / `mlp_dot`), weight decay `AdamW` chọn theo validation.
3. Mã hóa embedding downstream trên GPU (Ads ~2 giờ → vài phút).
4. Để sau: overfit next-action / pretrain v2.

## 9. Chẩn đoán: nghẽn ở embedding hay ở cách dùng? (2026-10-19)

Mã chẩn đoán là mã dùng một lần (không vào repo), chạy trên 4 kernel Kaggle private:
`vric-sim-diag-export` (GPU), `vric-sim-diag-heads`, `vric-sim-diag-b1emb-{food,ads}` (CPU).

**Bước 0 — xuất embedding.** Checkpoint `epoch3` (best) và encoder `random` (seed 42), đầu vào token
hóa bằng chính `load_food_inputs` / `load_ad_inputs` của downstream. Food 80.000 request, Ads 260.000;
trên T4 mỗi nhánh 48 s / 137 s (token hóa trên CPU mới là phần chậm: 6 / 20 phút).

**Bước 1 — probe.** Hồi quy logistic từ trung bình `[long_term, short_term]` các request train của
mỗi user ra đặc điểm ẩn mà bộ mô phỏng gieo cho user đó (dựng lại world bằng seed 42). 17.652 user,
chia 70/30 theo user. `history_counts` = tỷ lệ đơn theo loại món + click theo ngành quảng cáo trước
tuần 7 (không có thông tin giá).

| Độ chính xác (test 30% user) | Đa số | `random` | `epoch3` | `history_counts` |
|---|---|---|---|---|
| persona (5 lớp) | 0,411 | 0,811 | **0,919** | 0,870 |
| mức chi tiêu (3 lớp) | 0,455 | 0,660 | **0,733** | 0,507 |
| loại món thích nhất (12 lớp) | 0,250 | 0,423 | 0,498 | **0,571** |
| ngành quảng cáo thích nhất (15 lớp) | 0,378 | 0,487 | 0,545 | **0,586** |

Embedding pretrained hơn `random` ở cả 4 đặc điểm và hơn đếm đơn giản ở persona và mức chi tiêu,
nhưng **thua đếm đơn giản ở sở thích chi tiết** (loại món, ngành quảng cáo).

**Bước 2 — B1 + embedding** (cùng runner, cutoff, evaluator với B1 chính thức; B1 tái lập đúng số
chính thức). `pca` = 32 thành phần PCA của `[long_term, short_term]` (fit trên train); `proto` =
cosine giữa embedding user và "nguyên mẫu" của ứng viên (trung bình embedding các request train mà
ứng viên đó là positive, bỏ chính request đó với hàng train).

| | Food NDCG@10 | Ads ROC AUC |
|---|---|---|
| B1 | 0,7465 | 0,7003 |
| B1 + `pca` (`epoch3`) | 0,7500 | **0,7099** |
| B1 + `pca` + `proto` (`epoch3`) | **0,7893** (MRR 0,721, HitRate@1 57,3%) | 0,7097 |
| B1 + `pca` + `proto` (`random`, đối chứng) | 0,7469 | 0,7020 |

**Bước 3 — head trên embedding đóng băng** (AdamW, 10 epoch, chọn theo validation, báo test).
H0 = head hiện tại (concat → MLP); H1 = H0 + thuộc tính ứng viên (Food: loại món, mức giá, khu vực;
Ads: ngành, nhà quảng cáo); H2 = H1 + tích vô hướng user × ứng viên; C1 = H1 không có user.

| | Food `epoch3` | Food `random` | Ads `epoch3` | Ads `random` |
|---|---|---|---|---|
| H0 | 0,733 | 0,548 | 0,630 | 0,635 |
| H1 | **0,766** | 0,608 | 0,676 | 0,650 |
| H2 | 0,766 | 0,645 | **0,683** | 0,653 |
| C1 (không user) | 0,516 | — | 0,644 | — |

H0 tái lập đúng số downstream chính thức (0,726 / 0,628).

**Kết luận.**
- Nghẽn chính **ở cách dùng embedding**, không phải ở embedding. Embedding có thông tin (probe), và
  khi dùng đúng cách thì pretraining giúp rõ, còn encoder ngẫu nhiên thì không:
  - B1 + embedding pretrained: Food **+0,043 NDCG@10**, Ads **+0,010 AUC** so với B1. Đối chứng
    `random` ≈ B1 (+0,000 / +0,002), nên phần tăng đến từ pretraining.
  - Foundation một mình với head có thuộc tính ứng viên: Food 0,766 **vượt B1** (0,747); Ads 0,683
    vẫn dưới B1 (0,700).
- Head hiện tại thiếu thuộc tính ứng viên (chỉ có id): thêm vào được +0,034 (Food) / +0,046 (Ads).
  Tích vô hướng thêm ít khi đã có thuộc tính.
- Điểm yếu của pretrain: sở thích chi tiết (loại món, ngành quảng cáo) mã hóa kém hơn đếm đơn giản.
  Ứng viên số một: cửa sổ 30 sự kiện quá ngắn. Probe này là thước đo rẻ cho pretrain v2.
- Không rò nhãn test: dữ liệu pretrain dừng ở ngày 33, trước validation (ngày 43) và test.
  Phơi nhiễm R8 chỉ chạm hàng train nên không làm đẹp số test; bước 4 (chỉ train từ ngày 34) không
  cần cho kết luận này.
- Giới hạn: một seed. Food +0,043 trên 10.000 request là chắc; Ads +0,010 trên 30.000 impression
  cần lặp seed để chắc.

**Bước tiếp theo đề xuất.**
1. Đưa vào repo: biến thể B1 + đặc trưng foundation (`pca` + `proto`), và thuộc tính ứng viên cho
   head Food/Ads. Lặp 3 seed.
2. Pretrain v2 (sau): cửa sổ history dài hơn; đánh giá trước bằng probe sở thích chi tiết.

## 10. Kiểm định chính thức trong ver2: bootstrap và 3 seed (2026-10-20)

Code: branch `feat/foundation-features` (`d4ec225`): thuộc tính ứng viên cho head, `encoder_device`,
`src.downstream.user_embeddings`, B1 với `foundation_features`. Kernel CPU private
`hunhthhichu/vric-ver2-cpu-{food,ads}`. Mỗi nhánh foundation được mã hóa một lần; mọi lượt head dùng lại
embedding đó, nên nhánh `random` là cùng một encoder ngẫu nhiên (seed 42) cho cả 3 seed của head.
Bootstrap ghép cặp: 2.000 lần lấy mẫu lại các request test.

| | Food NDCG@10 | Ads ROC AUC |
|---|---|---|
| B1 | 0,7465 | 0,7003 |
| **B1 + embedding pretrained** | **0,7899** | **0,7092** |
| chênh so với B1 (KTC 95%) | **+0,0434 [+0,0400; +0,0471]** | **+0,0090 [+0,0044; +0,0135]** |
| B1 + embedding encoder ngẫu nhiên | 0,7469 | 0,7019 |
| chênh so với B1 (KTC 95%) | +0,0004 [−0,0011; +0,0020] | +0,0016 [−0,0020; +0,0051] |

Head trên foundation đóng băng, trung bình ± độ lệch chuẩn qua seed 42, 43, 44:

| | Food `pretrained` | Food `random` | Ads `pretrained` | Ads `random` |
|---|---|---|---|---|
| H1: ID + thuộc tính ứng viên (config sim hiện tại) | **0,760 ± 0,003** | 0,569 ± 0,003 | **0,678 ± 0,004** | 0,653 ± 0,001 |
| H0: chỉ ID (head cũ) | 0,725 ± 0,002 | 0,533 ± 0,002 | 0,627 ± 0,002 | 0,627 ± 0,002 |

**Kết luận.**
- Trên cả hai nhiệm vụ, thêm embedding pretrained vào B1 cải thiện có ý nghĩa thống kê; cùng cách thêm
  nhưng với encoder ngẫu nhiên thì không. Đây là bằng chứng kiểu Grab rằng pretraining có ích.
- Food: foundation một mình với head có thuộc tính (0,760) vượt B1 (0,7465).
- Ads: mức tăng nhỏ hơn nhiều (+0,009 AUC). Foundation một mình (0,678) vẫn dưới B1 (0,700). Head chỉ ID
  không phân biệt được pretrained với random (0,627 cả hai); khi có thuộc tính ứng viên thì pretrained
  hơn random +0,025.
- Kết quả chẩn đoán ở mục 9 được tái lập: Food 0,789 → 0,790, Ads 0,710 → 0,709.
- Giới hạn: một lần pretrain (seed 42), dữ liệu mô phỏng, chưa so với baseline chuỗi học có giám sát (B2–B4).

## 11. Pretrain v2: `long_term` từ lịch sử dài (2026-10-20)

**Vì sao.** Probe trên bộ full cho thấy `long_term` của v1 (bảng theo user ID) không mang thông tin
gì về user: persona 0,404, loại món 0,243, ngành quảng cáo 0,373, đều bằng mức đoán lớp đa số.
User ở downstream khớp 100% từ điển, nên không phải lỗi tra cứu: bảng ID thừa, vì user ID đã có
trong chuỗi đầu vào. Cửa sổ 30 sự kiện thì bỏ mất lịch sử dài: 76% request Ads test không có click
quảng cáo nào trong cửa sổ.

**Thay đổi** (branch `feat/pretrain-v2`, `9217473`, `8357953`, `15f0f1b`):
- `long_term = MLP(71 đặc trưng lịch sử dài)` thay cho bảng user ID. Đặc trưng tính từ mọi sự kiện
  trước thời điểm cắt: log số sự kiện theo dịch vụ và loại, tỷ lệ đơn theo loại món và mức giá,
  tỷ lệ click/hiển thị và CTR theo ngành quảng cáo, độ gần đây.
- Mục tiêu phụ (trọng số 0,5): từ `long_term` dự đoán phân phối loại món đã đặt và ngành quảng cáo
  đã click trong 7 ngày sau thời điểm cắt.
- Food đọc lịch sử dài từ `events.parquet` của gói Ads cùng bộ mô phỏng (`data.events_root`).

**Pretrain** (`vric-sim-pretrain-v2`, T4, 5 epoch, 2 giờ 35 phút): val loss giảm đều 2,38 → 1,73,
tốt nhất ở epoch 5. Val loss next-action giảm 0,73 → 0,64; ở v1 nó tăng 0,69 → 1,14. Overfit của
v1 đến từ việc học thuộc theo user ID.

**Đánh giá** (`vric-sim-v2-eval`, cùng tập test, cùng phương pháp với mục 10):

| | Food NDCG@10 | Ads ROC AUC |
|---|---|---|
| B1 | 0,7465 | 0,7003 |
| B1 + embedding v1 | 0,7899 (+0,043) | 0,7092 (+0,009) |
| **B1 + embedding v2** | **0,7874 (+0,041 [0,037; 0,045])** | **0,7273 (+0,027 [0,021; 0,034])** |
| B1 + embedding v2 encoder ngẫu nhiên | 0,7478 (+0,001, không ý nghĩa) | 0,7038 (+0,004, không ý nghĩa) |

Head trên embedding đóng băng (cùng head với thí nghiệm "thử rẻ", 3 seed):

| Head | Food v1 | Food v2 | Ads v1 | Ads v2 |
|---|---|---|---|---|
| A: embedding + thuộc tính ứng viên | 0,767 | **0,797** | 0,669 | **0,716** |
| C: A + thống kê lịch sử + đặc trưng cặp | 0,818 | **0,821** | 0,708 | **0,719** |
| D: như C, không embedding | 0,783 | 0,785 | 0,717 | 0,718 |
| E: như C, embedding encoder ngẫu nhiên | 0,785 | 0,786 | 0,713 | 0,713 |

Probe (độ chính xác, 17.652 user; đếm toàn lịch sử: 0,809 / 0,498 / 0,515 / 0,543):

| | Persona | Mức chi tiêu | Loại món | Ngành QC |
|---|---|---|---|---|
| `long_term` v1 | 0,404 | 0,452 | 0,243 | 0,373 |
| `long_term` v2 | 0,930 | 0,583 | 0,546 | 0,579 |
| `long_term` + `short_term` v2 | **0,968** | **0,763** | **0,569** | **0,594** |

**Kết luận.**
- `long_term` v2 mang thông tin thật; embedding v2 hơn cách đếm lịch sử ở cả 4 đặc điểm.
- Ads: B1 + embedding tăng gấp 3 so với v1 (+0,027 so với +0,009), có ý nghĩa thống kê. Foundation
  một mình (head A) đạt 0,716, vượt B1 (0,700) và ngang head dùng thống kê tay (D, 0,718).
- Food: foundation một mình 0,797, vượt cả B1 (0,747) lẫn B1 + embedding v1 (0,790).
- Nhánh encoder ngẫu nhiên của v2 vẫn có MLP trên đặc trưng lịch sử, nên `long_term` ngẫu nhiên
  cũng mang thông tin (probe persona 0,923). Phần pretraining đóng góp được đo bằng chênh lệch so
  với nhánh này: B1 + embedding +0,041 (Food) / +0,027 (Ads) so với +0,001 / +0,004.
- Đang train thêm 5 epoch (warm start, `vric-sim-pretrain-v2b`) vì val loss vẫn giảm ở epoch 5.
