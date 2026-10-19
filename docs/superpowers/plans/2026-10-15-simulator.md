# Simulator dữ liệu siêu app — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Viết package `src/simulation/` sinh một thế giới user dùng chung ride + food + ads có quy luật cài sẵn và xác suất thật, rồi xuất 5 gói (pretrain `omnibehavior_pretraining_v2`, Food, Ads, Baseline JSON, oracle) kèm kiểm tra và báo cáo độ khó.

**Architecture:** `config` → `world` (thành phố, catalog, user + tính cách ẩn) → `behavior` (sinh event theo ngày, ý định, phễu; lưu trạng thái để tính xác suất thật) → các `export_*` độc lập đọc cùng một `Simulation` → `checks`/`calibrate` đọc lại file đã xuất. Mọi ngẫu nhiên đi qua một `numpy.random.Generator` duy nhất có seed để tái lập byte-for-byte.

**Tech Stack:** Python 3.11 (cú pháp py310), numpy, pyarrow, pyyaml, pytest, ruff. Không thêm phụ thuộc.

**Spec:** `docs/superpowers/specs/2026-10-15-simulator-design.md` (gọi tắt "spec §N").

## Global Constraints

- Branch `feat/sim-data`; không push. Chỉ tạo/sửa `src/simulation/**`, `tests/simulation/**`, `configs/simulation/**`, `docs/simulation_data_contract.md`. Không sửa `data/**`, `src/data/**`, `src/foundation_input/**`, `src/models/**`, `src/training/**`, `src/downstream/**`, `src/baselines/**`, `src/evaluation/**` (chỉ được import).
- Namespace `sim_v1`; ID dạng `sim:user:<n>`, `sim:merchant:<n>`, `sim:dish:<n>`, `sim:ad:<n>`, `sim:driver:<n>`, `sim:event:<n>`, `sim:food_req:<n>`, `sim:ad_req:<n>` (`<n>` số nguyên không đệm 0).
- Giờ địa phương cố định `+07:00` (`Asia/Ho_Chi_Minh`); ngày 1 = 2026-01-05 (thứ Hai); tuần 7 bắt đầu ngày 43. Bên trong dùng epoch giây UTC (`int`); mọi phép gán ngày/tuần/split dùng **giờ địa phương**.
- Cùng `SimConfig` + `seed` → mọi file đầu ra giống hệt byte (không ghi thời điểm chạy vào file trừ `data_report.json` mục `generated_at_utc`, mục đó bị loại khỏi so sánh SHA).
- Đầu vào model không bao giờ chứa: nhóm người, khẩu vị, sở thích quảng cáo, ý định, vị trí hiện tại, khoảng cách từ vị trí hiện tại, xác suất thật.
- Ruff `line-length = 100`, `target-version = "py310"`; kiểm tra bằng `ruff check src tests configs` (lỗi có sẵn ở `data/`, `notebooks/`, `.history/` nằm ngoài phạm vi).
- Commit message kết thúc bằng `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Lệnh Python: `.venv/Scripts/python`.

## Review Focus

1. User không có chỗ làm (ở nhà/hưu trí) → nhóm ứng viên "chỗ làm" thay bằng khu TTTM gần nhà nhất, vẫn đủ 9 âm (test ở Task 4).
2. Ranh giới ngày theo giờ `+07:00`: một request lúc 00:30 ngày 43 giờ địa phương (= 17:30 UTC ngày 42) phải thuộc split `validation` (test ở Task 2).
3. Phiên đang dở dang tại cutoff của snapshot: chỉ nhận phiên mà **mọi** action đều trước cutoff; phiên dở dang không vào history (test ở Task 3).
4. Không đủ quán cùng thập phân vị phổ biến trong một nhóm khu → nới sang thập phân vị kề rồi khu kề theo thứ tự cố định, vẫn đủ 10 ứng viên, không trùng (test ở Task 4).
5. User chưa có event nào trước một cutoff snapshot → không sinh snapshot đó (không có snapshot rỗng) (test ở Task 3).

---

### Task 1: Cấu hình và thế giới tĩnh

**Files:**
- Create: `src/simulation/__init__.py`, `src/simulation/config.py`, `src/simulation/world.py`, `configs/simulation/smoke.yaml`, `configs/simulation/full.yaml`
- Test: `tests/simulation/__init__.py`, `tests/simulation/conftest.py`, `tests/simulation/test_world.py`

**Interfaces:**
- Produces:
  - `SimConfig` (frozen dataclass) với các trường và giá trị preset theo spec §3: `dataset_version: str`, `seed: int`, `n_users`, `n_zones=50`, `n_merchants`, `dishes_per_merchant=8`, `n_ads`, `n_ad_categories=15`, `n_drivers`, `start_date="2026-01-05"`, `weeks=8`, `utc_offset_hours=7`, `target_events_per_user=200` (trong tuần 1–6), `food_requests: dict[str, int] | None` (None = lấy tất cả), `ads_requests: dict[str, int] | None`, `target_ctr=0.05`, `snapshot_every_days=3`, `max_history_events=30`, `noise: NoiseConfig`.
  - `NoiseConfig` (frozen dataclass) chứa các hệ số dùng ở Task 2 và hiệu chỉnh ở Task 8: `food_cuisine_weight`, `food_price_weight`, `food_distance_weight`, `food_popularity_weight`, `food_logit_noise_sd`, `ad_interest_weight`, `ad_intent_weight`, `ad_zone_weight`, `ad_cross_service_weight`, `ad_logit_noise_sd` (giá trị khởi đầu do implementer chọn, ghi trong report).
  - `PRESETS: dict[str, SimConfig]` với khóa `"smoke"` (`sim-smoke-v1`, 500 user, 60 quán, 150 quảng cáo, 300 tài xế, request `None`) và `"full"` (`sim-v1`, 20.000 user, 600 quán, 1.500 quảng cáo, 3.000 tài xế, food `{"train": 60000, "validation": 10000, "test": 10000}`, ads `{"train": 200000, "validation": 30000, "test": 30000}`), cả hai `seed=42`.
  - `load_config(path) -> SimConfig` đọc YAML gồm `preset:` và các khóa ghi đè; `config_sha256(config) -> str` (SHA-256 của JSON chuẩn hóa `sort_keys=True`).
  - Hằng: `CUISINES` (12 chuỗi theo thứ tự spec §4.2), `AD_CATEGORIES` (15 chuỗi theo spec §4.2), `PERSONA_SHARES = {"office": 0.40, "student": 0.20, "night_shift": 0.10, "home": 0.15, "traveler": 0.15}`, `ZONE_TYPE_COUNTS = {"residential": 20, "office": 12, "university": 4, "nightlife": 6, "mall": 5, "transport_hub": 3}`.
  - Dataclass `Zone(zone_id: int, zone_type: str, lat: float, lon: float, geohash: str)`, `Merchant(merchant_id: str, zone_id: int, cuisine: str, price_level: int, name: str, base_popularity: float)`, `Dish(dish_id: str, merchant_id: str, name: str, description: str, price: float)`, `Ad(ad_id: str, category: str, advertiser: str, title: str, target_zone: int | None)`, `Driver(driver_id: str, vehicle_type: str, zone_id: int)`, `UserTraits(user_id: str, persona: str, age_group: str, home_zone: int, work_zone: int | None, activity: float, cuisine_pref: tuple[float, ...], spend_level: int, ad_interest: tuple[float, ...], commute_out_hour: float, commute_back_hour: float, workdays: tuple[int, ...])`, `World(config, zones, merchants, dishes, ads, drivers, users)`.
  - `build_world(config: SimConfig, rng: numpy.random.Generator) -> World`; `geohash_encode(lat: float, lon: float, precision: int = 6) -> str` (thuật toán geohash chuẩn base32); `haversine_km(a: Zone, b: Zone) -> float`.
- Test fixture (`conftest.py`): `tiny_config()` = preset smoke với `n_users=80, n_merchants=24, n_ads=40, n_drivers=40` (dùng cho test nhanh ở mọi task).

- [ ] **Step 1: Viết test (fail)**

```python
def test_presets_match_spec():
    s, f = PRESETS["smoke"], PRESETS["full"]
    assert (s.dataset_version, s.n_users, s.n_merchants, s.n_ads, s.n_drivers) == ("sim-smoke-v1", 500, 60, 150, 300)
    assert (f.dataset_version, f.n_users, f.n_merchants, f.n_ads, f.n_drivers) == ("sim-v1", 20000, 600, 1500, 3000)
    assert f.food_requests == {"train": 60000, "validation": 10000, "test": 10000}
    assert f.ads_requests == {"train": 200000, "validation": 30000, "test": 30000}
    assert s.seed == f.seed == 42 and s.snapshot_every_days == 3 and s.max_history_events == 30

def test_world_is_deterministic_and_shaped(tiny_config):
    a = build_world(tiny_config, np.random.default_rng(tiny_config.seed))
    b = build_world(tiny_config, np.random.default_rng(tiny_config.seed))
    assert a == b
    assert len(a.zones) == 50 and Counter(z.zone_type for z in a.zones) == ZONE_TYPE_COUNTS
    assert len(a.users) == 80 and len(a.merchants) == 24 and len(a.dishes) == 24 * 8
    assert all(len(u.cuisine_pref) == 12 and abs(sum(u.cuisine_pref) - 1) < 1e-9 for u in a.users)
    assert all((u.work_zone is None) == (u.persona == "home") for u in a.users)
    assert a.users[0].user_id == "sim:user:0" and a.merchants[0].merchant_id == "sim:merchant:0"

def test_geohash_known_value():
    assert geohash_encode(10.7769, 106.7009, 6) == "w3gv2u"   # giá trị tham chiếu tính bằng thuật toán chuẩn

def test_persona_shares_on_smoke():
    world = build_world(PRESETS["smoke"], np.random.default_rng(42))
    shares = Counter(u.persona for u in world.users)
    assert all(abs(shares[p] / 500 - q) < 0.06 for p, q in PERSONA_SHARES.items())
```

(Nếu giá trị geohash tham chiếu khác do sai tọa độ tham chiếu, implementer tính lại bằng bảng base32 chuẩn và ghi rõ nguồn trong report; không đổi thuật toán.)

- [ ] **Step 2: Chạy test để thấy fail** — `.venv/Scripts/python -m pytest tests/simulation/test_world.py -q` → FAIL (`ModuleNotFoundError`).
- [ ] **Step 3: Implement `config.py`, `world.py`, hai file YAML** (YAML chỉ gồm `preset: smoke|full`). Tâm khu nằm trong hộp ±0,09° quanh (10.7769, 106.7009). Tên quán/món/tiêu đề quảng cáo ghép từ từ vựng cố định theo loại món/ngành hàng (có nghĩa, ví dụ "Cơm Tấm Phúc 12").
- [ ] **Step 4: Chạy test để thấy pass** — cùng lệnh → PASS.
- [ ] **Step 5: Commit** — `git add src/simulation configs/simulation tests/simulation && git commit -m "feat: add simulator config and static world"`

---

### Task 2: Hành vi theo ngày, ý định và xác suất thật

**Files:**
- Create: `src/simulation/behavior.py`, `src/simulation/timeutil.py`
- Test: `tests/simulation/test_behavior.py`

**Interfaces:**
- Consumes: `SimConfig`, `World`, `build_world`, `haversine_km`, hằng của Task 1.
- Produces:
  - `timeutil.py`: `local_day(ts: int, config) -> int` (ngày 1 = 2026-01-05 địa phương), `week_of(ts, config) -> int` (1-based), `split_of(ts, config) -> str` (`"train"` tuần 1–6, `"validation"` tuần 7, `"test"` tuần 8), `local_iso(ts, config) -> str` (ISO có `+07:00`), `naive_utc(ts) -> datetime`, `day_start_ts(day, config) -> int`.
  - `Intent(user_id: str, kind: str, start_ts: int, end_ts: int | None)` với `kind ∈ {"diet", "trip", "job_change", "payday"}`.
  - `Event(event_id: str, user_id: str, session_id: str, session_index: int, action_index: int, session_size: int, ts: int, service: str, event_type: str, entity_id: str, context: dict, action: dict)`. `service ∈ {"ride", "food", "ads"}`; `event_type` ∈ Ride `search_destination|get_quote|book_ride|complete_ride`, Food `search|view_merchant|click_dish|order`, Ads `ad_impression|ad_click`. `context`/`action` dùng đúng khóa của spec §6.1 cho scenario tương ứng.
  - `FoodSession(session_id: str, user_id: str, start_ts: int, order_ts: int, merchant_id: str, location_zone: int, spend_level: int, active_intents: tuple[str, ...])`.
  - `AdImpression(event_id: str, user_id: str, ts: int, ad_id: str, location_zone: int, active_intents: tuple[str, ...], recent_services: tuple[str, ...], true_probability: float, clicked: bool)`.
  - `Simulation(world, events: tuple[Event, ...], intents, food_sessions, ad_impressions)`; `events` sắp theo `(user_id số, ts, session_id, action_index)`.
  - `simulate(world: World, rng) -> Simulation` (dùng `world.config`).
  - `food_logits(world, session: FoodSession, merchant_ids: Sequence[str]) -> numpy.ndarray` — logit thật theo spec §4.5 (không gồm nhiễu đã rút); xác suất chọn trên một tập = softmax của logit trên tập đó.
  - `click_probability(world, impression-like state) -> float` theo spec §4.5.

- [ ] **Step 1: Viết test (fail)** — dùng fixture module-scope `sim = simulate(build_world(tiny_config, rng), rng)`:

```python
def test_simulation_is_deterministic(tiny_config):
    assert run(tiny_config).events == run(tiny_config).events          # run = build_world + simulate với seed

def test_density_and_ordering(sim):
    early = [e for e in sim.events if week_of(e.ts, sim.world.config) <= 6]
    per_user = len(early) / len(sim.world.users)
    assert 140 <= per_user <= 260
    keys = [(int(e.user_id.rsplit(":", 1)[1]), e.ts, e.session_id, e.action_index) for e in sim.events]
    assert keys == sorted(keys)

def test_food_orders_follow_current_location(sim):
    near = sum(haversine_km(zone(s.location_zone), zone(merchant(s.merchant_id).zone_id)) <= 3.0
               for s in sim.food_sessions)
    assert near / len(sim.food_sessions) >= 0.6

def test_ctr_and_intent_effects(sim):
    imps = sim.ad_impressions
    assert 0.03 <= sum(i.clicked for i in imps) / len(imps) <= 0.07
    assert healthy_share(sim, during="diet") > healthy_share(sim, during=None) + 0.10
    assert travel_ctr(sim, during="trip") > travel_ctr(sim, during=None)

def test_trip_starts_with_airport_ride(sim):
    for intent in (i for i in sim.intents if i.kind == "trip"):
        first = next(e for e in sim.events if e.user_id == intent.user_id and e.ts >= intent.start_ts
                     and e.event_type == "complete_ride")
        assert zone_by_geohash(first.context["dropoff_geohash"]).zone_type == "transport_hub"

def test_local_day_boundary(tiny_config):
    ts = day_start_ts(43, tiny_config) + 30 * 60          # 00:30 ngày 43 giờ +07:00
    assert split_of(ts, tiny_config) == "validation" and split_of(ts - 3600, tiny_config) == "train"
```

(`zone`, `merchant`, `healthy_share`, `travel_ctr`, `zone_by_geohash` là helper nhỏ trong file test; "healthy" = cuisine `healthy/salad` hoặc `đồ chay`; "travel" = ngành hàng du lịch hoặc khách sạn.)

- [ ] **Step 2: Chạy test để thấy fail** — `.venv/Scripts/python -m pytest tests/simulation/test_behavior.py -q` → FAIL.
- [ ] **Step 3: Implement** theo spec §4.3–4.5. Vị trí hiện tại = khu trả của `complete_ride` gần nhất trong ngày địa phương, mặc định khu nhà. Phiên Food được bỏ ngang (không có `order`) theo một tỷ lệ cố định — phiên đó không phải `FoodSession`. Hệ số nền quảng cáo tự giải (vd. tìm nhị phân trên một mẫu) để CTR ≈ `target_ctr`.
- [ ] **Step 4: Chạy test để thấy pass.**
- [ ] **Step 5: Commit** — `git commit -m "feat: simulate daily ride, food and ads behavior with true probabilities"`

---

### Task 3: Xuất gói pretrain `omnibehavior_pretraining_v2`

**Files:**
- Create: `src/simulation/export_pretrain.py`, `src/simulation/history.py`, `docs/simulation_data_contract.md` (mục "Pretrain")
- Test: `tests/simulation/test_export_pretrain.py`

**Interfaces:**
- Consumes: `Simulation`, `Event`, `timeutil`.
- Produces:
  - `history.py`: `completed_sessions_before(events_of_user: Sequence[Event], cutoff_ts: int) -> list[Event]` — chỉ giữ event thuộc phiên mà **mọi** action có `ts < cutoff_ts`; `last_history(events, cutoff_ts, max_events) -> tuple[list[Event], bool]` — tối đa `max_events` event gần nhất, giữ trọn phiên (bỏ nguyên phiên cũ nhất nếu phải cắt), trả cờ `group_truncated` đúng sự thật; `profile_at(events_of_user, traits, cutoff_ts, config) -> dict` theo spec §4.6 (chỉ dùng event trước cutoff).
  - `export_pretrain(sim: Simulation, out_dir: Path) -> dict` ghi `out_dir/pretrain/` theo spec §6.1, trả tóm tắt `{users, events, snapshots_by_split}`. Cutoff snapshot: ngày 6, 9, …, 33 (train) và 37, 40 (validation) lúc 23:59:59 địa phương; `preprocessing_config.splits` = {`start_inclusive`: `"2026-01-04T17:00:00"`, `train_end_exclusive`: `"2026-02-06T17:00:00"`, `validation_end_exclusive`: `"2026-02-15T17:00:00"`, `test_end_exclusive`: `"2026-02-15T17:00:00"`} (naive UTC); `manifest.source_dataset = "sim_v1"`.
  - Chỉ xuất event có `ts < day_start_ts(43)`. Mỗi phiên → một `source_record_id`; mỗi bước phễu → một action (`action_index`, `source_action_count = session_size`).

- [ ] **Step 1: Viết test (fail)** — fixture module-scope xuất `tiny_config` vào `tmp_path_factory`:

```python
def test_pretrain_loads_with_foundation_loader(pretrain_root):
    from src.foundation_input.loader import iter_normalized_samples
    from src.foundation_input.validation import validate_sample
    train = list(iter_normalized_samples(pretrain_root, "train"))
    val = list(iter_normalized_samples(pretrain_root, "validation"))
    assert train and val and all(validate_sample(s) is s for s in train[:50])
    assert {s.dataset_name for s in train} == {"sim_v1"}

def test_no_event_on_or_after_week_7(pretrain_root, tiny_config):
    limit = naive_utc(day_start_ts(43, tiny_config))
    assert max_event_upper_bound(pretrain_root) < limit           # đọc mọi events/*.parquet

def test_snapshot_cutoffs_and_history_rules(pretrain_root):
    cutoffs = snapshot_cutoffs(pretrain_root)                       # set các naive UTC
    assert cutoffs["train"] <= {datetime(2026, 1, 10, 16, 59, 59) + timedelta(days=3 * k) for k in range(10)}
    assert cutoffs["validation"] <= {datetime(2026, 2, 10, 16, 59, 59), datetime(2026, 2, 13, 16, 59, 59)}
    assert all(1 <= n <= 30 for n in history_lengths(pretrain_root))

def test_in_progress_session_excluded():
    events = session("s1", [100, 200, 300]) + session("s2", [400, 500])   # helper tạo Event giả
    kept = completed_sessions_before(events, cutoff_ts=450)
    assert {e.session_id for e in kept} == {"s1"}

def test_user_without_prior_events_gets_no_snapshot(tmp_path, tiny_config):
    sim = simulation_with_single_user(tiny_config, first_event_day=8)       # helper: 1 user, event từ ngày 8
    export_pretrain(sim, tmp_path)
    snaps = pq.read_table(tmp_path / "pretrain" / "snapshots" / "sim_user_0.parquet").to_pylist()
    assert min(s["cutoff_time"] for s in snaps) == datetime(2026, 1, 13, 16, 59, 59)   # ngày 9

def test_manifest_hashes_match_files(pretrain_root):
    for shard in json.loads((pretrain_root / "manifest.json").read_text())["shards"]:
        assert sha256(pretrain_root / shard["events_file"]) == shard["events_sha256"]
```

(Tên file shard: `user_id` với `:` thay bằng `_`, ví dụ `sim_user_0.parquet`; ghi quy ước này vào contract.)

- [ ] **Step 2: Chạy test để thấy fail.**
- [ ] **Step 3: Implement** — ghi từng user một (không giữ toàn bộ bảng trong RAM); Parquet nén `zstd`; ghi `preview.json`, `data_report` riêng của gói không cần. Viết mục "Pretrain" của `docs/simulation_data_contract.md` (cột, kiểu, ý nghĩa, ví dụ một dòng).
- [ ] **Step 4: Chạy test để thấy pass.**
- [ ] **Step 5: Commit** — `git commit -m "feat: export simulated pretraining dataset in omnibehavior v2 schema"`

---

### Task 4: Xuất gói Food

**Files:**
- Create: `src/simulation/export_food.py`; Modify: `docs/simulation_data_contract.md` (mục "Food")
- Test: `tests/simulation/test_export_food.py`

**Interfaces:**
- Consumes: `Simulation`, `FoodSession`, `food_logits`, `history.last_history`, `history.profile_at`, `timeutil`.
- Produces:
  - `FoodRequest(request_id: str, session: FoodSession, cutoff_ts: int, split: str, candidate_ids: tuple[str, ...], positive_id: str)`.
  - `build_food_requests(sim, rng) -> list[FoodRequest]` — `cutoff_ts = session.start_ts - 1`; lấy mẫu theo `config.food_requests` mỗi split (None = tất cả); ứng viên theo spec §5 (thập phân vị phổ biến từ số đơn tuần 1–6; 3 gần khu quán được đặt + 3 gần nhà + 3 gần chỗ làm/TTTM gần nhà; nới theo Review Focus 4); thứ tự ứng viên xáo bằng `rng`.
  - `export_food(sim, requests, out_dir) -> dict` ghi `out_dir/food/samples.jsonl`, `merchants.parquet`, `manifest.json` đúng spec §6.2. History lấy bằng `last_history(events_of_user, cutoff_ts, 30)` (phiên của request bị loại vì `start_ts > cutoff_ts`).

- [ ] **Step 1: Viết test (fail)**

```python
def test_candidates_are_ten_unique_popularity_matched(food_requests, popularity_decile):
    for r in food_requests:
        assert len(r.candidate_ids) == 10 == len(set(r.candidate_ids)) and r.positive_id in r.candidate_ids
        deciles = {popularity_decile[c] for c in r.candidate_ids}
        assert max(deciles) - min(deciles) <= 1                      # cùng hoặc kề thập phân vị

def test_positive_zone_group_has_three_negatives(food_requests, world):
    for r in food_requests:
        pz = world_zone_of(r.positive_id)
        assert sum(world_zone_of(c) == pz or near(c, pz) for c in r.candidate_ids if c != r.positive_id) >= 3

def test_no_location_leak_in_inputs(food_jsonl_rows):
    for row in food_jsonl_rows:
        assert set(row["input"]["context"]) == {"hour", "weekday"}
        assert all(set(c) == {"item_id", "cuisine", "price_level", "geohash"} for c in row["input"]["candidates"])

def test_session_events_and_future_excluded(food_jsonl_rows):
    for row in food_jsonl_rows:
        cutoff = datetime.fromisoformat(row["input"]["cutoff_time"])
        assert all(datetime.fromisoformat(e["event_time"]) < cutoff for e in row["input"]["history"])
        assert datetime.fromisoformat(row["label_metadata"]["label_time"]) > cutoff
        assert row["label_metadata"]["observation_complete"] is True

def test_homebody_uses_mall_group(world, food_requests):
    homebodies = {u.user_id: u for u in world.users if u.persona == "home"}
    checked = 0
    for r in (r for r in food_requests if r.session.user_id in homebodies):
        mall = nearest_mall_zone(world, homebodies[r.session.user_id].home_zone)   # helper trong test
        assert sum(near(c, mall) for c in r.candidate_ids if c != r.positive_id) >= 3
        checked += 1
    assert checked > 0

def test_sparse_decile_falls_back_deterministically(tiny_config):
    sparse = dataclasses.replace(tiny_config, n_merchants=12)                 # ít quán → thiếu cùng thập phân vị
    a, b = requests_for(sparse), requests_for(sparse)                          # helper: world+simulate+build_food_requests
    assert [r.candidate_ids for r in a] == [r.candidate_ids for r in b]
    assert all(len(r.candidate_ids) == 10 == len(set(r.candidate_ids)) for r in a)

def test_manifest_and_split_counts(food_root):
    manifest = json.loads((food_root / "manifest.json").read_text())
    assert manifest["split"]["kind"] == "temporal" and manifest["candidate_policy"] == "sim-popularity-matched-home-work-v1"
    assert sha256(food_root / "samples.jsonl") == manifest["files"]["samples"]["sha256"]
```

- [ ] **Step 2: Chạy test để thấy fail.**
- [ ] **Step 3: Implement** + mục "Food" của contract (định dạng từng khóa, ví dụ một dòng rút gọn).
- [ ] **Step 4: Chạy test để thấy pass.**
- [ ] **Step 5: Commit** — `git commit -m "feat: export simulated Food recommendation requests"`

---

### Task 5: Xuất gói Ads

**Files:**
- Create: `src/simulation/export_ads.py`; Modify: `docs/simulation_data_contract.md` (mục "Ads")
- Test: `tests/simulation/test_export_ads.py`

**Interfaces:**
- Consumes: `Simulation`, `AdImpression`, `timeutil`.
- Produces:
  - `AdRequest(request_id: str, impression: AdImpression, split: str, source_row_index: int)`; `build_ad_requests(sim, rng) -> list[AdRequest]` (lấy mẫu theo `config.ads_requests`).
  - `export_ads(sim, requests, out_dir) -> dict` ghi `out_dir/ads/` theo spec §6.3 với cột đúng tập reader Ads yêu cầu (`profiles`: `user_id, as_of_timestamp, age_group, saved_home_geohash, saved_work_geohash`; `ad_catalog`: `candidate_id, first_observed_timestamp, category, advertiser, title`; `events`: `event_id, source_row_index, user_id, timestamp, event_type, service, entity_id, event_index, attributes_json`; `ad_opportunities`: 13 cột của spec §6.3). `cutoff_timestamp = exposure_timestamp = impression.ts`; `history_start/stop` trỏ vào `events` của user, chỉ event `timestamp < cutoff` (loại cả event cùng giây), tối đa 30.
  - **Quyết định:** reader Ads giữ một dòng profile cho mỗi user, nên `profiles` chỉ chứa trường **tĩnh** (nhóm tuổi, geohash nhà/chỗ làm đã lưu) với `as_of_timestamp` = đầu ngày 1; không xuất các trường tổng hợp 30 ngày ở gói Ads (tránh dùng thông tin sau cutoff — spec §4.6 là ràng buộc gốc). `first_observed_timestamp` = đầu ngày 1.

- [ ] **Step 1: Viết test (fail)**

```python
REQUIRED = {"profiles": {"user_id", "as_of_timestamp"}, "ad_catalog": {"candidate_id", "first_observed_timestamp"},
            "events": {"event_id", "source_row_index", "user_id", "timestamp", "event_type", "service", "entity_id", "event_index"},
            "ad_opportunities": {"request_id", "source_row_index", "user_id", "cutoff_timestamp", "candidate_id", "clicked",
                                 "split", "exposure_timestamp", "candidate_timestamp", "profile_as_of_timestamp",
                                 "history_end_timestamp", "history_start", "history_stop"}}

def test_tables_have_reader_columns_and_hashes(ads_root):
    manifest = json.loads((ads_root / "manifest.json").read_text())
    for name, cols in REQUIRED.items():
        table = pq.read_table(ads_root / manifest["files"][name]["path"])
        assert cols <= set(table.column_names) and table.num_rows == manifest["files"][name]["row_count"]
    assert manifest["source_dataset"] == "sim_v1" and manifest["negative_sampling"] == "none"
    assert manifest["history_policy"] == {"timestamp_rule": "strictly < cutoff", "target_click_in_history": False,
                                          "equal_timestamp_events_excluded": True, "max_events": 30}

def test_history_pointers_strictly_before_exposure(ads_root):
    events, opps = read_events_by_user(ads_root), read_table(ads_root, "ad_opportunities")
    for o in opps:
        window = events[o["user_id"]][o["history_start"]:o["history_stop"]]
        assert o["history_stop"] - o["history_start"] <= 30
        assert all(e["timestamp"] < o["cutoff_timestamp"] for e in window)
        assert o["cutoff_timestamp"] == o["exposure_timestamp"]

def test_ctr_and_splits(ads_root, tiny_config):
    opps = read_table(ads_root, "ad_opportunities")
    assert 0.03 <= sum(o["clicked"] for o in opps) / len(opps) <= 0.07
    assert all(o["split"] == split_of(o["exposure_timestamp"], tiny_config) for o in opps)

def test_profiles_static(ads_root, tiny_config):
    profiles = read_table(ads_root, "profiles")
    assert {p["as_of_timestamp"] for p in profiles} == {day_start_ts(1, tiny_config)}
    assert not any(column.endswith("_30d") for column in profiles[0])
```

(`read_table`, `read_events_by_user` là helper đọc Parquet trong file test; `events[user]` sắp theo `event_index`.)

- [ ] **Step 2–5:** chạy fail → implement + mục "Ads" của contract → pass → commit `feat: export simulated Ads opportunities`.

---

### Task 6: Gói Baseline JSON và oracle

**Files:**
- Create: `src/simulation/export_baseline.py`, `src/simulation/export_oracle.py`
- Test: `tests/simulation/test_export_baseline_oracle.py`

**Interfaces:**
- Consumes: `FoodRequest`, `AdRequest`, `food_logits`, `history.last_history`.
- Produces:
  - `export_baseline(sim, food_requests, ad_requests, out_dir) -> dict` ghi `out_dir/baseline/food.json` và `ads.json` theo `src.baselines.runner.load_dataset`: `metadata` (`dataset_name="sim_v1"`, `data_version`, `task`, `item_kind` = `"merchant"` | `"ad"`, `timezone="Asia/Ho_Chi_Minh"`, `candidate_policy`, `availability_policy="handoff_timestamp"`), `queries` (`cutoff_time` ISO `+07:00`), `candidates`, `labels` (`available_at` = `label_time` Food / thời điểm hiển thị + 1 giây Ads), `history` (Interaction cho mọi event trước cutoff lớn nhất của user có request, `available_at = event_time`).
  - `export_oracle(sim, food_requests, ad_requests, out_dir) -> dict` ghi `out_dir/oracle/food_oracle.parquet` (`request_id, candidate_id, true_probability` = softmax `food_logits` trên 10 ứng viên) và `ads_oracle.parquet` (`request_id, true_click_probability`).

- [ ] **Step 1: Viết test (fail)**

```python
def test_baseline_json_loads_with_runner(baseline_root):
    from src.baselines.runner import load_dataset
    food, ads = load_dataset(baseline_root / "food.json"), load_dataset(baseline_root / "ads.json")
    assert food.metadata.task == "food_recommendation" and ads.metadata.task == "ad_optimization"
    assert len(food.candidates) == 10 * len(food.queries)

def test_oracle_probabilities(oracle_root):
    food = pq.read_table(oracle_root / "food_oracle.parquet").to_pylist()
    sums = defaultdict(float)
    for row in food: sums[row["request_id"]] += row["true_probability"]
    assert all(abs(v - 1) < 1e-9 for v in sums.values())
    ads = pq.read_table(oracle_root / "ads_oracle.parquet").column("true_click_probability").to_pylist()
    assert all(0 < p < 1 for p in ads)
```

- [ ] **Step 2–5:** fail → implement → pass → commit `feat: export Baseline datasets and oracle probabilities`.

---

### Task 7: Kiểm tra, hiệu chỉnh độ khó, CLI và báo cáo

**Files:**
- Create: `src/simulation/checks.py`, `src/simulation/calibrate.py`, `src/simulation/__main__.py`; Modify: `docs/simulation_data_contract.md` (mục "Chạy" và "data_report")
- Test: `tests/simulation/test_cli_and_checks.py`

**Interfaces:**
- Consumes: mọi export ở Task 3–6.
- Produces:
  - `run_checks(out_dir: Path, config: SimConfig) -> dict` — đọc lại file đã xuất và kiểm tra các mục "Không lộ dữ liệu" + "Đúng định dạng" của spec §8; trả dict kết quả; vi phạm → `ValueError` nêu kiểm tra và tối đa 20 ID.
  - `difficulty_report(out_dir: Path) -> dict` — trên split test: Food `popularity_ndcg_at_10` (điểm = số đơn tuần 1–6 của quán), `candidate_only_ndcg_at_10` (tỷ lệ positive theo `(cuisine, price_level)` fit trên train, cộng phổ biến), `oracle_ndcg_at_10`; Ads `popularity_auc` (CTR train làm mịn theo quảng cáo), `oracle_auc`. Dùng `src.evaluation.metrics.ranking_metrics` và `advertising_metrics`.
  - Fixture `tiny_config_file` (thêm vào `tests/simulation/conftest.py`): ghi YAML `{preset: smoke, n_users: 80, n_merchants: 24, n_ads: 40, n_drivers: 40}` vào `tmp_path` và trả đường dẫn; `load_config` phải áp các khóa ghi đè này.
  - `main(argv=None) -> int`: `python -m src.simulation --preset {smoke,full} | --config PATH, --out DIR, [--seed N], [--skip-pretrain]`; chạy world → simulate → export (pretrain trừ khi `--skip-pretrain`) → checks → difficulty → ghi `out_dir/<dataset_version>/data_report.json` (spec §6.6, có `generated_at_utc`, `difficulty`, `checks`, SHA mọi file); in đường dẫn report. Lỗi kiểm tra → exit 3.

- [ ] **Step 1: Viết test (fail)**

```python
def test_cli_end_to_end_tiny(tmp_path, tiny_config_file):
    assert main(["--config", str(tiny_config_file), "--out", str(tmp_path)]) == 0
    root = tmp_path / "sim-smoke-v1"
    report = json.loads((root / "data_report.json").read_text())
    assert {"pretrain", "food", "ads", "baseline", "oracle"} <= {p.name for p in root.iterdir()}
    d = report["difficulty"]
    assert d["food"]["oracle_ndcg_at_10"] > d["food"]["popularity_ndcg_at_10"]
    assert d["ads"]["oracle_auc"] > d["ads"]["popularity_auc"]
    assert d["food"]["candidate_only_ndcg_at_10"] <= d["food"]["popularity_ndcg_at_10"] + 0.05

def test_same_seed_same_bytes_different_seed_differs(tmp_path, tiny_config_file):
    a, b, c = (tmp_path / x for x in "abc")
    main(["--config", str(tiny_config_file), "--out", str(a)]); main(["--config", str(tiny_config_file), "--out", str(b)])
    main(["--config", str(tiny_config_file), "--out", str(c), "--seed", "7"])
    assert tree_hashes(a) == tree_hashes(b) != tree_hashes(c)        # bỏ qua data_report.json

def test_checks_catch_leak(tmp_path, tiny_config_file):
    main(["--config", str(tiny_config_file), "--out", str(tmp_path)])
    inject_future_history_event(tmp_path / "sim-smoke-v1" / "food" / "samples.jsonl")   # helper sửa 1 dòng
    with pytest.raises(ValueError, match="history"):
        run_checks(tmp_path / "sim-smoke-v1", load_config(tiny_config_file))
```

- [ ] **Step 2: Chạy test để thấy fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Chạy test, rồi toàn bộ** — `.venv/Scripts/python -m pytest -q` và `.venv/Scripts/python -m ruff check src tests configs` → PASS/clean.
- [ ] **Step 5: Chạy preset smoke thật** — `.venv/Scripts/python -m src.simulation --preset smoke --out experiments/sim` → exit 0; ghi thời gian chạy và `difficulty` vào report của task.
- [ ] **Step 6: Commit** — `git commit -m "feat: simulator checks, difficulty report and CLI"`

---

### Task 8: Hiệu chỉnh độ khó trên `full` (không xuất pretrain)

**Files:**
- Modify: `src/simulation/config.py` (giá trị mặc định `NoiseConfig`), `docs/simulation_data_contract.md` (bảng độ khó đạt được)
- Test: `tests/simulation/test_world.py` (pin giá trị mặc định mới)

- [ ] **Step 1: Đo** — `.venv/Scripts/python -m src.simulation --preset full --out experiments/sim --skip-pretrain` (chạy nền; ghi thời gian). Đọc `difficulty` trong `experiments/sim/sim-v1/data_report.json`.
- [ ] **Step 2: Hiệu chỉnh** chỉ các hệ số trong `NoiseConfig` để đạt mục tiêu spec §5: Food `popularity_ndcg_at_10 ∈ [0.40, 0.55]`, `oracle − popularity ≥ 0.20`, `candidate_only ≤ popularity + 0.05`; Ads `popularity_auc ∈ [0.55, 0.65]`, `oracle_auc ≥ 0.75`; CTR ∈ [0.045, 0.055]. Tối đa 6 vòng đo; ghi từng vòng (hệ số → chỉ số) vào report. Nếu sau 6 vòng chưa đạt, dừng và báo DONE_WITH_CONCERNS kèm vòng tốt nhất.
- [ ] **Step 3: Pin giá trị** — thêm test `test_noise_defaults_are_calibrated` khẳng định giá trị `NoiseConfig` mặc định bằng bộ đã chọn; chạy toàn bộ test + ruff.
- [ ] **Step 4: Commit** — `git commit -m "chore: calibrate simulator difficulty on full preset"` (không commit `experiments/`).
