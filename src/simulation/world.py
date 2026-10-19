"""Static simulated world: zones, catalog, drivers and latent user traits."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import (
    AD_CATEGORIES,
    CUISINES,
    PERSONA_SHARES,
    ZONE_TYPE_COUNTS,
    SimConfig,
)

CENTER_LAT = 10.7769
CENTER_LON = 106.7009
BOX_DEG = 0.09
_BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"
# Zone centres are at least this far apart and never share a 6-char geohash, so a zone is
# always recoverable from the geohash seen in event history.
MIN_ZONE_SEPARATION_KM = 1.0
_MAX_ZONE_REDRAWS = 10_000

AGE_GROUPS = ("18-24", "25-34", "35-44", "45-54", "55+")
VEHICLE_TYPES = ("bike", "car", "premium")

_CUISINE_IDX = {c: i for i, c in enumerate(CUISINES)}
_AD_IDX = {c: i for i, c in enumerate(AD_CATEGORIES)}

_MERCHANT_PREFIX = {
    "cơm văn phòng": ("Cơm Tấm", "Cơm Gà", "Cơm Sườn", "Cơm Phần"),
    "phở/bún": ("Phở", "Bún Bò", "Bún Chả", "Hủ Tiếu"),
    "healthy/salad": ("Salad", "Green Bowl", "Fresh Eat", "Healthy Kitchen"),
    "gà rán": ("Gà Rán", "Gà Giòn", "Chicken House", "Gà Sốt"),
    "pizza": ("Pizza", "Pizza Lò Củi", "Pizza Slice", "Pizza Box"),
    "lẩu": ("Lẩu", "Lẩu Thái", "Lẩu Nướng", "Lẩu Bò"),
    "đồ Nhật": ("Sushi", "Ramen", "Bento", "Izakaya"),
    "đồ Hàn": ("Tokbokki", "BBQ Hàn", "Kimbap", "Gà Hàn"),
    "trà sữa": ("Trà Sữa", "Trà Đào", "Milk Tea", "Trà Chanh"),
    "cà phê/bánh": ("Cà Phê", "Bánh Ngọt", "Coffee", "Tiệm Bánh"),
    "đồ chay": ("Quán Chay", "Cơm Chay", "Bún Chay", "Chay An Lạc"),
    "ăn vặt đêm": ("Ăn Vặt", "Bánh Tráng", "Xiên Que", "Cháo Đêm"),
}
_OWNER_NAMES = (
    "Phúc", "Lộc", "Thảo", "Minh", "An", "Hòa", "Bình", "Cúc", "Dũng", "Lan",
    "Hương", "Nam", "Tâm", "Vy", "Quân", "Mai",
)
_DISHES = {
    "cơm văn phòng": (
        "Cơm sườn nướng", "Cơm gà xối mỡ", "Cơm thịt kho", "Cơm cá kho", "Cơm bò xào",
        "Cơm tấm bì chả", "Cơm chiên dương châu", "Cơm gà teriyaki", "Cơm đùi gà",
        "Cơm trứng chiên",
    ),
    "phở/bún": (
        "Phở bò tái", "Phở gà", "Bún bò Huế", "Bún chả", "Bún riêu", "Hủ tiếu nam vang",
        "Bún thịt nướng", "Phở cuốn", "Bún mọc", "Miến gà",
    ),
    "healthy/salad": (
        "Salad ức gà", "Salad cá hồi", "Bowl quinoa", "Salad rau củ", "Gỏi ngó sen",
        "Bowl bơ trứng", "Sinh tố xanh", "Ức gà áp chảo", "Salad tôm", "Bowl đậu hũ",
    ),
    "gà rán": (
        "Gà rán giòn", "Gà sốt cay", "Cánh gà chiên mắm", "Burger gà", "Gà viên",
        "Khoai tây chiên", "Gà popcorn", "Gà sốt phô mai", "Combo gà 2 miếng", "Gà không xương",
    ),
    "pizza": (
        "Pizza hải sản", "Pizza phô mai", "Pizza bò", "Pizza xúc xích", "Pizza rau củ",
        "Mì Ý bò bằm", "Pizza gà nướng", "Salad kèm pizza", "Khoai wedge",
        "Pizza 4 loại phô mai",
    ),
    "lẩu": (
        "Lẩu Thái hải sản", "Lẩu bò nhúng dấm", "Lẩu nấm", "Lẩu gà lá é", "Lẩu cá kèo",
        "Lẩu riêu cua", "Lẩu thập cẩm", "Lẩu kim chi", "Lẩu mắm", "Lẩu dê",
    ),
    "đồ Nhật": (
        "Sushi cá hồi", "Ramen tonkotsu", "Cơm cà ri Nhật", "Tempura tôm", "Udon bò",
        "Cơm bento", "Sashimi tổng hợp", "Takoyaki", "Katsu don", "Gyoza chiên",
    ),
    "đồ Hàn": (
        "Tokbokki phô mai", "Kimbap chiên", "Mì lạnh", "Gà rán Hàn Quốc", "Canh kim chi",
        "Bibimbap", "Bulgogi bò", "Mì tương đen", "Cơm cuộn", "Lẩu tokbokki",
    ),
    "trà sữa": (
        "Trà sữa trân châu", "Trà đào cam sả", "Trà sữa matcha", "Trà vải",
        "Sữa tươi trân châu đường đen", "Trà chanh", "Hồng trà macchiato", "Trà dâu tằm",
        "Sữa chua dẻo", "Trà ô long",
    ),
    "cà phê/bánh": (
        "Cà phê sữa đá", "Bạc xỉu", "Cappuccino", "Bánh tiramisu", "Bánh croissant",
        "Cà phê muối", "Bánh mousse", "Cold brew", "Bánh cookie", "Latte",
    ),
    "đồ chay": (
        "Cơm chay thập cẩm", "Bún chay", "Đậu hũ sốt cà", "Phở chay", "Gỏi cuốn chay",
        "Nấm xào sả ớt", "Cháo chay", "Mì xào chay", "Chả giò chay", "Canh rong biển",
    ),
    "ăn vặt đêm": (
        "Bánh tráng trộn", "Xiên que nướng", "Cá viên chiên", "Bắp xào", "Cháo sườn",
        "Bánh mì nướng muối ớt", "Trứng cút lộn xào me", "Bò bía", "Chân gà sả tắc",
        "Hột vịt lộn",
    ),
}
_PRICE_BASE = (25000.0, 45000.0, 90000.0)

_AD_ADVERTISERS = {
    "đồ ăn": ("FoodHub", "BếpNhà", "VuaBurger"),
    "thể thao": ("RunFast", "GymPro", "SportZone"),
    "du lịch": ("ViVu", "BayXa", "TourPlus"),
    "khách sạn": ("StayInn", "NghỉDưỡng", "Hotelio"),
    "thời trang": ("StyleVN", "ThờiTrangMới", "FashionBox"),
    "làm đẹp": ("GlowUp", "SpaXinh", "Cosmo"),
    "điện tử": ("TechMart", "ĐiệnMáyXanh", "GadgetPro"),
    "tài chính": ("VayNhanh", "BảoHiểmAn", "TiếtKiệmPlus"),
    "giáo dục": ("EduViet", "AnhNgữPro", "CodeSchool"),
    "game": ("GameVerse", "PlayNow", "EsportVN"),
    "mẹ & bé": ("BéYêu", "MẹVàBé", "SữaTốt"),
    "nhà cửa": ("NộiThấtĐẹp", "NhàXinh", "HomeFix"),
    "xe cộ": ("AutoVN", "PhụTùngXe", "RửaXeNhanh"),
    "sức khỏe": ("VitaCare", "PhòngKhám24", "ThựcPhẩmXanh"),
    "giải trí": ("CinePlus", "NhạcSống", "VéSựKiện"),
}
_AD_TITLES = {
    "đồ ăn": ("Giảm 30% đơn đầu tiên", "Combo bữa trưa tiết kiệm", "Freeship cả tuần"),
    "thể thao": ("Giày chạy bộ giảm giá", "Đăng ký phòng gym 3 tháng", "Quần áo thể thao mới"),
    "du lịch": ("Vé máy bay giá rẻ", "Tour Phú Quốc 3 ngày", "Combo du lịch hè"),
    "khách sạn": ("Đặt phòng giảm 40%", "Khách sạn gần sân bay", "Resort cuối tuần"),
    "thời trang": ("Bộ sưu tập mới", "Sale đồ công sở", "Giày sneaker hot"),
    "làm đẹp": ("Serum dưỡng da", "Spa thư giãn", "Son môi mới ra mắt"),
    "điện tử": ("Tai nghe không dây", "Điện thoại trả góp 0%", "Laptop học tập"),
    "tài chính": ("Vay nhanh lãi thấp", "Bảo hiểm sức khỏe", "Gửi tiết kiệm lãi cao"),
    "giáo dục": ("Khóa tiếng Anh online", "Học lập trình cho người mới", "Luyện thi IELTS"),
    "game": ("Nạp thẻ game ưu đãi", "Game nhập vai mới", "Giải đấu esport"),
    "mẹ & bé": ("Sữa cho bé", "Tã giấy giảm giá", "Xe đẩy em bé"),
    "nhà cửa": ("Sofa phòng khách", "Dọn nhà theo giờ", "Đèn trang trí"),
    "xe cộ": ("Bảo dưỡng xe máy", "Mũ bảo hiểm mới", "Rửa xe tận nơi"),
    "sức khỏe": ("Vitamin tổng hợp", "Khám sức khỏe tổng quát", "Thực phẩm hữu cơ"),
    "giải trí": ("Vé xem phim 2D", "Đêm nhạc cuối tuần", "Xem phim không giới hạn"),
}

# persona -> boost on the cuisine Dirichlet concentration (base concentration 1.0)
_PERSONA_CUISINE_BOOST = {
    "office": {"cơm văn phòng": 4.0, "phở/bún": 2.0, "healthy/salad": 1.5, "cà phê/bánh": 1.5},
    "student": {"trà sữa": 4.0, "gà rán": 2.5, "ăn vặt đêm": 2.0, "đồ Hàn": 2.0, "pizza": 1.0},
    "night_shift": {"ăn vặt đêm": 4.0, "lẩu": 2.5, "phở/bún": 1.5, "gà rán": 1.5},
    "home": {"đồ chay": 3.0, "phở/bún": 3.0, "cơm văn phòng": 1.5, "cà phê/bánh": 1.0},
    "traveler": {"đồ Nhật": 3.0, "healthy/salad": 2.5, "cà phê/bánh": 2.0, "pizza": 1.5},
}
_PERSONA_AD_BOOST = {
    "office": {"tài chính": 2.0, "điện tử": 1.5, "thời trang": 1.0, "giáo dục": 1.0},
    "student": {"game": 2.5, "giáo dục": 2.0, "thời trang": 1.5, "điện tử": 1.0, "giải trí": 1.5},
    "night_shift": {"giải trí": 2.0, "game": 1.5, "sức khỏe": 1.0, "xe cộ": 1.0},
    "home": {"nhà cửa": 2.5, "sức khỏe": 2.0, "mẹ & bé": 1.5, "đồ ăn": 1.0},
    "traveler": {"du lịch": 2.5, "khách sạn": 2.5, "tài chính": 1.0, "điện tử": 1.0},
}
# cuisine taste -> ad-category affinity (taste-driven part of ad interest)
_CUISINE_AD_LINK = {
    "healthy/salad": {"thể thao": 1.0, "sức khỏe": 1.0},
    "đồ chay": {"sức khỏe": 1.0},
    "gà rán": {"đồ ăn": 0.6, "game": 0.4},
    "trà sữa": {"làm đẹp": 0.4, "thời trang": 0.4},
    "đồ Nhật": {"du lịch": 0.5},
    "cà phê/bánh": {"giải trí": 0.5},
}
# persona -> spend_level probabilities (low, mid, high)
_SPEND_PROBS = {
    "office": (0.2, 0.55, 0.25),
    "student": (0.6, 0.35, 0.05),
    "night_shift": (0.4, 0.45, 0.15),
    "home": (0.5, 0.4, 0.1),
    "traveler": (0.1, 0.4, 0.5),
}
# persona -> (gamma shape, gamma scale) for app activity
_ACTIVITY = {
    "office": (4.0, 0.25),
    "student": (4.0, 0.28),
    "night_shift": (3.5, 0.28),
    "home": (3.0, 0.3),
    "traveler": (4.0, 0.25),
}
_AGE_PROBS = {
    "office": (0.15, 0.40, 0.30, 0.13, 0.02),
    "student": (0.85, 0.13, 0.02, 0.0, 0.0),
    "night_shift": (0.25, 0.40, 0.25, 0.10, 0.0),
    "home": (0.0, 0.05, 0.15, 0.30, 0.50),
    "traveler": (0.05, 0.35, 0.35, 0.20, 0.05),
}
_WORK_ZONE_TYPE = {
    "office": "office",
    "student": "university",
    "night_shift": "nightlife",
    "traveler": "office",
    "home": None,
}
# persona -> (commute-out hour mean, commute-back hour mean, workdays Mon=0)
_SCHEDULE = {
    "office": (8.0, 17.5, (0, 1, 2, 3, 4)),
    "student": (7.5, 16.0, (0, 1, 2, 3, 4, 5)),
    "night_shift": (20.0, 5.0, (0, 1, 2, 3, 4)),
    "home": (9.0, 11.0, ()),
    "traveler": (8.5, 18.0, (0, 1, 2, 3, 4)),
}


@dataclass(frozen=True)
class Zone:
    zone_id: int
    zone_type: str
    lat: float
    lon: float
    geohash: str


@dataclass(frozen=True)
class Merchant:
    merchant_id: str
    zone_id: int
    cuisine: str
    price_level: int
    name: str
    base_popularity: float


@dataclass(frozen=True)
class Dish:
    dish_id: str
    merchant_id: str
    name: str
    description: str
    price: float


@dataclass(frozen=True)
class Ad:
    ad_id: str
    category: str
    advertiser: str
    title: str
    target_zone: int | None


@dataclass(frozen=True)
class Driver:
    driver_id: str
    vehicle_type: str
    zone_id: int


@dataclass(frozen=True)
class UserTraits:
    user_id: str
    persona: str
    age_group: str
    home_zone: int
    work_zone: int | None
    activity: float
    cuisine_pref: tuple[float, ...]
    spend_level: int
    ad_interest: tuple[float, ...]
    commute_out_hour: float
    commute_back_hour: float
    workdays: tuple[int, ...]


@dataclass(frozen=True)
class World:
    config: SimConfig
    zones: list[Zone]
    merchants: list[Merchant]
    dishes: list[Dish]
    ads: list[Ad]
    drivers: list[Driver]
    users: list[UserTraits]


def geohash_encode(lat: float, lon: float, precision: int = 6) -> str:
    """Standard geohash (base32, interleaved bits, longitude first)."""
    lat_lo, lat_hi, lon_lo, lon_hi = -90.0, 90.0, -180.0, 180.0
    out: list[str] = []
    bits = 0
    value = 0
    even = True
    while len(out) < precision:
        if even:
            mid = (lon_lo + lon_hi) / 2
            if lon >= mid:
                value = value * 2 + 1
                lon_lo = mid
            else:
                value *= 2
                lon_hi = mid
        else:
            mid = (lat_lo + lat_hi) / 2
            if lat >= mid:
                value = value * 2 + 1
                lat_lo = mid
            else:
                value *= 2
                lat_hi = mid
        even = not even
        bits += 1
        if bits == 5:
            out.append(_BASE32[value])
            bits = 0
            value = 0
    return "".join(out)


def haversine_km(a: Zone, b: Zone) -> float:
    r = 6371.0088
    p1, p2 = math.radians(a.lat), math.radians(b.lat)
    dphi = p2 - p1
    dlmb = math.radians(b.lon - a.lon)
    h = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def _build_zones(config: SimConfig, rng: np.random.Generator) -> list[Zone]:
    types: list[str] = []
    for zone_type, count in ZONE_TYPE_COUNTS.items():
        types.extend([zone_type] * count)
    if len(types) != config.n_zones:
        raise ValueError(f"n_zones={config.n_zones} must equal {len(types)} (ZONE_TYPE_COUNTS)")
    order = rng.permutation(len(types))
    types = [types[int(i)] for i in order]
    lats = rng.uniform(CENTER_LAT - BOX_DEG, CENTER_LAT + BOX_DEG, size=len(types))
    lons = rng.uniform(CENTER_LON - BOX_DEG, CENTER_LON + BOX_DEG, size=len(types))
    zones: list[Zone] = []
    used_geohashes: set[str] = set()
    for i, zone_type in enumerate(types):
        lat, lon = float(lats[i]), float(lons[i])
        for _ in range(_MAX_ZONE_REDRAWS):
            zone = Zone(i, zone_type, lat, lon, geohash_encode(lat, lon, 6))
            if zone.geohash not in used_geohashes and all(
                haversine_km(zone, other) >= MIN_ZONE_SEPARATION_KM for other in zones
            ):
                break
            # rejected: redraw this centre only (same rng, zone order => deterministic)
            lat = float(rng.uniform(CENTER_LAT - BOX_DEG, CENTER_LAT + BOX_DEG))
            lon = float(rng.uniform(CENTER_LON - BOX_DEG, CENTER_LON + BOX_DEG))
        else:
            raise RuntimeError(f"could not place zone {i} with distinct geohash and spacing")
        used_geohashes.add(zone.geohash)
        zones.append(zone)
    return zones


def _build_catalog(
    config: SimConfig, rng: np.random.Generator, zones: list[Zone]
) -> tuple[list[Merchant], list[Dish]]:
    merchants: list[Merchant] = []
    dishes: list[Dish] = []
    dish_n = 0
    for m in range(config.n_merchants):
        cuisine = CUISINES[int(rng.integers(len(CUISINES)))]
        zone_id = int(rng.integers(len(zones)))
        price_level = int(rng.choice(3, p=(0.35, 0.45, 0.20)))
        prefix = _MERCHANT_PREFIX[cuisine][int(rng.integers(4))]
        owner = _OWNER_NAMES[int(rng.integers(len(_OWNER_NAMES)))]
        popularity = float(rng.lognormal(0.0, 0.6))
        merchant_id = f"sim:merchant:{m}"
        name = f"{prefix} {owner} {m}"
        merchants.append(Merchant(merchant_id, zone_id, cuisine, price_level, name, popularity))
        vocab = _DISHES[cuisine]
        picks = rng.choice(len(vocab), size=config.dishes_per_merchant, replace=False)
        for k in picks:
            price = round(_PRICE_BASE[price_level] * float(rng.uniform(0.8, 1.4)), -3)
            dish_name = vocab[int(k)]
            desc = f"{dish_name} của quán {name}, món {cuisine} phục vụ nóng."
            dishes.append(Dish(f"sim:dish:{dish_n}", merchant_id, dish_name, desc, float(price)))
            dish_n += 1
    return merchants, dishes


def _build_ads(config: SimConfig, rng: np.random.Generator, zones: list[Zone]) -> list[Ad]:
    ads: list[Ad] = []
    for i in range(config.n_ads):
        category = AD_CATEGORIES[int(rng.integers(len(AD_CATEGORIES)))]
        advertiser = _AD_ADVERTISERS[category][int(rng.integers(3))]
        title = _AD_TITLES[category][int(rng.integers(3))]
        target = int(rng.integers(len(zones))) if rng.random() < 0.3 else None
        ads.append(Ad(f"sim:ad:{i}", category, advertiser, f"{title} - {advertiser}", target))
    return ads


def _build_drivers(config: SimConfig, rng: np.random.Generator, zones: list[Zone]) -> list[Driver]:
    drivers = []
    for i in range(config.n_drivers):
        vehicle = VEHICLE_TYPES[int(rng.choice(3, p=(0.7, 0.22, 0.08)))]
        drivers.append(Driver(f"sim:driver:{i}", vehicle, int(rng.integers(len(zones)))))
    return drivers


def _persona_assignment(n: int, rng: np.random.Generator) -> list[str]:
    """Largest-remainder counts per persona, then shuffled (keeps shares near-exact)."""
    names = list(PERSONA_SHARES)
    raw = [PERSONA_SHARES[p] * n for p in names]
    counts = [int(x) for x in raw]
    by_remainder = sorted(range(len(names)), key=lambda i: (-(raw[i] - counts[i]), i))
    for i in by_remainder[: n - sum(counts)]:
        counts[i] += 1
    personas = [p for p, c in zip(names, counts) for _ in range(c)]
    order = rng.permutation(n)
    return [personas[int(i)] for i in order]


def _build_users(
    config: SimConfig, rng: np.random.Generator, zones: list[Zone]
) -> list[UserTraits]:
    by_type: dict[str, list[int]] = {}
    for z in zones:
        by_type.setdefault(z.zone_type, []).append(z.zone_id)
    users: list[UserTraits] = []
    for i, persona in enumerate(_persona_assignment(config.n_users, rng)):
        age = AGE_GROUPS[int(rng.choice(5, p=_AGE_PROBS[persona]))]
        homes = by_type["residential"]
        home = homes[int(rng.integers(len(homes)))]
        work_type = _WORK_ZONE_TYPE[persona]
        work = None
        if work_type is not None:
            works = by_type[work_type]
            work = works[int(rng.integers(len(works)))]
        shape, scale = _ACTIVITY[persona]
        activity = float(rng.gamma(shape, scale))
        alpha = np.ones(len(CUISINES))
        for cuisine, boost in _PERSONA_CUISINE_BOOST[persona].items():
            alpha[_CUISINE_IDX[cuisine]] += boost
        pref = rng.dirichlet(alpha)
        pref = pref / pref.sum()
        spend = int(rng.choice(3, p=_SPEND_PROBS[persona]))
        ad = np.full(len(AD_CATEGORIES), 0.2)
        for cat, boost in _PERSONA_AD_BOOST[persona].items():
            ad[_AD_IDX[cat]] += boost * 0.3
        for cuisine, links in _CUISINE_AD_LINK.items():
            for cat, w in links.items():
                ad[_AD_IDX[cat]] += w * float(pref[_CUISINE_IDX[cuisine]]) * 4.0
        ad = ad + rng.gamma(2.0, 0.1, size=len(AD_CATEGORIES))
        ad = ad / ad.sum()
        out_mean, back_mean, workdays = _SCHEDULE[persona]
        out_h = float((out_mean + rng.normal(0.0, 0.5)) % 24.0)
        back_h = float((back_mean + rng.normal(0.0, 0.5)) % 24.0)
        users.append(
            UserTraits(
                user_id=f"sim:user:{i}",
                persona=persona,
                age_group=age,
                home_zone=home,
                work_zone=work,
                activity=activity,
                cuisine_pref=tuple(float(x) for x in pref),
                spend_level=spend,
                ad_interest=tuple(float(x) for x in ad),
                commute_out_hour=out_h,
                commute_back_hour=back_h,
                workdays=workdays,
            )
        )
    return users


def build_world(config: SimConfig, rng: np.random.Generator) -> World:
    """Build the static world; every random draw comes from ``rng`` in a fixed order."""
    zones = _build_zones(config, rng)
    merchants, dishes = _build_catalog(config, rng, zones)
    ads = _build_ads(config, rng, zones)
    drivers = _build_drivers(config, rng, zones)
    users = _build_users(config, rng, zones)
    return World(config, zones, merchants, dishes, ads, drivers, users)
