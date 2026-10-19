"""Daily ride / food / ads behaviour, short-term intents and true probabilities (spec §4.3-4.5).

Users are simulated one after another (in id order), day by day. Every random draw comes from
the single ``numpy.random.Generator`` passed to :func:`simulate`, consumed in a fixed order.

Cross-service link: a user's *current location* is the drop-off zone of the latest
``complete_ride`` of the current local day (home zone when there is none). Food merchant choice
(distance term) and ad clicks (zone match) both use it, so it is only recoverable from the ride
history.

True probabilities:

* Food: ``food_logits`` = cuisine taste (log, shifted to healthy/vegetarian during a diet)
  + price match with the effective spend level - distance from the current location
  + base popularity. The chosen merchant is a multinomial-logit draw over **all** merchants
  with logits ``food_logits + eps``, ``eps ~ N(0, food_logit_noise_sd)`` drawn fresh per session
  and merchant; the noise is never part of ``food_logits``. Probability of a choice inside any
  candidate set = softmax of ``food_logits`` on that set.
* Ads: ``z`` = interest (log) + active intents + zone match + recent rides/orders in the other
  services; ``click_probability = E_eps[sigmoid(base + z + eps)]`` with
  ``eps ~ N(0, ad_logit_noise_sd)`` integrated out by 16-node Gauss-Hermite quadrature.
  ``clicked`` is drawn from that probability. ``base`` is solved from the world alone (bisection
  on a fixed, deterministic sample of users x served-ad distribution) so that CTR ~ target_ctr.
"""

from __future__ import annotations

import math
import weakref
from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np

from .config import AD_CATEGORIES, CUISINES
from .timeutil import SECONDS_PER_DAY, day_start_ts, n_days
from .world import _PERSONA_AD_BOOST, VEHICLE_TYPES, World, haversine_km

INTENT_KINDS: tuple[str, ...] = ("diet", "trip", "job_change", "payday")
SERVICES: tuple[str, ...] = ("ride", "food", "ads")

# --- short-term intents (spec §4.4) -------------------------------------------------------
_P_DIET_WEEK = 0.05
_P_TRIP_WEEK = 0.04
_TRAVELER_TRIP_FACTOR = 4.0
_P_JOB_CHANGE_WEEK = 0.01
_DIET_DAYS = (5, 10)
_TRIP_DAYS = (3, 7)
_PAYDAY_FIRST_DOM = (25, 28)
_PAYDAY_DAYS = (2, 4)
_DIET_CUISINES = ("healthy/salad", "đồ chay")
_DIET_TASTE_SHIFT = 0.7  # share of taste mass moved onto _DIET_CUISINES during a diet

# --- daily activity (per day, before the user's activity multiplier) ------------------------
# The per-day rates below, unscaled, give about this many events per user in weeks 1-6
# (measured on smoke and a 2k-user full-catalog run); rates are scaled by
# target_events_per_user / this value.
_UNSCALED_EVENTS_PER_USER = 238.0
_COMMUTE_P = {"office": 0.5, "student": 0.35, "night_shift": 0.45, "home": 0.0, "traveler": 0.55}
_RETURN_P_AFTER_RIDE = 0.75
_RETURN_P_OTHERWISE = 0.2
_ERRAND_P = {"office": 0.06, "student": 0.06, "night_shift": 0.06, "home": 0.25, "traveler": 0.06}
_OUTING_P = {"office": 0.2, "student": 0.28, "night_shift": 0.12, "home": 0.07, "traveler": 0.2}
_OUTING_WEEKDAY_FACTOR = 0.25  # Mon-Thu evenings vs Fri-Sun
_ERRAND_RETURN_P = 0.8
_OUTING_RETURN_P = 0.7
_LAST_RIDE_HOUR = 22.5
_MEAL_SLOTS = ((11.0, 13.0), (14.5, 16.5), (18.0, 20.5), (21.5, 23.25))
_MEAL_P = {
    "office": (0.22, 0.05, 0.17, 0.02),
    "student": (0.18, 0.08, 0.15, 0.07),
    "night_shift": (0.18, 0.04, 0.12, 0.14),
    "home": (0.17, 0.04, 0.17, 0.02),
    "traveler": (0.2, 0.05, 0.17, 0.02),
}
_PAYDAY_FOOD_FACTOR = 1.25
_AT_WORK_LUNCH_FACTOR = 1.6
_FOOD_ABANDON_P = 0.15
_APP_OPENS_PER_DAY = 0.6
_APP_HOURS = (7.0, 23.5)
_AD_AFTER_SESSION_P = 0.3
# A user's next session starts at least this long after the previous session ended (within a
# day; days start hours after the last session of the previous day), so two ad exposures of
# one user are always >= 60 s apart.
SESSION_GAP_SECONDS = 60
_SESSION_GAP_S = SESSION_GAP_SECONDS
# An ad click follows its impression after this many seconds (inclusive bounds); the export
# treats an impression without a click as observable one second after the upper bound.
AD_CLICK_DELAY_SECONDS = (2, 26)
_MAX_P = 0.95

# --- ride funnel and pricing ----------------------------------------------------------------
_P_QUOTE, _P_BOOK, _P_COMPLETE = 0.92, 0.85, 0.96
_VEHICLE_CUM = ((0.85, 1.0, 1.0), (0.55, 0.95, 1.0), (0.2, 0.75, 1.0))  # by spend level
_FARE = {"bike": (12000.0, 4300.0, 24.0), "car": (26000.0, 9500.0, 20.0),
         "premium": (38000.0, 13500.0, 20.0)}  # base VND, VND/km, km/h

# --- ads ------------------------------------------------------------------------------------
_AD_TARGETED_SHARE = 0.5
_ZONE_MATCH_RADIUS_KM = 4.0
_RECENT_WINDOW_S = 6 * 3600
_MAX_RECENT = 3
_GH_NODES = 16
_BASE_SAMPLE_USERS = 400
_INTENT_AD_BOOST = {
    "diet": {"thể thao": 1.0, "sức khỏe": 1.0},
    "trip": {"du lịch": 1.0, "khách sạn": 1.0},
    "payday": {"thời trang": 0.5, "điện tử": 0.5, "làm đẹp": 0.5},
}
_CROSS_AD_BOOST = {
    "ride:transport_hub": {"du lịch": 1.0, "khách sạn": 1.0},
    "ride:nightlife": {"giải trí": 0.7},
    "ride:mall": {"thời trang": 0.7},
    "food:healthy/salad": {"thể thao": 1.0, "sức khỏe": 0.8},
    "food:đồ chay": {"sức khỏe": 1.0},
}

_SEARCH_TERMS = {
    "cơm văn phòng": ("cơm trưa văn phòng", "cơm sườn", "cơm gà"),
    "phở/bún": ("phở bò", "bún bò huế", "bún chả"),
    "healthy/salad": ("salad", "eat clean", "healthy bowl"),
    "gà rán": ("gà rán", "gà sốt cay", "burger gà"),
    "pizza": ("pizza", "pizza hải sản", "mì ý"),
    "lẩu": ("lẩu thái", "lẩu bò", "lẩu nấm"),
    "đồ Nhật": ("sushi", "ramen", "cơm bento"),
    "đồ Hàn": ("tokbokki", "kimbap", "gà hàn quốc"),
    "trà sữa": ("trà sữa", "trà đào", "trà sữa trân châu"),
    "cà phê/bánh": ("cà phê", "bánh ngọt", "bạc xỉu"),
    "đồ chay": ("đồ chay", "cơm chay", "bún chay"),
    "ăn vặt đêm": ("ăn vặt", "bánh tráng trộn", "cháo đêm"),
}
_DIET_QUERY_SUFFIX = " ít calo"

_CUISINE_IDX = {c: i for i, c in enumerate(CUISINES)}
_AD_IDX = {c: i for i, c in enumerate(AD_CATEGORIES)}

# Constant action payloads are shared between events (treat Event dicts as read-only).
_ACTION = {
    "search": {"type": "search"},
    "view_merchant": {"type": "view_merchant"},
    "click_dish": {"type": "click_dish"},
    "impression": {"type": "impression"},
    "click": {"type": "click"},
}


# ============================================================================================
# Public records
# ============================================================================================


@dataclass(frozen=True, slots=True)
class Intent:
    user_id: str
    kind: str
    start_ts: int
    end_ts: int | None

    def active_at(self, ts: int) -> bool:
        return self.start_ts <= ts and (self.end_ts is None or ts < self.end_ts)


@dataclass(frozen=True, slots=True)
class Event:
    """One funnel step. ``context``/``action`` use the spec §6.1 keys; treat them as read-only."""

    event_id: str
    user_id: str
    session_id: str
    session_index: int
    action_index: int
    session_size: int
    ts: int
    service: str
    event_type: str
    entity_id: str
    context: dict
    action: dict


@dataclass(frozen=True, slots=True)
class FoodSession:
    """A Food session that ended in an ``order`` (hidden state at session start included)."""

    session_id: str
    user_id: str
    start_ts: int
    order_ts: int
    merchant_id: str
    location_zone: int
    spend_level: int
    active_intents: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AdImpression:
    """One served ad. ``recent_services`` holds ``"<service>:<detail>"`` tokens of the user's
    completed rides (``ride:<dropoff zone type>``) and orders (``food:<cuisine>``) in the
    6 hours before ``ts``, most recent first (at most 3, deduplicated)."""

    event_id: str
    user_id: str
    ts: int
    ad_id: str
    location_zone: int
    active_intents: tuple[str, ...]
    recent_services: tuple[str, ...]
    true_probability: float
    clicked: bool


@dataclass(frozen=True)
class Simulation:
    world: World
    events: tuple[Event, ...]
    intents: tuple[Intent, ...]
    food_sessions: tuple[FoodSession, ...]
    ad_impressions: tuple[AdImpression, ...]


# ============================================================================================
# Precomputed world arrays (cached per World object)
# ============================================================================================


def _sigmoid(x: float) -> float:
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


class _Index:
    """Arrays derived from a World: true-probability terms, lookups and catalog payloads."""

    def __init__(self, world: World) -> None:
        self.world_ref = weakref.ref(world)
        noise = world.config.noise
        zones = world.zones
        self.zone_dist = [[haversine_km(a, b) for b in zones] for a in zones]
        self.zone_type = [z.zone_type for z in zones]
        self.zone_geohash = [z.geohash for z in zones]
        self.zones_by_type: dict[str, list[int]] = {}
        for z in zones:
            self.zones_by_type.setdefault(z.zone_type, []).append(z.zone_id)

        # Food: logit = taste[cuisine] + price[spend] + distance[zone] + popularity
        merchants = world.merchants
        self.n_merchants = len(merchants)
        self.merchant_pos = {m.merchant_id: i for i, m in enumerate(merchants)}
        self.m_zone = [m.zone_id for m in merchants]
        self.m_cuisine = np.array([_CUISINE_IDX[m.cuisine] for m in merchants], dtype=np.int64)
        m_price = np.array([m.price_level for m in merchants], dtype=float)
        dist = np.asarray(self.zone_dist, dtype=float)
        self.dist_term = -noise.food_distance_weight * dist[:, np.asarray(self.m_zone, dtype=int)]
        spend_levels = np.arange(3.0)[:, None]
        self.price_term = -noise.food_price_weight * np.abs(m_price[None, :] - spend_levels)
        self.pop_term = noise.food_popularity_weight * np.log(
            np.array([m.base_popularity for m in merchants], dtype=float))
        self.merchant_ctx = [
            {"merchant_id": m.merchant_id, "merchant_name": m.name, "cuisine": m.cuisine,
             "price_level": m.price_level}
            for m in merchants
        ]
        self.dishes: list[list] = [[] for _ in merchants]
        for dish in world.dishes:
            self.dishes[self.merchant_pos[dish.merchant_id]].append(dish)

        users = world.users
        self.user_pos = {u.user_id: i for i, u in enumerate(users)}
        taste = np.array([u.cuisine_pref for u in users], dtype=float).reshape(len(users), -1)
        diet_mix = np.zeros(len(CUISINES))
        for c in _DIET_CUISINES:
            diet_mix[_CUISINE_IDX[c]] = 1.0 / len(_DIET_CUISINES)
        diet_taste = (1.0 - _DIET_TASTE_SHIFT) * taste + _DIET_TASTE_SHIFT * diet_mix
        w_c = noise.food_cuisine_weight
        self.taste_term = w_c * np.log(np.maximum(taste, 1e-12))
        self.taste_term_diet = w_c * np.log(np.maximum(diet_taste, 1e-12))
        self.taste_cum = [np.cumsum(t).tolist() for t in taste]
        self.diet_taste_cum = [np.cumsum(t).tolist() for t in diet_taste]

        # Ads
        ads = world.ads
        self.n_ads = len(ads)
        self.ad_pos = {a.ad_id: i for i, a in enumerate(ads)}
        self.ad_cat = [_AD_IDX[a.category] for a in ads]
        self.ad_target = [a.target_zone for a in ads]
        self.ads_by_cat: list[list[int]] = [[] for _ in AD_CATEGORIES]
        for i, cat in enumerate(self.ad_cat):
            self.ads_by_cat[cat].append(i)
        self.ad_ctx = [
            {"product": a.ad_id, "industry_main": a.category, "industry_sub": a.advertiser,
             "caption": a.title}
            for a in ads
        ]
        self.persona_cats = {
            p: [_AD_IDX[c] for c in boosts] for p, boosts in _PERSONA_AD_BOOST.items()
        }
        interest = np.array([u.ad_interest for u in users], dtype=float).reshape(len(users), -1)
        self.log_interest = np.log(np.maximum(len(AD_CATEGORIES) * interest, 1e-12)).tolist()
        self.w_int = noise.ad_interest_weight
        self.w_intent = noise.ad_intent_weight
        self.w_zone = noise.ad_zone_weight
        self.w_cross = noise.ad_cross_service_weight
        self.intent_boost = {k: _category_vector(v) for k, v in _INTENT_AD_BOOST.items()}
        self.cross_boost = {k: _category_vector(v) for k, v in _CROSS_AD_BOOST.items()}
        nodes, weights = np.polynomial.hermite_e.hermegauss(_GH_NODES)
        weights = weights / weights.sum()
        self.gh = list(zip((noise.ad_logit_noise_sd * nodes).tolist(), weights.tolist()))
        self.gh_nodes = noise.ad_logit_noise_sd * nodes
        self.gh_weights = weights

        # Drivers
        self.drivers = world.drivers
        self.drivers_by_zv: dict[tuple[int, str], list] = {}
        self.drivers_by_v: dict[str, list] = {}
        for drv in world.drivers:
            self.drivers_by_zv.setdefault((drv.zone_id, drv.vehicle_type), []).append(drv)
            self.drivers_by_v.setdefault(drv.vehicle_type, []).append(drv)

        self.ad_base = self._solve_ad_base(world)

    # ---- food ----------------------------------------------------------------------------
    def food_logit_vector(self, user: int, zone: int, spend: int, diet: bool) -> np.ndarray:
        taste = (self.taste_term_diet if diet else self.taste_term)[user]
        return taste[self.m_cuisine] + self.price_term[spend] + self.dist_term[zone] + self.pop_term

    # ---- ads -----------------------------------------------------------------------------
    def zone_match(self, ad: int, zone: int) -> float:
        target = self.ad_target[ad]
        if target is None:
            return 0.0
        return max(-0.5, 1.0 - self.zone_dist[target][zone] / _ZONE_MATCH_RADIUS_KM)

    def ad_logit(self, user: int, ad: int, zone: int, intents: Sequence[str],
                 recent: Sequence[str]) -> float:
        """Click logit without the base term (spec §4.5)."""
        cat = self.ad_cat[ad]
        z = self.w_int * self.log_interest[user][cat]
        boost = 0.0
        for kind in intents:
            vec = self.intent_boost.get(kind)
            if vec is not None:
                boost += vec[cat]
        z += self.w_intent * boost
        z += self.w_zone * self.zone_match(ad, zone)
        cross = 0.0
        for token in recent:
            vec = self.cross_boost.get(token)
            if vec is not None and vec[cat] > cross:
                cross = vec[cat]
        return z + self.w_cross * cross

    def prob(self, x: float) -> float:
        """E[sigmoid(x + eps)], eps ~ N(0, ad_logit_noise_sd) (Gauss-Hermite)."""
        total = 0.0
        for shift, weight in self.gh:
            total += weight * _sigmoid(x + shift)
        return total

    def _prob_vector(self, x: np.ndarray) -> np.ndarray:
        v = x[:, None] + self.gh_nodes[None, :]
        return (1.0 / (1.0 + np.exp(-np.clip(v, -500.0, 500.0)))) @ self.gh_weights

    def served_distribution(self, persona: str) -> np.ndarray:
        """Serving policy: 50% coarse persona targeting, 50% uniform over all ads."""
        q = np.full(self.n_ads, (1.0 - _AD_TARGETED_SHARE) / self.n_ads)
        cats = self.persona_cats[persona]
        for cat in cats:
            pool = self.ads_by_cat[cat]
            share = _AD_TARGETED_SHARE / len(cats)
            if pool:
                q[pool] += share / len(pool)
            else:
                q += share / self.n_ads
        return q

    def _solve_ad_base(self, world: World) -> float:
        """Base logit so that mean click probability over a fixed reference sample = target.

        Reference sample (deterministic, no RNG): evenly spaced users, each with the serving
        policy's ad distribution, located at home (and at work 30% of the time when employed),
        no active intents and no recent cross-service events.
        """
        users = world.users
        step = max(1, len(users) // _BASE_SAMPLE_USERS)
        ad_cat = np.asarray(self.ad_cat, dtype=int)
        zone_terms: dict[int, np.ndarray] = {}
        zs: list[np.ndarray] = []
        ws: list[np.ndarray] = []
        for ui in range(0, len(users), step):
            user = users[ui]
            q = self.served_distribution(user.persona)
            z_int = self.w_int * np.asarray(self.log_interest[ui])[ad_cat]
            locs = ((user.home_zone, 1.0),) if user.work_zone is None else (
                (user.home_zone, 0.7), (user.work_zone, 0.3))
            for zone, w_loc in locs:
                if zone not in zone_terms:
                    zone_terms[zone] = self.w_zone * np.array(
                        [self.zone_match(a, zone) for a in range(self.n_ads)])
                zs.append(z_int + zone_terms[zone])
                ws.append(w_loc * q)
        z = np.round(np.concatenate(zs), 4)
        w = np.concatenate(ws)
        z, inverse = np.unique(z, return_inverse=True)
        w = np.bincount(inverse.ravel(), weights=w)
        w = w / w.sum()
        target = world.config.target_ctr
        lo, hi = -30.0, 10.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if float(self._prob_vector(mid + z) @ w) < target:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)


def _category_vector(boosts: dict[str, float]) -> list[float]:
    vec = [0.0] * len(AD_CATEGORIES)
    for cat, value in boosts.items():
        vec[_AD_IDX[cat]] = value
    return vec


_INDEX_CACHE: dict[int, _Index] = {}


def _index(world: World) -> _Index:
    cached = _INDEX_CACHE.get(id(world))
    if cached is not None and cached.world_ref() is world:
        return cached
    for key in [k for k, v in _INDEX_CACHE.items() if v.world_ref() is None]:
        del _INDEX_CACHE[key]
    idx = _Index(world)
    _INDEX_CACHE[id(world)] = idx
    return idx


# ============================================================================================
# Public true-probability functions
# ============================================================================================


def food_logits(world: World, session: FoodSession, merchant_ids: Sequence[str]) -> np.ndarray:
    """True (noise-free) choice logits for ``merchant_ids`` in ``session``'s hidden state.

    The probability of choosing each merchant within the set is the softmax of the result.
    """
    idx = _index(world)
    full = idx.food_logit_vector(
        idx.user_pos[session.user_id],
        session.location_zone,
        session.spend_level,
        "diet" in session.active_intents,
    )
    return full[[idx.merchant_pos[m] for m in merchant_ids]]


def ad_base_logit(world: World) -> float:
    """The solved base term of the click logit (deterministic in ``world``)."""
    return _index(world).ad_base


def click_probability(world: World, state) -> float:
    """True click probability of an impression-like ``state``.

    ``state`` needs ``user_id, ad_id, location_zone, active_intents, recent_services``.
    """
    idx = _index(world)
    z = idx.ad_logit(
        idx.user_pos[state.user_id],
        idx.ad_pos[state.ad_id],
        state.location_zone,
        state.active_intents,
        state.recent_services,
    )
    return idx.prob(idx.ad_base + z)


# ============================================================================================
# Simulation
# ============================================================================================


class _Draws:
    """Buffered scalar draws from one Generator (cheap per call, fixed order)."""

    __slots__ = ("_i", "_j", "_rng", "_u", "_z")
    _BLOCK = 4096

    def __init__(self, rng: np.random.Generator) -> None:
        self._rng = rng
        self._u: list[float] = []
        self._i = 0
        self._z: list[float] = []
        self._j = 0

    def u(self) -> float:
        if self._i >= len(self._u):
            self._u = self._rng.random(self._BLOCK).tolist()
            self._i = 0
        value = self._u[self._i]
        self._i += 1
        return value

    def normal(self) -> float:
        if self._j >= len(self._z):
            self._z = self._rng.standard_normal(self._BLOCK).tolist()
            self._j = 0
        value = self._z[self._j]
        self._j += 1
        return value

    def uniform(self, lo: float, hi: float) -> float:
        return lo + (hi - lo) * self.u()

    def int(self, n: int) -> int:
        return min(int(self.u() * n), n - 1)

    def between(self, lo: int, hi: int) -> int:
        """Integer uniform on [lo, hi]."""
        return lo + self.int(hi - lo + 1)

    def categorical(self, cum: Sequence[float]) -> int:
        return min(bisect_right(cum, self.u() * cum[-1]), len(cum) - 1)

    def poisson(self, lam: float) -> int:
        limit = math.exp(-lam)
        k, p = 0, self.u()
        while p > limit:
            k += 1
            p *= self.u()
        return k


class _IntentRec:
    """Mutable intent record while a user is simulated (trip times are set during layout)."""

    __slots__ = ("end_ts", "kind", "planned_end_ts", "start_ts")

    def __init__(self, kind: str, start_ts: int | None, end_ts: int | None,
                 planned_end_ts: int | None = None) -> None:
        self.kind = kind
        self.start_ts = start_ts
        self.end_ts = end_ts
        self.planned_end_ts = planned_end_ts

    def active(self, ts: int) -> bool:
        if self.start_ts is None or ts < self.start_ts:
            return False
        end = self.end_ts if self.end_ts is not None else self.planned_end_ts
        return end is None or ts < end


@dataclass(slots=True)
class _Trip:
    start_day: int
    return_day: int
    start_hour: float
    return_hour: float
    hub: int
    rec: _IntentRec


# A raw row: (ts, session_id, action_index, session_size, session_index, service, event_type,
#             entity_id, context, action, impression_payload_or_None)


class _UserSimulator:
    def __init__(self, world: World, idx: _Index, rng: np.random.Generator) -> None:
        self.world = world
        self.cfg = world.config
        self.idx = idx
        self.rng = rng
        self.d = _Draws(rng)
        self.noise_sd = self.cfg.noise.food_logit_noise_sd
        self.scale = self.cfg.target_events_per_user / _UNSCALED_EVENTS_PER_USER
        self.n_days = n_days(self.cfg)
        self.day0 = day_start_ts(1, self.cfg)
        self.paydays = self._payday_windows()

    # ---- setup ---------------------------------------------------------------------------
    def _payday_windows(self) -> list[int]:
        """Local day numbers of the 1st of each month touching the simulated window."""
        start = date.fromisoformat(self.cfg.start_date)
        last = start.toordinal() + self.n_days - 1
        out = []
        year, month = start.year, start.month
        while date(year, month, 1).toordinal() <= last:
            out.append(date(year, month, 1).toordinal() - start.toordinal() + 1)
            year, month = (year + 1, 1) if month == 12 else (year, month + 1)
        return out

    def _ts(self, day: int, hour: float) -> int:
        return self.day0 + (day - 1) * SECONDS_PER_DAY + round(hour * 3600.0)

    def _sample_intents(self) -> None:
        d, user = self.d, self.user
        self.intents: list[_IntentRec] = []
        self.trips: dict[int, tuple[_Trip, str]] = {}
        self.trip_list: list[_Trip] = []
        self.work_changes: dict[int, int] = {}
        diet_until = 0
        trip_free_from = 1
        work = user.work_zone
        p_trip = _P_TRIP_WEEK * (_TRAVELER_TRIP_FACTOR if user.persona == "traveler" else 1.0)
        for week in range(self.cfg.weeks):
            first = 7 * week + 1
            if d.u() < _P_DIET_WEEK:
                start = first + d.int(7)
                days = d.between(*_DIET_DAYS)
                if start >= diet_until:
                    diet_until = start + days
                    self.intents.append(_IntentRec(
                        "diet", self._ts(start, 0.0), self._ts(start + days, 0.0)))
            if d.u() < p_trip:
                start = first + d.int(7)
                days = d.between(*_TRIP_DAYS)
                h_out = d.uniform(6.0, 16.0)
                h_back = d.uniform(9.0, 21.0)
                hubs = self.idx.zones_by_type["transport_hub"]
                hub = hubs[d.int(len(hubs))]
                if start >= trip_free_from:
                    back = start + days
                    rec = _IntentRec("trip", None, None, self._ts(back, h_back))
                    trip = _Trip(start, back, h_out, h_back, hub, rec)
                    self.intents.append(rec)
                    self.trip_list.append(trip)
                    self.trips[start] = (trip, "start")
                    for day in range(start + 1, back):
                        self.trips[day] = (trip, "away")
                    self.trips[back] = (trip, "return")
                    trip_free_from = back + 2
            if work is not None and d.u() < _P_JOB_CHANGE_WEEK:
                day = first + d.int(7)
                same_type = self.idx.zones_by_type[self.idx.zone_type[work]]
                choices = [z for z in same_type if z != work]
                if choices and day not in self.work_changes:
                    work = choices[d.int(len(choices))]
                    self.work_changes[day] = work
                    self.intents.append(_IntentRec("job_change", self._ts(day, 0.0), None))
        for month_first in self.paydays:
            start = month_first + d.between(*_PAYDAY_FIRST_DOM) - 1
            days = d.between(*_PAYDAY_DAYS)
            if start <= self.n_days and start + days > 1:
                self.intents.append(_IntentRec(
                    "payday", self._ts(start, 0.0), self._ts(start + days, 0.0)))

    # ---- state queries -------------------------------------------------------------------
    def _active(self, ts: int) -> tuple[str, ...]:
        found = []
        for kind in INTENT_KINDS:
            for rec in self.intents:
                if rec.kind == kind and rec.active(ts):
                    found.append(kind)
                    break
        return tuple(found)

    def _away(self, ts: int) -> bool:
        for trip in self.trip_list:
            if trip.rec.active(ts):
                return True
        return False

    def _recent(self, ts: int) -> tuple[str, ...]:
        out: list[str] = []
        floor = ts - _RECENT_WINDOW_S
        for when, token in reversed(self.recent):
            if when < floor:
                break
            if when < ts and token not in out:
                out.append(token)
                if len(out) == _MAX_RECENT:
                    break
        return tuple(out)

    def _spend(self, intents: tuple[str, ...]) -> int:
        return min(2, self.user.spend_level + (1 if "payday" in intents else 0))

    # ---- one user ------------------------------------------------------------------------
    def run(self, user_index: int):
        self.user = user = self.world.users[user_index]
        self.u_pos = user_index
        self.uid = user.user_id
        self.rate = user.activity * self.scale
        self.rows: list[tuple] = []
        self.food: list[FoodSession] = []
        self.recent: list[tuple[int, str]] = []
        self.session_index = 0
        self.work = user.work_zone
        self._sample_intents()
        for day in range(1, self.n_days + 1):
            if day in self.work_changes:
                self.work = self.work_changes[day]
            self._run_day(day)
        intents = []
        for rec in self.intents:
            if rec.start_ts is None:
                continue
            end = rec.end_ts if rec.end_ts is not None else rec.planned_end_ts
            intents.append(Intent(self.uid, rec.kind, rec.start_ts, end))
        intents.sort(key=lambda i: (i.start_ts, INTENT_KINDS.index(i.kind)))
        return self.rows, intents, self.food

    def _p(self, p: float) -> float:
        return min(_MAX_P, p * self.rate)

    def _meal_p(self, slot: int, ts: int) -> float:
        """Probability of a Food session in ``slot``; lunch is likelier when at work."""
        p = _MEAL_P[self.user.persona][slot]
        if slot == 0 and self.work is not None and self.loc == self.work:
            p *= _AT_WORK_LUNCH_FACTOR
        if any(rec.kind == "payday" and rec.active(ts) for rec in self.intents):
            p *= _PAYDAY_FOOD_FACTOR
        return self._p(p)

    def _plan_day(self, day: int) -> list[tuple]:
        d, user = self.d, self.user
        persona = user.persona
        weekday = (day - 1) % 7
        trip_role = self.trips.get(day)
        plans: list[tuple] = []  # (desired_ts, seq, kind, args)

        def add(hour: float, kind: str, *args) -> None:
            plans.append((self._ts(day, hour), len(plans), kind, args))

        home, work = user.home_zone, self.work
        ride_out_h = ride_back_h = None
        if trip_role is not None:
            trip, role = trip_role
            if role == "start":
                ride_out_h = trip.start_hour
                add(trip.start_hour, "ride", "airport_out", None, trip.hub, trip)
            elif role == "return":
                ride_back_h = trip.return_hour
                add(trip.return_hour, "ride", "airport_back", trip.hub, home, trip)
        else:
            if work is not None and persona == "night_shift":
                if (weekday - 1) % 7 in user.workdays and d.u() < self._p(_COMMUTE_P[persona]):
                    add(user.commute_back_hour + 0.25 * d.normal(), "ride", "shift_back", work,
                        home, None)
                if weekday in user.workdays and d.u() < self._p(_COMMUTE_P[persona]):
                    add(user.commute_out_hour + 0.25 * d.normal(), "ride", "commute_out", None,
                        work, None)
            elif work is not None and weekday in user.workdays:
                if d.u() < self._p(_COMMUTE_P[persona]):
                    add(user.commute_out_hour + 0.25 * d.normal(), "ride", "commute_out", None,
                        work, None)
                add(user.commute_back_hour + 0.3 * d.normal(), "ride", "commute_back", work,
                    home, None)
            workday = work is not None and weekday in user.workdays
            if not workday and d.u() < self._p(_ERRAND_P[persona]):
                h = d.uniform(8.5, 15.0)
                types = ("mall", "residential")
                zones = self.idx.zones_by_type[types[d.int(2)]]
                dest = zones[d.int(len(zones))]
                add(h, "ride", "errand_out", None, dest, None)
                add(h + d.uniform(1.0, 2.5), "ride", "errand_back", dest, home, None)
            night_work = persona == "night_shift" and weekday in user.workdays
            outing_p = _OUTING_P[persona] * (1.0 if weekday >= 4 else _OUTING_WEEKDAY_FACTOR)
            if not night_work and d.u() < self._p(outing_p):
                h = d.uniform(18.5, 21.0)
                types = ("nightlife", "nightlife", "mall")
                zones = self.idx.zones_by_type[types[d.int(3)]]
                dest = zones[d.int(len(zones))]
                add(h, "ride", "outing_out", None, dest, None)
                add(min(_LAST_RIDE_HOUR, h + d.uniform(1.5, 3.0)), "ride", "outing_back", dest,
                    home, None)

        for slot, (lo, hi) in enumerate(_MEAL_SLOTS):
            if trip_role is not None:
                role = trip_role[1]
                if role == "away" or (role == "start" and lo > ride_out_h - 1.5) or (
                        role == "return" and hi < ride_back_h + 0.5):
                    continue
            add(d.uniform(lo, hi), "food", slot)  # ordered or not is decided at layout time

        for _ in range(d.poisson(_APP_OPENS_PER_DAY * self.rate)):
            add(d.uniform(*_APP_HOURS), "app")
        plans.sort()
        return plans

    def _run_day(self, day: int) -> None:
        d = self.d
        day_end = self._ts(day + 1, 0.0)
        self.loc = self.user.home_zone
        completed: set[str] = set()
        cursor = self._ts(day, 0.0)
        for desired, _seq, kind, args in self._plan_day(day):
            start = max(desired, cursor)
            if kind == "app":
                end = self._ad(start, day_end)
            elif kind == "food":
                if self._away(start) or d.u() >= self._meal_p(args[0], start):
                    continue
                end = self._food(start, day_end)
            else:
                purpose, pickup, dropoff, trip = args
                forced = trip is not None
                if not forced:
                    if self._away(start):
                        continue
                    if purpose == "commute_back":
                        if "commute_out" in completed:
                            p = _RETURN_P_AFTER_RIDE
                        else:
                            p = self._p(_RETURN_P_OTHERWISE)
                        if d.u() >= p:
                            continue
                    elif purpose == "errand_back":
                        if "errand_out" not in completed or d.u() >= _ERRAND_RETURN_P:
                            continue
                    elif purpose == "outing_back":
                        if "outing_out" not in completed or d.u() >= _OUTING_RETURN_P:
                            continue
                    if start > self._ts(day, _LAST_RIDE_HOUR):
                        continue
                pickup = self.loc if pickup is None else pickup
                if pickup == dropoff:
                    continue
                result = self._ride(start, day_end, pickup, dropoff, forced)
                if result is None:
                    continue
                end, done = result
                if done:
                    completed.add(purpose)
                    if purpose == "airport_out":
                        trip.rec.start_ts = start
                    elif purpose == "airport_back":
                        trip.rec.end_ts = end
            if end is None:
                continue
            cursor = end + _SESSION_GAP_S
            if kind != "app" and d.u() < _AD_AFTER_SESSION_P:
                ad_end = self._ad(end + 5 + d.int(115), day_end)
                if ad_end is not None:
                    cursor = ad_end + _SESSION_GAP_S

    # ---- sessions ------------------------------------------------------------------------
    def _new_session(self) -> tuple[str, int]:
        k = self.session_index
        self.session_index += 1
        return f"sim:session:{self.u_pos}:{k}", k

    def _ride(self, start: int, day_end: int, pickup: int, dropoff: int, forced: bool):
        d, idx = self.d, self.idx
        dist_km = round(idx.zone_dist[pickup][dropoff] * 1.3 + d.uniform(0.4, 1.5), 2)
        intents = self._active(start)
        vehicle = VEHICLE_TYPES[d.categorical(_VEHICLE_CUM[self._spend(intents)])]
        pool = idx.drivers_by_zv.get((pickup, vehicle))
        if pool and d.u() < 0.7:
            driver = pool[d.int(len(pool))]
        else:
            pool = idx.drivers_by_v.get(vehicle) or idx.drivers
            driver = pool[d.int(len(pool))]
        vehicle = driver.vehicle_type
        base, per_km, speed = _FARE[vehicle]
        fare = float(round((base + per_km * dist_km) * d.uniform(1.0, 1.2), -3))
        hub = idx.zone_type[pickup] == "transport_hub" or idx.zone_type[dropoff] == "transport_hub"
        wait = round(d.uniform(2.0, 9.0) + (3.0 if hub else 0.0), 1)
        n = 1
        if forced or d.u() < _P_QUOTE:
            n = 2
            if forced or d.u() < _P_BOOK:
                n = 3
                if forced or d.u() < _P_COMPLETE:
                    n = 4
        t1 = start + 5 + d.int(30)
        t2 = t1 + 5 + d.int(45)
        t3 = t2 + int(wait * 60 + dist_km / speed * 3600) + 60 + d.int(300)
        times = (start, t1, t2, t3)[:n]
        if times[-1] >= day_end:
            return None
        session_id, s_index = self._new_session()
        base_ctx = {
            "pickup_geohash": idx.zone_geohash[pickup],
            "dropoff_geohash": idx.zone_geohash[dropoff],
            "pickup_zone_type": idx.zone_type[pickup],
            "dropoff_zone_type": idx.zone_type[dropoff],
            "vehicle_type": vehicle,
        }
        booked_ctx = {**base_ctx, "driver_id": driver.driver_id}
        dropoff_gh = idx.zone_geohash[dropoff]
        steps = (
            ("search_destination", dropoff_gh, base_ctx,
             {"type": "search_destination", "distance_km": dist_km}),
            ("get_quote", dropoff_gh, base_ctx,
             {"type": "get_quote", "fare": fare, "distance_km": dist_km}),
            ("book_ride", driver.driver_id, booked_ctx,
             {"type": "book_ride", "fare": fare, "distance_km": dist_km, "wait_minutes": wait}),
            ("complete_ride", driver.driver_id, booked_ctx,
             {"type": "complete_ride", "fare": fare, "distance_km": dist_km,
              "wait_minutes": wait}),
        )
        for k in range(n):
            event_type, entity, ctx, action = steps[k]
            self.rows.append((times[k], session_id, k, n, s_index, "ride", event_type, entity,
                              ctx, action, None))
        done = n == 4
        if done:
            self.loc = dropoff
            self.recent.append((t3, f"ride:{idx.zone_type[dropoff]}"))
        return times[-1], done

    def _food(self, start: int, day_end: int):
        d, idx = self.d, self.idx
        loc = self.loc
        intents = self._active(start)
        diet = "diet" in intents
        spend = self._spend(intents)
        logits = idx.food_logit_vector(self.u_pos, loc, spend, diet)
        if self.noise_sd > 0.0:
            logits = logits + self.noise_sd * self.rng.standard_normal(idx.n_merchants)
        cum = np.cumsum(np.exp(logits - logits.max()))
        total = float(cum[-1])
        last = idx.n_merchants - 1

        def draw() -> int:
            return min(int(cum.searchsorted(d.u() * total, side="right")), last)

        chosen = draw()
        n_views = 1 + d.categorical((0.5, 0.8, 1.0))
        viewed: list[int] = []
        for _ in range(n_views - 1):
            k = draw()
            if k != chosen and k not in viewed:
                viewed.append(k)
        viewed.append(chosen)
        stage = 3  # 0: stop after search, 1: after views, 2: after dish clicks, 3: order
        if d.u() < _FOOD_ABANDON_P:
            stage = d.int(3)
        menu = idx.dishes[chosen]
        clicks = [menu[d.int(len(menu))]]
        if d.u() < 0.35:
            extra = menu[d.int(len(menu))]
            if extra is not clicks[0]:
                clicks.append(extra)
        merchant = self.world.merchants[chosen]
        cuisine = merchant.cuisine
        if d.u() >= 0.75:
            cum_taste = (idx.diet_taste_cum if diet else idx.taste_cum)[self.u_pos]
            cuisine = CUISINES[d.categorical(cum_taste)]
        terms = _SEARCH_TERMS[cuisine]
        query = terms[d.int(len(terms))]
        if diet and d.u() < 0.6:
            query += _DIET_QUERY_SUFFIX
        delivery_km = round(idx.zone_dist[loc][idx.m_zone[chosen]] * 1.25 + d.uniform(0.4, 1.5), 2)
        qty = 1 + (1 if d.u() < 0.1 + 0.15 * spend else 0)

        steps: list[tuple] = [("search", query, {"search_query": query}, _ACTION["search"])]
        t = start
        times = [t]
        if stage >= 1:
            for m in viewed:
                t += 8 + d.int(53)
                times.append(t)
                steps.append(("view_merchant", self.world.merchants[m].merchant_id,
                              idx.merchant_ctx[m], _ACTION["view_merchant"]))
        if stage >= 2:
            for dish in clicks:
                t += 5 + d.int(36)
                times.append(t)
                ctx = {**idx.merchant_ctx[chosen], "dish_id": dish.dish_id, "dish_name": dish.name}
                steps.append(("click_dish", dish.dish_id, ctx, _ACTION["click_dish"]))
        if stage == 3:
            dish = clicks[-1]
            t += 15 + d.int(76)
            times.append(t)
            amount = float(round(dish.price * qty + 8000.0 + 2500.0 * delivery_km, -3))
            ctx = {**idx.merchant_ctx[chosen], "dish_id": dish.dish_id, "dish_name": dish.name}
            steps.append(("order", merchant.merchant_id, ctx,
                          {"type": "order", "order_amount": amount,
                           "delivery_distance_km": delivery_km}))
        if times[-1] >= day_end:
            return None
        session_id, s_index = self._new_session()
        n = len(steps)
        for k, (event_type, entity, ctx, action) in enumerate(steps):
            self.rows.append((times[k], session_id, k, n, s_index, "food", event_type, entity,
                              ctx, action, None))
        if stage == 3:
            self.food.append(FoodSession(session_id, self.uid, start, times[-1],
                                         merchant.merchant_id, loc, spend, intents))
            self.recent.append((times[-1], f"food:{merchant.cuisine}"))
        return times[-1]

    def _ad(self, start: int, day_end: int):
        d, idx = self.d, self.idx
        low, high = AD_CLICK_DELAY_SECONDS
        click_ts = start + low + d.int(high - low + 1)  # = start + 2 + d.int(25)
        if click_ts >= day_end:
            return None
        if d.u() < _AD_TARGETED_SHARE:
            cats = idx.persona_cats[self.user.persona]
            pool = idx.ads_by_cat[cats[d.int(len(cats))]]
            ad = pool[d.int(len(pool))] if pool else d.int(idx.n_ads)
        else:
            ad = d.int(idx.n_ads)
        intents = self._active(start)
        recent = self._recent(start)
        prob = idx.prob(idx.ad_base + idx.ad_logit(self.u_pos, ad, self.loc, intents, recent))
        clicked = d.u() < prob
        session_id, s_index = self._new_session()
        n = 2 if clicked else 1
        ad_id, ctx = self.world.ads[ad].ad_id, idx.ad_ctx[ad]
        payload = (ad_id, self.loc, intents, recent, prob, clicked)
        self.rows.append((start, session_id, 0, n, s_index, "ads", "ad_impression", ad_id, ctx,
                          _ACTION["impression"], payload))
        if clicked:
            self.rows.append((click_ts, session_id, 1, n, s_index, "ads", "ad_click", ad_id, ctx,
                              _ACTION["click"], None))
            return click_ts
        return start


def _row_key(row: tuple) -> tuple:
    return row[0], row[1], row[2]


def simulate(world: World, rng: np.random.Generator) -> Simulation:
    """Simulate all users over ``world.config.weeks`` weeks (uses ``world.config``)."""
    idx = _index(world)
    runner = _UserSimulator(world, idx, rng)
    events: list[Event] = []
    intents: list[Intent] = []
    food_sessions: list[FoodSession] = []
    impressions: list[AdImpression] = []
    counter = 0
    for user_index, user in enumerate(world.users):
        rows, user_intents, user_food = runner.run(user_index)
        rows.sort(key=_row_key)
        uid = user.user_id
        for (ts, session_id, action_index, size, s_index, service, event_type, entity, ctx,
             action, payload) in rows:
            event_id = f"sim:event:{counter}"
            counter += 1
            events.append(Event(event_id, uid, session_id, s_index, action_index, size, ts,
                                service, event_type, entity, ctx, action))
            if payload is not None:
                ad_id, zone, active, recent, prob, clicked = payload
                impressions.append(AdImpression(event_id, uid, ts, ad_id, zone, active, recent,
                                                prob, clicked))
        intents.extend(user_intents)
        food_sessions.extend(user_food)
    return Simulation(world, tuple(events), tuple(intents), tuple(food_sessions),
                      tuple(impressions))
