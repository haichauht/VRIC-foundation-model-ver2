"""History selection and as-of profiles shared by every export (spec §4.6, §5, §6.1).

All helpers take a cutoff as epoch seconds and use **only** events with ``ts < cutoff_ts``.
A session (one ``session_id``) is either entirely in a history or not at all.
"""

from __future__ import annotations

from collections.abc import Sequence

from .behavior import Event
from .timeutil import SECONDS_PER_DAY, local_day

PROFILE_WINDOW_DAYS = 30
# Order of the keys in a profile dict (spec §4.6). Hidden traits (persona, taste, ad interest,
# activity, intents) are never part of a profile.
PROFILE_FIELDS: tuple[str, ...] = (
    "age_group",
    "saved_home_geohash",
    "saved_work_geohash",
    "avg_monthly_spend",
    "most_used_service",
    "ride_count_30d",
    "food_order_count_30d",
    "ad_click_count_30d",
    "active_days_30d",
)
# Services a user chooses to use; ads are served to the user, so they never count as "used".
_USED_SERVICES: tuple[str, ...] = ("ride", "food")


def event_order_key(event: Event) -> tuple[int, int, int]:
    """Chronological order of a user's events (sessions of one user never overlap)."""
    return event.ts, event.session_index, event.action_index


def completed_sessions_before(events_of_user: Sequence[Event], cutoff_ts: int) -> list[Event]:
    """Events (input order) of the sessions whose every action has ``ts < cutoff_ts``.

    A session counts only when all of its ``session_size`` actions are in ``events_of_user``,
    so a session that is still in progress at the cutoff is dropped even when its later
    actions were already filtered out by the caller.
    """
    present: dict[str, int] = {}
    late: set[str] = set()
    for event in events_of_user:
        present[event.session_id] = present.get(event.session_id, 0) + 1
        if event.ts >= cutoff_ts:
            late.add(event.session_id)
    return [
        e for e in events_of_user
        if e.session_id not in late and present[e.session_id] == e.session_size
    ]


def last_history(
    events: Sequence[Event], cutoff_ts: int, max_events: int
) -> tuple[list[Event], bool]:
    """Most recent completed sessions before the cutoff, at most ``max_events`` events.

    Whole sessions only: when the ``max_events`` most recent events would start inside a
    session, that whole (oldest) session is dropped and the returned flag
    (``group_truncated``) is True. The flag is False when nothing had to be cut or the cut
    fell exactly on a session boundary. Events are returned in chronological order. The Food
    package reports this flag as ``history_group_truncated``; the pretrain package does not
    (ruling R11: no session is ever split there, so it writes False).
    """
    if max_events < 1:
        raise ValueError(f"max_events must be >= 1, got {max_events}")
    kept = sorted(completed_sessions_before(events, cutoff_ts), key=event_order_key)
    if len(kept) <= max_events:
        return kept, False
    chosen: set[str] = set()
    total = 0
    sizes: dict[str, int] = {}
    for event in kept:
        sizes[event.session_id] = sizes.get(event.session_id, 0) + 1
    for event in reversed(kept):
        sid = event.session_id
        if sid in chosen:
            continue
        if total + sizes[sid] > max_events:
            break
        chosen.add(sid)
        total += sizes[sid]
    history = [e for e in kept if e.session_id in chosen]
    return history, len(history) < max_events


def profile_at(events_of_user: Sequence[Event], traits, cutoff_ts: int, config, *,
               zones) -> dict:
    """As-of profile (spec §4.6) from the events with ``ts < cutoff_ts`` only.

    ``traits`` is the user's ``UserTraits``; only its observable fields are used (age group,
    saved home and initial work place). ``zones`` is ``world.zones`` (zone id -> geohash).
    Windowed fields use ``[cutoff_ts - 30 days, cutoff_ts)``, clipped by the simulation start
    (no extrapolation). ``avg_monthly_spend`` = fares of completed rides + amounts of food
    orders in that window (VND). ``most_used_service`` = ``ride`` or ``food``, whichever has
    more sessions (any funnel depth) in the window, ties -> ``ride``, None when neither.
    """
    window_start = cutoff_ts - PROFILE_WINDOW_DAYS * SECONDS_PER_DAY
    rides = orders = clicks = 0
    spend = 0.0
    days: set[int] = set()
    sessions: dict[str, set[str]] = {s: set() for s in _USED_SERVICES}
    for event in events_of_user:
        if not window_start <= event.ts < cutoff_ts:
            continue
        days.add(local_day(event.ts, config))
        if event.service in sessions:
            sessions[event.service].add(event.session_id)
        if event.event_type == "complete_ride":
            rides += 1
            spend += float(event.action["fare"])
        elif event.event_type == "order":
            orders += 1
            spend += float(event.action["order_amount"])
        elif event.event_type == "ad_click":
            clicks += 1
    counts = {s: len(ids) for s, ids in sessions.items()}
    best = max(_USED_SERVICES, key=lambda s: counts[s])  # first wins ties
    work = traits.work_zone
    return {
        "age_group": traits.age_group,
        "saved_home_geohash": zones[traits.home_zone].geohash,
        "saved_work_geohash": None if work is None else zones[work].geohash,
        "avg_monthly_spend": spend,
        "most_used_service": best if counts[best] > 0 else None,
        "ride_count_30d": rides,
        "food_order_count_30d": orders,
        "ad_click_count_30d": clicks,
        "active_days_30d": len(days),
    }
