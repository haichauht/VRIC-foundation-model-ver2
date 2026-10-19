"""Local-time helpers: fixed UTC offset, day 1 = ``config.start_date`` 00:00 local time."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from functools import lru_cache

SECONDS_PER_DAY = 86400


@lru_cache(maxsize=16)
def _start_ts(start_date: str, utc_offset_hours: int) -> int:
    d = date.fromisoformat(start_date)
    tz = timezone(timedelta(hours=utc_offset_hours))
    return int(datetime(d.year, d.month, d.day, tzinfo=tz).timestamp())


def start_ts(config) -> int:
    """Epoch seconds of day 1, 00:00 local time."""
    return _start_ts(config.start_date, config.utc_offset_hours)


def n_days(config) -> int:
    return config.weeks * 7


def day_start_ts(day: int, config) -> int:
    """Epoch seconds of local midnight that starts ``day`` (1-based)."""
    return start_ts(config) + (day - 1) * SECONDS_PER_DAY


def local_day(ts: int, config) -> int:
    """1-based local day number of ``ts`` (day 1 = start_date)."""
    return (int(ts) - start_ts(config)) // SECONDS_PER_DAY + 1


def week_of(ts: int, config) -> int:
    """1-based local week number (weeks start on day 1, a Monday)."""
    return (local_day(ts, config) - 1) // 7 + 1


def split_of(ts: int, config) -> str:
    """``train`` for weeks 1..weeks-2, ``validation`` for the next week, ``test`` for the last."""
    week = week_of(ts, config)
    if week < 1 or week > config.weeks:
        raise ValueError(f"timestamp {ts} is outside the simulated {config.weeks} weeks")
    if week <= config.weeks - 2:
        return "train"
    if week == config.weeks - 1:
        return "validation"
    return "test"


def local_hour(ts: int, config) -> int:
    return (int(ts) - start_ts(config)) % SECONDS_PER_DAY // 3600


def local_weekday(ts: int, config) -> int:
    """Monday = 0 (day 1 is a Monday)."""
    return (local_day(ts, config) - 1) % 7


def local_iso(ts: int, config) -> str:
    """ISO-8601 local time with explicit offset, e.g. ``2026-01-05T00:30:00+07:00``."""
    tz = timezone(timedelta(hours=config.utc_offset_hours))
    return datetime.fromtimestamp(int(ts), tz).isoformat()


def naive_utc(ts: int) -> datetime:
    """UTC wall-clock time without tzinfo (the pretrain parquet convention)."""
    return datetime.fromtimestamp(int(ts), timezone.utc).replace(tzinfo=None)
