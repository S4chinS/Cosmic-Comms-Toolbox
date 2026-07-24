"""Small shared helpers for globe math transforms."""

from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Tuple


def rotate_vector_z(
    vector: tuple[float, float, float],
    angle_rad: float,
) -> tuple[float, float, float]:
    """Rotate the provided vector around the +Z axis by the given angle."""
    cos_ang = math.cos(angle_rad)
    sin_ang = math.sin(angle_rad)
    x, y, z = vector
    return (cos_ang * x - sin_ang * y, sin_ang * x + cos_ang * y, z)


def julian_date_utc(dt: datetime) -> float:
    """Return Julian Date for a UTC datetime (proleptic Gregorian calendar)."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)

    year = dt.year
    month = dt.month
    day = dt.day

    a = (14 - month) // 12
    y = year + 4800 - a
    m = month + 12 * a - 3

    jdn = (
        day
        + (153 * m + 2) // 5
        + 365 * y
        + y // 4
        - y // 100
        + y // 400
        - 32045
    )

    frac_day = (
        (dt.hour - 12) / 24.0
        + dt.minute / 1440.0
        + (dt.second + dt.microsecond / 1_000_000.0) / 86400.0
    )
    return float(jdn) + float(frac_day)


def gmst_rad_utc(dt: datetime) -> float:
    """Greenwich Mean Sidereal Time angle (radians) from UTC time (approx)."""
    jd = julian_date_utc(dt)
    d = jd - 2451545.0
    t = d / 36525.0
    gmst_deg = (
        280.46061837
        + 360.98564736629 * d
        + 0.000387933 * t * t
        - (t * t * t) / 38710000.0
    )
    gmst_deg = float(gmst_deg % 360.0)
    return math.radians(gmst_deg)


__all__: Tuple[str, ...] = ("rotate_vector_z", "julian_date_utc", "gmst_rad_utc")

