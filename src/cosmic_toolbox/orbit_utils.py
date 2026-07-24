"""Small closed-form orbit helpers shared by the GUI and analysis scripts.

Canonical home of the Sun-synchronous inclination solve and the basic
Keplerian relations that were previously copy-pasted per script.  All
functions are pure and Orekit-free.
"""

from __future__ import annotations

import math

# WGS84 / EGM96 constants used by every consumer of these helpers.
WGS84_EARTH_EQUATORIAL_RADIUS_M = 6_378_137.0
WGS84_EARTH_MU_M3_S2 = 3.986004418e14
WGS84_J2 = 1.08262668355e-3
TROPICAL_YEAR_DAYS = 365.2421897


def sso_raan_rate_rad_s() -> float:
    """Nodal-regression rate (rad/s) required for a Sun-synchronous orbit."""

    return 2.0 * math.pi / (TROPICAL_YEAR_DAYS * 86400.0)


def sso_inclination_deg(*, altitude_km: float, eccentricity: float = 0.0) -> float | None:
    """Sun-synchronous inclination (deg) from the J2 nodal-regression solve.

    Returns ``None`` when no physical solution exists (non-finite inputs,
    e outside [0, 1), altitude <= 0, or |cos i| > 1).
    """

    alt = float(altitude_km)
    e = float(eccentricity)
    if not (math.isfinite(alt) and math.isfinite(e)):
        return None
    if e < 0.0 or e >= 1.0:
        return None
    if alt <= 0.0:
        return None
    re_m = WGS84_EARTH_EQUATORIAL_RADIUS_M
    a_m = re_m + alt * 1000.0
    p_m = a_m * (1.0 - e * e)
    if p_m <= 0.0:
        return None
    n = math.sqrt(WGS84_EARTH_MU_M3_S2 / (a_m**3))  # rad/s
    if not math.isfinite(n) or n <= 0.0:
        return None
    cos_i = (
        -sso_raan_rate_rad_s()
        * (2.0 / (3.0 * WGS84_J2))
        * ((p_m / re_m) ** 2)
        * (1.0 / n)
    )
    if not math.isfinite(cos_i):
        return None
    if cos_i < -1.0 - 1e-12 or cos_i > 1.0 + 1e-12:
        return None
    cos_i = min(1.0, max(-1.0, cos_i))
    return math.degrees(math.acos(cos_i))


def orbital_period_s(*, altitude_km: float) -> float:
    """Keplerian orbital period (s) for a circular orbit at the given altitude."""

    a_m = WGS84_EARTH_EQUATORIAL_RADIUS_M + float(altitude_km) * 1000.0
    return 2.0 * math.pi * math.sqrt(a_m**3 / WGS84_EARTH_MU_M3_S2)


def orbital_velocity_km_s(*, altitude_km: float) -> float:
    """Circular orbital velocity (km/s) at the given altitude."""

    r_m = WGS84_EARTH_EQUATORIAL_RADIUS_M + float(altitude_km) * 1000.0
    return math.sqrt(WGS84_EARTH_MU_M3_S2 / r_m) / 1000.0
