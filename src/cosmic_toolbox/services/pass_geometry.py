"""Pass-window trimming and station-geometry helpers.

Canonical home of the link-close zero-crossing interpolation previously
private to the GUI main window, plus the geocentric elevation model that
several batch scripts had re-implemented locally.  Everything here is pure
NumPy — no Orekit, no Qt.
"""

from __future__ import annotations

import numpy as np

from cosmic_toolbox.services.antenna_pattern import station_ecef_m

__all__ = [
    "station_ecef_m",
    "interpolate_zero_crossing_time",
    "resolve_positive_rate_bounds",
    "elevation_deg_geocentric",
]


def interpolate_zero_crossing_time(
    *, t0: float, y0: float, t1: float, y1: float
) -> float:
    """Linear-interpolate the time where y crosses zero between two samples."""

    if not all(np.isfinite(value) for value in (t0, y0, t1, y1)):
        raise ValueError("Zero-crossing interpolation requires finite inputs.")
    if t1 <= t0:
        raise ValueError("Zero-crossing interpolation requires increasing time.")
    delta = y1 - y0
    if delta == 0.0:
        return float(t1 if y1 > 0.0 else t0)
    fraction = -y0 / delta
    fraction = min(1.0, max(0.0, float(fraction)))
    return float(t0 + fraction * (t1 - t0))


def resolve_positive_rate_bounds(
    *,
    time_axis: np.ndarray,
    rates: np.ndarray,
    start_sec: float,
    end_sec: float,
) -> tuple[float, float] | None:
    """Trim [start_sec, end_sec] to the span where the rate series is positive.

    Used to shrink geometric pass windows to the interval where the link
    actually closes.  Returns ``None`` when the rate never goes positive in
    the window (or the window is empty after clamping to the time axis).
    """

    if end_sec <= start_sec:
        return None
    if time_axis.ndim != 1 or rates.shape != time_axis.shape or time_axis.size < 2:
        raise ValueError(
            "Positive-rate bounds require aligned 1-D time and rate series."
        )
    # Clamp to the cached timeline so trimming never raises when passes or
    # scenario bounds extend slightly past the sampled axis (e.g. after station edits).
    t_axis_lo = float(time_axis[0])
    t_axis_hi = float(time_axis[-1])
    start_sec = max(float(start_sec), t_axis_lo)
    end_sec = min(float(end_sec), t_axis_hi)
    if end_sec <= start_sec:
        return None
    interior_mask = (time_axis > start_sec) & (time_axis < end_sec)
    interval_times = time_axis[interior_mask]
    interval_rates = rates[interior_mask]
    start_rate = float(np.interp(start_sec, time_axis, rates))
    end_rate = float(np.interp(end_sec, time_axis, rates))
    times = np.concatenate(([start_sec], interval_times, [end_sec]))
    values = np.concatenate(([start_rate], interval_rates, [end_rate]))
    positive_mask = np.isfinite(values) & (values > 0.0)
    if not np.any(positive_mask):
        return None
    first_positive = int(np.argmax(positive_mask))
    last_positive = int(len(positive_mask) - 1 - np.argmax(positive_mask[::-1]))
    trimmed_start = float(times[first_positive])
    if first_positive > 0:
        trimmed_start = interpolate_zero_crossing_time(
            t0=float(times[first_positive - 1]),
            y0=float(values[first_positive - 1]),
            t1=float(times[first_positive]),
            y1=float(values[first_positive]),
        )
    trimmed_end = float(times[last_positive])
    if last_positive < len(times) - 1:
        trimmed_end = interpolate_zero_crossing_time(
            t0=float(times[last_positive]),
            y0=float(values[last_positive]),
            t1=float(times[last_positive + 1]),
            y1=float(values[last_positive + 1]),
        )
    trimmed_start = max(float(start_sec), trimmed_start)
    trimmed_end = min(float(end_sec), trimmed_end)
    if trimmed_end <= trimmed_start:
        return None
    return (trimmed_start, trimmed_end)


def elevation_deg_geocentric(
    sat_ecef_m: np.ndarray, station_ecef: np.ndarray
) -> np.ndarray:
    """Elevation (deg) of satellite positions using the geocentric up vector.

    Approximates geodetic "up" by the station position unit vector; the error
    versus a true topocentric frame is below ~0.2 deg — acceptable for pass
    statistics, not for antenna pointing.

    Args:
        sat_ecef_m: (N, 3) satellite ECEF positions in metres.
        station_ecef: (3,) station ECEF position in metres.
    """

    sat = np.asarray(sat_ecef_m, dtype=float)
    stn = np.asarray(station_ecef, dtype=float).reshape(3)
    if sat.ndim == 1:
        sat = sat.reshape(1, 3)
    if sat.ndim != 2 or sat.shape[1] != 3:
        raise ValueError("sat_ecef_m must have shape (N, 3).")
    rel = sat - stn[None, :]
    rel_norm = np.linalg.norm(rel, axis=1)
    up = stn / np.linalg.norm(stn)
    sin_el = (rel @ up) / np.where(rel_norm > 0.0, rel_norm, np.inf)
    return np.degrees(np.arcsin(np.clip(sin_el, -1.0, 1.0)))
