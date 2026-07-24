"""Orbit-averaged summary series (same smoothing as the GUI secular plots)."""

from __future__ import annotations

from typing import Any

import numpy as np


def _json_list(arr: np.ndarray) -> list[float | None]:
    a = np.asarray(arr, dtype=float)
    out: list[float | None] = []
    for v in a.tolist():
        fv = float(v)
        out.append(fv if np.isfinite(fv) else None)
    return out


def compute_orbit_averaged_orbit_summary(
    *, times_s: np.ndarray, inst: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compute orbit-averaged series using the same smoothing/windowing as the UI."""

    times_s = np.asarray(times_s, dtype=float)
    n = int(times_s.size)
    if n < 2:
        return {}, {"window_samples": 0, "one_orbit_s": None, "dt_s": None}

    def _rehydrate(key: str) -> np.ndarray:
        raw = inst.get(key, [])
        a = np.asarray(raw, dtype=float)
        if a.size != n:
            a = np.full((n,), np.nan, dtype=float)
        return a

    period = _rehydrate("orbital_period_s")
    valid_periods = period[np.isfinite(period) & (period > 0.0)]
    dt_s = float(np.median(np.diff(times_s))) if n > 1 else float("nan")
    if not np.isfinite(dt_s) or dt_s <= 0.0:
        dt_s = float(times_s[-1] - times_s[0]) / max(n - 1, 1)

    if valid_periods.size and dt_s > 0.0:
        one_orbit_s = float(np.median(valid_periods))
        window = int(round(one_orbit_s / dt_s))
    else:
        one_orbit_s = None
        window = 0

    window = max(5, min(window, max(5, n))) if window and window > 0 else 0

    if window > 1:
        kernel = np.ones(window, dtype=float) / float(window)

        def _smooth(values: np.ndarray) -> np.ndarray:
            if values.size == 0:
                return values
            smoothed = np.convolve(values, kernel, mode="same")
            half = window // 2
            if values.size > window and half > 0:
                smoothed[:half] = np.nan
                smoothed[-half:] = np.nan
            return smoothed

        def _smooth_angle_deg(values_deg: np.ndarray) -> np.ndarray:
            if values_deg.size == 0:
                return values_deg
            rad = np.deg2rad(np.mod(values_deg, 360.0))
            cos_v = _smooth(np.cos(rad))
            sin_v = _smooth(np.sin(rad))
            out = np.rad2deg(np.arctan2(sin_v, cos_v))
            return np.mod(out, 360.0)

        alt = _smooth(_rehydrate("orbital_altitude_km"))
        sma = _smooth(_rehydrate("semi_major_axis_km"))
        per = _smooth(_rehydrate("perigee_altitude_km"))
        apo = _smooth(_rehydrate("apogee_altitude_km"))
        ecc = _smooth(_rehydrate("eccentricity"))
        inc = _smooth(_rehydrate("inclination_deg"))
        argp = _smooth_angle_deg(_rehydrate("argument_of_perigee_deg"))
        per_s = _smooth(period)
        aoa = _smooth(_rehydrate("angle_of_attack_deg"))
    else:
        alt = _rehydrate("orbital_altitude_km")
        sma = _rehydrate("semi_major_axis_km")
        per = _rehydrate("perigee_altitude_km")
        apo = _rehydrate("apogee_altitude_km")
        ecc = _rehydrate("eccentricity")
        inc = _rehydrate("inclination_deg")
        argp = np.mod(_rehydrate("argument_of_perigee_deg"), 360.0)
        per_s = period
        aoa = _rehydrate("angle_of_attack_deg")

    if one_orbit_s is not None and np.isfinite(one_orbit_s) and one_orbit_s > 0.0:
        t0 = float(times_s[0])
        tN = float(times_s[-1])
        inner_mask = (times_s - t0 >= one_orbit_s) & (tN - times_s >= one_orbit_s)
        for arr in (alt, sma, per, apo, ecc, inc, argp, per_s, aoa):
            if arr.size == inner_mask.size:
                arr[~inner_mask] = np.nan

    averaged = {
        "orbital_altitude_km": _json_list(alt),
        "semi_major_axis_km": _json_list(sma),
        "perigee_altitude_km": _json_list(per),
        "apogee_altitude_km": _json_list(apo),
        "eccentricity": _json_list(ecc),
        "inclination_deg": _json_list(inc),
        "argument_of_perigee_deg": _json_list(argp),
        "orbital_period_s": _json_list(per_s),
        "angle_of_attack_deg": _json_list(aoa),
    }
    meta = {
        "dt_s": None if not np.isfinite(dt_s) else float(dt_s),
        "one_orbit_s": None if one_orbit_s is None else float(one_orbit_s),
        "window_samples": int(window),
    }
    return averaged, meta


def compute_series_stats(series: dict[str, Any]) -> dict[str, Any]:
    """Basic stats for each series entry (lists of numbers or nulls)."""

    out: dict[str, Any] = {}
    for key, values in series.items():
        if not isinstance(values, list):
            continue
        arr = np.array([np.nan if v is None else float(v) for v in values], dtype=float)
        finite = arr[np.isfinite(arr)]
        out[key] = {
            "count_total": int(arr.size),
            "count_valid": int(finite.size),
            "min": None if finite.size == 0 else float(np.min(finite)),
            "max": None if finite.size == 0 else float(np.max(finite)),
            "mean": None if finite.size == 0 else float(np.mean(finite)),
            "std": None if finite.size == 0 else float(np.std(finite)),
        }
    return out


__all__ = [
    "compute_orbit_averaged_orbit_summary",
    "compute_series_stats",
]
