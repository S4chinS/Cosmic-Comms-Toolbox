"""Pure-Python helpers for cached access recomputation."""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from functools import partial
from typing import List

import numpy as np

from cosmic_toolbox.models import (
    AnalysisConfig,
    AnalysisResult,
    AnalysisSummary,
    DerivedAccessResult,
    GroundStationConfig,
    PassStatistic,
    PropagatedEphemeris,
    StationSummary,
)
from cosmic_toolbox.services.antenna_pattern import station_ecef_m
from cosmic_toolbox.services.horizon_mask_io import horizon_elevations_at_azimuths

DEFAULT_MIN_ELEVATION_DEG: float = 5.0
"""Elevation cutoff (degrees) applied when no horizon mask CSV is configured."""


def _union_interval_seconds(passes: List[PassStatistic]) -> float:
    """Total seconds covered by the union of all pass [aos, los] intervals.

    Overlapping passes (e.g. simultaneous contacts at different stations) are
    merged so each instant of access counts once.
    """
    intervals = sorted(
        ((p.aos, p.los) for p in passes if p.los > p.aos),
        key=lambda iv: iv[0],
    )
    if not intervals:
        return 0.0
    total = timedelta(0)
    cur_start, cur_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= cur_end:
            if end > cur_end:
                cur_end = end
        else:
            total += cur_end - cur_start
            cur_start, cur_end = start, end
    total += cur_end - cur_start
    return total.total_seconds()


def _build_summary(
    passes: List[PassStatistic], scenario_duration_seconds: float
) -> AnalysisSummary:
    if not passes:
        return AnalysisSummary(
            total_passes=0,
            total_access_minutes=0.0,
            coverage_percent=0.0,
            avg_duration_minutes=0.0,
            min_duration_minutes=0.0,
            max_duration_minutes=0.0,
        )
    durations = np.array([p.duration_minutes for p in passes], dtype=float)
    total_access_minutes = float(np.sum(durations))
    # Coverage is the fraction of the scenario during which at least one
    # station has access.  Summing per-pass durations would double-count
    # time when passes from different stations overlap, so compute the union
    # of access intervals instead.
    union_access_seconds = _union_interval_seconds(passes)
    coverage_percent = (
        100.0 * union_access_seconds / scenario_duration_seconds
        if scenario_duration_seconds > 0.0
        else 0.0
    )
    return AnalysisSummary(
        total_passes=len(passes),
        total_access_minutes=total_access_minutes,
        coverage_percent=coverage_percent,
        avg_duration_minutes=float(np.mean(durations)),
        min_duration_minutes=float(np.min(durations)),
        max_duration_minutes=float(np.max(durations)),
    )




def _empty_derived_access_result() -> DerivedAccessResult:
    return DerivedAccessResult(
        passes=[],
        summary=AnalysisSummary(0, 0.0, 0.0, 0.0, 0.0, 0.0),
        station_summaries=[],
        station_elevation_series={},
    )


def analysis_result_from_components(
    *,
    ephemeris: PropagatedEphemeris,
    derived_access: DerivedAccessResult,
) -> AnalysisResult:
    """Flatten nested ephemeris/access caches into the UI-facing result shape."""
    return AnalysisResult(
        passes=list(derived_access.passes),
        summary=derived_access.summary,
        station_summaries=list(derived_access.station_summaries),
        timeline_seconds=ephemeris.timeline_seconds,
        station_elevation_series=dict(derived_access.station_elevation_series),
        orbit_period_seconds=float(ephemeris.orbit_period_seconds),
        mean_ic_report=ephemeris.mean_ic_report,
        station_azimuth_series=dict(derived_access.station_azimuth_series),
        station_az_rate_series=dict(derived_access.station_az_rate_series),
        station_el_rate_series=dict(derived_access.station_el_rate_series),
        station_range_rate_series=dict(derived_access.station_range_rate_series),
        station_range_accel_series=dict(derived_access.station_range_accel_series),
        station_above_horizon_series=dict(derived_access.station_above_horizon_series),
        orbital_altitude_km=ephemeris.orbital_altitude_km,
        semi_major_axis_km=ephemeris.semi_major_axis_km,
        perigee_altitude_km=ephemeris.perigee_altitude_km,
        apogee_altitude_km=ephemeris.apogee_altitude_km,
        eccentricity=ephemeris.eccentricity,
        inclination_deg=ephemeris.inclination_deg,
        argument_of_perigee_deg=ephemeris.argument_of_perigee_deg,
        orbital_period_series_s=ephemeris.orbital_period_series_s,
        true_anomaly_deg=ephemeris.true_anomaly_deg,
        angle_of_attack_deg=ephemeris.angle_of_attack_deg,
        ephemeris=ephemeris,
        derived_access=derived_access,
    )


def _station_topocentric_basis(
    station: GroundStationConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat_rad = math.radians(float(station.latitude_deg))
    lon_rad = math.radians(float(station.longitude_deg))
    east = np.array([-math.sin(lon_rad), math.cos(lon_rad), 0.0], dtype=float)
    north = np.array(
        [
            -math.sin(lat_rad) * math.cos(lon_rad),
            -math.sin(lat_rad) * math.sin(lon_rad),
            math.cos(lat_rad),
        ],
        dtype=float,
    )
    up = np.array(
        [
            math.cos(lat_rad) * math.cos(lon_rad),
            math.cos(lat_rad) * math.sin(lon_rad),
            math.sin(lat_rad),
        ],
        dtype=float,
    )
    return east, north, up


def _extract_pass_statistics_from_series(
    *,
    timeline_seconds: np.ndarray,
    elevations_deg: np.ndarray,
    az_rate_deg_s: np.ndarray,
    el_rate_deg_s: np.ndarray,
    scenario_start: datetime,
    station_name: str,
    start_index: int,
    above_horizon: "np.ndarray | None" = None,
    horizon_threshold: "np.ndarray | float" = 0.0,
) -> List[PassStatistic]:
    t = np.asarray(timeline_seconds, dtype=float)
    el = np.asarray(elevations_deg, dtype=float)
    az_rate = np.asarray(az_rate_deg_s, dtype=float)
    el_rate = np.asarray(el_rate_deg_s, dtype=float)
    if t.ndim != 1 or el.shape != t.shape or az_rate.shape != t.shape or el_rate.shape != t.shape:
        raise ValueError("Pass extraction from series requires aligned time/elevation/rate arrays.")
    if t.size == 0:
        return []
    if scenario_start.tzinfo is None:
        scenario_start = scenario_start.replace(tzinfo=timezone.utc)
    scenario_start = scenario_start.astimezone(timezone.utc)

    # Expand threshold to a per-sample array for generalised crossing interpolation.
    if isinstance(horizon_threshold, np.ndarray) and horizon_threshold.shape == t.shape:
        thresh_arr = horizon_threshold.astype(float)
    else:
        thresh_arr = np.full(t.size, float(horizon_threshold), dtype=float)

    def _crossing_time(i0: int, i1: int) -> float:
        t0 = float(t[i0])
        t1 = float(t[i1])
        y0 = float(el[i0])
        y1 = float(el[i1])
        if not (math.isfinite(t0) and math.isfinite(t1) and math.isfinite(y0) and math.isfinite(y1)):
            return t1
        if t1 == t0:
            return t1
        # Interpolate to find t where el(t) == threshold(t), both varying linearly.
        th0 = float(thresh_arr[i0])
        th1 = float(thresh_arr[i1])
        lhs = y0 - th0
        slope = (y1 - y0) - (th1 - th0)
        if slope == 0.0:
            return t1
        frac = -lhs / slope
        return t0 + max(0.0, min(1.0, frac)) * (t1 - t0)

    # Use the precomputed above_horizon mask when available; otherwise derive from threshold.
    if above_horizon is not None:
        positive = np.asarray(above_horizon, dtype=bool)
    else:
        positive = np.isfinite(el) & (el > thresh_arr)
    diff = np.diff(positive.astype(np.int8))
    rise_idx = np.where(diff == 1)[0]   # index i where positive[i+1] becomes True
    fall_idx = np.where(diff == -1)[0]  # index i where positive[i+1] becomes False

    # Handle passes that start or end at the timeline boundary.
    if positive[0]:
        rise_idx = np.concatenate((np.array([-1], dtype=rise_idx.dtype), rise_idx))
    if positive[-1]:
        fall_idx = np.concatenate((fall_idx, np.array([t.size - 1], dtype=fall_idx.dtype)))

    # Guard against malformed data producing mismatched rise/fall counts.
    if rise_idx.size != fall_idx.size:
        n_pairs = min(rise_idx.size, fall_idx.size)
        rise_idx = rise_idx[:n_pairs]
        fall_idx = fall_idx[:n_pairs]

    passes: List[PassStatistic] = []
    next_index = int(start_index)

    for r_i, f_i in zip(rise_idx.tolist(), fall_idx.tolist()):
        # Interpolate exact AOS/LOS crossing times.
        pass_start_sec = float(t[0]) if r_i == -1 else _crossing_time(r_i, r_i + 1)
        pass_end_sec = float(t[-1]) if f_i == t.size - 1 else _crossing_time(f_i, f_i + 1)

        # O(log n) slice bounds — replaces O(n) boolean mask per pass.
        i0 = int(np.searchsorted(t, pass_start_sec))
        i1 = int(np.searchsorted(t, pass_end_sec, side="right"))
        sl = slice(i0, i1)

        if i1 > i0:
            max_el = float(np.max(el[sl]))
            el_rad = np.radians(el[sl])
            az_r = az_rate[sl]
            el_r = el_rate[sl]
            valid = np.isfinite(az_r) & np.isfinite(el_r)
            if np.any(valid):
                slew = np.sqrt(
                    np.square(el_r[valid])
                    + np.square(az_r[valid] * np.cos(el_rad[valid]))
                )
                finite_slew = np.isfinite(slew)
                max_slew = float(np.max(slew[finite_slew])) if np.any(finite_slew) else float("nan")
            else:
                max_slew = float("nan")
        else:
            max_el = float("-inf")
            max_slew = float("nan")

        passes.append(
            PassStatistic(
                index=next_index,
                aos=scenario_start + timedelta(seconds=pass_start_sec),
                los=scenario_start + timedelta(seconds=pass_end_sec),
                duration_minutes=(pass_end_sec - pass_start_sec) / 60.0,
                max_elevation_deg=max_el,
                max_sc_slew_rate_deg_s=max_slew,
                station_name=station_name,
            )
        )
        next_index += 1

    return passes


def _compute_station_geometry(
    station: GroundStationConfig,
    sat_ecef_m: np.ndarray,
    sat_ecef_v_mps: np.ndarray,
    timeline_seconds: np.ndarray,
    min_elevation_deg: float,
) -> dict[str, np.ndarray]:
    """Compute all geometry series for a single ground station.

    Two-pass design for large timelines:
    - **Pass 1** (full timeline): elevation only — needed for pass detection.
    - **Pass 2** (in-pass samples only): azimuth, rates, range-rate on the
      subset where ``elevation_deg > 0``.  Out-of-pass samples are filled with
      ``NaN`` so consumers can index by time without shape changes.

    Each call only reads the shared ``sat_ecef_m`` / ``sat_ecef_v_mps`` arrays
    and writes to its own local arrays, so multiple calls can run concurrently
    under ``ThreadPoolExecutor`` — NumPy releases the GIL for most operations.
    """
    n = timeline_seconds.shape[0]
    gs_ecef_m = station_ecef_m(station)
    east, north, up = _station_topocentric_basis(station)

    # --- Pass 1: elevation on full timeline -----------------------------------
    los = sat_ecef_m - gs_ecef_m[None, :]          # (N, 3)
    ranges_m = np.linalg.norm(los, axis=1)          # (N,)
    if np.any(ranges_m <= 0.0) or not np.all(np.isfinite(ranges_m)):
        raise ValueError(f"Invalid station LOS ranges for station {station.name!r}")
    east_c = los @ east
    north_c = los @ north
    up_c = los @ up
    elevation_deg = np.degrees(np.arctan2(up_c, np.hypot(east_c, north_c)))

    # --- Horizon mask threshold (per-azimuth terrain cutoff) -----------------
    if station.horizon_mask_path:
        # Compute full-timeline azimuth cheaply (atan2 only; no rate needed).
        az_full = np.mod(np.degrees(np.arctan2(east_c, north_c)), 360.0)
        horizon_threshold: np.ndarray | float = horizon_elevations_at_azimuths(
            az_full, station.horizon_mask_path
        )
    else:
        horizon_threshold = min_elevation_deg

    # --- Pass 2: detail series only for in-pass samples ----------------------
    above = elevation_deg > horizon_threshold       # boolean mask (N,)
    n_above = int(np.count_nonzero(above))

    azimuth_out = np.full(n, np.nan)
    az_rate_out = np.full(n, np.nan)
    el_rate_out = np.full(n, np.nan)
    range_rate_out = np.full(n, np.nan)
    range_accel_out = np.full(n, np.nan)

    if n_above > 0:
        los_p = los[above]                          # (M, 3)
        ranges_p = ranges_m[above]                  # (M,)
        v_p = sat_ecef_v_mps[above]                 # (M, 3)
        t_p = timeline_seconds[above]               # (M,)

        los_hat_p = los_p / ranges_p[:, None]
        east_p = los_p @ east
        north_p = los_p @ north
        up_p = los_p @ up

        azimuth_p = np.mod(np.degrees(np.arctan2(east_p, north_p)), 360.0)
        el_p = np.degrees(np.arctan2(up_p, np.hypot(east_p, north_p)))

        if n_above > 1:
            az_rad_unwrapped = np.unwrap(np.radians(azimuth_p))
            rr_p = np.sum(los_hat_p * v_p, axis=1)
            # np.gradient divides by the time spacing; duplicate timestamps
            # (dt == 0) produce divide-by-zero.  Suppress the numpy warning and
            # replace any resulting non-finite values with zero.
            with np.errstate(divide="ignore", invalid="ignore"):
                az_rate_p = np.degrees(np.gradient(az_rad_unwrapped, t_p))
                el_rate_p = np.degrees(np.gradient(np.radians(el_p), t_p))
                ra_p = np.gradient(rr_p, t_p)
            az_rate_p = np.where(np.isfinite(az_rate_p), az_rate_p, 0.0)
            el_rate_p = np.where(np.isfinite(el_rate_p), el_rate_p, 0.0)
            ra_p = np.where(np.isfinite(ra_p), ra_p, 0.0)
        else:
            az_rate_p = np.zeros(1)
            el_rate_p = np.zeros(1)
            rr_p = np.sum(los_hat_p * v_p, axis=1)
            ra_p = np.zeros(1)

        azimuth_out[above] = azimuth_p
        az_rate_out[above] = az_rate_p
        el_rate_out[above] = el_rate_p
        range_rate_out[above] = rr_p
        range_accel_out[above] = ra_p

    return {
        "elevation_deg": elevation_deg,
        "azimuth_deg": azimuth_out,
        "az_rate_deg_s": az_rate_out,
        "el_rate_deg_s": el_rate_out,
        "range_rate_mps": range_rate_out,
        "range_accel_mps2": range_accel_out,
        "above_horizon": above,
        "horizon_threshold": horizon_threshold,
    }


def derive_access_results_from_ephemeris(
    *,
    ephemeris: PropagatedEphemeris,
    config: AnalysisConfig,
    stations: List[GroundStationConfig] | None = None,
    _prebuilt_sat_ecef_m: "np.ndarray | None" = None,
    _prebuilt_sat_ecef_v_mps: "np.ndarray | None" = None,
) -> DerivedAccessResult:
    """Derive station access outputs from cached ephemeris samples."""
    options = getattr(config, "options", None)
    compute_passes = bool(
        getattr(options, "compute_ground_station_passes", True) if options else True
    )
    if not compute_passes:
        return _empty_derived_access_result()
    station_configs = list(stations) if stations is not None else (
        [config.ground_station] if config.ground_station is not None else []
    )
    if not station_configs:
        return _empty_derived_access_result()

    timeline_seconds = np.asarray(ephemeris.timeline_seconds, dtype=float)
    if timeline_seconds.ndim != 1 or timeline_seconds.size == 0:
        raise ValueError("Cached ephemeris timeline is empty or invalid.")
    if ephemeris.ecef_pos_km is None or ephemeris.ecef_vel_km_s is None:
        raise ValueError("Cached ephemeris is missing ECEF position/velocity arrays.")
    if ephemeris.ecef_pos_km.shape[0] != timeline_seconds.size:
        raise ValueError("Cached ephemeris arrays must match timeline length.")

    sat_ecef_m = _prebuilt_sat_ecef_m if _prebuilt_sat_ecef_m is not None else ephemeris.ecef_pos_km * 1000.0
    sat_ecef_v_mps = _prebuilt_sat_ecef_v_mps if _prebuilt_sat_ecef_v_mps is not None else ephemeris.ecef_vel_km_s * 1000.0

    station_elevation_series: dict[str, np.ndarray] = {}
    station_azimuth_series: dict[str, np.ndarray] = {}
    station_az_rate_series: dict[str, np.ndarray] = {}
    station_el_rate_series: dict[str, np.ndarray] = {}
    station_range_rate_series: dict[str, np.ndarray] = {}
    station_range_accel_series: dict[str, np.ndarray] = {}
    station_above_horizon_series: dict[str, np.ndarray] = {}
    passes: List[PassStatistic] = []
    per_station_passes: dict[str, List[PassStatistic]] = {}
    scenario_duration_seconds = (
        config.scenario.end_time - config.scenario.start_time
    ).total_seconds()

    # Elevation cutoff for stations without a horizon mask comes from the
    # propagation config; fall back to the documented default only when the
    # config does not provide one.
    propagation = getattr(config, "propagation", None)
    min_elevation_deg = getattr(propagation, "min_elevation_deg", None)
    if min_elevation_deg is None:
        min_elevation_deg = DEFAULT_MIN_ELEVATION_DEG
    min_elevation_deg = float(min_elevation_deg)

    # Compute geometry for each station concurrently.  NumPy releases the GIL
    # for most array operations so threads run in true parallel.  Pass indices
    # are assigned sequentially afterwards to preserve deterministic ordering.
    _compute = partial(
        _compute_station_geometry,
        sat_ecef_m=sat_ecef_m,
        sat_ecef_v_mps=sat_ecef_v_mps,
        timeline_seconds=timeline_seconds,
        min_elevation_deg=min_elevation_deg,
    )
    n_workers = min(len(station_configs), 8)
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        station_geos = list(pool.map(_compute, station_configs))

    next_index = 1
    for station, geo in zip(station_configs, station_geos):
        elevation_deg = geo["elevation_deg"]
        azimuth_deg = geo["azimuth_deg"]
        az_rate_deg_s = geo["az_rate_deg_s"]
        el_rate_deg_s = geo["el_rate_deg_s"]
        range_rate_mps = geo["range_rate_mps"]
        range_accel_mps2 = geo["range_accel_mps2"]

        station_elevation_series[station.name] = elevation_deg
        station_azimuth_series[station.name] = azimuth_deg
        station_az_rate_series[station.name] = az_rate_deg_s
        station_el_rate_series[station.name] = el_rate_deg_s
        station_range_rate_series[station.name] = range_rate_mps
        station_range_accel_series[station.name] = range_accel_mps2
        station_above_horizon_series[station.name] = geo["above_horizon"]

        station_passes = _extract_pass_statistics_from_series(
            timeline_seconds=timeline_seconds,
            elevations_deg=elevation_deg,
            az_rate_deg_s=az_rate_deg_s,
            el_rate_deg_s=el_rate_deg_s,
            scenario_start=config.scenario.start_time,
            station_name=station.name,
            start_index=next_index,
            above_horizon=geo["above_horizon"],
            horizon_threshold=geo["horizon_threshold"],
        )
        next_index += len(station_passes)
        passes.extend(station_passes)
        per_station_passes[station.name] = station_passes

    return DerivedAccessResult(
        passes=passes,
        summary=_build_summary(passes, scenario_duration_seconds),
        station_summaries=[
            StationSummary(
                station_name=name,
                total_passes=len(items),
                total_access_minutes=float(sum(p.duration_minutes for p in items)),
            )
            for name, items in per_station_passes.items()
        ],
        station_elevation_series=station_elevation_series,
        station_azimuth_series=station_azimuth_series,
        station_az_rate_series=station_az_rate_series,
        station_el_rate_series=station_el_rate_series,
        station_range_rate_series=station_range_rate_series,
        station_range_accel_series=station_range_accel_series,
        station_above_horizon_series=station_above_horizon_series,
    )
