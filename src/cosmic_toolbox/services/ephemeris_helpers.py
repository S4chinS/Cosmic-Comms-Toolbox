"""Lightweight helpers for indexing into columnar PropagatedEphemeris arrays."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from cosmic_toolbox.models.results import PropagatedEphemeris


def ephemeris_time_slice(
    ephemeris: "PropagatedEphemeris",
    t_start_unix: float,
    t_end_unix: float,
) -> slice:
    """Return a slice covering samples whose unix timestamp falls within [t_start, t_end]."""
    ts = ephemeris.timestamps_unix
    if ts is None or ts.size == 0:
        return slice(0, 0)
    start_idx = int(np.searchsorted(ts, t_start_unix, side="left"))
    end_idx = int(np.searchsorted(ts, t_end_unix, side="right"))
    return slice(start_idx, end_idx)


def ephemeris_pass_mask(
    ephemeris: "PropagatedEphemeris",
    t_start_unix: float,
    t_end_unix: float,
) -> np.ndarray:
    """Return a boolean mask for samples within [t_start, t_end] (unix seconds)."""
    ts = ephemeris.timestamps_unix
    if ts is None or ts.size == 0:
        return np.zeros(0, dtype=bool)
    return (ts >= t_start_unix) & (ts <= t_end_unix)


def ecef_to_geodetic_deg(ecef_pos_km: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Spherical ECEF → (lat_deg, lon_deg).

    Uses a spherical Earth assumption, which is sufficient for ground-track
    display.  Accuracy is within ~0.2° of geodetic latitude.

    Args:
        ecef_pos_km: (N, 3) array of ECEF positions in kilometres.

    Returns:
        ``(lat_deg, lon_deg)`` — each shape ``(N,)``.
    """
    x, y, z = ecef_pos_km[:, 0], ecef_pos_km[:, 1], ecef_pos_km[:, 2]
    lat_deg = np.degrees(np.arctan2(z, np.hypot(x, y)))
    lon_deg = np.degrees(np.arctan2(y, x))
    return lat_deg, lon_deg


def body_axes_at_index(
    ephemeris: "PropagatedEphemeris",
    i: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Return (body_x, body_y, body_z) ECEF unit vectors at sample index i.

    Returns None if attitude data is missing or contains non-finite values.
    """
    bx_arr = ephemeris.body_x_ecef
    by_arr = ephemeris.body_y_ecef
    bz_arr = ephemeris.body_z_ecef
    if bx_arr is None or by_arr is None or bz_arr is None:
        return None
    bx = bx_arr[i].copy()
    by = by_arr[i].copy()
    bz = bz_arr[i].copy()
    if not (np.all(np.isfinite(bx)) and np.all(np.isfinite(by)) and np.all(np.isfinite(bz))):
        return None
    return bx, by, bz
