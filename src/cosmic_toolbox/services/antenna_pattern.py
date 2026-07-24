"""Helpers for spacecraft body-frame antenna pattern lookup."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cosmic_toolbox.models import GroundStationConfig


@dataclass(frozen=True)
class SphericalGainLut:
    """Regular Az/El gain lookup table.

    ``gain_dbi_grid`` holds the gain (dBi) as stored in the NPZ — this may be
    total gain, or the boresight-dominant circular polarisation gain (LHCP or
    RHCP) when the LUT was produced from a FEKO FFE file.
    """

    az_deg: np.ndarray
    el_deg: np.ndarray
    gain_dbi_grid: np.ndarray

    def _interpolate(self, grid: np.ndarray, az_deg: np.ndarray, el_deg: np.ndarray) -> np.ndarray:
        """Bilinearly interpolate *grid* at the requested az/el coordinates."""
        az = np.asarray(az_deg, dtype=float)
        el = np.asarray(el_deg, dtype=float)
        if az.shape != el.shape:
            raise ValueError("Azimuth and elevation arrays must have identical shapes")
        if az.ndim == 0:
            raise ValueError("Azimuth/elevation queries must be array-like")
        if not np.all(np.isfinite(az)) or not np.all(np.isfinite(el)):
            raise ValueError("Azimuth/elevation query contains non-finite values")
        if not np.all((el >= self.el_deg[0]) & (el <= self.el_deg[-1])):
            raise ValueError("Elevation query is outside LUT bounds")

        az_wrapped = np.mod(az, 360.0)
        az_step = float(self.az_deg[1] - self.az_deg[0])
        el_step = float(self.el_deg[1] - self.el_deg[0])

        az_idx_f = az_wrapped / az_step
        el_idx_f = (el - float(self.el_deg[0])) / el_step

        ia0 = np.floor(az_idx_f).astype(int) % self.az_deg.size
        ia1 = (ia0 + 1) % self.az_deg.size
        ie0 = np.floor(el_idx_f).astype(int)
        ie1 = np.clip(ie0 + 1, 0, self.el_deg.size - 1)

        fa = az_idx_f - np.floor(az_idx_f)
        fe = el_idx_f - np.floor(el_idx_f)

        return (
            (1.0 - fa) * (1.0 - fe) * grid[ia0, ie0]
            + fa * (1.0 - fe) * grid[ia1, ie0]
            + (1.0 - fa) * fe * grid[ia0, ie1]
            + fa * fe * grid[ia1, ie1]
        )

    def gain_dbi(self, az_deg: np.ndarray, el_deg: np.ndarray) -> np.ndarray:
        """Bilinearly interpolate total gain (dBi) at the requested az/el coordinates."""
        return self._interpolate(self.gain_dbi_grid, az_deg, el_deg)



def default_synthesized_lut_path() -> Path:
    """Return the default Anywaves S-band TTC antenna LUT shipped with the package.

    The active file is the manufacturer-measured 4-phi-cut pattern at 2.200 GHz
    (LHCP co-polar), converted to a 361x181 Az/El grid by
    scripts/convert_manufacturer_xlsx_to_lut.py.  The legacy WebPlotDigitizer-
    derived gain_AzEl_synth_from_phi0_phi90.npz (2 phi cuts, irregular sampling)
    is retained in the same directory for reference.
    """
    from cosmic_toolbox.paths import package_resources_root

    return (
        package_resources_root()
        / "antenna_luts"
        / "Anywaves_S_band_TTC_6dB"
        / "gain_AzEl_mfr_LHCP_2200MHz.npz"
    )


def load_spherical_gain_lut(path: str | Path) -> SphericalGainLut:
    """Load and strictly validate a regular Az/El antenna LUT NPZ."""
    lut_path = Path(path).expanduser().resolve()
    if not lut_path.exists():
        raise FileNotFoundError(f"Antenna LUT NPZ not found: {lut_path}")

    with np.load(lut_path, allow_pickle=True) as data:
        required = {"az_deg", "el_deg", "gain_dbi_grid"}
        missing = required.difference(data.files)
        if missing:
            raise KeyError(f"LUT NPZ is missing required keys: {sorted(missing)}")

        az = np.asarray(data["az_deg"], dtype=float)
        el = np.asarray(data["el_deg"], dtype=float)
        grid = np.asarray(data["gain_dbi_grid"], dtype=float)

    if az.ndim != 1 or el.ndim != 1 or grid.ndim != 2:
        raise ValueError("Invalid LUT dimensionality")
    if grid.shape != (az.size, el.size):
        raise ValueError(
            f"Invalid LUT grid shape: {grid.shape}, expected ({az.size}, {el.size})"
        )
    if az.size < 2 or el.size < 2:
        raise ValueError("LUT must contain at least 2 samples per axis")
    if not np.all(np.isfinite(az)) or not np.all(np.isfinite(el)):
        raise ValueError("LUT axes contain non-finite values")
    if not np.all(np.isfinite(grid)):
        raise ValueError("LUT gain grid contains non-finite values")
    if not np.all(np.diff(az) > 0.0):
        raise ValueError("LUT azimuth axis must be strictly increasing")
    if not np.all(np.diff(el) > 0.0):
        raise ValueError("LUT elevation axis must be strictly increasing")
    if not np.isclose(az[0], 0.0) or not np.isclose(az[-1], 360.0):
        raise ValueError("LUT azimuth axis must span 0..360 deg")
    if not np.isclose(el[0], -90.0) or not np.isclose(el[-1], 90.0):
        raise ValueError("LUT elevation axis must span -90..90 deg")

    az_steps = np.diff(az)
    el_steps = np.diff(el)
    if not np.allclose(az_steps, az_steps[0], rtol=0.0, atol=1e-9):
        raise ValueError("LUT azimuth axis must use uniform spacing")
    if not np.allclose(el_steps, el_steps[0], rtol=0.0, atol=1e-9):
        raise ValueError("LUT elevation axis must use uniform spacing")

    return SphericalGainLut(az_deg=az, el_deg=el, gain_dbi_grid=grid)


def station_ecef_m(station: GroundStationConfig) -> np.ndarray:
    """Convert a WGS84 station geodetic location to ECEF meters."""
    a = 6378137.0
    f = 1.0 / 298.257223563
    e2 = f * (2.0 - f)
    lat = math.radians(float(station.latitude_deg))
    lon = math.radians(float(station.longitude_deg))
    alt = float(station.altitude_m)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)
    radius = a / math.sqrt(1.0 - e2 * sin_lat * sin_lat)
    x = (radius + alt) * cos_lat * cos_lon
    y = (radius + alt) * cos_lat * sin_lon
    z = (radius * (1.0 - e2) + alt) * sin_lat
    return np.array([x, y, z], dtype=float)


def body_vectors_to_az_el(body_vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert unit body-frame vectors to azimuth/elevation look angles."""
    vectors = np.asarray(body_vectors, dtype=float)
    if vectors.ndim != 2 or vectors.shape[1] != 3:
        raise ValueError("Body vectors must have shape (N, 3)")
    if not np.all(np.isfinite(vectors)):
        raise ValueError("Body vectors contain non-finite values")
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms <= 0.0):
        raise ValueError("Body vectors must have positive norm")
    unit_vectors = vectors / norms[:, None]

    x = unit_vectors[:, 0]
    y = unit_vectors[:, 1]
    z = unit_vectors[:, 2]
    az_deg = np.mod(np.degrees(np.arctan2(y, x)), 360.0)
    el_deg = np.degrees(np.arctan2(z, np.hypot(x, y)))
    if np.any(el_deg < -90.0) or np.any(el_deg > 90.0):
        raise ValueError("Computed body-frame elevation is outside physical range")
    return az_deg, el_deg


def apply_roll_toward_station_about_x(
    body_vectors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Roll body vectors about +X so the station lies in the X/Z plane with +Z maximized."""
    vectors = np.asarray(body_vectors, dtype=float)
    if vectors.ndim != 2 or vectors.shape[1] != 3:
        raise ValueError("Body vectors must have shape (N, 3)")
    if not np.all(np.isfinite(vectors)):
        raise ValueError("Body vectors contain non-finite values")
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms <= 0.0):
        raise ValueError("Body vectors must have positive norm")
    unit_vectors = vectors / norms[:, None]

    x = unit_vectors[:, 0]
    y = unit_vectors[:, 1]
    z = unit_vectors[:, 2]
    roll_rad = np.arctan2(y, z)
    cos_roll = np.cos(roll_rad)
    sin_roll = np.sin(roll_rad)
    y_rot = y * cos_roll - z * sin_roll
    z_rot = y * sin_roll + z * cos_roll
    rotated = np.column_stack([x, y_rot, z_rot])
    rotated_norms = np.linalg.norm(rotated, axis=1)
    if np.any(rotated_norms <= 0.0):
        raise ValueError("Rolled body vectors have non-positive norm")
    return rotated / rotated_norms[:, None], np.degrees(roll_rad)


def apply_constrained_aoa_toward_station(
    body_vectors: np.ndarray,
    *,
    max_aoa_deg: float,
) -> np.ndarray:
    """Steer toward the station while constraining body +X away from prograde."""
    vectors = np.asarray(body_vectors, dtype=float)
    if vectors.ndim != 2 or vectors.shape[1] != 3:
        raise ValueError("Body vectors must have shape (N, 3)")
    if not np.all(np.isfinite(vectors)):
        raise ValueError("Body vectors contain non-finite values")
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms <= 0.0):
        raise ValueError("Body vectors must have positive norm")
    if not (np.isfinite(max_aoa_deg) and 0.0 <= float(max_aoa_deg) <= 180.0):
        raise ValueError(f"Invalid max_aoa_deg: {max_aoa_deg!r}")
    unit_vectors = vectors / norms[:, None]

    x = unit_vectors[:, 0]
    y = unit_vectors[:, 1]
    z = unit_vectors[:, 2]
    yz_norm = np.hypot(y, z)
    aoa_limit_rad = np.radians(float(max_aoa_deg))
    aoa_opt_rad = np.arctan2(np.abs(x), yz_norm)
    aoa_cmd_rad = np.minimum(aoa_limit_rad, aoa_opt_rad)

    perp_dir = np.zeros_like(unit_vectors)
    valid_perp = yz_norm > 1e-12
    perp_dir[valid_perp, 1] = y[valid_perp] / yz_norm[valid_perp]
    perp_dir[valid_perp, 2] = z[valid_perp] / yz_norm[valid_perp]
    perp_dir[~valid_perp, 2] = 1.0

    x_sign = np.where(x >= 0.0, 1.0, -1.0)
    cos_aoa = np.cos(aoa_cmd_rad)
    sin_aoa = np.sin(aoa_cmd_rad)
    commanded_x = np.column_stack(
        [
            cos_aoa,
            -x_sign * sin_aoa * perp_dir[:, 1],
            -x_sign * sin_aoa * perp_dir[:, 2],
        ]
    )
    x_norm = np.linalg.norm(commanded_x, axis=1)
    if np.any(x_norm <= 0.0):
        raise ValueError("Commanded AoA steering produced invalid body +X vectors")
    commanded_x = commanded_x / x_norm[:, None]

    los_x = np.sum(unit_vectors * commanded_x, axis=1)
    los_z_vec = unit_vectors - los_x[:, None] * commanded_x
    los_z_norm = np.linalg.norm(los_z_vec, axis=1)
    fallback_mask = los_z_norm <= 1e-12
    if np.any(fallback_mask):
        los_z_vec[fallback_mask] = np.array([0.0, 0.0, 1.0], dtype=float)
        los_z_vec[fallback_mask] -= (
            np.sum(los_z_vec[fallback_mask] * commanded_x[fallback_mask], axis=1)[:, None]
            * commanded_x[fallback_mask]
        )
        los_z_norm = np.linalg.norm(los_z_vec, axis=1)
    if np.any(los_z_norm <= 0.0):
        raise ValueError("Commanded AoA steering produced invalid body +Z vectors")
    los_z = los_z_norm
    steered = np.column_stack([los_x, np.zeros_like(los_x), los_z])
    steered_norm = np.linalg.norm(steered, axis=1)
    if np.any(steered_norm <= 0.0):
        raise ValueError("Commanded AoA steering produced invalid LOS vectors")
    return steered / steered_norm[:, None]


def project_los_to_body_frame(
    los_hat_ecef: np.ndarray,
    body_x_ecef: np.ndarray,
    body_y_ecef: np.ndarray,
    body_z_ecef: np.ndarray,
) -> np.ndarray:
    """Project ECEF LOS unit vectors onto spacecraft body axes."""
    los = np.asarray(los_hat_ecef, dtype=float)
    bx = np.asarray(body_x_ecef, dtype=float)
    by = np.asarray(body_y_ecef, dtype=float)
    bz = np.asarray(body_z_ecef, dtype=float)
    if los.ndim != 2 or los.shape[1] != 3:
        raise ValueError("LOS vectors must have shape (N, 3)")
    if bx.shape != los.shape or by.shape != los.shape or bz.shape != los.shape:
        raise ValueError("Body-axis arrays must match LOS vector shape")
    if not (
        np.all(np.isfinite(los))
        and np.all(np.isfinite(bx))
        and np.all(np.isfinite(by))
        and np.all(np.isfinite(bz))
    ):
        raise ValueError("LOS/body-axis arrays contain non-finite values")

    los_body = np.column_stack(
        [
            np.sum(los * bx, axis=1),
            np.sum(los * by, axis=1),
            np.sum(los * bz, axis=1),
        ]
    )
    norms = np.linalg.norm(los_body, axis=1)
    if np.any(norms <= 0.0):
        raise ValueError("Projected LOS vectors in body frame have zero norm")
    return los_body / norms[:, None]


def evaluate_station_gain_series(
    *,
    lut: SphericalGainLut,
    station: GroundStationConfig,
    sat_ecef_m: np.ndarray,
    body_x_ecef: np.ndarray,
    body_y_ecef: np.ndarray,
    body_z_ecef: np.ndarray,
    pointing_mode: str = "prograde_pointing",
    max_aoa_deg: float | None = None,
    steering_active_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate time-varying gain to a station using spacecraft ECEF/body samples."""
    sat = np.asarray(sat_ecef_m, dtype=float)
    if sat.ndim != 2 or sat.shape[1] != 3:
        raise ValueError("Satellite ECEF samples must have shape (N, 3)")
    if not np.all(np.isfinite(sat)):
        raise ValueError("Satellite ECEF samples contain non-finite values")

    gs_ecef_m = station_ecef_m(station)
    los_ecef = gs_ecef_m[None, :] - sat
    los_norm = np.linalg.norm(los_ecef, axis=1)
    if np.any(los_norm <= 0.0) or not np.all(np.isfinite(los_norm)):
        raise ValueError("Invalid station line-of-sight vectors")
    los_hat_ecef = los_ecef / los_norm[:, None]

    los_body = project_los_to_body_frame(
        los_hat_ecef=los_hat_ecef,
        body_x_ecef=body_x_ecef,
        body_y_ecef=body_y_ecef,
        body_z_ecef=body_z_ecef,
    )
    roll_deg = np.zeros(los_body.shape[0], dtype=float)
    if steering_active_mask is None:
        active_mask = np.ones(los_body.shape[0], dtype=bool)
    else:
        active_mask = np.asarray(steering_active_mask, dtype=bool)
        if active_mask.shape != (los_body.shape[0],):
            raise ValueError("Steering active mask must have shape (N,)")

    mode = str(pointing_mode).strip().lower()
    steered_body = np.array(los_body, copy=True)
    if mode == "prograde_pointing":
        pass
    elif mode == "free_to_roll":
        steered_body[active_mask], roll_deg[active_mask] = apply_roll_toward_station_about_x(
            los_body[active_mask]
        )
    elif mode == "constrained_aoa":
        if max_aoa_deg is None:
            raise ValueError("Constrained AoA mode requires max_aoa_deg")
        steered_body[active_mask] = apply_constrained_aoa_toward_station(
            los_body[active_mask],
            max_aoa_deg=max_aoa_deg,
        )
    else:
        raise ValueError(f"Unsupported pointing_mode: {pointing_mode!r}")
    az_deg, el_deg = body_vectors_to_az_el(steered_body)
    gains_dbi = lut.gain_dbi(az_deg, el_deg)
    if not np.all(np.isfinite(gains_dbi)):
        raise ValueError("Non-finite gains computed from LUT")
    return gains_dbi, az_deg, el_deg, roll_deg
