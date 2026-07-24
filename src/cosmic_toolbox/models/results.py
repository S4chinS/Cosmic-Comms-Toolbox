"""Result dataclasses returned by analysis services."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List

import numpy as np


def _empty_f64() -> np.ndarray:
    return np.empty(0, dtype=np.float64)


@dataclass
class StateVectorSample:
    """One timestep of spacecraft position/velocity in ECI and ECEF frames."""

    timestamp: datetime
    eci_x_km: float
    eci_y_km: float
    eci_z_km: float
    eci_vx_km_s: float
    eci_vy_km_s: float
    eci_vz_km_s: float
    ecef_x_km: float
    ecef_y_km: float
    ecef_z_km: float
    ecef_vx_km_s: float
    ecef_vy_km_s: float
    ecef_vz_km_s: float
    body_x_ecef_x: float = 1.0
    body_x_ecef_y: float = 0.0
    body_x_ecef_z: float = 0.0
    body_y_ecef_x: float = 0.0
    body_y_ecef_y: float = 1.0
    body_y_ecef_z: float = 0.0
    body_z_ecef_x: float = 0.0
    body_z_ecef_y: float = 0.0
    body_z_ecef_z: float = 1.0


@dataclass
class GroundTrackPoint:
    """One timestep of satellite sub-satellite ground track position."""

    timestamp: datetime
    latitude_deg: float
    longitude_deg: float
    x_km: float = 0.0
    y_km: float = 0.0
    z_km: float = 0.0


@dataclass
class PassStatistic:
    """Represents a single access window."""

    index: int
    aos: datetime
    los: datetime
    duration_minutes: float
    max_elevation_deg: float
    max_sc_slew_rate_deg_s: float = float("nan")
    station_name: str | None = None


@dataclass
class AnalysisSummary:
    """High-level statistics derived from all access windows."""

    total_passes: int
    total_access_minutes: float
    coverage_percent: float
    avg_duration_minutes: float
    min_duration_minutes: float
    max_duration_minutes: float


@dataclass
class StationSummary:
    """Aggregated statistics for an individual ground station."""

    station_name: str
    total_passes: int
    total_access_minutes: float


@dataclass
class PropagatedEphemeris:
    """Orbit-only outputs that can be reused across cached access recomputes.

    Internally stores all arrays in columnar numpy form for performance.
    ``state_vectors`` and ``ground_track`` constructor kwargs are accepted for
    backward compatibility and are converted to columnar arrays on init.
    """

    timeline_seconds: np.ndarray
    orbit_period_seconds: float
    mean_ic_report: str | None = None

    # Columnar numpy arrays populated by propagation or NPZ load.
    # Shape: timestamps_unix (N,), pos/vel arrays (N, 3), gt scalars (N,).
    timestamps_unix: np.ndarray | None = field(default=None, compare=False, repr=False)
    eci_pos_km: np.ndarray | None = field(default=None, compare=False, repr=False)
    eci_vel_km_s: np.ndarray | None = field(default=None, compare=False, repr=False)
    ecef_pos_km: np.ndarray | None = field(default=None, compare=False, repr=False)
    ecef_vel_km_s: np.ndarray | None = field(default=None, compare=False, repr=False)
    body_x_ecef: np.ndarray | None = field(default=None, compare=False, repr=False)
    body_y_ecef: np.ndarray | None = field(default=None, compare=False, repr=False)
    body_z_ecef: np.ndarray | None = field(default=None, compare=False, repr=False)
    gt_lat_deg: np.ndarray | None = field(default=None, compare=False, repr=False)
    gt_lon_deg: np.ndarray | None = field(default=None, compare=False, repr=False)

    # Backward-compatibility shims: accept list-based inputs.
    # These are consumed by __post_init__ and converted to numpy arrays.
    state_vectors: List["StateVectorSample"] | None = field(
        default=None, compare=False, repr=False
    )
    ground_track: List["GroundTrackPoint"] | None = field(
        default=None, compare=False, repr=False
    )

    def __post_init__(self) -> None:
        if self.state_vectors is not None and len(self.state_vectors) > 0:
            svs = self.state_vectors
            self.timestamps_unix = np.array(
                [sv.timestamp.timestamp() for sv in svs], dtype=np.float64
            )
            if self.timeline_seconds is None:
                self.timeline_seconds = self.timestamps_unix - self.timestamps_unix[0]
            self.eci_pos_km = np.array(
                [[sv.eci_x_km, sv.eci_y_km, sv.eci_z_km] for sv in svs], dtype=np.float64
            )
            self.eci_vel_km_s = np.array(
                [[sv.eci_vx_km_s, sv.eci_vy_km_s, sv.eci_vz_km_s] for sv in svs], dtype=np.float64
            )
            self.ecef_pos_km = np.array(
                [[sv.ecef_x_km, sv.ecef_y_km, sv.ecef_z_km] for sv in svs], dtype=np.float64
            )
            self.ecef_vel_km_s = np.array(
                [[sv.ecef_vx_km_s, sv.ecef_vy_km_s, sv.ecef_vz_km_s] for sv in svs], dtype=np.float64
            )
            self.body_x_ecef = np.array(
                [[sv.body_x_ecef_x, sv.body_x_ecef_y, sv.body_x_ecef_z] for sv in svs], dtype=np.float64
            )
            self.body_y_ecef = np.array(
                [[sv.body_y_ecef_x, sv.body_y_ecef_y, sv.body_y_ecef_z] for sv in svs], dtype=np.float64
            )
            self.body_z_ecef = np.array(
                [[sv.body_z_ecef_x, sv.body_z_ecef_y, sv.body_z_ecef_z] for sv in svs], dtype=np.float64
            )
            self.orbital_altitude_km = np.sqrt(
                self.ecef_pos_km[:, 0] ** 2
                + self.ecef_pos_km[:, 1] ** 2
                + self.ecef_pos_km[:, 2] ** 2
            ) - 6378.137
        if self.ground_track is not None and len(self.ground_track) > 0:
            gts = self.ground_track
            self.gt_lat_deg = np.array([gt.latitude_deg for gt in gts], dtype=np.float64)
            self.gt_lon_deg = np.array([gt.longitude_deg for gt in gts], dtype=np.float64)
        # Ensure timeline_seconds is always an ndarray.
        if not isinstance(self.timeline_seconds, np.ndarray):
            self.timeline_seconds = np.asarray(self.timeline_seconds, dtype=np.float64)

    # Orbital element time series stored as compact numpy arrays.
    # All consumers call np.asarray() on access so list vs ndarray is transparent.
    orbital_altitude_km: np.ndarray = field(default_factory=_empty_f64)
    semi_major_axis_km: np.ndarray = field(default_factory=_empty_f64)
    perigee_altitude_km: np.ndarray = field(default_factory=_empty_f64)
    apogee_altitude_km: np.ndarray = field(default_factory=_empty_f64)
    eccentricity: np.ndarray = field(default_factory=_empty_f64)
    inclination_deg: np.ndarray = field(default_factory=_empty_f64)
    argument_of_perigee_deg: np.ndarray = field(default_factory=_empty_f64)
    orbital_period_series_s: np.ndarray = field(default_factory=_empty_f64)
    true_anomaly_deg: np.ndarray = field(default_factory=_empty_f64)
    angle_of_attack_deg: np.ndarray = field(default_factory=_empty_f64)


@dataclass
class DerivedAccessResult:
    """Station-dependent outputs derived from a propagated trajectory."""

    passes: List[PassStatistic]
    summary: AnalysisSummary
    station_summaries: List[StationSummary]
    station_elevation_series: dict[str, np.ndarray]
    station_azimuth_series: dict[str, np.ndarray] = field(default_factory=dict)
    station_az_rate_series: dict[str, np.ndarray] = field(default_factory=dict)
    station_el_rate_series: dict[str, np.ndarray] = field(default_factory=dict)
    station_range_rate_series: dict[str, np.ndarray] = field(default_factory=dict)
    station_range_accel_series: dict[str, np.ndarray] = field(default_factory=dict)
    station_above_horizon_series: dict[str, np.ndarray] = field(default_factory=dict)
    """Boolean mask per station: True where the satellite is above the horizon threshold."""


def propagated_ephemeris_sample_count(eph: PropagatedEphemeris) -> int:
    """Return number of propagation samples stored in the columnar numpy arrays."""
    if eph.timestamps_unix is not None:
        return int(eph.timestamps_unix.shape[0])
    return int(eph.timeline_seconds.shape[0])


@dataclass
class AnalysisResult:
    """Final payload returned after running the analysis."""

    passes: List[PassStatistic]
    summary: AnalysisSummary
    station_summaries: List[StationSummary]
    timeline_seconds: np.ndarray
    station_elevation_series: dict[str, np.ndarray]
    orbit_period_seconds: float
    mean_ic_report: str | None = None
    station_azimuth_series: dict[str, np.ndarray] = field(default_factory=dict)
    station_az_rate_series: dict[str, np.ndarray] = field(default_factory=dict)
    station_el_rate_series: dict[str, np.ndarray] = field(default_factory=dict)
    station_range_rate_series: dict[str, np.ndarray] = field(default_factory=dict)
    station_range_accel_series: dict[str, np.ndarray] = field(default_factory=dict)
    station_above_horizon_series: dict[str, np.ndarray] = field(default_factory=dict)
    """Boolean mask per station: True where the satellite is above the horizon threshold."""
    orbital_altitude_km: np.ndarray = field(default_factory=_empty_f64)
    semi_major_axis_km: np.ndarray = field(default_factory=_empty_f64)
    perigee_altitude_km: np.ndarray = field(default_factory=_empty_f64)
    apogee_altitude_km: np.ndarray = field(default_factory=_empty_f64)
    eccentricity: np.ndarray = field(default_factory=_empty_f64)
    inclination_deg: np.ndarray = field(default_factory=_empty_f64)
    argument_of_perigee_deg: np.ndarray = field(default_factory=_empty_f64)
    orbital_period_series_s: np.ndarray = field(default_factory=_empty_f64)
    true_anomaly_deg: np.ndarray = field(default_factory=_empty_f64)
    angle_of_attack_deg: np.ndarray = field(default_factory=_empty_f64)
    ephemeris: PropagatedEphemeris | None = None
    derived_access: DerivedAccessResult | None = None


def analysis_result_sample_count(result: AnalysisResult) -> int:
    """Return number of propagation samples from an AnalysisResult."""
    if result.ephemeris is not None:
        return propagated_ephemeris_sample_count(result.ephemeris)
    return int(result.timeline_seconds.shape[0])

