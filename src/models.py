"""Dataclasses shared between the UI layer and the analysis services."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List


@dataclass
class GroundStationConfig:
    """Represents a user-defined ground station."""

    name: str
    latitude_deg: float
    longitude_deg: float
    altitude_m: float


@dataclass
class OrbitConfig:
    """Stores the classical Keplerian orbital elements."""

    semi_major_axis_km: float
    eccentricity: float
    inclination_deg: float
    raan_deg: float
    arg_perigee_deg: float
    mean_anomaly_deg: float


@dataclass
class PropagationConfig:
    """Holds configuration for the propagator and access detection."""

    propagator_type: str  # e.g. "keplerian" or "numerical"
    min_elevation_deg: float
    sample_step_seconds: float
    enable_drag: bool
    # Optional pre-step for numerical propagation: solve for an initial Cartesian
    # state whose mean (h̄, k̄, alt̄_geo) matches the requested orbit under the
    # selected force model.
    use_mean_ic_finder: bool = False
    mean_ic_target_altitude_km: float | None = None
    # Spacecraft drag properties (used when enable_drag is True).
    # Defaults match the previous hard-coded IsotropicDrag(1.0, 2.2) setup.
    drag_area_m2: float = 0.43
    drag_cd: float = 3.0


@dataclass
class AnalysisOptions:
    """Toggle which high-level analyses are performed for a run."""

    compute_ground_station_passes: bool = True


@dataclass
class ScenarioConfig:
    """Defines the time span for the analysis."""

    start_time: datetime
    end_time: datetime


@dataclass
class AnalysisConfig:
    """Aggregates all input required to run the Orekit analysis."""

    # Primary ground station used when ground-station analysis is enabled.
    # When ground-station analysis is disabled, this may be None.
    ground_station: GroundStationConfig | None
    orbit: OrbitConfig
    propagation: PropagationConfig
    scenario: ScenarioConfig
    options: AnalysisOptions = field(default_factory=AnalysisOptions)


@dataclass
class PassStatistic:
    """Represents a single access window."""

    index: int
    aos: datetime
    los: datetime
    duration_minutes: float
    max_elevation_deg: float
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
class GroundTrackPoint:
    """Represents a sampled point along the satellite trajectory."""

    timestamp: datetime
    latitude_deg: float
    longitude_deg: float
    x_km: float
    y_km: float
    z_km: float


@dataclass
class AnalysisResult:
    """Final payload returned to the UI after running the analysis."""

    passes: List[PassStatistic]
    summary: AnalysisSummary
    station_summaries: List[StationSummary]
    ground_track: List[GroundTrackPoint]
    timeline_seconds: List[float]
    station_elevation_series: dict[str, List[float]]
    orbit_period_seconds: float
    mean_ic_report: str | None = None
    station_azimuth_series: dict[str, List[float]] = field(default_factory=dict)
    station_az_rate_series: dict[str, List[float]] = field(default_factory=dict)
    station_el_rate_series: dict[str, List[float]] = field(default_factory=dict)
    station_range_rate_series: dict[str, List[float]] = field(default_factory=dict)
    station_range_accel_series: dict[str, List[float]] = field(default_factory=dict)
    # Optional per-sample orbital metrics for the Orbit Summary tab.
    orbital_altitude_km: List[float] = field(default_factory=list)
    semi_major_axis_km: List[float] = field(default_factory=list)
    perigee_altitude_km: List[float] = field(default_factory=list)
    apogee_altitude_km: List[float] = field(default_factory=list)
    eccentricity: List[float] = field(default_factory=list)
    argument_of_perigee_deg: List[float] = field(default_factory=list)
    orbital_period_series_s: List[float] = field(default_factory=list)
    density_kg_m3: List[float] = field(default_factory=list)
    dynamic_pressure_pa: List[float] = field(default_factory=list)
    true_anomaly_deg: List[float] = field(default_factory=list)
    # Optional per-sample force diagnostics (positive magnitudes).
    drag_force_N: List[float] = field(default_factory=list)
