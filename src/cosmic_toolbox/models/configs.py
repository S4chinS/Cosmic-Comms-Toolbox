"""Configuration dataclasses for propagation and scenario analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class GroundStationConfig:
    """Represents a user-defined ground station."""

    name: str
    latitude_deg: float
    longitude_deg: float
    altitude_m: float
    horizon_mask_path: str | None = None
    """Optional path to a horizon mask CSV (360 azimuth/elevation pairs).

    When set, the per-azimuth terrain elevation is used as the visibility
    threshold for pass detection, antenna slew-rate calculation, and link-close
    rate-series masking.  When absent, a fixed default minimum elevation is
    applied instead (see DEFAULT_MIN_ELEVATION_DEG in cached_access_recompute).
    Absolute paths are used as-is; relative paths are resolved relative to the
    scenario package directory at load time.
    """
    supplier: str = ""
    """Semicolon-delimited supplier label(s) from the master database.

    Examples: ``"Acme Ground Network"``, ``"Acme;Beta;Gamma"``.  Empty for
    custom or legacy stations that have no database entry.
    """


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

    propagator_type: str  # "keplerian" or "brouwer_lyddane"
    min_elevation_deg: float
    sample_step_seconds: float
    attitude_mode: str = "prograde"
    contact_elevation_deg: float = 10.0
    enable_contact_attitude_switching: bool = True
    comms_pointing_mode: str = "prograde_pointing"
    comms_pointing_aoa_limit_deg: float = 5.0
    sensor_fov_cone_total_deg: float | None = None
    max_off_nadir_slew_deg: float | None = None
    drag_coefficient: float = 0.0
    """BrouwerLyddane secular drag (M2) term. 0.0 = no drag. Ignored for "keplerian"."""
    apply_mean_orbit_correction: bool = True
    """BrouwerLyddane only: apply the closed-form J2 short-period SMA correction so the
    propagator's *mean* orbit matches the requested target elements rather than its
    osculating initial state. Ignored for "keplerian" (no short-period correction needed)."""


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

    ground_station: GroundStationConfig | None
    orbit: OrbitConfig
    propagation: PropagationConfig
    scenario: ScenarioConfig
    options: AnalysisOptions = field(default_factory=AnalysisOptions)
