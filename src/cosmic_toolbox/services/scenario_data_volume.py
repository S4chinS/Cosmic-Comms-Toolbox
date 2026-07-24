"""Scenario-level downlink data-volume utilities for batch analysis."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from cosmic_toolbox import link_budget_math
from cosmic_toolbox.itu_losses import estimate_slant_path_loss
from cosmic_toolbox.link_budget_defaults import load_link_budget_defaults
from cosmic_toolbox.models import (
    AnalysisResult,
    GroundStationConfig,
    PassStatistic,
    PropagatedEphemeris,
)
from cosmic_toolbox.services import antenna_pattern

_LB_DEFAULTS = load_link_budget_defaults()

OPERATING_MODE_VCM = "VCM"
OPERATING_MODE_FIXED = "Fixed MODCOD"
_VALID_POINTING_MODES = {"prograde_pointing", "free_to_roll", "constrained_aoa"}


@dataclass(frozen=True)
class AccessSeries:
    """Strictly validated analysis payload needed for batch data-volume evaluation."""

    time_seconds: np.ndarray
    station_series: dict[str, np.ndarray]
    orbit_period_s: float
    altitude_km: np.ndarray
    sat_ecef_m: np.ndarray
    body_x_ecef: np.ndarray
    body_y_ecef: np.ndarray
    body_z_ecef: np.ndarray


@dataclass(frozen=True)
class ArchitectureConfig:
    """One link architecture to evaluate against a shared scenario."""

    architecture_id: str
    band: str
    frequency_GHz: float
    symbol_rate_Msps: float
    operating_mode: str
    margin_dB: float
    antenna_lut_path: Path
    fixed_modcod_name: str | None = None
    tx_power_dBw: float = _LB_DEFAULTS.tx_power_dBw
    tx_losses_dB: float = _LB_DEFAULTS.tx_losses_dB
    tx_backoff_dB: float = _LB_DEFAULTS.tx_backoff_dB
    rx_antenna_gain_dBi: float = _LB_DEFAULTS.rx_antenna_gain_dBi
    receiver_noise_figure_dB: float = _LB_DEFAULTS.receiver_noise_figure_dB
    sky_background_temperature_K: float = _LB_DEFAULTS.sky_background_temperature_K
    receiver_losses_dB: float = _LB_DEFAULTS.receiver_losses_dB
    polarization_loss_dB: float = _LB_DEFAULTS.polarization_loss_dB
    implementation_loss_dB: float = _LB_DEFAULTS.implementation_loss_dB
    unavailability_percent: float = _LB_DEFAULTS.unavailability_percent

    @property
    def symbol_rate_sps(self) -> float:
        return float(self.symbol_rate_Msps) * 1e6


@dataclass(frozen=True)
class ArchitectureResult:
    """Output metrics for one evaluated architecture."""

    architecture_id: str
    band: str
    frequency_ghz: float
    symbol_rate_msps: float
    operating_mode: str
    fixed_modcod: str | None
    total_gbit: float
    gbit_per_orbit: float | None
    num_filtered_passes: int


def build_access_series(result: AnalysisResult) -> AccessSeries:
    """Build the strictly validated access payload used by batch evaluators."""

    time_seconds = np.asarray(result.timeline_seconds, dtype=float)
    if time_seconds.ndim != 1 or time_seconds.size < 2:
        raise ValueError("Analysis result must contain at least two timeline samples")
    if not np.all(np.isfinite(time_seconds)):
        raise ValueError("Timeline contains non-finite values")

    station_series: dict[str, np.ndarray] = {}
    for name, samples in result.station_elevation_series.items():
        if not name:
            raise ValueError("Station elevation series contains an empty station name")
        series = np.asarray(samples, dtype=float)
        if series.shape != time_seconds.shape:
            raise ValueError(
                f"Station elevation series length mismatch for {name!r}: "
                f"{series.shape} != {time_seconds.shape}"
            )
        if not np.all(np.isfinite(series)):
            raise ValueError(f"Station elevation series contains non-finite values for {name!r}")
        station_series[name] = series
    if not station_series:
        raise ValueError("Analysis result does not contain any station elevation series")

    altitude_km = np.asarray(result.orbital_altitude_km, dtype=float)
    if altitude_km.shape != time_seconds.shape:
        raise ValueError(
            f"Orbital altitude series length mismatch: {altitude_km.shape} != {time_seconds.shape}"
        )
    if not np.all(np.isfinite(altitude_km)):
        raise ValueError("Orbital altitude series contains non-finite values")

    eph: PropagatedEphemeris | None = result.ephemeris
    if (
        eph is None
        or eph.ecef_pos_km is None
        or eph.body_x_ecef is None
        or eph.body_y_ecef is None
        or eph.body_z_ecef is None
    ):
        raise ValueError(
            "State-vector samples must be present for every timeline sample to evaluate antenna gains"
        )
    if eph.ecef_pos_km.shape[0] != time_seconds.size:
        raise ValueError(
            "State-vector samples must be present for every timeline sample to evaluate antenna gains"
        )

    sat_ecef_m = eph.ecef_pos_km * 1000.0
    body_x_ecef = eph.body_x_ecef
    body_y_ecef = eph.body_y_ecef
    body_z_ecef = eph.body_z_ecef

    for name, array in {
        "sat_ecef_m": sat_ecef_m,
        "body_x_ecef": body_x_ecef,
        "body_y_ecef": body_y_ecef,
        "body_z_ecef": body_z_ecef,
    }.items():
        if array.shape != (time_seconds.size, 3):
            raise ValueError(f"{name} must have shape ({time_seconds.size}, 3)")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} contains non-finite values")

    orbit_period_s = float(result.orbit_period_seconds)
    if not np.isfinite(orbit_period_s) or orbit_period_s <= 0.0:
        raise ValueError(f"Invalid orbit period: {orbit_period_s!r}")

    return AccessSeries(
        time_seconds=time_seconds,
        station_series=station_series,
        orbit_period_s=orbit_period_s,
        altitude_km=altitude_km,
        sat_ecef_m=sat_ecef_m,
        body_x_ecef=body_x_ecef,
        body_y_ecef=body_y_ecef,
        body_z_ecef=body_z_ecef,
    )


def filter_passes_by_max_elevation(
    passes: list[PassStatistic], lower_deg: float, upper_deg: float
) -> list[PassStatistic]:
    """Filter passes using the same max-elevation semantics as the UI."""

    lower = float(lower_deg)
    upper = float(upper_deg)
    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError("Elevation bounds must be finite")
    if lower < 0.0 or upper > 90.0 or lower > upper:
        raise ValueError(f"Invalid elevation bounds: lower={lower!r}, upper={upper!r}")

    filtered: list[PassStatistic] = []
    for item in passes:
        max_elev = float(item.max_elevation_deg)
        if not np.isfinite(max_elev):
            continue
        if lower <= max_elev <= upper:
            filtered.append(item)
    return filtered


def integrate_data_volume_interval(
    time_axis: np.ndarray, data_rates: np.ndarray, start_sec: float, end_sec: float
) -> float:
    """Integrate Mbps samples over the requested interval and return Gigabits."""

    if time_axis.ndim != 1 or data_rates.ndim != 1:
        raise ValueError("Time axis and data-rate arrays must be one-dimensional")
    if time_axis.shape != data_rates.shape:
        raise ValueError("Time axis and data-rate arrays must match in shape")
    if not np.all(np.isfinite(time_axis)) or not np.all(np.isfinite(data_rates)):
        raise ValueError("Integration inputs must be finite")
    if end_sec <= start_sec:
        raise ValueError("Integration end time must be greater than start time")

    # O(log n) slice instead of O(n) boolean mask — critical for large timelines.
    i0 = int(np.searchsorted(time_axis, start_sec, side="right"))
    i1 = int(np.searchsorted(time_axis, end_sec))
    interval_times = time_axis[i0:i1]
    interval_rates = data_rates[i0:i1]
    start_rate = float(np.interp(start_sec, time_axis, data_rates))
    end_rate = float(np.interp(end_sec, time_axis, data_rates))
    times = np.concatenate(([start_sec], interval_times, [end_sec]))
    rates = np.concatenate(([start_rate], interval_rates, [end_rate]))
    return link_budget_math.integrate_data_volume_gb(times, rates)


def compute_pass_downlink_volumes_gbit(
    *,
    access_series: AccessSeries,
    station_rate_lookup: dict[str, np.ndarray],
    passes: list[PassStatistic],
    scenario_start_time: datetime,
) -> list[float]:
    """Return Gbit/pass values using the same pass integration semantics as the UI."""

    if not station_rate_lookup:
        raise ValueError("Station rate lookup cannot be empty")
    if scenario_start_time.tzinfo is None:
        raise ValueError("Scenario start time must be timezone-aware")

    time_seconds = access_series.time_seconds
    start_axis = float(time_seconds[0])
    end_axis = float(time_seconds[-1])
    volumes_gbit: list[float] = []
    for entry in passes:
        station_name = str(entry.station_name or "").strip()
        if not station_name:
            raise ValueError("Filtered pass is missing a station_name")
        if station_name not in station_rate_lookup:
            raise ValueError(f"Missing rate series for station {station_name!r}")
        rates_array = np.asarray(station_rate_lookup[station_name], dtype=float)
        if rates_array.shape != time_seconds.shape:
            raise ValueError(
                f"Rate series length mismatch for station {station_name!r}: "
                f"{rates_array.shape} != {time_seconds.shape}"
            )
        aos_seconds = (entry.aos - scenario_start_time).total_seconds()
        los_seconds = (entry.los - scenario_start_time).total_seconds()
        if not np.isfinite(aos_seconds) or not np.isfinite(los_seconds):
            raise ValueError("Pass AOS/LOS values must be finite")
        pass_start = max(start_axis, float(aos_seconds))
        pass_end = min(end_axis, float(los_seconds))
        if pass_end <= pass_start:
            continue
        volume_gbit = integrate_data_volume_interval(
            time_seconds, rates_array, pass_start, pass_end
        )
        if volume_gbit <= 0.0:
            continue
        volumes_gbit.append(volume_gbit)
    return volumes_gbit


def compute_total_and_per_orbit_gbit(
    *, access_series: AccessSeries, pass_volumes_gbit: list[float]
) -> tuple[float, float | None]:
    """Mirror the current UI total/per-orbit aggregation exactly."""

    if not pass_volumes_gbit:
        return 0.0, None

    total_gbit = float(np.sum(pass_volumes_gbit))
    duration_s = float(access_series.time_seconds[-1] - access_series.time_seconds[0])
    if duration_s <= 0.0:
        raise ValueError("Scenario duration must be positive")
    num_orbits = duration_s / float(access_series.orbit_period_s)
    if num_orbits <= 0.0:
        raise ValueError("Number of orbits must be positive")
    return total_gbit, total_gbit / num_orbits


class ScenarioDataVolumeEvaluator:
    """Evaluate batch downlink architectures on a shared scenario payload."""

    def __init__(
        self,
        *,
        access_series: AccessSeries,
        station_lookup: dict[str, GroundStationConfig],
        scenario_start_time: datetime,
        comms_pointing_mode: str,
        comms_pointing_aoa_limit_deg: float,
        contact_elevation_deg: float,
    ) -> None:
        if scenario_start_time.tzinfo is None:
            raise ValueError("Scenario start time must be timezone-aware")
        mode = str(comms_pointing_mode).strip().lower()
        if mode not in _VALID_POINTING_MODES:
            raise ValueError(f"Unsupported comms pointing mode: {comms_pointing_mode!r}")
        aoa_limit = float(comms_pointing_aoa_limit_deg)
        if not np.isfinite(aoa_limit) or aoa_limit < 0.0 or aoa_limit > 180.0:
            raise ValueError(f"Invalid comms AoA limit: {comms_pointing_aoa_limit_deg!r}")
        contact_elevation = float(contact_elevation_deg)
        if not np.isfinite(contact_elevation) or contact_elevation < 0.0 or contact_elevation > 90.0:
            raise ValueError(f"Invalid contact elevation threshold: {contact_elevation_deg!r}")
        if not station_lookup:
            raise ValueError("Station lookup cannot be empty")

        self._access_series = access_series
        self._station_lookup = station_lookup
        self._scenario_start_time = scenario_start_time
        self._comms_pointing_mode = mode
        self._comms_pointing_aoa_limit_deg = aoa_limit
        self._contact_elevation_deg = contact_elevation
        self._loss_curve_cache: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
        self._lut_cache: dict[Path, antenna_pattern.SphericalGainLut] = {}
        self._boresight_cache: dict[Path, float] = {}
        self._gain_series_cache: dict[tuple[str, Path], np.ndarray] = {}

    def evaluate_architecture(
        self,
        *,
        architecture: ArchitectureConfig,
        filtered_passes: list[PassStatistic],
    ) -> ArchitectureResult:
        """Evaluate one architecture and return its scenario-level data-volume metrics."""

        operating_mode = str(architecture.operating_mode).strip()
        if operating_mode not in {OPERATING_MODE_VCM, OPERATING_MODE_FIXED}:
            raise ValueError(f"Unsupported operating mode: {architecture.operating_mode!r}")
        if operating_mode == OPERATING_MODE_FIXED and not architecture.fixed_modcod_name:
            raise ValueError("Fixed MODCOD mode requires a fixed_modcod_name")
        if architecture.frequency_GHz <= 0.0:
            raise ValueError(f"Invalid frequency_GHz: {architecture.frequency_GHz!r}")
        if architecture.symbol_rate_sps <= 0.0:
            raise ValueError(f"Invalid symbol_rate_Msps: {architecture.symbol_rate_Msps!r}")
        if not architecture.antenna_lut_path.exists():
            raise FileNotFoundError(f"Antenna LUT NPZ not found: {architecture.antenna_lut_path}")

        tx_gain_dBi = self._get_tx_boresight_gain_dBi(architecture.antenna_lut_path)
        station_rates: dict[str, np.ndarray] = {}
        for station_name, elevations in self._access_series.station_series.items():
            station = self._resolve_station_config(station_name)
            antenna_gains = self._get_station_antenna_gain_series(
                station=station,
                antenna_lut_path=architecture.antenna_lut_path,
            )
            station_rates[station_name] = self._evaluate_station_data_rate(
                station=station,
                elevations_deg=elevations,
                altitude_km=self._access_series.altitude_km,
                antenna_gains_dBi=antenna_gains,
                frequency_GHz=architecture.frequency_GHz,
                symbol_rate_sps=architecture.symbol_rate_sps,
                tx_power_dBw=architecture.tx_power_dBw,
                tx_gain_dBi=tx_gain_dBi,
                tx_losses_dB=architecture.tx_losses_dB,
                tx_backoff_dB=architecture.tx_backoff_dB,
                rx_antenna_gain_dBi=architecture.rx_antenna_gain_dBi,
                receiver_noise_figure_dB=architecture.receiver_noise_figure_dB,
                sky_background_temperature_K=architecture.sky_background_temperature_K,
                rx_losses_dB=architecture.receiver_losses_dB,
                polarization_loss_dB=architecture.polarization_loss_dB,
                implementation_loss_dB=architecture.implementation_loss_dB,
                margin_dB=architecture.margin_dB,
                unavailability_percent=architecture.unavailability_percent,
                fixed_modcod_name=architecture.fixed_modcod_name
                if operating_mode == OPERATING_MODE_FIXED
                else None,
            )

        pass_volumes_gbit = compute_pass_downlink_volumes_gbit(
            access_series=self._access_series,
            station_rate_lookup=station_rates,
            passes=filtered_passes,
            scenario_start_time=self._scenario_start_time,
        )
        total_gbit, gbit_per_orbit = compute_total_and_per_orbit_gbit(
            access_series=self._access_series,
            pass_volumes_gbit=pass_volumes_gbit,
        )
        return ArchitectureResult(
            architecture_id=architecture.architecture_id,
            band=architecture.band,
            frequency_ghz=float(architecture.frequency_GHz),
            symbol_rate_msps=float(architecture.symbol_rate_Msps),
            operating_mode=operating_mode,
            fixed_modcod=architecture.fixed_modcod_name,
            total_gbit=total_gbit,
            gbit_per_orbit=gbit_per_orbit,
            num_filtered_passes=len(filtered_passes),
        )

    def _resolve_station_config(self, station_name: str) -> GroundStationConfig:
        station = self._station_lookup.get(station_name)
        if station is None:
            raise ValueError(f"Active station lookup does not contain {station_name!r}")
        return station

    def _load_lut(self, path: Path) -> antenna_pattern.SphericalGainLut:
        lut_path = path.expanduser().resolve()
        lut = self._lut_cache.get(lut_path)
        if lut is None:
            lut = antenna_pattern.load_spherical_gain_lut(lut_path)
            self._lut_cache[lut_path] = lut
        return lut

    def _get_tx_boresight_gain_dBi(self, antenna_lut_path: Path) -> float:
        lut_path = antenna_lut_path.expanduser().resolve()
        cached = self._boresight_cache.get(lut_path)
        if cached is not None:
            return cached
        peak_gain_dBi = float(np.max(self._load_lut(lut_path).gain_dbi_grid))
        if not np.isfinite(peak_gain_dBi):
            raise ValueError("Antenna LUT peak gain is non-finite")
        self._boresight_cache[lut_path] = peak_gain_dBi
        return peak_gain_dBi

    def _get_comms_pointing_active_mask(self, station_name: str) -> np.ndarray:
        elevations = self._access_series.station_series.get(station_name)
        if elevations is None:
            raise ValueError(f"Missing station elevation series for {station_name!r}")
        return elevations >= self._contact_elevation_deg

    def _get_in_view_mask(self, station_name: str) -> np.ndarray:
        """Samples where the satellite is above the local horizon for a station.

        Antenna gain and the link budget only contribute to downlink volume while
        the satellite is geometrically visible (elevation > 0); below-horizon
        samples are zero-rated.  Computing only on this subset is numerically
        identical to evaluating the full timeline and then zeroing elevation <= 0,
        while avoiding the LUT/link-budget work on the (typically ~95%+) of samples
        that never contribute.
        """
        elevations = self._access_series.station_series.get(station_name)
        if elevations is None:
            raise ValueError(f"Missing station elevation series for {station_name!r}")
        return elevations > 0.0

    def _get_station_antenna_gain_series(
        self,
        *,
        station: GroundStationConfig,
        antenna_lut_path: Path,
    ) -> np.ndarray:
        cache_key = (station.name, antenna_lut_path.expanduser().resolve())
        cached = self._gain_series_cache.get(cache_key)
        if cached is not None:
            return cached

        # Only evaluate the (expensive) pointing + LUT lookup where the satellite
        # is above the horizon; off-view gains are never consumed by the data-rate
        # evaluation (which masks to the same in-view subset).  Off-view entries
        # are filled with 0.0 dBi purely to keep a full-length, finite series.
        in_view = self._get_in_view_mask(station.name)
        n = int(in_view.size)
        gains_dbi = np.zeros(n, dtype=float)
        if np.any(in_view):
            steering_mask = self._get_comms_pointing_active_mask(station.name)
            gains_sub, _az_deg, _el_deg, _roll_deg = antenna_pattern.evaluate_station_gain_series(
                lut=self._load_lut(cache_key[1]),
                station=station,
                sat_ecef_m=self._access_series.sat_ecef_m[in_view],
                body_x_ecef=self._access_series.body_x_ecef[in_view],
                body_y_ecef=self._access_series.body_y_ecef[in_view],
                body_z_ecef=self._access_series.body_z_ecef[in_view],
                pointing_mode=self._comms_pointing_mode,
                max_aoa_deg=self._comms_pointing_aoa_limit_deg,
                steering_active_mask=steering_mask[in_view],
            )
            gains_dbi[in_view] = gains_sub
        self._gain_series_cache[cache_key] = gains_dbi
        return gains_dbi

    def _get_timeseries_atmospheric_loss_curve(
        self,
        *,
        station: GroundStationConfig,
        frequency_GHz: float,
        unavailability_percent: float,
        min_elev_deg: float = 0.1,
        max_elev_deg: float = 90.0,
        num_samples: int = 361,
    ) -> tuple[np.ndarray, np.ndarray]:
        if frequency_GHz <= 0.0:
            raise ValueError(f"Invalid frequency_GHz: {frequency_GHz!r}")
        if num_samples < 10:
            raise ValueError("num_samples must be at least 10")
        grid = np.linspace(float(min_elev_deg), float(max_elev_deg), int(num_samples))
        key = (
            round(float(frequency_GHz), 6),
            round(float(unavailability_percent), 6),
            round(float(station.latitude_deg), 6),
            round(float(station.longitude_deg), 6),
            round(float(station.altitude_m), 3),
            int(grid.size),
            round(float(grid[0]), 6),
            round(float(grid[-1]), 6),
        )
        cached = self._loss_curve_cache.get(key)
        if cached is not None:
            return cached
        losses = np.asarray(
            estimate_slant_path_loss(
                frequency_GHz=frequency_GHz,
                elevations_deg=grid,
                lat_deg=station.latitude_deg,
                lon_deg=station.longitude_deg,
                altitude_m=station.altitude_m,
                unavailability_percent=unavailability_percent,
            ),
            dtype=float,
        )
        if losses.shape != grid.shape:
            raise ValueError(
                f"Atmospheric loss curve shape mismatch: {losses.shape} != {grid.shape}"
            )
        if not np.all(np.isfinite(losses)):
            raise ValueError("Non-finite atmospheric loss values returned by ITU-R model")
        self._loss_curve_cache[key] = (grid, losses)
        return grid, losses

    def _evaluate_station_data_rate(
        self,
        *,
        station: GroundStationConfig,
        elevations_deg: np.ndarray,
        altitude_km: np.ndarray,
        antenna_gains_dBi: np.ndarray,
        frequency_GHz: float,
        symbol_rate_sps: float,
        tx_power_dBw: float,
        tx_gain_dBi: float,
        tx_losses_dB: float,
        tx_backoff_dB: float,
        rx_antenna_gain_dBi: float,
        receiver_noise_figure_dB: float,
        sky_background_temperature_K: float,
        rx_losses_dB: float,
        polarization_loss_dB: float,
        implementation_loss_dB: float,
        margin_dB: float,
        unavailability_percent: float,
        fixed_modcod_name: str | None,
    ) -> np.ndarray:
        if elevations_deg.shape != self._access_series.time_seconds.shape:
            raise ValueError("Elevation series must match the scenario timeline shape")
        if altitude_km.shape != elevations_deg.shape:
            raise ValueError("Altitude series must match the elevation series shape")
        if antenna_gains_dBi.shape != elevations_deg.shape:
            raise ValueError("Antenna gain series must match the elevation series shape")
        if not np.all(np.isfinite(elevations_deg)):
            raise ValueError("Elevation series contains non-finite values")
        if not np.all(np.isfinite(altitude_km)):
            raise ValueError("Altitude series contains non-finite values")
        if not np.all(np.isfinite(antenna_gains_dBi)):
            raise ValueError("Antenna gain series contains non-finite values")

        # Below-horizon samples are zero-rated, so only evaluate the link budget
        # where the satellite is in view.  This is numerically identical to
        # evaluating the full timeline and zeroing elevation <= 0 afterwards,
        # because pass integration only ever reads in-view (elevation > 0)
        # samples and their AOS/LOS-bracketing neighbours.
        rates = np.zeros_like(elevations_deg, dtype=float)
        in_view = elevations_deg > 0.0
        if not np.any(in_view):
            return rates

        sanitized_elev = np.clip(elevations_deg[in_view], 0.0, 90.0)
        loss_grid, loss_values = self._get_timeseries_atmospheric_loss_curve(
            station=station,
            frequency_GHz=frequency_GHz,
            unavailability_percent=unavailability_percent,
            min_elev_deg=0.1,
            max_elev_deg=90.0,
        )
        losses = np.interp(
            np.maximum(sanitized_elev, float(loss_grid[0])),
            loss_grid,
            loss_values,
            left=float(loss_values[0]),
            right=float(loss_values[-1]),
        )
        results = link_budget_math.calculate_link_budget(
            elevations_deg=sanitized_elev,
            antenna_gains_dBi=antenna_gains_dBi[in_view],
            atmospheric_losses_dB=losses,
            tx_power_dBw=tx_power_dBw,
            tx_boresight_gain_dBi=tx_gain_dBi,
            tx_losses_dB=tx_losses_dB,
            tx_backoff_dB=tx_backoff_dB,
            frequency_GHz=frequency_GHz,
            satellite_altitude_km=altitude_km[in_view],
            ground_altitude_m=station.altitude_m,
            rx_antenna_gain_dBi=rx_antenna_gain_dBi,
            receiver_noise_figure_dB=receiver_noise_figure_dB,
            sky_background_temperature_K=sky_background_temperature_K,
            receiver_losses_dB=rx_losses_dB,
            polarization_loss_dB=polarization_loss_dB,
            symbol_rate_sps=symbol_rate_sps,
            implementation_loss_dB=implementation_loss_dB,
            margin_dB=margin_dB,
            fixed_modcod_name=fixed_modcod_name,
        )
        rates_sub = np.asarray(results.get("data_rate_mbps", []), dtype=float)
        if rates_sub.shape != sanitized_elev.shape:
            raise ValueError(
                f"Link-budget result length mismatch: {rates_sub.shape} != {sanitized_elev.shape}"
            )
        if not np.all(np.isfinite(rates_sub)):
            raise ValueError("Link-budget results contain non-finite data rates")
        rates[in_view] = rates_sub
        return rates
