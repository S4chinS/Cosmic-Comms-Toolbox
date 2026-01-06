"""Orekit-backed ground-station access analysis service."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import wraps
from typing import Callable, List

import numpy as np
import orekit
from org.hipparchus.ode.nonstiff import DormandPrince853Integrator
from org.hipparchus.geometry.euclidean.threed import Vector3D
from org.orekit.attitudes import FrameAlignedProvider
from java.io import File
from org.orekit.bodies import CelestialBodyFactory, GeodeticPoint, OneAxisEllipsoid
from org.orekit.data import DataContext, DirectoryCrawler
from org.orekit.forces.drag import DragForce, IsotropicDrag
from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel
from org.orekit.forces.gravity.potential import GravityFieldFactory
from org.orekit.frames import FramesFactory, TopocentricFrame
from org.orekit.models.earth.atmosphere import NRLMSISE00
from org.orekit.models.earth.atmosphere.data import MarshallSolarActivityFutureEstimation
from org.orekit.orbits import CartesianOrbit, KeplerianOrbit, PositionAngleType
from org.orekit.propagation import Propagator, SpacecraftState
from org.orekit.propagation.analytical import KeplerianPropagator
from org.orekit.propagation.events import ElevationDetector, EventsLogger
from org.orekit.propagation.numerical import NumericalPropagator
from org.orekit.time import AbsoluteDate, TimeScale, TimeScalesFactory
from org.orekit.utils import Constants, IERSConventions, PVCoordinates

import orekitdata

from src.tools.ic_finder.initial_condition_finder import (
    ICFinderConfig,
    MeanElementTargets,
    find_initial_cartesian_state_for_mean_elements,
)

from src.models import (
    AnalysisConfig,
    AnalysisResult,
    AnalysisSummary,
    GroundStationConfig,
    GroundTrackPoint,
    PassStatistic,
    PropagationConfig,
    StationSummary,
)

orekit.initVM()

_OREKIT_BOOTSTRAPPED = False

# Altitude below which we consider the spacecraft to have re-entered and stop
# sampling / propagation-driven diagnostics. This is expressed in meters above
# the WGS84 ellipsoid.
_REENTRY_CUTOFF_ALTITUDE_M = 80_000.0


def ensure_orekit_bootstrapped() -> None:
    """Load the Orekit data files exactly once."""

    global _OREKIT_BOOTSTRAPPED
    if _OREKIT_BOOTSTRAPPED:
        return

    data_manager = DataContext.getDefault().getDataProvidersManager()
    crawler = DirectoryCrawler(File(orekitdata.__path__[0]))
    data_manager.addProvider(crawler)
    _OREKIT_BOOTSTRAPPED = True


@dataclass
class _StationContext:
    config: GroundStationConfig
    frame: TopocentricFrame
    logger: EventsLogger


def _attach_thread(func: Callable[..., object]) -> Callable[..., object]:
    """Ensure Orekit's JVM is attached for the current thread."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            orekit.getVMEnv().attachCurrentThread()
        except Exception:
            pass
        return func(*args, **kwargs)

    return wrapper


@_attach_thread
def run_access_analysis(
    config: AnalysisConfig,
    stations: List[GroundStationConfig] | None = None,
    progress_callback: Callable[[float], None] | None = None,
) -> AnalysisResult:
    """Execute the Orekit propagation (once) and return multi-station pass statistics."""

    ensure_orekit_bootstrapped()

    utc = TimeScalesFactory.getUTC()
    start_date = _to_absolute_date(config.scenario.start_time, utc)
    end_date = _to_absolute_date(config.scenario.end_time, utc)

    inertial_frame = FramesFactory.getEME2000()
    orbit = _build_initial_orbit(config, start_date, inertial_frame)
    initial_state = SpacecraftState(orbit)

    # Decide whether ground-station pass analysis is enabled.
    options = getattr(config, "options", None)
    compute_passes = bool(
        getattr(options, "compute_ground_station_passes", True) if options else True
    )

    station_configs: List[GroundStationConfig] = []
    if compute_passes:
        if stations is not None:
            station_configs = list(stations)
        elif config.ground_station is not None:
            station_configs = [config.ground_station]

    earth = _build_earth()
    atmosphere = _build_atmosphere(earth)

    # Optional pre-step: adjust the osculating initial state so mean (h̄, k̄, alt̄_geo)
    # match the requested orbit under the numerical force model.
    propagation_cfg = config.propagation
    mean_ic_report: str | None = None
    if (
        getattr(propagation_cfg, "use_mean_ic_finder", False)
        and str(getattr(propagation_cfg, "propagator_type", "")).lower() == "keplerian"
    ):
        mean_ic_report = "Mean IC: not applied (requires numerical propagator)"
    if (
        getattr(propagation_cfg, "use_mean_ic_finder", False)
        and str(getattr(propagation_cfg, "propagator_type", "")).lower() != "keplerian"
    ):
        try:
            alt_target_km = getattr(propagation_cfg, "mean_ic_target_altitude_km", None)
            if alt_target_km is None:
                alt_target_km = max(
                    0.0,
                    float(config.orbit.semi_major_axis_km)
                    - float(Constants.WGS84_EARTH_EQUATORIAL_RADIUS) / 1000.0,
                )
            e = float(config.orbit.eccentricity)
            argp_rad = math.radians(float(config.orbit.arg_perigee_deg))
            h_bar = e * math.cos(argp_rad)
            k_bar = e * math.sin(argp_rad)
            targets = MeanElementTargets(
                # Use the requested mean altitude to build a closer initial guess
                # for the solver; this significantly improves convergence when the
                # user adjusts the target altitude independently of the SMA input.
                a_guess_m=(
                    float(Constants.WGS84_EARTH_EQUATORIAL_RADIUS) + float(alt_target_km) * 1000.0
                ),
                alt_bar_m=float(alt_target_km) * 1000.0,
                i_bar_rad=math.radians(float(config.orbit.inclination_deg)),
                raan_bar_rad=math.radians(float(config.orbit.raan_deg)),
                h_bar=float(h_bar),
                k_bar=float(k_bar),
            )
            # Heuristic: if the target is far from the SMA-implied altitude, allow
            # larger Newton steps and more iterations so the solver can reach the
            # solution in one go.
            sma_implied_alt_km = float(config.orbit.semi_major_axis_km) - float(
                Constants.WGS84_EARTH_EQUATORIAL_RADIUS
            ) / 1000.0
            alt_mismatch_km = abs(float(alt_target_km) - float(sma_implied_alt_km))
            max_iterations = 12 if alt_mismatch_km > 25.0 else 6
            max_step_dv = 200.0 if alt_mismatch_km > 25.0 else 50.0
            cfg = ICFinderConfig(
                atmosphere=atmosphere if bool(getattr(propagation_cfg, "enable_drag", False)) else None,
                enable_drag=bool(getattr(propagation_cfg, "enable_drag", False)),
                drag_area_m2=float(getattr(propagation_cfg, "drag_area_m2", 0.43)),
                drag_cd=float(getattr(propagation_cfg, "drag_cd", 3.0)),
                gravity_model="hf",
                gravity_degree=10,
                gravity_order=10,
                mass_kg=float(initial_state.getMass()),
                unlock_r0_radial=True,
                max_iterations=int(max_iterations),
                max_step_dv_m_s=float(max_step_dv),
            )
            result_ic = find_initial_cartesian_state_for_mean_elements(
                epoch=start_date, targets=targets, cfg=cfg
            )
            converged = bool(getattr(result_ic, "converged", False))
            mean_ic_report = (
                f"Mean IC: {'applied' if converged else 'not applied'} "
                f"(target {float(alt_target_km):.1f} km, "
                f"mean {float(result_ic.alt_bar_m) / 1000.0:.2f} km, "
                f"Δalt {float(result_ic.residual_alt_m):+.1f} m, "
                f"iters {int(getattr(result_ic, 'iterations', 0))})"
            )
            if converged:
                r0 = result_ic.r0_m
                v0 = result_ic.v0_m
                pv0 = PVCoordinates(
                    Vector3D(float(r0[0]), float(r0[1]), float(r0[2])),
                    Vector3D(float(v0[0]), float(v0[1]), float(v0[2])),
                )
                corrected_orbit = CartesianOrbit(pv0, inertial_frame, start_date, Constants.WGS84_EARTH_MU)
                initial_state = SpacecraftState(corrected_orbit, float(initial_state.getMass()))
        except Exception:
            # Keep the original initial state if the solver fails for any reason.
            mean_ic_report = "Mean IC: failed (see console for details)"

    propagator = _build_propagator(propagation_cfg, initial_state, earth, atmosphere)

    station_contexts: List[_StationContext] = []
    if compute_passes and station_configs:
        # Pass detection is intentionally ungated by any elevation mask so that
        # downstream tools (e.g., link budget) can apply their own elevation
        # filtering bounds without needing to re-run propagation.
        min_elevation_rad = 0.0
        for station_cfg in station_configs:
            station_frame = _build_station_frame(station_cfg, earth)
            logger = EventsLogger()
            elevation_detector = ElevationDetector(
                station_frame
            ).withConstantElevation(min_elevation_rad)
            propagator.addEventDetector(logger.monitorDetector(elevation_detector))
            station_contexts.append(
                _StationContext(config=station_cfg, frame=station_frame, logger=logger)
            )

    scenario_duration_seconds = (
        config.scenario.end_time - config.scenario.start_time
    ).total_seconds()

    (
        times_hours,
        elevations_deg,
        track_points,
        orbit_elements,
        azimuths_deg,
        az_rate_deg_s,
        el_rate_deg_s,
        range_rate_mps,
        range_accel_mps2,
    ) = _sample_elevation_time_series(
        propagator=propagator,
        station_frames=[(ctx.config.name, ctx.frame) for ctx in station_contexts],
        start_date=start_date,
        end_date=end_date,
        sample_step=config.propagation.sample_step_seconds,
        earth=earth,
        atmosphere=atmosphere,
        total_duration=scenario_duration_seconds,
        progress_callback=progress_callback,
        drag_enabled=bool(getattr(propagation_cfg, "enable_drag", False)),
        drag_area_m2=float(getattr(propagation_cfg, "drag_area_m2", 1.0)),
        drag_cd=float(getattr(propagation_cfg, "drag_cd", 2.2)),
    )

    passes: List[PassStatistic] = []
    per_station_passes: dict[str, List[PassStatistic]] = {}
    next_index = 1
    for ctx in station_contexts:
        station_events = ctx.logger.getLoggedEvents()
        elevations = elevations_deg.get(ctx.config.name)
        if elevations is None:
            continue
        station_passes = _extract_pass_statistics(
            events=station_events,
            times_hours=times_hours,
            elevations_deg=elevations,
            start_date=start_date,
            utc=utc,
            station_name=ctx.config.name,
            start_index=next_index,
        )
        next_index += len(station_passes)
        passes.extend(station_passes)
        per_station_passes[ctx.config.name] = station_passes

    summary = _build_summary(
        passes=passes,
        scenario_duration_seconds=scenario_duration_seconds,
    )

    station_summaries = [
        StationSummary(
            station_name=name,
            total_passes=len(station_pass_list),
            total_access_minutes=float(
                sum(p.duration_minutes for p in station_pass_list)
            ),
        )
        for name, station_pass_list in per_station_passes.items()
    ]

    ground_track = [
        GroundTrackPoint(
            timestamp=_absolute_to_datetime(date, utc),
            latitude_deg=lat,
            longitude_deg=_wrap_longitude(lon),
            x_km=x / 1000.0,
            y_km=y / 1000.0,
            z_km=z / 1000.0,
        )
        for date, lat, lon, x, y, z in track_points
    ]

    timeline_seconds = (times_hours * 3600.0).tolist()
    station_series = {name: values.tolist() for name, values in elevations_deg.items()}
    station_az_series = {name: values.tolist() for name, values in azimuths_deg.items()}
    station_az_rate_series = {name: values.tolist() for name, values in az_rate_deg_s.items()}
    station_el_rate_series = {name: values.tolist() for name, values in el_rate_deg_s.items()}
    station_range_rate_series = {name: values.tolist() for name, values in range_rate_mps.items()}
    station_range_accel_series = {
        name: values.tolist() for name, values in range_accel_mps2.items()
    }
    semi_major_axis_m = config.orbit.semi_major_axis_km * 1000.0
    orbit_period_seconds = (
        2 * math.pi * math.sqrt(semi_major_axis_m**3 / Constants.WGS84_EARTH_MU)
    )

    return AnalysisResult(
        passes=passes,
        summary=summary,
        station_summaries=station_summaries,
        ground_track=ground_track,
        timeline_seconds=timeline_seconds,
        station_elevation_series=station_series,
        station_azimuth_series=station_az_series,
        station_az_rate_series=station_az_rate_series,
        station_el_rate_series=station_el_rate_series,
        station_range_rate_series=station_range_rate_series,
        station_range_accel_series=station_range_accel_series,
        orbit_period_seconds=float(orbit_period_seconds),
        mean_ic_report=mean_ic_report,
        orbital_altitude_km=orbit_elements["altitude_km"].tolist(),
        semi_major_axis_km=orbit_elements["semi_major_axis_km"].tolist(),
        perigee_altitude_km=orbit_elements["perigee_altitude_km"].tolist(),
        apogee_altitude_km=orbit_elements["apogee_altitude_km"].tolist(),
        eccentricity=orbit_elements["eccentricity"].tolist(),
        argument_of_perigee_deg=orbit_elements["argument_of_perigee_deg"].tolist(),
        orbital_period_series_s=orbit_elements["period_seconds"].tolist(),
        density_kg_m3=orbit_elements["density_kg_m3"].tolist(),
        dynamic_pressure_pa=orbit_elements["dynamic_pressure_pa"].tolist(),
        true_anomaly_deg=orbit_elements.get("true_anomaly_deg", []).tolist()
        if isinstance(orbit_elements.get("true_anomaly_deg"), np.ndarray)
        else list(orbit_elements.get("true_anomaly_deg", [])),
        drag_force_N=orbit_elements.get("drag_force_N", []).tolist()
        if isinstance(orbit_elements.get("drag_force_N"), np.ndarray)
        else list(orbit_elements.get("drag_force_N", [])),
    )


def _build_initial_orbit(
    config: AnalysisConfig, epoch: AbsoluteDate, frame
) -> KeplerianOrbit:
    """Create the Keplerian orbit from the user inputs."""

    orbit_cfg = config.orbit
    semi_major_axis = orbit_cfg.semi_major_axis_km * 1000.0
    inclination = math.radians(orbit_cfg.inclination_deg)
    raan = math.radians(orbit_cfg.raan_deg)
    arg_perigee = math.radians(orbit_cfg.arg_perigee_deg)
    mean_anomaly = math.radians(orbit_cfg.mean_anomaly_deg)

    return KeplerianOrbit(
        semi_major_axis,
        orbit_cfg.eccentricity,
        inclination,
        arg_perigee,
        raan,
        mean_anomaly,
        PositionAngleType.MEAN,
        frame,
        epoch,
        Constants.WGS84_EARTH_MU,
    )


def _build_earth() -> OneAxisEllipsoid:
    """Return the Earth model used for all stations."""

    itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
    return OneAxisEllipsoid(
        Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
        Constants.WGS84_EARTH_FLATTENING,
        itrf,
    )


def _build_atmosphere(earth: OneAxisEllipsoid) -> NRLMSISE00:
    """Create the NRLMSISE-00 atmosphere model used for drag and diagnostics."""

    msafe = MarshallSolarActivityFutureEstimation(
        MarshallSolarActivityFutureEstimation.DEFAULT_SUPPORTED_NAMES,
        MarshallSolarActivityFutureEstimation.StrengthLevel.AVERAGE,
    )
    sun = CelestialBodyFactory.getSun()
    return NRLMSISE00(msafe, sun, earth)


def _build_station_frame(
    station_cfg: GroundStationConfig, earth: OneAxisEllipsoid
) -> TopocentricFrame:
    """Create a TopocentricFrame for the provided station config."""

    station_point = GeodeticPoint(
        math.radians(station_cfg.latitude_deg),
        math.radians(station_cfg.longitude_deg),
        station_cfg.altitude_m,
    )

    return TopocentricFrame(earth, station_point, station_cfg.name)


def _build_propagator(
    propagation_cfg: PropagationConfig,
    initial_state: SpacecraftState,
    earth: OneAxisEllipsoid,
    atmosphere,
) -> Propagator:
    """Instantiate the requested propagator."""

    if propagation_cfg.propagator_type.lower() == "keplerian":
        return KeplerianPropagator(initial_state.getOrbit())

    min_step = 0.1
    max_step = 300.0
    position_tolerance = 10.0
    integrator = DormandPrince853Integrator(
        min_step, max_step, position_tolerance, position_tolerance
    )

    propagator = NumericalPropagator(integrator)
    propagator.setInitialState(initial_state)
    propagator.setAttitudeProvider(FrameAlignedProvider(initial_state.getFrame()))

    itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
    gravity_provider = GravityFieldFactory.getNormalizedProvider(10, 10)
    propagator.addForceModel(HolmesFeatherstoneAttractionModel(itrf, gravity_provider))

    if propagation_cfg.enable_drag:
        # Use configured spacecraft drag properties, falling back to legacy
        # defaults if they are missing for any reason.
        area_m2 = float(getattr(propagation_cfg, "drag_area_m2", 1.0))
        cd = float(getattr(propagation_cfg, "drag_cd", 2.2))
        propagator.addForceModel(DragForce(atmosphere, IsotropicDrag(area_m2, cd)))

    return propagator


def _sample_elevation_time_series(
    propagator: Propagator,
    station_frames: List[tuple[str, TopocentricFrame]],
    start_date: AbsoluteDate,
    end_date: AbsoluteDate,
    sample_step: float,
    earth: OneAxisEllipsoid,
    atmosphere,
    total_duration: float,
    progress_callback: Callable[[float], None] | None = None,
    *,
    drag_enabled: bool,
    drag_area_m2: float,
    drag_cd: float,
) -> tuple[
    np.ndarray,
    dict[str, np.ndarray],
    list[tuple[AbsoluteDate, float, float, float, float, float]],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    """Propagate the orbit and collect elevation and orbital element samples."""

    # Storage for sampled values.
    times_hours: List[float] = []
    elevations_deg: dict[str, List[float]] = {name: [] for name, _ in station_frames}
    azimuths_deg: dict[str, List[float]] = {name: [] for name, _ in station_frames}
    az_rate_deg_s: dict[str, List[float]] = {name: [] for name, _ in station_frames}
    el_rate_deg_s: dict[str, List[float]] = {name: [] for name, _ in station_frames}
    range_rates_mps: dict[str, List[float]] = {name: [] for name, _ in station_frames}
    range_accels_mps2: dict[str, List[float]] = {
        name: [] for name, _ in station_frames
    }
    track_points: list[tuple[AbsoluteDate, float, float, float, float, float]] = []

    altitude_km: List[float] = []
    semi_major_axis_km: List[float] = []
    perigee_altitude_km: List[float] = []
    apogee_altitude_km: List[float] = []
    eccentricity: List[float] = []
    argument_of_perigee_deg: List[float] = []
    true_anomaly_deg: List[float] = []
    orbital_period_s: List[float] = []
    density_kg_m3: List[float] = []
    dynamic_pressure_pa: List[float] = []
    drag_force_N: List[float] = []

    # Start from the propagator's initial state.
    try:
        state = propagator.getInitialState()
    except Exception:
        # Fallback: no accessible initial state; return empty diagnostics.
        return np.array([]), {}, [], {}

    current_date = state.getDate()
    elapsed_seconds = 0.0

    while current_date.compareTo(end_date) <= 0:
        # Sample geodetic position and orbital elements at the current state.
        geodetic = earth.transform(
            state.getPVCoordinates().getPosition(), state.getFrame(), current_date
        )
        alt_m = float(geodetic.getAltitude())
        if alt_m < _REENTRY_CUTOFF_ALTITUDE_M:
            break

        pv_earth = state.getPVCoordinates(earth.getBodyFrame())
        position = pv_earth.getPosition()
        track_points.append(
            (
                state.getDate(),
                math.degrees(geodetic.getLatitude()),
                math.degrees(geodetic.getLongitude()),
                position.getX(),
                position.getY(),
                position.getZ(),
            )
        )

        times_hours.append(elapsed_seconds / 3600.0)
        altitude_km.append(alt_m / 1000.0)

        # Atmospheric density and dynamic pressure (assuming co-rotation with Earth).
        try:
            rho = float(
                atmosphere.getDensity(state.getDate(), position, earth.getBodyFrame())
            )
        except Exception:
            rho = float("nan")
        density_kg_m3.append(rho)
        v_vec = pv_earth.getVelocity()
        v_norm = v_vec.getNorm()
        q = 0.5 * rho * v_norm * v_norm if rho == rho else float("nan")
        dynamic_pressure_pa.append(q)

        # Drag force magnitude (approx): q * Cd * A.
        if drag_enabled:
            if math.isfinite(q):
                drag_force_N.append(float(q) * float(drag_area_m2) * float(drag_cd))
            else:
                drag_force_N.append(float("nan"))
        else:
            drag_force_N.append(0.0)

        # Osculating Keplerian elements.
        kepler = KeplerianOrbit(state.getOrbit())
        a_m = kepler.getA()
        semi_major_axis_km.append(a_m / 1000.0)
        e = kepler.getE()
        ecc = float(e)
        eccentricity.append(ecc)
        r_perigee = a_m * (1.0 - ecc)
        r_apogee = a_m * (1.0 + ecc)
        perigee_altitude_km.append(r_perigee / 1000.0)
        apogee_altitude_km.append(r_apogee / 1000.0)
        argument_of_perigee_deg.append(math.degrees(kepler.getPerigeeArgument()))
        true_anomaly_deg.append(math.degrees(kepler.getTrueAnomaly()))
        period_sample = float(kepler.getKeplerianPeriod())
        orbital_period_s.append(period_sample)

        # Station look angles and dynamics.
        for name, frame in station_frames:
            pv_topo = state.getPVCoordinates(frame)
            pos_topo = pv_topo.getPosition()
            vel_topo = pv_topo.getVelocity()
            acc_topo = pv_topo.getAcceleration()
            x = pos_topo.getX()
            y = pos_topo.getY()
            z = pos_topo.getZ()
            vx = vel_topo.getX()
            vy = vel_topo.getY()
            vz = vel_topo.getZ()
            ax = acc_topo.getX()
            ay = acc_topo.getY()
            az = acc_topo.getZ()
            xy_norm = math.hypot(x, y)
            rho_sq = x * x + y * y + z * z
            rho = math.sqrt(rho_sq) if rho_sq > 0.0 else 0.0
            elevation_rad = math.atan2(z, xy_norm)
            elevations_deg[name].append(math.degrees(elevation_rad))
            azimuth_rad = math.atan2(x, y)
            if azimuth_rad < 0.0:
                azimuth_rad += 2.0 * math.pi
            azimuths_deg[name].append(math.degrees(azimuth_rad))
            if xy_norm > 1e-3:
                az_rate = (y * vx - x * vy) / (xy_norm * xy_norm)
            else:
                az_rate = 0.0
            az_rate_deg_s[name].append(math.degrees(az_rate))
            if rho_sq > 1e-6 and xy_norm > 1e-3:
                rh_dot = (x * vx + y * vy) / xy_norm
                el_rate = (vz * xy_norm - z * rh_dot) / rho_sq
            else:
                el_rate = 0.0
            el_rate_deg_s[name].append(math.degrees(el_rate))
            if rho > 1e-3:
                pos_dot_vel = x * vx + y * vy + z * vz
                range_rate = pos_dot_vel / rho
                vel_sq = vx * vx + vy * vy + vz * vz
                pos_dot_acc = x * ax + y * ay + z * az
                range_accel = (vel_sq + pos_dot_acc) / rho - (pos_dot_vel * pos_dot_vel) / (
                    rho * rho * rho
                )
            else:
                range_rate = 0.0
                range_accel = 0.0
            range_rates_mps[name].append(range_rate)
            range_accels_mps2[name].append(range_accel)

        if progress_callback and total_duration > 0:
            progress = max(0.0, min(100.0, (elapsed_seconds / total_duration) * 100.0))
            progress_callback(progress)

        # Advance to the next sample.
        if current_date.compareTo(end_date) >= 0:
            break

        next_date = current_date.shiftedBy(sample_step)
        try:
            state = propagator.propagate(next_date)
        except Exception:
            break

        current_date = state.getDate()
        elapsed_seconds = (current_date.durationFrom(start_date))  # seconds since start

    if progress_callback:
        progress_callback(100.0)

    orbit_elements = {
        "altitude_km": np.array(altitude_km, dtype=float),
        "semi_major_axis_km": np.array(semi_major_axis_km, dtype=float),
        "perigee_altitude_km": np.array(perigee_altitude_km, dtype=float),
        "apogee_altitude_km": np.array(apogee_altitude_km, dtype=float),
        "eccentricity": np.array(eccentricity, dtype=float),
        "argument_of_perigee_deg": np.array(argument_of_perigee_deg, dtype=float),
        "true_anomaly_deg": np.array(true_anomaly_deg, dtype=float),
        "period_seconds": np.array(orbital_period_s, dtype=float),
        "density_kg_m3": np.array(density_kg_m3, dtype=float),
        "dynamic_pressure_pa": np.array(dynamic_pressure_pa, dtype=float),
        "drag_force_N": np.array(drag_force_N, dtype=float),
    }

    return (
        np.array(times_hours),
        {name: np.array(values) for name, values in elevations_deg.items()},
        track_points,
        orbit_elements,
        {name: np.array(values) for name, values in azimuths_deg.items()},
        {name: np.array(values) for name, values in az_rate_deg_s.items()},
        {name: np.array(values) for name, values in el_rate_deg_s.items()},
        {name: np.array(values) for name, values in range_rates_mps.items()},
        {name: np.array(values) for name, values in range_accels_mps2.items()},
    )


def _extract_pass_statistics(
    events,
    times_hours: np.ndarray,
    elevations_deg: np.ndarray,
    start_date: AbsoluteDate,
    utc: TimeScale,
    station_name: str,
    start_index: int,
) -> List[PassStatistic]:
    """Convert logged elevation detector events into pass statistics."""

    passes: List[PassStatistic] = []
    index = start_index
    i = 0
    while i < events.size():
        event = events.get(i)
        if event.isIncreasing() and i + 1 < events.size():
            next_event = events.get(i + 1)
            if not next_event.isIncreasing():
                aos_seconds = event.getState().getDate().durationFrom(start_date)
                los_seconds = next_event.getState().getDate().durationFrom(start_date)
                aos_hours = aos_seconds / 3600.0
                los_hours = los_seconds / 3600.0

                max_elev = _max_elevation_between(
                    times_hours, elevations_deg, aos_hours, los_hours
                )

                passes.append(
                    PassStatistic(
                        index=index,
                        aos=_absolute_to_datetime(event.getState().getDate(), utc),
                        los=_absolute_to_datetime(next_event.getState().getDate(), utc),
                        duration_minutes=(los_seconds - aos_seconds) / 60.0,
                        max_elevation_deg=max_elev,
                        station_name=station_name,
                    )
                )
                index += 1
                i += 2
                continue
        i += 1

    return passes


def _max_elevation_between(
    times_hours: np.ndarray,
    elevations_deg: np.ndarray,
    aos_hours: float,
    los_hours: float,
) -> float:
    """Compute the peak elevation in the provided interval."""

    mask = (times_hours >= aos_hours) & (times_hours <= los_hours)
    if np.any(mask):
        return float(np.max(elevations_deg[mask]))
    return float("-inf")


def _build_summary(
    passes: List[PassStatistic], scenario_duration_seconds: float
) -> AnalysisSummary:
    """Create aggregated statistics for display in the GUI."""

    if not passes:
        return AnalysisSummary(
            total_passes=0,
            total_access_minutes=0.0,
            coverage_percent=0.0,
            avg_duration_minutes=0.0,
            min_duration_minutes=0.0,
            max_duration_minutes=0.0,
        )

    durations = np.array([p.duration_minutes for p in passes])
    total_access_minutes = float(np.sum(durations))
    coverage_percent = (
        100.0 * (total_access_minutes * 60.0) / scenario_duration_seconds
        if scenario_duration_seconds > 0
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


def _to_absolute_date(moment: datetime, utc: TimeScale) -> AbsoluteDate:
    """Convert a timezone-aware datetime into an Orekit AbsoluteDate."""

    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    moment = moment.astimezone(timezone.utc)

    return AbsoluteDate(
        moment.year,
        moment.month,
        moment.day,
        moment.hour,
        moment.minute,
        float(moment.second + moment.microsecond / 1_000_000.0),
        utc,
    )


def _absolute_to_datetime(date: AbsoluteDate, utc: TimeScale) -> datetime:
    """Convert AbsoluteDate back into a Python datetime (UTC)."""

    java_date = date.toDate(utc)
    return datetime.fromtimestamp(java_date.getTime() / 1000.0, tz=timezone.utc)


def _wrap_longitude(lon: float) -> float:
    """Normalize longitude into [-180, 180) degrees."""

    wrapped = ((lon + 180.0) % 360.0) - 180.0
    return wrapped if wrapped != -180.0 else 180.0
