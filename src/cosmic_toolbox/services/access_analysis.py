"""Orekit-backed ground-station access analysis service."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from functools import wraps
from typing import Callable, List

import numpy as np

import orekitdata  # type: ignore

from cosmic_toolbox.models import (
    AnalysisConfig,
    AnalysisResult,
    GroundStationConfig,
    PropagatedEphemeris,
    PropagationConfig,
)

from cosmic_toolbox.services.cancel import CancelToken
from cosmic_toolbox.services.orekit_vm import ensure_orekit_vm_started
from cosmic_toolbox.services.progress import (
    ProgressReporter,
    legacy_callback_from_progress,
    progress_from_legacy_callback,
)

_OREKIT_BOOTSTRAPPED = False

# NOTE:
# JCC-backed Java packages (java.*, org.orekit.*, org.hipparchus.*) are only importable
# after the JVM is started (orekit.initVM()). Therefore, we import them lazily from
# ensure_orekit_bootstrapped() and bind them into module globals.

File = None  # type: ignore[assignment]
Vector3D = None  # type: ignore[assignment]
AlignedAndConstrained = None  # type: ignore[assignment]
AttitudesSwitcher = None  # type: ignore[assignment]
GroundPointTarget = None  # type: ignore[assignment]
PredefinedTarget = None  # type: ignore[assignment]
CelestialBodyFactory = None  # type: ignore[assignment]
GeodeticPoint = None  # type: ignore[assignment]
OneAxisEllipsoid = None  # type: ignore[assignment]
DataContext = None  # type: ignore[assignment]
DirectoryCrawler = None  # type: ignore[assignment]
GravityFieldFactory = None  # type: ignore[assignment]
FramesFactory = None  # type: ignore[assignment]
TopocentricFrame = None  # type: ignore[assignment]
KeplerianOrbit = None  # type: ignore[assignment]
PositionAngleType = None  # type: ignore[assignment]
Propagator = None  # type: ignore[assignment]
SpacecraftState = None  # type: ignore[assignment]
KeplerianPropagator = None  # type: ignore[assignment]
ElevationDetector = None  # type: ignore[assignment]
EventsLogger = None  # type: ignore[assignment]
BrouwerLyddanePropagator = None  # type: ignore[assignment]
AbsoluteDate = None  # type: ignore[assignment]
TimeScale = None  # type: ignore[assignment]
TimeScalesFactory = None  # type: ignore[assignment]
Constants = None  # type: ignore[assignment]
IERSConventions = None  # type: ignore[assignment]

def _import_orekit_symbols() -> None:
    global File
    global Vector3D
    global AlignedAndConstrained
    global AttitudesSwitcher
    global GroundPointTarget
    global PredefinedTarget
    global CelestialBodyFactory
    global GeodeticPoint
    global OneAxisEllipsoid
    global DataContext
    global DirectoryCrawler
    global GravityFieldFactory
    global FramesFactory
    global TopocentricFrame
    global KeplerianOrbit
    global PositionAngleType
    global Propagator
    global SpacecraftState
    global KeplerianPropagator
    global ElevationDetector
    global EventsLogger
    global BooleanDetector
    global PythonAbstractDetector
    global BrouwerLyddanePropagator
    global AbsoluteDate
    global TimeScale
    global TimeScalesFactory
    global Constants
    global IERSConventions
    global ContinueOnEvent

    if File is not None:
        return

    from java.io import File as _File
    from org.hipparchus.geometry.euclidean.threed import Vector3D as _Vector3D
    from org.orekit.attitudes import (
        AlignedAndConstrained as _AlignedAndConstrained,
        AttitudesSwitcher as _AttitudesSwitcher,
        GroundPointTarget as _GroundPointTarget,
        PredefinedTarget as _PredefinedTarget,
    )
    from org.orekit.bodies import (
        CelestialBodyFactory as _CelestialBodyFactory,
        GeodeticPoint as _GeodeticPoint,
        OneAxisEllipsoid as _OneAxisEllipsoid,
    )
    from org.orekit.data import DataContext as _DataContext, DirectoryCrawler as _DirectoryCrawler
    from org.orekit.forces.gravity.potential import GravityFieldFactory as _GravityFieldFactory
    from org.orekit.frames import FramesFactory as _FramesFactory, TopocentricFrame as _TopocentricFrame
    from org.orekit.orbits import (
        KeplerianOrbit as _KeplerianOrbit,
        PositionAngleType as _PositionAngleType,
    )
    from org.orekit.propagation import Propagator as _Propagator, SpacecraftState as _SpacecraftState
    from org.orekit.propagation.analytical import (
        BrouwerLyddanePropagator as _BrouwerLyddanePropagator,
        KeplerianPropagator as _KeplerianPropagator,
    )
    from org.orekit.propagation.events import (
        BooleanDetector as _BooleanDetector,
        ElevationDetector as _ElevationDetector,
        EventsLogger as _EventsLogger,
        PythonAbstractDetector as _PythonAbstractDetector,
    )
    from org.orekit.propagation.events.handlers import ContinueOnEvent as _ContinueOnEvent
    from org.orekit.time import AbsoluteDate as _AbsoluteDate, TimeScale as _TimeScale, TimeScalesFactory as _TimeScalesFactory
    from org.orekit.utils import Constants as _Constants, IERSConventions as _IERSConventions

    File = _File
    Vector3D = _Vector3D
    AlignedAndConstrained = _AlignedAndConstrained
    AttitudesSwitcher = _AttitudesSwitcher
    GroundPointTarget = _GroundPointTarget
    PredefinedTarget = _PredefinedTarget
    CelestialBodyFactory = _CelestialBodyFactory
    GeodeticPoint = _GeodeticPoint
    OneAxisEllipsoid = _OneAxisEllipsoid
    DataContext = _DataContext
    DirectoryCrawler = _DirectoryCrawler
    GravityFieldFactory = _GravityFieldFactory
    FramesFactory = _FramesFactory
    TopocentricFrame = _TopocentricFrame
    KeplerianOrbit = _KeplerianOrbit
    PositionAngleType = _PositionAngleType
    Propagator = _Propagator
    SpacecraftState = _SpacecraftState
    KeplerianPropagator = _KeplerianPropagator
    ElevationDetector = _ElevationDetector
    EventsLogger = _EventsLogger
    BooleanDetector = _BooleanDetector
    PythonAbstractDetector = _PythonAbstractDetector
    BrouwerLyddanePropagator = _BrouwerLyddanePropagator
    AbsoluteDate = _AbsoluteDate
    TimeScale = _TimeScale
    TimeScalesFactory = _TimeScalesFactory
    Constants = _Constants
    IERSConventions = _IERSConventions
    ContinueOnEvent = _ContinueOnEvent

# Altitude below which we consider the spacecraft to have re-entered and stop
# sampling / propagation-driven diagnostics. This is expressed in meters above
# the WGS84 ellipsoid.
_REENTRY_CUTOFF_ALTITUDE_M = 80_000.0


def _enable_station_pointing_during_contact(
    *,
    propagator: Propagator,
    station_cfg: GroundStationConfig,
    station_frame: TopocentricFrame,
    inertial_frame,
    earth: OneAxisEllipsoid,
    contact_elevation_rad: float,
    gate_enabled: bool,
    max_off_nadir_slew_deg: float | None,
) -> None:
    """Switch attitude prograde <-> station-pointing based on elevation crossings.

    - Default mode: body +Z "down" (nadir), +X "forward" (velocity-constrained).
    - In contact: body +Z points to the station line-of-sight (ground point target).

    The switch is triggered by an ElevationDetector crossing the configured
    elevation threshold.
    """

    if contact_elevation_rad <= -math.pi / 2 or contact_elevation_rad >= math.pi / 2:
        return

    # Default: +Z nadir, +X as close as possible to velocity while ⟂ +Z.
    prograde = AlignedAndConstrained(
        Vector3D.PLUS_K,
        PredefinedTarget.NADIR,
        Vector3D.PLUS_I,
        PredefinedTarget.VELOCITY,
        inertial_frame,
        CelestialBodyFactory.getSun(),
        earth,
    )

    # Contact: point body +Z at the station.
    station_point = GeodeticPoint(
        math.radians(station_cfg.latitude_deg),
        math.radians(station_cfg.longitude_deg),
        station_cfg.altitude_m,
    )
    try:
        station_location = earth.transform(station_point)
    except Exception as exc:
        raise RuntimeError(
            "Failed to transform station geodetic point to cartesian location for "
            f"station={getattr(station_cfg, 'name', None)!r}, "
            f"lat_deg={station_cfg.latitude_deg!r}, lon_deg={station_cfg.longitude_deg!r}, "
            f"altitude_m={station_cfg.altitude_m!r}"
        ) from exc

    station_pointing = AlignedAndConstrained(
        # Primary: align spacecraft +Z with line-of-sight to the station.
        Vector3D.PLUS_K,
        GroundPointTarget(station_location),
        # Secondary: constrain spacecraft +X to lie in the plane formed by the
        # target line-of-sight (+Z) and the inertial velocity vector.
        Vector3D.PLUS_I,
        PredefinedTarget.VELOCITY,
        inertial_frame,
        CelestialBodyFactory.getSun(),
        earth,
    )

    # Use AttitudesSwitcher (instantaneous, but robust in the Python bindings).
    # The same detector instance is used for both directions; switches are
    # gated by switchOnIncrease / switchOnDecrease.
    elevation_detector = ElevationDetector(station_frame).withConstantElevation(
        float(contact_elevation_rad)
    )

    detector = elevation_detector
    if gate_enabled:
        if max_off_nadir_slew_deg is None:
            raise ValueError(
                "Sensor-FOV gating is enabled but max_off_nadir_slew_deg is not set."
            )
        ms = float(max_off_nadir_slew_deg)
        if not (math.isfinite(ms) and ms >= 0.0):
            raise ValueError(f"Invalid max_off_nadir_slew_deg: {max_off_nadir_slew_deg!r}")
        max_slew_rad = math.radians(ms)

        class _OffNadirSlewEnvelopeDetector(PythonAbstractDetector):
            """Event detector: g>0 when station is within max off-nadir slew envelope.

            g(state) = max_slew_rad - angle(nadir, los_to_station)
            """

            def __init__(
                self,
                *,
                station_frame: TopocentricFrame,
                inertial_frame,
                max_slew_rad: float,
                max_check: float,
                threshold: float,
                max_iter: int,
                handler,
            ) -> None:
                super().__init__(float(max_check), float(threshold), int(max_iter), handler)
                self._station_frame = station_frame
                self._inertial_frame = inertial_frame
                self._max_slew_rad = float(max_slew_rad)

            def g(self, state: SpacecraftState) -> float:  # type: ignore[override]
                pv_sc = state.getPVCoordinates(self._inertial_frame)
                r_sc = pv_sc.getPosition()
                r_gs = self._station_frame.getPVCoordinates(
                    state.getDate(), self._inertial_frame
                ).getPosition()
                los = r_gs.subtract(r_sc)
                nadir = r_sc.scalarMultiply(-1.0)
                off_nadir = float(Vector3D.angle(nadir, los))
                return float(self._max_slew_rad - off_nadir)

            def create(self, newMaxCheck, newThreshold, newMaxIter, newHandler):  # type: ignore[override]
                # Orekit clones detectors through this hook when adjusting settings.
                return _OffNadirSlewEnvelopeDetector(
                    station_frame=self._station_frame,
                    inertial_frame=self._inertial_frame,
                    max_slew_rad=self._max_slew_rad,
                    max_check=float(newMaxCheck),
                    threshold=float(newThreshold),
                    max_iter=int(newMaxIter),
                    handler=newHandler,
                )

        slew_detector = _OffNadirSlewEnvelopeDetector(
            station_frame=station_frame,
            inertial_frame=inertial_frame,
            max_slew_rad=max_slew_rad,
            max_check=60.0,
            threshold=1e-6,
            max_iter=100,
            handler=ContinueOnEvent(),
        )
        detector = BooleanDetector.andCombine([elevation_detector, slew_detector])

    switcher = AttitudesSwitcher()
    switcher.addSwitchingCondition(prograde, station_pointing, detector, True, False, None)
    switcher.addSwitchingCondition(station_pointing, prograde, detector, False, True, None)
    switcher.resetActiveProvider(prograde)
    propagator.setAttitudeProvider(switcher)


def _angle_of_attack_deg(
    state: SpacecraftState,
    inertial_frame,
    *,
    attitude=None,
) -> float:
    """Angle between body +X and inertial velocity direction (ECI), in degrees.

    Definition: angle( x_body(ECI), v_hat(ECI) ).

    Notes:
    - If an attitude is available, we transform the spacecraft-frame +X axis
      into the inertial frame using the attitude rotation.
    - If attitude is unavailable (e.g. some analytical propagators), we fall
      back to assuming body frame is aligned with the inertial frame.
    """

    pv_eci = state.getPVCoordinates(inertial_frame)
    v_eci = pv_eci.getVelocity()
    v_norm = float(v_eci.getNorm())
    if not np.isfinite(v_norm) or v_norm <= 0.0:
        raise ValueError(f"Invalid inertial velocity norm for AoA computation: {v_norm!r}")

    # In our Orekit Python bindings, Attitude rotation maps reference-frame
    # vectors -> spacecraft-frame vectors (verified with LofOffset(TNW)).
    try:
        att = state.getAttitude() if attitude is None else attitude
        rot = att.getRotation()
    except Exception as exc:
        raise RuntimeError("Failed to access attitude/rotation for AoA computation") from exc

    v_body = rot.applyTo(v_eci)
    vn = float(v_body.getNorm())
    if not np.isfinite(vn) or vn <= 0.0:
        raise ValueError(f"Invalid body-frame velocity norm for AoA computation: {vn!r}")
    v_hat_body = v_body.scalarMultiply(1.0 / vn)
    dot = float(Vector3D.dotProduct(Vector3D.PLUS_I, v_hat_body))
    dot = max(-1.0, min(1.0, dot))
    return float(math.degrees(math.acos(dot)))


def ensure_orekit_bootstrapped() -> None:
    """Load the Orekit data files exactly once."""

    global _OREKIT_BOOTSTRAPPED
    if _OREKIT_BOOTSTRAPPED:
        return

    # Start JVM (and attach thread) before touching Orekit data providers.
    ensure_orekit_vm_started()
    _import_orekit_symbols()

    data_manager = DataContext.getDefault().getDataProvidersManager()
    crawler = DirectoryCrawler(File(orekitdata.__path__[0]))
    data_manager.addProvider(crawler)
    _OREKIT_BOOTSTRAPPED = True


def _attach_thread(func: Callable[..., object]) -> Callable[..., object]:
    """Ensure Orekit's JVM is attached for the current thread."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        # `orekit.getVMEnv()` may be None until `orekit.initVM()` has been called.
        # Use our helper which starts the JVM (once) and safely attaches the thread.
        ensure_orekit_vm_started()
        return func(*args, **kwargs)

    return wrapper


def _build_ephemeris_from_samples(
    *,
    config: AnalysisConfig,
    utc,
    track_points,
    state_vector_points,
    orbit_elements: dict[str, np.ndarray],
    times_hours: np.ndarray,
    mean_ic_report: str | None,
) -> PropagatedEphemeris:
    """Build the orbit-only ephemeris cache from sampled propagation outputs."""
    # Build columnar numpy arrays directly — no intermediate Python dataclasses.
    timestamps_unix = np.array(
        [_absolute_to_datetime(tp[0], utc).timestamp() for tp in track_points],
        dtype=np.float64,
    )
    gt_lat_deg = np.array([tp[1] for tp in track_points], dtype=np.float64)
    gt_lon_deg = np.array([_wrap_longitude(tp[2]) for tp in track_points], dtype=np.float64)

    sv_raw = np.array(
        [p[1:] for p in state_vector_points],
        dtype=np.float64,
    )
    # Columns: eci_x_m, eci_y_m, eci_z_m, eci_vx_mps, eci_vy_mps, eci_vz_mps,
    #          ecef_x_m, ecef_y_m, ecef_z_m, ecef_vx_mps, ecef_vy_mps, ecef_vz_mps,
    #          bx_x, bx_y, bx_z, by_x, by_y, by_z, bz_x, bz_y, bz_z
    eci_pos_km = sv_raw[:, 0:3] / 1000.0
    eci_vel_km_s = sv_raw[:, 3:6] / 1000.0
    ecef_pos_km = sv_raw[:, 6:9] / 1000.0
    ecef_vel_km_s = sv_raw[:, 9:12] / 1000.0
    body_x_ecef = sv_raw[:, 12:15]
    body_y_ecef = sv_raw[:, 15:18]
    body_z_ecef = sv_raw[:, 18:21]

    semi_major_axis_m = config.orbit.semi_major_axis_km * 1000.0
    orbit_period_seconds = (
        2 * math.pi * math.sqrt(semi_major_axis_m**3 / Constants.WGS84_EARTH_MU)
    )
    return PropagatedEphemeris(
        timeline_seconds=np.asarray(times_hours, dtype=float) * 3600.0,
        orbit_period_seconds=float(orbit_period_seconds),
        mean_ic_report=mean_ic_report,
        timestamps_unix=timestamps_unix,
        eci_pos_km=eci_pos_km,
        eci_vel_km_s=eci_vel_km_s,
        ecef_pos_km=ecef_pos_km,
        ecef_vel_km_s=ecef_vel_km_s,
        body_x_ecef=body_x_ecef,
        body_y_ecef=body_y_ecef,
        body_z_ecef=body_z_ecef,
        gt_lat_deg=gt_lat_deg,
        gt_lon_deg=gt_lon_deg,
        orbital_altitude_km=orbit_elements["altitude_km"],
        semi_major_axis_km=orbit_elements["semi_major_axis_km"],
        perigee_altitude_km=orbit_elements["perigee_altitude_km"],
        apogee_altitude_km=orbit_elements["apogee_altitude_km"],
        eccentricity=orbit_elements["eccentricity"],
        inclination_deg=orbit_elements["inclination_deg"],
        argument_of_perigee_deg=orbit_elements["argument_of_perigee_deg"],
        orbital_period_series_s=orbit_elements["period_seconds"],
        true_anomaly_deg=orbit_elements.get("true_anomaly_deg", np.empty(0, dtype=float)),
        angle_of_attack_deg=orbit_elements.get("angle_of_attack_deg", np.empty(0, dtype=float)),
    )


@_attach_thread
def run_access_analysis(
    config: AnalysisConfig,
    stations: List[GroundStationConfig] | None = None,
    *,
    progress: ProgressReporter | None = None,
    cancel: CancelToken | None = None,
    progress_callback: Callable[[float], None] | None = None,
) -> AnalysisResult:
    """Propagate the orbit and return multi-station pass statistics.

    The Orekit propagation loop collects state vectors only; all per-station
    geometry (elevation, azimuth, range rate, pass detection) is derived
    post-propagation via the vectorised NumPy path in
    ``cached_access_recompute``.  This keeps the hot JVM loop as thin as
    possible and means adding or changing stations never touches the propagator.
    """

    ensure_orekit_bootstrapped()

    eff_progress: ProgressReporter | None = progress
    if eff_progress is None and progress_callback is not None:
        eff_progress = progress_from_legacy_callback(progress_callback)
    legacy_cb = legacy_callback_from_progress(eff_progress)

    utc = TimeScalesFactory.getUTC()
    start_date = _to_absolute_date(config.scenario.start_time, utc)
    end_date = _to_absolute_date(config.scenario.end_time, utc)

    propagation_cfg = config.propagation
    propagator_type = str(getattr(propagation_cfg, "propagator_type", "")).lower()
    if propagator_type not in ("keplerian", "brouwer_lyddane"):
        raise ValueError(
            f"Unsupported propagator_type={propagation_cfg.propagator_type!r}. "
            "Expected 'keplerian' or 'brouwer_lyddane'."
        )

    # BrouwerLyddane's initial orbit is interpreted as osculating, but scenario
    # inputs specify the desired *mean* semi-major axis. Apply the closed-form
    # J2 short-period correction so the propagated mean orbit matches the
    # requested target instead of drifting by twice the correction magnitude.
    apply_correction = propagator_type == "brouwer_lyddane" and bool(
        getattr(propagation_cfg, "apply_mean_orbit_correction", True)
    )
    osculating_sma_km = float(config.orbit.semi_major_axis_km)
    mean_ic_report: str | None = None
    if apply_correction:
        delta_a_km = _brouwer_lyddane_mean_orbit_correction_km(
            mean_semi_major_axis_km=float(config.orbit.semi_major_axis_km),
            inclination_deg=float(config.orbit.inclination_deg),
            arg_perigee_deg=float(config.orbit.arg_perigee_deg),
            mean_anomaly_deg=float(config.orbit.mean_anomaly_deg),
        )
        osculating_sma_km = float(config.orbit.semi_major_axis_km) + delta_a_km
        mean_ic_report = (
            f"Mean-orbit correction: applied (da_sp={delta_a_km:+.3f} km, "
            f"a_osc={osculating_sma_km:.3f} km, "
            f"target mean a={float(config.orbit.semi_major_axis_km):.3f} km)"
        )
    elif propagator_type == "brouwer_lyddane":
        mean_ic_report = "Mean-orbit correction: disabled"

    inertial_frame = FramesFactory.getEME2000()
    orbit = _build_initial_orbit(
        config, start_date, inertial_frame, semi_major_axis_km=osculating_sma_km
    )
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

    propagator = _build_propagator(propagation_cfg, initial_state, earth)

    # Attach contact-attitude switching for the primary station so that body
    # axes are computed correctly during contact windows.  All station geometry
    # (elevation, azimuth, passes) is derived post-propagation via the
    # vectorised NumPy path; only the attitude switcher needs a station frame
    # at propagation time.
    attitude_mode = str(getattr(propagation_cfg, "attitude_mode", "prograde") or "prograde").lower()
    enable_switching = bool(getattr(propagation_cfg, "enable_contact_attitude_switching", True))
    if station_configs and enable_switching and attitude_mode != "nadir":
        if propagation_cfg.propagator_type.lower() == "brouwer_lyddane":
            # Orekit's BrouwerLyddanePropagator.resetIntermediateState() -- invoked
            # automatically whenever an event detector (the contact-attitude
            # switcher below) resets state mid-propagation -- always uses a fixed
            # internal tolerance/iteration cap (1e-13, 200 iterations) for its
            # osculating<->mean fixed-point conversion, with no public way to
            # loosen it. That conversion is prone to non-convergence for
            # near-circular orbits, so contact-attitude switching is not
            # supported with this propagator.
            raise ValueError(
                "enable_contact_attitude_switching=True is not supported with "
                "propagator_type='brouwer_lyddane': Orekit's BrouwerLyddanePropagator "
                "can fail to converge its internal osculating<->mean conversion on "
                "every attitude-switch event reset. Set "
                "enable_contact_attitude_switching=False (or attitude_mode='nadir', "
                "or leave ground stations unconfigured) to skip attitude switching, "
                "or use propagator_type='keplerian' if contact-attitude switching "
                "is required."
            )
        contact_elevation_rad = math.radians(
            float(getattr(propagation_cfg, "contact_elevation_deg", 10.0))
        )
        primary_frame = _build_station_frame(station_configs[0], earth)
        _enable_station_pointing_during_contact(
            propagator=propagator,
            station_cfg=station_configs[0],
            station_frame=primary_frame,
            inertial_frame=inertial_frame,
            earth=earth,
            contact_elevation_rad=contact_elevation_rad,
            gate_enabled=False,
            max_off_nadir_slew_deg=None,
        )

    scenario_duration_seconds = (
        config.scenario.end_time - config.scenario.start_time
    ).total_seconds()

    # Propagate – collect state vectors, ground track and orbital elements only.
    # Per-station geometry is intentionally absent from this loop; see docstring.
    (
        times_hours,
        track_points,
        state_vector_points,
        orbit_elements,
    ) = _propagate_and_collect_state_vectors(
        propagator=propagator,
        start_date=start_date,
        end_date=end_date,
        sample_step=config.propagation.sample_step_seconds,
        inertial_frame=inertial_frame,
        earth=earth,
        total_duration=scenario_duration_seconds,
        progress_callback=legacy_cb,
        cancel=cancel,
    )

    ephemeris = _build_ephemeris_from_samples(
        config=config,
        utc=utc,
        track_points=track_points,
        state_vector_points=state_vector_points,
        orbit_elements=orbit_elements,
        times_hours=times_hours,
        mean_ic_report=mean_ic_report,
    )

    # Derive all station-dependent outputs from the ephemeris via the same
    # vectorised NumPy path used by the cached fast path.  This is equivalent
    # in accuracy to the former in-loop sampling for pass geometry, and is
    # substantially faster because it avoids JVM frame-transform round-trips
    # at every propagation timestep.
    from cosmic_toolbox.services.cached_access_recompute import (
        analysis_result_from_components,
        derive_access_results_from_ephemeris,
    )

    derived_access = derive_access_results_from_ephemeris(
        ephemeris=ephemeris,
        config=config,
        stations=station_configs if station_configs else None,
    )

    return analysis_result_from_components(ephemeris=ephemeris, derived_access=derived_access)


_WGS84_J2 = 1.08262668355e-3
_WGS84_EQUATORIAL_RADIUS_KM = 6378.137


def _brouwer_lyddane_mean_orbit_correction_km(
    mean_semi_major_axis_km: float,
    inclination_deg: float,
    arg_perigee_deg: float,
    mean_anomaly_deg: float,
) -> float:
    """J2 short-period semi-major-axis correction ``delta_a_sp`` at the initial epoch.

    The osculating SMA to feed BrouwerLyddanePropagator is
    ``a_osc = a_mean_target + delta_a_sp``. Matches
    ``BrouwerLyddanePropagator.computeMeanOrbit()`` to <0.01 km.

        delta_a_sp(u) = (3/2) * J2 * (Re/a)^2 * a * sin^2(i) * cos(2u)
    """
    u_rad = math.radians(arg_perigee_deg + mean_anomaly_deg)
    i_rad = math.radians(inclination_deg)
    delta_a_max = (
        1.5
        * _WGS84_J2
        * (_WGS84_EQUATORIAL_RADIUS_KM / mean_semi_major_axis_km) ** 2
        * mean_semi_major_axis_km
        * math.sin(i_rad) ** 2
    )
    return delta_a_max * math.cos(2.0 * u_rad)


def _build_initial_orbit(
    config: AnalysisConfig,
    epoch: AbsoluteDate,
    frame,
    *,
    semi_major_axis_km: float | None = None,
) -> KeplerianOrbit:
    """Create the Keplerian (osculating) orbit from the user inputs.

    ``semi_major_axis_km`` overrides ``config.orbit.semi_major_axis_km`` when
    provided, which lets callers feed BrouwerLyddane a corrected osculating SMA
    while keeping the other elements (inclination, RAAN, ...) untouched.
    """

    orbit_cfg = config.orbit
    semi_major_axis = (
        semi_major_axis_km if semi_major_axis_km is not None else orbit_cfg.semi_major_axis_km
    ) * 1000.0
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
) -> Propagator:
    """Instantiate the requested propagator."""

    attitude_mode = str(getattr(propagation_cfg, "attitude_mode", "prograde") or "prograde").lower()

    # Body-frame convention used throughout this repo:
    # - +X: forward
    # - +Z: down (nadir in nominal Earth-pointing modes)
    # - +Y: completes right-handed set
    #
    # For "prograde", we keep +Z as close to "down" as possible while making +X
    # track the velocity direction (by constraining +X in the (nadir, velocity) plane).
    def _prograde_provider():
        return AlignedAndConstrained(
            Vector3D.PLUS_K,
            PredefinedTarget.NADIR,
            Vector3D.PLUS_I,
            PredefinedTarget.VELOCITY,
            initial_state.getFrame(),
            CelestialBodyFactory.getSun(),
            earth,
        )

    def _zenith_provider():
        # Map body -X to nadir so body +X points to zenith.
        # Keep body +Z aligned with velocity (prograde) to resolve roll.
        return AlignedAndConstrained(
            Vector3D.MINUS_I,
            PredefinedTarget.NADIR,
            Vector3D.PLUS_K,
            PredefinedTarget.VELOCITY,
            initial_state.getFrame(),
            CelestialBodyFactory.getSun(),
            earth,
        )

    if attitude_mode not in ("prograde", "nadir", "zenith"):
        raise ValueError(
            f"Unknown attitude_mode={attitude_mode!r}. Expected 'prograde', 'nadir', or 'zenith'."
        )

    propagator_type = propagation_cfg.propagator_type.lower()
    attitude_provider = _zenith_provider() if attitude_mode == "zenith" else _prograde_provider()

    if propagator_type == "keplerian":
        propagator = KeplerianPropagator(initial_state.getOrbit())
    elif propagator_type == "brouwer_lyddane":
        drag_coefficient = float(getattr(propagation_cfg, "drag_coefficient", 0.0))
        # BrouwerLyddanePropagator always evaluates the J2-J5 zonal terms
        # internally (it reads (5, 0) unconditionally), so the gravity
        # provider must supply degree 5 regardless of which terms actually
        # matter for a given orbit.
        gravity_provider = GravityFieldFactory.getUnnormalizedProvider(5, 0)
        propagator = BrouwerLyddanePropagator(
            initial_state.getOrbit(), gravity_provider, drag_coefficient
        )
    else:
        raise ValueError(
            f"Unsupported propagator_type={propagation_cfg.propagator_type!r}. "
            "Expected 'keplerian' or 'brouwer_lyddane'."
        )
    propagator.setAttitudeProvider(attitude_provider)
    return propagator


def _propagate_and_collect_state_vectors(
    propagator: Propagator,
    start_date: AbsoluteDate,
    end_date: AbsoluteDate,
    sample_step: float,
    inertial_frame,
    earth: OneAxisEllipsoid,
    total_duration: float,
    progress_callback: Callable[[float], None] | None = None,
    cancel: CancelToken | None = None,
) -> tuple[
    np.ndarray,
    list[tuple],
    list[tuple],
    dict[str, np.ndarray],
]:
    """Propagate the orbit and collect state vectors, ground track and orbital elements.

    Per-station geometry is intentionally absent.  After this function returns,
    callers should build a ``PropagatedEphemeris`` and then call
    ``cached_access_recompute.derive_access_results_from_ephemeris`` for all
    station-dependent outputs.  This keeps the JVM propagation loop as tight as
    possible: every iteration makes only the minimum number of Orekit
    (JVM round-trip) calls required to fully describe the orbit state.
    """

    times_hours: List[float] = []
    track_points: list[tuple] = []
    state_vector_points: list[tuple] = []

    altitude_km: List[float] = []
    semi_major_axis_km: List[float] = []
    perigee_altitude_km: List[float] = []
    apogee_altitude_km: List[float] = []
    eccentricity: List[float] = []
    inclination_deg: List[float] = []
    argument_of_perigee_deg: List[float] = []
    true_anomaly_deg: List[float] = []
    orbital_period_s: List[float] = []
    angle_of_attack_deg: List[float] = []

    # Propagate to the start date so attitude providers are evaluated
    # consistently before the first sample is taken.
    try:
        state = propagator.propagate(start_date)
    except Exception as exc:
        raise RuntimeError("Failed to propagate to the scenario start date") from exc

    current_date = state.getDate()
    elapsed_seconds = 0.0

    while current_date.compareTo(end_date) <= 0:
        # Geodetic position and re-entry guard.
        geodetic = earth.transform(
            state.getPVCoordinates().getPosition(), state.getFrame(), current_date
        )
        alt_m = float(geodetic.getAltitude())
        if alt_m < _REENTRY_CUTOFF_ALTITUDE_M:
            break

        # Ground track in ECEF.
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

        # ECI state vector.
        pv_eci = state.getPVCoordinates(inertial_frame)
        pos_eci = pv_eci.getPosition()
        vel_eci = pv_eci.getVelocity()
        vel_ecef = pv_earth.getVelocity()

        # Body axes and angle of attack from the propagator's attitude provider.
        att_provider = propagator.getAttitudeProvider()
        if att_provider is None:
            raise RuntimeError("Propagator has no attitude provider for AoA computation")
        attitude = att_provider.getAttitude(
            state.getOrbit(),
            state.getDate(),
            state.getFrame(),
        )
        aoa = _angle_of_attack_deg(state, inertial_frame, attitude=attitude)
        angle_of_attack_deg.append(aoa)

        # Extract body axes in ECEF for downstream visualization.
        # Attitude rotation maps reference-frame -> body-frame vectors, so
        # applyInverseTo on body unit axes yields those axes in the reference frame.
        att_ecef = attitude.withReferenceFrame(earth.getBodyFrame())
        rot_ecef = att_ecef.getRotation()
        bx_vec = rot_ecef.applyInverseTo(Vector3D.PLUS_I)
        by_vec = rot_ecef.applyInverseTo(Vector3D.PLUS_J)
        bz_vec = rot_ecef.applyInverseTo(Vector3D.PLUS_K)

        bx = np.array([float(bx_vec.getX()), float(bx_vec.getY()), float(bx_vec.getZ())], dtype=float)
        by = np.array([float(by_vec.getX()), float(by_vec.getY()), float(by_vec.getZ())], dtype=float)
        bz = np.array([float(bz_vec.getX()), float(bz_vec.getY()), float(bz_vec.getZ())], dtype=float)
        for name, vec in (("bx", bx), ("by", by), ("bz", bz)):
            n = float(np.linalg.norm(vec))
            if not np.isfinite(n) or n <= 0.0:
                raise ValueError(f"Non-unit attitude axis: {name} norm={n}")
            vec /= n

        state_vector_points.append(
            (
                state.getDate(),
                pos_eci.getX(),
                pos_eci.getY(),
                pos_eci.getZ(),
                vel_eci.getX(),
                vel_eci.getY(),
                vel_eci.getZ(),
                position.getX(),
                position.getY(),
                position.getZ(),
                vel_ecef.getX(),
                vel_ecef.getY(),
                vel_ecef.getZ(),
                float(bx[0]),
                float(bx[1]),
                float(bx[2]),
                float(by[0]),
                float(by[1]),
                float(by[2]),
                float(bz[0]),
                float(bz[1]),
                float(bz[2]),
            )
        )

        times_hours.append(elapsed_seconds / 3600.0)
        altitude_km.append(alt_m / 1000.0)

        # Osculating Keplerian elements.
        kepler = KeplerianOrbit(state.getOrbit())
        a_m = kepler.getA()
        semi_major_axis_km.append(a_m / 1000.0)
        e = kepler.getE()
        ecc = float(e)
        eccentricity.append(ecc)
        inclination_deg.append(math.degrees(kepler.getI()))
        # Perigee/apogee radii are measured from Earth's centre; convert to
        # altitude above the WGS84 equatorial radius (spherical approximation,
        # consistent with the conventional "perigee altitude" definition).
        earth_radius_m = float(Constants.WGS84_EARTH_EQUATORIAL_RADIUS)
        r_perigee = a_m * (1.0 - ecc)
        r_apogee = a_m * (1.0 + ecc)
        perigee_altitude_km.append((r_perigee - earth_radius_m) / 1000.0)
        apogee_altitude_km.append((r_apogee - earth_radius_m) / 1000.0)
        argument_of_perigee_deg.append(math.degrees(kepler.getPerigeeArgument()))
        true_anomaly_deg.append(math.degrees(kepler.getTrueAnomaly()))
        orbital_period_s.append(float(kepler.getKeplerianPeriod()))

        if cancel is not None:
            cancel.check()

        if progress_callback and total_duration > 0:
            progress_pct = max(0.0, min(100.0, (elapsed_seconds / total_duration) * 100.0))
            progress_callback(progress_pct)

        # Advance to the next sample.
        if current_date.compareTo(end_date) >= 0:
            break

        next_date = current_date.shiftedBy(sample_step)
        try:
            state = propagator.propagate(next_date)
        except Exception as exc:
            raise RuntimeError(
                "Propagation failed while stepping the propagator "
                f"at t={float(elapsed_seconds):.3f}s, step={float(sample_step):.3f}s."
            ) from exc

        current_date = state.getDate()
        elapsed_seconds = float(current_date.durationFrom(start_date))

    if progress_callback:
        progress_callback(100.0)

    orbit_elements = {
        "altitude_km": np.array(altitude_km, dtype=float),
        "semi_major_axis_km": np.array(semi_major_axis_km, dtype=float),
        "perigee_altitude_km": np.array(perigee_altitude_km, dtype=float),
        "apogee_altitude_km": np.array(apogee_altitude_km, dtype=float),
        "eccentricity": np.array(eccentricity, dtype=float),
        "inclination_deg": np.array(inclination_deg, dtype=float),
        "argument_of_perigee_deg": np.array(argument_of_perigee_deg, dtype=float),
        "true_anomaly_deg": np.array(true_anomaly_deg, dtype=float),
        "period_seconds": np.array(orbital_period_s, dtype=float),
        "angle_of_attack_deg": np.array(angle_of_attack_deg, dtype=float),
    }

    return (np.array(times_hours), track_points, state_vector_points, orbit_elements)


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
