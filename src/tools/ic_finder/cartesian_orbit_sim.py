from __future__ import annotations

"""
Minimal Orekit Cartesian propagation helper used by the initial-condition finder.

This module intentionally provides the API expected by
`initial_condition_finder.py` without changing the solver implementation.
"""

import math
from typing import Any, Sequence

import numpy as np
import orekit  # type: ignore[import-untyped]
from java.io import File  # type: ignore[import-untyped]
from org.hipparchus.geometry.euclidean.threed import Vector3D  # type: ignore[import-untyped]
from org.hipparchus.ode.nonstiff import DormandPrince853Integrator  # type: ignore[import-untyped]
from org.orekit.attitudes import FrameAlignedProvider  # type: ignore[import-untyped]
from org.orekit.data import DataContext, DirectoryCrawler  # type: ignore[import-untyped]
from org.orekit.forces.drag import DragForce, IsotropicDrag  # type: ignore[import-untyped]
from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel  # type: ignore[import-untyped]
from org.orekit.forces.gravity.potential import GravityFieldFactory  # type: ignore[import-untyped]
from org.orekit.frames import FramesFactory  # type: ignore[import-untyped]
from org.orekit.models.earth.atmosphere import NRLMSISE00  # type: ignore[import-untyped]
from org.orekit.orbits import CartesianOrbit, KeplerianOrbit  # type: ignore[import-untyped]
from org.orekit.propagation import SpacecraftState  # type: ignore[import-untyped]
from org.orekit.propagation.numerical import NumericalPropagator  # type: ignore[import-untyped]
from org.orekit.time import AbsoluteDate  # type: ignore[import-untyped]
from org.orekit.utils import Constants, IERSConventions, PVCoordinates  # type: ignore[import-untyped]
from org.orekit.bodies import OneAxisEllipsoid  # type: ignore[import-untyped]

import orekitdata  # type: ignore[import-untyped]


_BOOTSTRAPPED = False


def init_orekit_vm() -> None:
    """Initialize the JVM and load Orekit data if needed."""
    global _BOOTSTRAPPED
    try:
        orekit.initVM()
    except Exception:
        pass
    try:
        orekit.getVMEnv().attachCurrentThread()
    except Exception:
        pass
    if _BOOTSTRAPPED:
        return
    manager = DataContext.getDefault().getDataProvidersManager()
    crawler = DirectoryCrawler(File(orekitdata.__path__[0]))
    manager.addProvider(crawler)
    _BOOTSTRAPPED = True


def _build_earth() -> OneAxisEllipsoid:
    itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
    return OneAxisEllipsoid(
        Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
        Constants.WGS84_EARTH_FLATTENING,
        itrf,
    )


def run_cartesian_orbit_simulation(
    *,
    epoch: AbsoluteDate,
    r0_m: Sequence[float],
    v0_m: Sequence[float],
    mass_kg: float,
    profiles: Sequence[Any],
    duration_s: float,
    step_s: float,
    atmosphere: object | None,
    enable_drag: bool,
    drag_area_m2: float,
    drag_cd: float,
    gravity_model: str,
    gravity_degree: int,
    gravity_order: int,
    progress_print: bool = False,
    make_plots: bool = False,
    make_globe_view: bool = False,
) -> dict[str, dict[str, np.ndarray]]:
    """
    Propagate a Cartesian initial state with the requested force model.

    The API matches what `initial_condition_finder.py` expects. Thrust profiles
    are accepted but ignored here (the IC finder uses `profiles=[]`).
    """
    _ = profiles, make_plots, make_globe_view
    init_orekit_vm()

    if duration_s <= 0 or step_s <= 0:
        raise ValueError("duration_s and step_s must be positive")
    if mass_kg <= 0:
        raise ValueError("mass_kg must be positive")

    inertial = FramesFactory.getEME2000()
    mu = Constants.WGS84_EARTH_MU

    pv0 = PVCoordinates(
        Vector3D(float(r0_m[0]), float(r0_m[1]), float(r0_m[2])),
        Vector3D(float(v0_m[0]), float(v0_m[1]), float(v0_m[2])),
    )
    orbit0 = CartesianOrbit(pv0, inertial, epoch, mu)
    state0 = SpacecraftState(orbit0, float(mass_kg))

    min_step = 0.1
    max_step = max(300.0, float(step_s))
    position_tolerance = 10.0
    integrator = DormandPrince853Integrator(
        min_step, max_step, position_tolerance, position_tolerance
    )
    propagator = NumericalPropagator(integrator)
    propagator.setInitialState(state0)
    propagator.setAttitudeProvider(FrameAlignedProvider(inertial))

    gravity_name = (gravity_model or "").lower()
    if gravity_name in ("hf", "holmes-featherstone", "holmesfeatherstone", ""):
        provider = GravityFieldFactory.getNormalizedProvider(
            int(gravity_degree), int(gravity_order)
        )
        itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
        propagator.addForceModel(HolmesFeatherstoneAttractionModel(itrf, provider))
    else:
        raise ValueError(f"Unsupported gravity_model: {gravity_model!r}")

    if enable_drag:
        if atmosphere is None:
            raise ValueError("enable_drag=True requires an atmosphere instance")
        if not isinstance(atmosphere, NRLMSISE00) and progress_print:
            print("[cartesian_orbit_sim] atmosphere is not NRLMSISE00; continuing")
        propagator.addForceModel(
            DragForce(atmosphere, IsotropicDrag(float(drag_area_m2), float(drag_cd)))
        )

    earth = _build_earth()

    steps = int(math.floor(float(duration_s) / float(step_s)))
    t_samples: list[float] = []
    r_samples: list[list[float]] = []
    v_samples: list[list[float]] = []
    a_samples: list[float] = []
    alt_samples: list[float] = []

    state = state0
    for k in range(steps + 1):
        t = float(k) * float(step_s)
        date = epoch.shiftedBy(t)
        state = propagator.propagate(date)
        pv = state.getPVCoordinates(inertial)
        pos = pv.getPosition()
        vel = pv.getVelocity()
        t_samples.append(t)
        r_samples.append([pos.getX(), pos.getY(), pos.getZ()])
        v_samples.append([vel.getX(), vel.getY(), vel.getZ()])
        kep = KeplerianOrbit(state.getOrbit())
        a_samples.append(float(kep.getA()))
        geodetic = earth.transform(pv.getPosition(), inertial, date)
        alt_samples.append(float(geodetic.getAltitude()))

    return {
        "no-thrust": {
            "t_s": np.asarray(t_samples, dtype=float),
            "r_eci_m": np.asarray(r_samples, dtype=float),
            "v_eci_m_s": np.asarray(v_samples, dtype=float),
            "a_m": np.asarray(a_samples, dtype=float),
            "altitude_m": np.asarray(alt_samples, dtype=float),
        }
    }


