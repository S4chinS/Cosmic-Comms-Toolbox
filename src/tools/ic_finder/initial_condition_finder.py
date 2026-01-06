from __future__ import annotations

"""
Initial condition finder for the Orekit thrust study.

Given a target set of *mean* orbital quantities
  - ā  (semi‑major axis)
  - ī  (inclination)
  - Ω̄  (right ascension of the ascending node)
  - ē = [h̄, k̄]  with h = e cos ω, k = e sin ω

this module searches for a Cartesian initial state (r0, v0) such that,
when propagated with the full force model, the time‑averaged h̄ and k̄
over ~1–2 orbits match the requested targets (to within a tolerance).

The algorithm:
  1. Build an initial Keplerian guess state from the target mean elements.
  2. Convert to Cartesian (r0, v0).
  3. Run a Newton / differential‑correction loop in two in‑plane “knobs”
     (radial and along‑track components of v0) using finite‑difference
     sensitivities of [h̄, k̄].
  4. Return the corrected (r0, v0) and diagnostics (residuals, iterations).
"""

import math
import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np

# IMPORTANT: importing `orekit` is required before importing Java packages like `org.*`
# (it bootstraps the JCC bridge so `org.orekit...` is importable).
import orekit  # type: ignore[import-untyped]

from org.hipparchus.geometry.euclidean.threed import Vector3D  # type: ignore[import-untyped]
from org.orekit.frames import FramesFactory  # type: ignore[import-untyped]
from org.orekit.orbits import (  # type: ignore[import-untyped]
    CartesianOrbit,
    KeplerianOrbit,
    PositionAngleType,
)
from org.orekit.time import AbsoluteDate  # type: ignore[import-untyped]
from org.orekit.utils import Constants, PVCoordinates  # type: ignore[import-untyped]

from .cartesian_orbit_sim import init_orekit_vm, run_cartesian_orbit_simulation


@dataclass(frozen=True)
class MeanElementTargets:
    """Container for the target mean orbital quantities."""

    # Used only to build the initial Keplerian guess (not part of the matching objective).
    a_guess_m: float
    # Mean geodetic altitude target above the WGS-84 ellipsoid [m].
    alt_bar_m: float
    i_bar_rad: float
    raan_bar_rad: float
    h_bar: float
    k_bar: float


@dataclass(frozen=True)
class ICFinderConfig:
    """
    Configuration for the initial‑condition finder.

    This keeps all non‑orbital configuration explicit: force models,
    propagation duration, time step, and Newton iteration controls.
    """

    # Force‑model / environment (must be consistent with the “real” sim)
    atmosphere: object | None
    enable_drag: bool
    drag_area_m2: float
    drag_cd: float
    gravity_model: str
    gravity_degree: int
    gravity_order: int
    mass_kg: float

    # Diagnostics propagation settings
    step_s: float = 60.0
    # Use an integer number of orbital periods for the averaging window so
    # short-period terms do not bias the mean.
    num_orbital_periods: int = 2  # duration = num_orbital_periods * T

    # Logging / UX
    progress_print: bool = False

    # Newton / finite‑difference settings
    max_iterations: int = 6
    tol_hk: float = 1e-5
    # Tolerance for mean geodetic altitude matching [m]. Used together with tol_hk
    # in a scaled least-squares objective so (h̄, k̄, alt̄_geo) have comparable weight.
    tol_alt_m: float = 1.0
    fd_dv_m_s: float = 0.1  # finite‑difference step in dv_R, dv_T [m/s]
    max_step_dv_m_s: float = 50.0  # optional safeguard per Newton update

    # Optional extra design variable: allow shifting the initial radius along the
    # (instantaneous) radial direction by dr_R [m] during the search.
    unlock_r0_radial: bool = False
    fd_dr_m: float = 10.0
    max_step_dr_m: float = 2_000.0


@dataclass(frozen=True)
class ICFinderResult:
    """Result and diagnostics from the initial‑condition search."""

    r0_m: np.ndarray  # shape (3,)
    v0_m: np.ndarray  # shape (3,)
    h_bar: float
    k_bar: float
    # Diagnostic: mean osculating semi-major axis [m] over the averaging window.
    a_bar_m: float
    # Primary requirement: mean geodetic altitude [m] over the averaging window.
    alt_bar_m: float
    residual_h: float
    residual_k: float
    residual_alt_m: float
    iterations: int
    converged: bool


def _build_keplerian_guess_from_mean(
    *,
    epoch: AbsoluteDate,
    targets: MeanElementTargets,
) -> CartesianOrbit:
    """
    Construct an initial Keplerian guess orbit from the target mean elements
    and return it as a CartesianOrbit at `epoch`.
    """
    mu = Constants.WGS84_EARTH_MU
    inertial = FramesFactory.getEME2000()

    e_mag = math.hypot(targets.h_bar, targets.k_bar)
    omega = math.atan2(targets.k_bar, targets.h_bar) if e_mag > 0.0 else 0.0

    kep = KeplerianOrbit(
        float(targets.a_guess_m),
        float(e_mag),
        float(targets.i_bar_rad),
        float(omega),
        float(targets.raan_bar_rad),
        0.0,  # true anomaly guess; mean tuning is done via velocity tweaks
        PositionAngleType.TRUE,
        inertial,
        epoch,
        mu,
    )
    return CartesianOrbit(kep)


def _compute_time_averaged_hk_a_and_alt(
    *,
    epoch: AbsoluteDate,
    r0_m: Sequence[float],
    v0_m: Sequence[float],
    cfg: ICFinderConfig,
) -> tuple[float, float, float, float]:
    """
    Propagate a no‑thrust orbit from (r0, v0) and compute time‑averaged
    (h̄, k̄), ā (diagnostic), and mean geodetic altitude alt̄_geo.
    """
    r0 = np.asarray(r0_m, dtype=float)
    v0 = np.asarray(v0_m, dtype=float)
    if r0.shape != (3,) or v0.shape != (3,):
        raise ValueError("r0_m and v0_m must be length‑3 sequences")

    mu = Constants.WGS84_EARTH_MU
    inertial = FramesFactory.getEME2000()

    pv0 = PVCoordinates(
        Vector3D(float(r0[0]), float(r0[1]), float(r0[2])),
        Vector3D(float(v0[0]), float(v0[1]), float(v0[2])),
    )
    orbit0 = CartesianOrbit(pv0, inertial, epoch, mu)
    # Use the Keplerian period from the instantaneous orbit to define the
    # averaging window as an (integer) number of orbital periods.
    kep0 = KeplerianOrbit(orbit0)
    orbital_period_s = float(kep0.getKeplerianPeriod())

    n_orbits = max(1, int(cfg.num_orbital_periods))
    duration_s = n_orbits * orbital_period_s

    # Use the existing cartesian simulation helper with an empty thrust list.
    results = run_cartesian_orbit_simulation(
        epoch=epoch,
        r0_m=r0,
        v0_m=v0,
        mass_kg=cfg.mass_kg,
        profiles=[],
        duration_s=duration_s,
        step_s=cfg.step_s,
        atmosphere=cfg.atmosphere,
        enable_drag=cfg.enable_drag,
        drag_area_m2=cfg.drag_area_m2,
        drag_cd=cfg.drag_cd,
        gravity_model=cfg.gravity_model,
        gravity_degree=cfg.gravity_degree,
        gravity_order=cfg.gravity_order,
        progress_print=False,
        make_plots=False,
        make_globe_view=False,
    )

    baseline = results.get("no-thrust")
    if baseline is None:
        raise RuntimeError("Expected 'no-thrust' baseline in results")

    t_s = baseline["t_s"]
    r_hist = baseline["r_eci_m"]
    v_hist = baseline["v_eci_m_s"]
    a_hist = baseline["a_m"]
    alt_hist = baseline["altitude_m"]

    if len(t_s) == 0:
        raise RuntimeError("Propagation returned no samples")

    # Compute h = e cos ω, k = e sin ω at each sample using the instantaneous
    # Keplerian elements, then time‑average them.
    h_samples = np.empty_like(t_s, dtype=float)
    k_samples = np.empty_like(t_s, dtype=float)

    for i in range(len(t_s)):
        t = float(t_s[i])
        r_vec = r_hist[i]
        v_vec = v_hist[i]
        date = epoch.shiftedBy(t)

        pv = PVCoordinates(
            Vector3D(float(r_vec[0]), float(r_vec[1]), float(r_vec[2])),
            Vector3D(float(v_vec[0]), float(v_vec[1]), float(v_vec[2])),
        )
        # Convert the instantaneous Cartesian orbit to a Keplerian orbit to
        # access classical elements (e, ω) in a version‑independent way.
        cart = CartesianOrbit(pv, inertial, date, mu)
        kep = KeplerianOrbit(cart)
        e_val = float(kep.getE())
        omega = float(kep.getPerigeeArgument())

        h_samples[i] = e_val * math.cos(omega)
        k_samples[i] = e_val * math.sin(omega)

    h_bar = float(np.mean(h_samples))
    k_bar = float(np.mean(k_samples))
    a_bar = float(np.mean(a_hist))
    alt_bar = float(np.mean(alt_hist))

    return h_bar, k_bar, a_bar, alt_bar


def _rt_basis_from_state(r_m: np.ndarray, v_m_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Build radial‑transverse (R‑T) unit vectors from a Cartesian state.

    R̂ is along r, T̂ lies in the orbital plane 90° ahead of R̂ in the
    direction of motion (approximately “along‑track”).
    """
    r = np.asarray(r_m, dtype=float)
    v = np.asarray(v_m_s, dtype=float)

    h_vec = np.cross(r, v)
    if np.linalg.norm(r) == 0.0 or np.linalg.norm(h_vec) == 0.0:
        raise ValueError("Cannot build RT basis from degenerate state")

    r_hat = r / np.linalg.norm(r)
    n_hat = h_vec / np.linalg.norm(h_vec)
    t_hat = np.cross(n_hat, r_hat)
    return r_hat, t_hat


def find_initial_cartesian_state_for_mean_elements(
    *,
    epoch: AbsoluteDate,
    targets: MeanElementTargets,
    cfg: ICFinderConfig,
) -> ICFinderResult:
    """
    Find an initial Cartesian state (r0, v0) whose mean (h̄, k̄, ā) over ~1–2
    orbits matches the requested targets under the specified force model.

    The search adjusts:
      - Always: two in‑plane components of v0 (radial and along‑track) via dv_R, dv_T
      - Optionally: the initial radial position via dr_R (cfg.unlock_r0_radial)

    The update uses a scaled least-squares Gauss–Newton step so the residuals in
    h̄, k̄, alt̄_geo have comparable importance after scaling by (tol_hk, tol_hk, tol_alt_m).
    """

    if cfg.mass_kg <= 0.0 or not math.isfinite(cfg.mass_kg):
        raise ValueError("cfg.mass_kg must be positive and finite")
    if cfg.step_s <= 0.0:
        raise ValueError("cfg.step_s must be positive")
    if cfg.num_orbital_periods <= 0.0:
        raise ValueError("cfg.num_orbital_periods must be positive")
    if cfg.max_iterations <= 0:
        raise ValueError("cfg.max_iterations must be a positive integer")
    if cfg.tol_hk <= 0.0 or not math.isfinite(cfg.tol_hk):
        raise ValueError("cfg.tol_hk must be positive and finite")
    if cfg.tol_alt_m <= 0.0 or not math.isfinite(cfg.tol_alt_m):
        raise ValueError("cfg.tol_alt_m must be positive and finite")
    if cfg.fd_dv_m_s <= 0.0 or not math.isfinite(cfg.fd_dv_m_s):
        raise ValueError("cfg.fd_dv_m_s must be positive and finite")
    if cfg.max_step_dv_m_s <= 0.0 or not math.isfinite(cfg.max_step_dv_m_s):
        raise ValueError("cfg.max_step_dv_m_s must be positive and finite")
    if cfg.fd_dr_m <= 0.0 or not math.isfinite(cfg.fd_dr_m):
        raise ValueError("cfg.fd_dr_m must be positive and finite")
    if cfg.max_step_dr_m <= 0.0 or not math.isfinite(cfg.max_step_dr_m):
        raise ValueError("cfg.max_step_dr_m must be positive and finite")

    if cfg.progress_print:
        vary = "dr_R,dv_R,dv_T" if bool(cfg.unlock_r0_radial) else "dv_R,dv_T"
        print(
            "[ICFinder] START  "
            f"vary=[{vary}]  "
            f"tol_hk={cfg.tol_hk:.1e}  tol_alt={cfg.tol_alt_m:.1f}m  "
            f"fd_dv={cfg.fd_dv_m_s:.2f}m/s"
            + (f"  fd_dr={cfg.fd_dr_m:.1f}m" if bool(cfg.unlock_r0_radial) else "")
        )

    # Ensure Orekit JVM is initialized before we touch any Orekit classes.
    init_orekit_vm()

    # Step 1: Keplerian guess at epoch.
    guess_orbit = _build_keplerian_guess_from_mean(epoch=epoch, targets=targets)
    pv_guess = guess_orbit.getPVCoordinates()
    r0 = np.array(
        [
            pv_guess.getPosition().getX(),
            pv_guess.getPosition().getY(),
            pv_guess.getPosition().getZ(),
        ],
        dtype=float,
    )
    v0_nominal = np.array(
        [
            pv_guess.getVelocity().getX(),
            pv_guess.getVelocity().getY(),
            pv_guess.getVelocity().getZ(),
        ],
        dtype=float,
    )

    r_hat0, t_hat0 = _rt_basis_from_state(r0, v0_nominal)

    # Newton loop in dv_R, dv_T.
    dv = np.zeros(2, dtype=float)  # [dv_R, dv_T]
    dr_r = 0.0  # radial position shift [m] (optional)
    target_vec = np.array([targets.h_bar, targets.k_bar, targets.alt_bar_m], dtype=float)

    h_bar = k_bar = a_bar = math.nan
    alt_bar = math.nan
    converged = False
    t_start = time.perf_counter()

    for it in range(cfg.max_iterations):
        r0_use = r0 + float(dr_r) * r_hat0
        r_hat, t_hat = _rt_basis_from_state(r0_use, v0_nominal)
        v0 = v0_nominal + dv[0] * r_hat + dv[1] * t_hat

        h_bar, k_bar, a_bar, alt_bar = _compute_time_averaged_hk_a_and_alt(
            epoch=epoch,
            r0_m=r0_use,
            v0_m=v0,
            cfg=cfg,
        )
        f_vec = np.array([h_bar, k_bar, alt_bar], dtype=float)
        residual = f_vec - target_vec

        if cfg.progress_print:
            scale_dbg = np.array(
                [1.0 / float(cfg.tol_hk), 1.0 / float(cfg.tol_hk), 1.0 / float(cfg.tol_alt_m)],
                dtype=float,
            )
            res_scaled = residual * scale_dbg
            res_scaled_norm = float(np.linalg.norm(res_scaled))
            elapsed_s = float(time.perf_counter() - t_start)
            prefix = f"[ICFinder] it {it + 1:02d}/{cfg.max_iterations:02d}"
            x_part = (
                f"dr={dr_r: .0f}m " if bool(cfg.unlock_r0_radial) else ""
            ) + f"dvR={dv[0]: .3f} dvT={dv[1]: .3f} (m/s)"
            res_part = (
                f"dh={float(residual[0]): .2e} dk={float(residual[1]): .2e} dalt={float(residual[2]): .1f}m"
            )
            print(
                f"{prefix}  {x_part}  {res_part}  ||s||={res_scaled_norm:.2e}  t={elapsed_s:.1f}s"
            )
        # Convergence: each component within tolerance.
        if (
            (abs(float(residual[0])) <= cfg.tol_hk)
            and (abs(float(residual[1])) <= cfg.tol_hk)
            and (abs(float(residual[2])) <= cfg.tol_alt_m)
        ):
            converged = True
            break

        # Finite‑difference Jacobian J ≈ d[h̄, k̄, ā]/d[vars]
        # vars = [dv_R, dv_T] or [dr_R, dv_R, dv_T]
        if bool(cfg.unlock_r0_radial):
            J = np.zeros((3, 3), dtype=float)
        else:
            J = np.zeros((3, 2), dtype=float)

        def _eval_at(dr_r_m: float, dv_vec: np.ndarray) -> np.ndarray:
            r0_p = r0 + float(dr_r_m) * r_hat0
            r_hat_p, t_hat_p = _rt_basis_from_state(r0_p, v0_nominal)
            v0_p = v0_nominal + float(dv_vec[0]) * r_hat_p + float(dv_vec[1]) * t_hat_p
            h_p, k_p, _a_p, alt_p = _compute_time_averaged_hk_a_and_alt(
                epoch=epoch,
                r0_m=r0_p,
                v0_m=v0_p,
                cfg=cfg,
            )
            return np.array([h_p, k_p, alt_p], dtype=float)

        if bool(cfg.unlock_r0_radial):
            f_pert = _eval_at(dr_r + float(cfg.fd_dr_m), dv)
            J[:, 0] = (f_pert - f_vec) / float(cfg.fd_dr_m)
            col0 = 1
        else:
            col0 = 0

        for j in range(2):
            dv_pert = dv.copy()
            dv_pert[j] += float(cfg.fd_dv_m_s)
            f_pert = _eval_at(dr_r, dv_pert)
            J[:, col0 + j] = (f_pert - f_vec) / float(cfg.fd_dv_m_s)

        # Scale residuals so h̄,k̄,alt̄ have equal importance in the objective.
        scale = np.array([1.0 / cfg.tol_hk, 1.0 / cfg.tol_hk, 1.0 / cfg.tol_alt_m], dtype=float)
        A = (J.T * scale).T  # scale rows
        b = -(residual * scale)

        # Least-squares Gauss-Newton step (3 residuals, 2 controls).
        try:
            delta, *_ = np.linalg.lstsq(A, b, rcond=None)
        except Exception:
            break

        if cfg.progress_print:
            if bool(cfg.unlock_r0_radial):
                print(
                    "[ICFinder]   step  "
                    f"d_dr={float(delta[0]): .0f}m  d_dvR={float(delta[1]): .3f}  d_dvT={float(delta[2]): .3f}"
                )
            else:
                print(
                    "[ICFinder]   step  "
                    f"d_dvR={float(delta[0]): .3f}  d_dvT={float(delta[1]): .3f}"
                )

        # Optional step‑length safeguard(s).
        if bool(cfg.unlock_r0_radial):
            d_dr = float(delta[0])
            d_dv = np.array([float(delta[1]), float(delta[2])], dtype=float)
            dv_step_norm = float(np.linalg.norm(d_dv))
            if abs(d_dr) > float(cfg.max_step_dr_m):
                d_dr *= float(cfg.max_step_dr_m) / abs(d_dr)
            if dv_step_norm > float(cfg.max_step_dv_m_s):
                d_dv *= float(cfg.max_step_dv_m_s) / dv_step_norm
            dr_r += d_dr
            dv += d_dv
        else:
            step_norm = float(np.linalg.norm(delta))
            if step_norm > cfg.max_step_dv_m_s:
                delta *= cfg.max_step_dv_m_s / step_norm
            dv += delta

    # Final state and residuals.
    r0_final = r0 + float(dr_r) * r_hat0
    r_hat_f, t_hat_f = _rt_basis_from_state(r0_final, v0_nominal)
    v0_final = v0_nominal + dv[0] * r_hat_f + dv[1] * t_hat_f
    residual_h = float(h_bar - targets.h_bar)
    residual_k = float(k_bar - targets.k_bar)
    residual_alt = float(alt_bar - targets.alt_bar_m)

    return ICFinderResult(
        r0_m=r0_final,
        v0_m=v0_final,
        h_bar=float(h_bar),
        k_bar=float(k_bar),
        a_bar_m=float(a_bar),
        alt_bar_m=float(alt_bar),
        residual_h=residual_h,
        residual_k=residual_k,
        residual_alt_m=residual_alt,
        iterations=it + 1,
        converged=converged,
    )


