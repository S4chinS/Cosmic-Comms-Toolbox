"""Orbit summary tab mixin for visualizing basic orbit metrics over time."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QComboBox, QVBoxLayout, QWidget

from src.models import AnalysisResult, GroundTrackPoint

WGS84_EQUATORIAL_RADIUS_KM = 6378.137


class OrbitSummaryTabMixin:
    """Provides a tab with orbit summary plots (altitude and basic elements)."""

    def _build_orbit_summary_tab(self) -> QWidget:
        """Create the Orbit Summary tab with secular and instantaneous element plots."""
        # Avoid heavy antialiasing for dense time series to keep things responsive.
        pg.setConfigOptions(antialias=False)

        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(4, 4, 4, 4)

        # --- Short-period averaged (Brouwer–Lyddane-style) elements ---
        self.orbit_secular_combo = QComboBox()
        self.orbit_secular_combo.addItems(
            [
                "Geodetic altitude (orbit-averaged)",
                "Semi-major axis and apsides (Brouwer–Lyddane mean)",
                "Eccentricity (Brouwer–Lyddane mean)",
                "Argument of perigee (Brouwer–Lyddane mean)",
                "Orbital period (Brouwer–Lyddane mean)",
                "Atmospheric density (Brouwer–Lyddane mean)",
                "Dynamic pressure (Brouwer–Lyddane mean)",
            ]
        )
        self.orbit_secular_combo.currentIndexChanged.connect(  # type: ignore[arg-type]
            self._handle_secular_metric_changed
        )
        layout.addWidget(self.orbit_secular_combo)

        self.orbit_secular_plot = pg.PlotWidget(title="Geodetic Altitude (orbit-averaged)")
        self.orbit_secular_plot.setLabel("bottom", "Time since start", units="h")
        self.orbit_secular_plot.setLabel("left", "Geodetic altitude", units="km")
        self.orbit_secular_plot.showGrid(x=True, y=True, alpha=0.3)
        self._orbit_secular_legend = self.orbit_secular_plot.addLegend()
        layout.addWidget(self.orbit_secular_plot, stretch=1)

        # --- Instantaneous (osculating) elements ---
        self.orbit_instant_combo = QComboBox()
        self.orbit_instant_combo.addItems(
            [
                "Geodetic altitude (instantaneous)",
                "Semi-major axis and apsides (instantaneous)",
                "Eccentricity (instantaneous)",
                "Argument of perigee (instantaneous)",
                "True anomaly (instantaneous)",
                "Orbital period (instantaneous)",
                "Atmospheric density (instantaneous)",
                "Dynamic pressure (instantaneous)",
                "Drag force (instantaneous)",
            ]
        )
        self.orbit_instant_combo.currentIndexChanged.connect(  # type: ignore[arg-type]
            self._handle_instant_metric_changed
        )
        layout.addWidget(self.orbit_instant_combo)

        self.orbit_instant_plot = pg.PlotWidget(title="Geodetic Altitude (instantaneous)")
        self.orbit_instant_plot.setLabel("bottom", "Time since start", units="h")
        self.orbit_instant_plot.setLabel("left", "Geodetic altitude", units="km")
        self.orbit_instant_plot.showGrid(x=True, y=True, alpha=0.3)
        self._orbit_instant_legend = self.orbit_instant_plot.addLegend()
        layout.addWidget(self.orbit_instant_plot, stretch=1)

        # Cache for latest time series (instantaneous + secular).
        self._orbit_summary_data: Dict[str, np.ndarray] | None = None

        # Initial empty state.
        self._update_orbit_summary(None)
        return tab

    def _downsample_series(
        self, times: np.ndarray, *series: np.ndarray, max_points: int = 2000
    ) -> tuple[np.ndarray, List[np.ndarray]]:
        """Return downsampled time and series arrays to keep plotting responsive."""
        if times.size == 0:
            return times, [s for s in series]
        if times.size <= max_points:
            return times, [s for s in series]
        indices = np.linspace(0, times.size - 1, max_points, dtype=int)
        indices = np.unique(indices)
        return (
            times[indices],
            [s[indices] if s.size == times.size else s for s in series],
        )

    def _update_orbit_summary(self, result: AnalysisResult | None) -> None:
        """Refresh the orbit summary plots from the latest analysis result."""
        if not getattr(self, "orbit_secular_plot", None) or not getattr(
            self, "orbit_instant_plot", None
        ):
            return

        if result is None or not result.ground_track:
            # Clear plots and show placeholders.
            for plot, legend, title in [
                (
                    self.orbit_secular_plot,
                    getattr(self, "_orbit_secular_legend", None),
                    "Orbit-averaged orbit summary\n(Run a simulation to view)",
                ),
                (
                    self.orbit_instant_plot,
                    getattr(self, "_orbit_instant_legend", None),
                    "Instantaneous orbit summary\n(Run a simulation to view)",
                ),
            ]:
                plot.clear()
                plot.setTitle(title)
                plot.setLabel("bottom", "Time since start", units="h")
                plot.setLabel("left", "", units="")
                if legend is not None:
                    legend.clear()

            self._orbit_summary_data = None
            if getattr(self, "orbit_secular_combo", None) is not None:
                self.orbit_secular_combo.setEnabled(False)
            if getattr(self, "orbit_instant_combo", None) is not None:
                self.orbit_instant_combo.setEnabled(False)
            return

        if getattr(self, "orbit_secular_combo", None) is not None:
            self.orbit_secular_combo.setEnabled(True)
        if getattr(self, "orbit_instant_combo", None) is not None:
            self.orbit_instant_combo.setEnabled(True)

        # Use timeline seconds as the common time base.
        times_hours = np.asarray(result.timeline_seconds, dtype=float) / 3600.0

        alt = np.asarray(getattr(result, "orbital_altitude_km", []), dtype=float)
        sma = np.asarray(getattr(result, "semi_major_axis_km", []), dtype=float)
        perigee = np.asarray(getattr(result, "perigee_altitude_km", []), dtype=float)
        apogee = np.asarray(getattr(result, "apogee_altitude_km", []), dtype=float)
        ecc = np.asarray(getattr(result, "eccentricity", []), dtype=float)
        argp = np.asarray(getattr(result, "argument_of_perigee_deg", []), dtype=float)
        # Normalize instantaneous argument of perigee and true anomaly to [0, 360) degrees for plotting.
        if argp.size:
            argp = np.mod(argp, 360.0)
        true_anom = np.asarray(getattr(result, "true_anomaly_deg", []), dtype=float)
        if true_anom.size:
            true_anom = np.mod(true_anom, 360.0)
        period = np.asarray(getattr(result, "orbital_period_series_s", []), dtype=float)
        density = np.asarray(getattr(result, "density_kg_m3", []), dtype=float)
        dyn_press = np.asarray(getattr(result, "dynamic_pressure_pa", []), dtype=float)
        drag_force = np.asarray(getattr(result, "drag_force_N", []), dtype=float)

        # Fallback for older results without per-sample metrics.
        if alt.size == 0 or alt.size != times_hours.size:
            track: List[GroundTrackPoint] = result.ground_track
            positions = np.array(
                [[pt.x_km, pt.y_km, pt.z_km] for pt in track], dtype=float
            )
            radii = np.linalg.norm(positions, axis=1)
            alt = radii - WGS84_EQUATORIAL_RADIUS_KM

        # Cache full-resolution data for per-metric rendering.
        self._orbit_summary_data = {
            "times_hours": times_hours,
            # Instantaneous
            "inst_altitude_km": alt,
            "inst_semi_major_axis_km": sma,
            "inst_perigee_altitude_km": perigee,
            "inst_apogee_altitude_km": apogee,
            "inst_eccentricity": ecc,
            "inst_argument_of_perigee_deg": argp,
            "inst_true_anomaly_deg": true_anom,
            "inst_period_seconds": period,
            "inst_density_kg_m3": density,
            "inst_dynamic_pressure_pa": dyn_press,
            "inst_drag_force_N": drag_force,
        }

        # Estimate a representative orbital period (in seconds) from the
        # instantaneous Keplerian period series. This lets us build a
        # smoothing window that roughly spans one orbit, which is a simple
        # numerical approximation to Brouwer–Lyddane-style mean elements
        # (short-period averaged quantities).
        times_sec = times_hours * 3600.0
        valid_periods = period[np.isfinite(period) & (period > 0.0)]
        if valid_periods.size and times_sec.size > 1:
            one_orbit_s = float(np.median(valid_periods))
            dt_s = float(np.median(np.diff(times_sec)))
            if not np.isfinite(dt_s) or dt_s <= 0.0:
                dt_s = float(times_sec[-1] - times_sec[0]) / max(times_sec.size - 1, 1)
            window = int(round(one_orbit_s / dt_s)) if dt_s > 0.0 else 0
        else:
            one_orbit_s = 0.0
            window = 0

        # Clamp window to a sensible range.
        window = max(5, min(window, max(5, times_hours.size))) if window > 0 else 0

        # Compute orbit-period-based smoothed "Brouwer–Lyddane mean" versions
        # of each quantity. We explicitly drop samples that don't have a full
        # window of support to avoid edge artefacts near the start and end of
        # the simulation.
        if window > 1:
            kernel = np.ones(window, dtype=float) / float(window)

            def _smooth(values: np.ndarray) -> np.ndarray:
                if values.size == 0:
                    return values
                smoothed = np.convolve(values, kernel, mode="same")
                # Mask out the first/last half-window where the convolution
                # relies on partially defined windows, which otherwise creates
                # artificial behaviour close to the simulation boundaries.
                half = window // 2
                if values.size > window and half > 0:
                    smoothed[:half] = np.nan
                    smoothed[-half:] = np.nan
                return smoothed

            alt_sec = _smooth(alt)
            sma_sec = _smooth(sma)
            per_sec = _smooth(perigee)
            apo_sec = _smooth(apogee)
            ecc_sec = _smooth(ecc)
            # For angular quantities, use a circular mean based on the complex
            # exponential to avoid artificial jumps near 0/360 deg. We smooth
            # cos(ω) and sin(ω) separately and then recover the mean angle.
            if argp.size:
                argp_rad = np.deg2rad(argp)
                cos_w = np.cos(argp_rad)
                sin_w = np.sin(argp_rad)
                cos_w_sec = _smooth(cos_w)
                sin_w_sec = _smooth(sin_w)
                argp_sec = np.rad2deg(np.arctan2(sin_w_sec, cos_w_sec))
                argp_sec = np.mod(argp_sec, 360.0)
            else:
                argp_sec = np.array([], dtype=float)
            period_sec = _smooth(period)
            density_sec = _smooth(density)
            dynp_sec = _smooth(dyn_press)
        else:
            alt_sec = alt
            sma_sec = sma
            per_sec = perigee
            apo_sec = apogee
            ecc_sec = ecc
            argp_sec = argp
            period_sec = period
            density_sec = density
            dynp_sec = dyn_press

        # For Brouwer–Lyddane-mean curves, omit the first and last orbit instead of relying on
        # partially defined smoothing windows at the boundaries.
        if period.size and times_hours.size and one_orbit_s > 0.0:
            t_start = float(times_sec[0])
            t_end = float(times_sec[-1])
            inner_mask = (times_sec - t_start >= one_orbit_s) & (
                t_end - times_sec >= one_orbit_s
            )
            # Apply mask: mark edge samples as NaN so they are visually omitted.
            for arr in (
                alt_sec,
                sma_sec,
                per_sec,
                apo_sec,
                ecc_sec,
                argp_sec,
                period_sec,
                density_sec,
                dynp_sec,
            ):
                if arr.size == inner_mask.size:
                    arr[~inner_mask] = np.nan

        self._orbit_summary_data.update(
            {
                "sec_altitude_km": alt_sec,
                "sec_semi_major_axis_km": sma_sec,
                "sec_perigee_altitude_km": per_sec,
                "sec_apogee_altitude_km": apo_sec,
                "sec_eccentricity": ecc_sec,
                "sec_argument_of_perigee_deg": argp_sec,
                "sec_period_seconds": period_sec,
                "sec_density_kg_m3": density_sec,
                "sec_dynamic_pressure_pa": dynp_sec,
            }
        )

        # Render the currently selected metrics for both plots.
        sec_index = (
            self.orbit_secular_combo.currentIndex()
            if getattr(self, "orbit_secular_combo", None) is not None
            else 0
        )
        inst_index = (
            self.orbit_instant_combo.currentIndex()
            if getattr(self, "orbit_instant_combo", None) is not None
            else 0
        )
        self._render_secular_metric(sec_index)
        self._render_instant_metric(inst_index)

    def _handle_secular_metric_changed(self, index: int) -> None:
        """React to dropdown changes for secular elements."""
        self._render_secular_metric(index)

    def _handle_instant_metric_changed(self, index: int) -> None:
        """React to dropdown changes for instantaneous elements."""
        self._render_instant_metric(index)

    def _render_secular_metric(self, index: int) -> None:
        """Render the short-period averaged (Brouwer–Lyddane-style) metric selected in the combo box."""
        if not self._orbit_summary_data or not getattr(self, "orbit_secular_plot", None):
            return

        times_hours = self._orbit_summary_data["times_hours"]
        alt = self._orbit_summary_data.get("sec_altitude_km", np.array([], dtype=float))
        sma = self._orbit_summary_data.get(
            "sec_semi_major_axis_km", np.array([], dtype=float)
        )
        perigee = self._orbit_summary_data.get(
            "sec_perigee_altitude_km", np.array([], dtype=float)
        )
        apogee = self._orbit_summary_data.get(
            "sec_apogee_altitude_km", np.array([], dtype=float)
        )
        ecc = self._orbit_summary_data.get("sec_eccentricity", np.array([], dtype=float))
        argp = self._orbit_summary_data.get(
            "sec_argument_of_perigee_deg", np.array([], dtype=float)
        )
        period = self._orbit_summary_data.get("sec_period_seconds", np.array([], dtype=float))
        density = self._orbit_summary_data.get(
            "sec_density_kg_m3", np.array([], dtype=float)
        )
        dynp = self._orbit_summary_data.get(
            "sec_dynamic_pressure_pa", np.array([], dtype=float)
        )

        plot = self.orbit_secular_plot
        legend = getattr(self, "_orbit_secular_legend", None)
        plot.clear()
        if legend is not None:
            legend.clear()

        # Geodetic altitude only (orbit-averaged).
        if index == 0:
            plot.setTitle("Geodetic Altitude (orbit-averaged)")
            plot.setLabel("bottom", "Time since start", units="h")
            plot.setLabel("left", "Geodetic altitude", units="km")
            times_ds, (alt_ds,) = self._downsample_series(times_hours, alt)
            plot.plot(
                times_ds,
                alt_ds,
                pen=pg.mkPen("#76c7ff", width=2),
                name="Geodetic altitude (orbit-averaged)",
            )
            plot.getViewBox().autoRange()
            return

        # Semi-major axis and apsides from orbital elements (Brouwer–Lyddane mean).
        if index == 1:
            plot.setTitle("Semi-major Axis and Apsides (Brouwer–Lyddane mean)")
            plot.setLabel("bottom", "Time since start", units="h")
            plot.setLabel("left", "Altitude (from elements)", units="km")
            # Convert element-derived radii to an altitude-like quantity so this
            # plot aligns with the Mean-IC altitude target and the geodetic altitude plots.
            sma_plot = sma - WGS84_EQUATORIAL_RADIUS_KM if sma.size else sma
            per_plot = perigee - WGS84_EQUATORIAL_RADIUS_KM if perigee.size else perigee
            apo_plot = apogee - WGS84_EQUATORIAL_RADIUS_KM if apogee.size else apogee
            times_ds, (sma_ds, per_ds, apo_ds) = self._downsample_series(
                times_hours, sma_plot, per_plot, apo_plot
            )
            if sma_ds.size == times_ds.size:
                plot.plot(
                    times_ds,
                    sma_ds,
                    pen=pg.mkPen("#2196f3", width=2),
                    name="Semi-major axis altitude",
                )
            if per_ds.size == times_ds.size:
                plot.plot(
                    times_ds,
                    per_ds,
                    pen=pg.mkPen(
                        "#ff9800",
                        width=1.5,
                        style=pg.QtCore.Qt.PenStyle.DashLine,  # type: ignore[attr-defined]
                    ),  # type: ignore[arg-type]
                    name="Perigee altitude (from elements)",
                )
            if apo_ds.size == times_ds.size:
                plot.plot(
                    times_ds,
                    apo_ds,
                    pen=pg.mkPen(
                        "#4caf50",
                        width=1.5,
                        style=pg.QtCore.Qt.PenStyle.DashLine,  # type: ignore[attr-defined]
                    ),  # type: ignore[arg-type]
                    name="Apogee altitude (from elements)",
                )
            plot.getViewBox().autoRange()
            return

        # Eccentricity (Brouwer–Lyddane mean).
        if index == 2:
            plot.setTitle("Eccentricity (Brouwer–Lyddane mean)")
            plot.setLabel("bottom", "Time since start", units="h")
            plot.setLabel("left", "e", units="")
            times_ds, (ecc_ds,) = self._downsample_series(times_hours, ecc)
            plot.plot(
                times_ds,
                ecc_ds,
                pen=pg.mkPen("#f06292", width=2),
                name="Eccentricity (Brouwer–Lyddane mean)",
            )
            plot.getViewBox().autoRange()
            return

        # Argument of perigee (Brouwer–Lyddane mean).
        if index == 3:
            plot.setTitle("Argument of Perigee (Brouwer–Lyddane mean)")
            plot.setLabel("bottom", "Time since start", units="h")
            plot.setLabel("left", "ω", units="deg")
            times_ds, (argp_ds,) = self._downsample_series(times_hours, argp)
            plot.plot(
                times_ds,
                argp_ds,
                pen=pg.mkPen("#ba68c8", width=2),
                name="Argument of perigee (Brouwer–Lyddane mean)",
            )
            plot.getViewBox().autoRange()
            return

        # Orbital period (Brouwer–Lyddane mean).
        if index == 4:
            plot.setTitle("Orbital Period (Brouwer–Lyddane mean)")
            plot.setLabel("bottom", "Time since start", units="h")
            plot.setLabel("left", "Period", units="s")
            times_ds, (period_ds,) = self._downsample_series(times_hours, period)
            plot.plot(
                times_ds,
                period_ds,
                pen=pg.mkPen("#ffb74d", width=2),
                name="Orbital period (Brouwer–Lyddane mean)",
            )
            plot.getViewBox().autoRange()
            return

        # Atmospheric density (Brouwer–Lyddane mean).
        if index == 5:
            plot.setTitle("Atmospheric Density (Brouwer–Lyddane mean)")
            plot.setLabel("bottom", "Time since start", units="h")
            plot.setLabel("left", "Density", units="kg/m³")
            times_ds, (rho_ds,) = self._downsample_series(times_hours, density)
            plot.plot(
                times_ds,
                rho_ds,
                pen=pg.mkPen("#90caf9", width=2),
                name="Density (Brouwer–Lyddane mean)",
            )
            plot.getViewBox().autoRange()
            return

        # Dynamic pressure (Brouwer–Lyddane mean).
        plot.setTitle("Dynamic Pressure (Brouwer–Lyddane mean)")
        plot.setLabel("bottom", "Time since start", units="h")
        plot.setLabel("left", "q", units="Pa")
        times_ds, (dynp_ds,) = self._downsample_series(times_hours, dynp)
        plot.plot(
            times_ds,
            dynp_ds,
            pen=pg.mkPen("#ffb300", width=2),
            name="Dynamic pressure (Brouwer–Lyddane mean)",
        )
        plot.getViewBox().autoRange()

    def _render_instant_metric(self, index: int) -> None:
        """Render the instantaneous (osculating) metric selected in the combo box."""
        if not self._orbit_summary_data or not getattr(self, "orbit_instant_plot", None):
            return

        times_hours = self._orbit_summary_data["times_hours"]
        alt = self._orbit_summary_data.get("inst_altitude_km", np.array([], dtype=float))
        sma = self._orbit_summary_data.get(
            "inst_semi_major_axis_km", np.array([], dtype=float)
        )
        perigee = self._orbit_summary_data.get(
            "inst_perigee_altitude_km", np.array([], dtype=float)
        )
        apogee = self._orbit_summary_data.get(
            "inst_apogee_altitude_km", np.array([], dtype=float)
        )
        ecc = self._orbit_summary_data.get("inst_eccentricity", np.array([], dtype=float))
        argp = self._orbit_summary_data.get(
            "inst_argument_of_perigee_deg", np.array([], dtype=float)
        )
        true_anom = self._orbit_summary_data.get(
            "inst_true_anomaly_deg", np.array([], dtype=float)
        )
        period = self._orbit_summary_data.get("inst_period_seconds", np.array([], dtype=float))
        density = self._orbit_summary_data.get(
            "inst_density_kg_m3", np.array([], dtype=float)
        )
        dynp = self._orbit_summary_data.get(
            "inst_dynamic_pressure_pa", np.array([], dtype=float)
        )
        drag_force = self._orbit_summary_data.get(
            "inst_drag_force_N", np.array([], dtype=float)
        )

        plot = self.orbit_instant_plot
        legend = getattr(self, "_orbit_instant_legend", None)
        plot.clear()
        if legend is not None:
            legend.clear()

        # Geodetic altitude only (instantaneous).
        if index == 0:
            plot.setTitle("Geodetic Altitude (instantaneous)")
            plot.setLabel("bottom", "Time since start", units="h")
            plot.setLabel("left", "Geodetic altitude", units="km")
            times_ds, (alt_ds,) = self._downsample_series(
                times_hours, alt, max_points=2000
            )
            plot.plot(
                times_ds,
                alt_ds,
                pen=pg.mkPen("#76c7ff", width=2),
                name="Geodetic altitude",
            )
            plot.getViewBox().autoRange()
            return

        # Semi-major axis and apsides from orbital elements (instantaneous).
        if index == 1:
            plot.setTitle("Semi-major Axis and Apsides (instantaneous)")
            plot.setLabel("bottom", "Time since start", units="h")
            plot.setLabel("left", "Altitude (from elements)", units="km")
            sma_plot = sma - WGS84_EQUATORIAL_RADIUS_KM if sma.size else sma
            per_plot = perigee - WGS84_EQUATORIAL_RADIUS_KM if perigee.size else perigee
            apo_plot = apogee - WGS84_EQUATORIAL_RADIUS_KM if apogee.size else apogee
            times_ds, (sma_ds, per_ds, apo_ds) = self._downsample_series(
                times_hours, sma_plot, per_plot, apo_plot, max_points=2000
            )
            if sma_ds.size == times_ds.size:
                plot.plot(
                    times_ds,
                    sma_ds,
                    pen=pg.mkPen("#2196f3", width=2),
                    name="Semi-major axis altitude",
                )
            if per_ds.size == times_ds.size:
                plot.plot(
                    times_ds,
                    per_ds,
                    pen=pg.mkPen(
                        "#ff9800",
                        width=1.5,
                        style=pg.QtCore.Qt.PenStyle.DashLine,  # type: ignore[attr-defined]
                    ),
                    name="Perigee altitude (from elements)",
                )
            if apo_ds.size == times_ds.size:
                plot.plot(
                    times_ds,
                    apo_ds,
                    pen=pg.mkPen(
                        "#4caf50",
                        width=1.5,
                        style=pg.QtCore.Qt.PenStyle.DashLine,  # type: ignore[attr-defined]
                    ),
                    name="Apogee altitude (from elements)",
                )
            plot.getViewBox().autoRange()
            return

        # Eccentricity (instantaneous).
        if index == 2:
            plot.setTitle("Eccentricity (instantaneous)")
            plot.setLabel("bottom", "Time since start", units="h")
            plot.setLabel("left", "e", units="")
            times_ds, (ecc_ds,) = self._downsample_series(times_hours, ecc, max_points=2000)
            plot.plot(
                times_ds,
                ecc_ds,
                pen=pg.mkPen("#f06292", width=2),
                name="Eccentricity",
            )
            plot.getViewBox().autoRange()
            return

        # Argument of perigee (instantaneous).
        if index == 3:
            plot.setTitle("Argument of Perigee (instantaneous)")
            plot.setLabel("bottom", "Time since start", units="h")
            plot.setLabel("left", "ω", units="deg")
            times_ds, (argp_ds,) = self._downsample_series(times_hours, argp, max_points=2000)
            plot.plot(
                times_ds,
                argp_ds,
                pen=pg.mkPen("#ba68c8", width=2),
                name="Argument of perigee",
            )
            plot.getViewBox().autoRange()
            return

        # True anomaly (instantaneous).
        if index == 4:
            plot.setTitle("True Anomaly (instantaneous)")
            plot.setLabel("bottom", "Time since start", units="h")
            plot.setLabel("left", "ν", units="deg")
            times_ds, (ta_ds,) = self._downsample_series(
                times_hours, true_anom, max_points=2000
            )
            plot.plot(
                times_ds,
                ta_ds,
                pen=pg.mkPen("#26a69a", width=2),
                name="True anomaly",
            )
            plot.getViewBox().autoRange()
            return

        # Orbital period (instantaneous).
        if index == 5:
            plot.setTitle("Orbital Period (instantaneous)")
            plot.setLabel("bottom", "Time since start", units="h")
            plot.setLabel("left", "Period", units="s")
            times_ds, (period_ds,) = self._downsample_series(
                times_hours, period, max_points=2000
            )
            plot.plot(
                times_ds,
                period_ds,
                pen=pg.mkPen("#ffb74d", width=2),
                name="Orbital period",
            )
            plot.getViewBox().autoRange()
            return

        # Atmospheric density (instantaneous).
        if index == 6:
            plot.setTitle("Atmospheric Density (instantaneous)")
            plot.setLabel("bottom", "Time since start", units="h")
            plot.setLabel("left", "Density", units="kg/m³")
            times_ds, (rho_ds,) = self._downsample_series(
                times_hours, density, max_points=2000
            )
            plot.plot(
                times_ds,
                rho_ds,
                pen=pg.mkPen("#90caf9", width=2),
                name="Density",
            )
            plot.getViewBox().autoRange()
            return

        # Dynamic pressure (instantaneous).
        if index == 7:
            plot.setTitle("Dynamic Pressure (instantaneous)")
            plot.setLabel("bottom", "Time since start", units="h")
            plot.setLabel("left", "q", units="Pa")
            times_ds, (dynp_ds,) = self._downsample_series(
                times_hours, dynp, max_points=2000
            )
            plot.plot(
                times_ds,
                dynp_ds,
                pen=pg.mkPen("#ffb300", width=2),
                name="Dynamic pressure",
            )
            plot.getViewBox().autoRange()
            return

        # Drag force (instantaneous, positive magnitude).
        plot.setTitle("Drag Force (instantaneous)")
        plot.setLabel("bottom", "Time since start", units="h")
        plot.setLabel("left", "Force", units="N")
        times_ds, (drag_ds,) = self._downsample_series(times_hours, drag_force, max_points=2000)
        if drag_ds.size == times_ds.size:
            plot.plot(
                times_ds,
                drag_ds,
                pen=pg.mkPen("#ff7043", width=2),
                name="Drag force",
            )
        plot.getViewBox().autoRange()
