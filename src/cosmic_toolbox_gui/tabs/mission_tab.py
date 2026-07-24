"""Mission analysis tab mixin."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
    QSizePolicy,
)

from cosmic_toolbox import orbit_utils
from cosmic_toolbox.models import PropagatedEphemeris
from cosmic_toolbox_gui.constants import EARTH_ROTATION_RATE_RAD_PER_SEC
from cosmic_toolbox_gui.globe_math import rotate_vector_z
from cosmic_toolbox_gui.tabs.visualization_tab import ecef_to_globe_coords
from cosmic_toolbox_gui.opengl import GlobeWidget


class MissionTabMixin:
    """Logic for building and updating the mission analysis tab."""

    _mission_ground_track: PropagatedEphemeris | None

    # Orbit constants and the SSO solve now live in cosmic_toolbox.orbit_utils
    # so scripts share one implementation; these aliases keep the old names.
    _WGS84_J2 = orbit_utils.WGS84_J2
    _WGS84_EARTH_EQUATORIAL_RADIUS_M = orbit_utils.WGS84_EARTH_EQUATORIAL_RADIUS_M
    _WGS84_EARTH_MU_M3_S2 = orbit_utils.WGS84_EARTH_MU_M3_S2
    _TROPICAL_YEAR_DAYS = orbit_utils.TROPICAL_YEAR_DAYS

    def _build_mission_analysis_tab(self) -> QWidget:
        """Create the Mission Configuration tab with configuration and mission map."""
        tab = QWidget()
        tab_layout = QVBoxLayout(tab)
        config_widget = QWidget()
        config_layout = QVBoxLayout(config_widget)
        config_layout.addWidget(self._build_orbit_group())
        config_layout.addWidget(self._build_scenario_group())
        config_layout.addWidget(self._build_propagation_group())
        button_row = QHBoxLayout()
        # Stack Run/Stop vertically so the stop button appears directly under Run.
        button_column = QVBoxLayout()
        self.run_button = QPushButton("Run Analysis")
        self._set_run_button_state("dirty")
        self.run_button.clicked.connect(self._handle_run_clicked)  # type: ignore[arg-type]
        button_column.addWidget(self.run_button)
        # Stop button is disabled until a run is in progress.
        from PySide6.QtWidgets import QPushButton as _QPushButtonAlias  # local alias to avoid circular import hints

        self.stop_button = getattr(self, "stop_button", None)
        if self.stop_button is None:
            self.stop_button = _QPushButtonAlias("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self._handle_stop_clicked)  # type: ignore[arg-type]
        button_column.addWidget(self.stop_button)
        self.refresh_cached_access_button = QPushButton(
            "Refresh Access From Cached Trajectory"
        )
        self.refresh_cached_access_button.setEnabled(False)
        self.refresh_cached_access_button.clicked.connect(
            self._handle_refresh_cached_access_clicked
        )  # type: ignore[arg-type]
        button_column.addWidget(self.refresh_cached_access_button)
        self.export_cached_package_button = QPushButton(
            "Export Cached Trajectory Package"
        )
        self.export_cached_package_button.setEnabled(False)
        self.export_cached_package_button.clicked.connect(
            self._handle_export_cached_package_clicked
        )  # type: ignore[arg-type]
        button_column.addWidget(self.export_cached_package_button)
        self.import_cached_package_button = QPushButton(
            "Import Cached Trajectory Package"
        )
        self.import_cached_package_button.clicked.connect(
            self._handle_import_cached_package_clicked
        )  # type: ignore[arg-type]
        button_column.addWidget(self.import_cached_package_button)
        button_row.addLayout(button_column)
        self.run_progress = QProgressBar()
        self.run_progress.setRange(0, 100)
        self.run_progress.setValue(0)
        self.run_progress.setFormat("0%")
        self.run_progress.setTextVisible(True)
        self.run_progress.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.run_progress.setStyleSheet(
            """
            QProgressBar {
                border: 1px solid #1E90FF;
                border-radius: 4px;
                background-color: #0b1f2d;
                color: white;
                padding: 2px;
                min-height: 28px;
            }
            QProgressBar::chunk {
                background-color: #76c7ff;
                border-radius: 4px;
            }
            """
        )
        self.run_progress.hide()
        button_row.addWidget(self.run_progress)
        config_layout.addLayout(button_row)
        self.analysis_cache_status_label = QLabel(
            "Run a mission scenario to cache a reusable trajectory."
        )
        self.analysis_cache_status_label.setWordWrap(True)
        config_layout.addWidget(self.analysis_cache_status_label)
        config_layout.addStretch(1)

        # Make the configuration column scrollable so the main window can be resized
        # to smaller heights without forcing all controls to remain visible.
        config_scroll = QScrollArea()
        config_scroll.setWidgetResizable(True)
        config_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        config_scroll.setWidget(config_widget)
        config_scroll.setMinimumWidth(0)
        config_scroll.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding
        )

        analysis_widget = QWidget()
        analysis_layout = QVBoxLayout(analysis_widget)
        # Mission tab now focuses on configuration + orbit overview globe.
        analysis_layout.addWidget(self._build_mission_globe_panel())
        analysis_widget.setMinimumWidth(0)
        analysis_widget.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding
        )
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(True)
        splitter.addWidget(config_scroll)
        splitter.addWidget(analysis_widget)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 5)
        splitter.setSizes([420, 980])
        tab_layout.addWidget(splitter)
        return tab

    def _build_mission_globe_panel(self) -> QWidget:
        """Create the mission analysis globe with frame toggle and orbit slider."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.mission_globe_widget = GlobeWidget()
        self.mission_globe_widget.setMinimumWidth(0)
        self.mission_globe_widget.set_earth_render_mode("wireframe")
        self.mission_globe_widget.reset_camera()
        self.mission_globe_widget.set_day_night_enabled(False)
        self.mission_globe_widget.set_uniform_lighting(True)
        self._update_mission_earth_rotation(None)
        layout.addWidget(self.mission_globe_widget, stretch=1)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Frame:"))
        self.mission_frame_combo = QComboBox()
        self.mission_frame_combo.addItems(["ECI (inertial)", "ECEF (earth-fixed)"])
        self.mission_frame_combo.setCurrentIndex(0)
        self.mission_frame_combo.currentIndexChanged.connect(
            self._handle_mission_frame_changed
        )  # type: ignore[attr-defined]
        controls.addWidget(self.mission_frame_combo)
        controls.addWidget(QLabel("Scenario window:"))
        self.mission_window_slider = QSlider(Qt.Orientation.Horizontal)
        self.mission_window_slider.setRange(0, 1000)
        self.mission_window_slider.setSingleStep(5)
        self.mission_window_slider.setValue(0)
        self.mission_window_slider.valueChanged.connect(
            self._handle_mission_window_changed
        )  # type: ignore[attr-defined]
        controls.addWidget(self.mission_window_slider, stretch=1)
        self.mission_window_label = QLabel("Start → End")
        controls.addWidget(self.mission_window_label)
        layout.addLayout(controls)
        self.mission_globe_status_label = QLabel("Run analysis to view ground track.")
        self.mission_globe_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.mission_globe_status_label)
        self._apply_mission_control_state()
        return tab

    def _build_orbit_group(self) -> QGroupBox:
        """Create widgets for the orbital elements."""
        group = QGroupBox("Initial State")
        form = QFormLayout(group)
        # Primary orbit size input is geodetic altitude above the WGS-84 ellipsoid.
        self.altitude_input = QDoubleSpinBox()
        self.altitude_input.setRange(80.0, 50_000.0)
        self.altitude_input.setDecimals(1)
        self.altitude_input.setSingleStep(5.0)
        self.altitude_input.setValue(250.0)
        self.altitude_input.setSuffix(" km")

        from PySide6.QtWidgets import QCheckBox

        self.sso_checkbox = QCheckBox("Sun-synchronous (SSO)")
        self.sso_checkbox.setChecked(True)
        self.sso_checkbox.stateChanged.connect(self._handle_sso_toggle)  # type: ignore[arg-type]
        self.ecc_input = QDoubleSpinBox()
        self.ecc_input.setRange(0.0, 0.99)
        self.ecc_input.setDecimals(5)
        self.ecc_input.setValue(0.0)
        self.inc_input = QDoubleSpinBox()
        self.inc_input.setRange(0.0, 180.0)
        self.inc_input.setValue(97.4)
        self.inc_input.setSuffix(" °")
        self.raan_input = QDoubleSpinBox()
        self.raan_input.setRange(0.0, 360.0)
        self.raan_input.setValue(0.0)
        self.raan_input.setSuffix(" °")
        self.argp_input = QDoubleSpinBox()
        self.argp_input.setRange(0.0, 360.0)
        self.argp_input.setValue(90.0)
        self.argp_input.setSuffix(" °")
        self.mean_anom_input = QDoubleSpinBox()
        self.mean_anom_input.setRange(0.0, 360.0)
        self.mean_anom_input.setValue(0.0)
        self.mean_anom_input.setSuffix(" °")
        form.addRow("Geodetic altitude:", self.altitude_input)
        self.altitude_input.valueChanged.connect(self._handle_orbit_size_changed)  # type: ignore[arg-type]
        form.addRow("", self.sso_checkbox)
        form.addRow("Eccentricity:", self.ecc_input)
        self.ecc_input.valueChanged.connect(self._handle_orbit_shape_changed)  # type: ignore[arg-type]
        form.addRow("Inclination:", self.inc_input)
        self.inc_input.valueChanged.connect(lambda *_: self._mark_dirty())
        form.addRow("RAAN:", self.raan_input)
        self.raan_input.valueChanged.connect(lambda *_: self._mark_dirty())
        form.addRow("Arg Perigee:", self.argp_input)
        self.argp_input.valueChanged.connect(lambda *_: self._mark_dirty())
        form.addRow("Mean Anomaly:", self.mean_anom_input)
        self.mean_anom_input.valueChanged.connect(lambda *_: self._mark_dirty())
        return group

    def _sso_rate_rad_s(self) -> float:
        return orbit_utils.sso_raan_rate_rad_s()

    def _compute_sso_inclination_deg(self, *, altitude_km: float, eccentricity: float) -> float | None:
        """Return the Sun-synchronous inclination (deg) for the requested orbit size/shape."""
        return orbit_utils.sso_inclination_deg(
            altitude_km=altitude_km, eccentricity=eccentricity
        )

    def _apply_sso_if_enabled(self) -> None:
        """If SSO is enabled, compute and lock the inclination."""
        if getattr(self, "sso_checkbox", None) is None or getattr(self, "inc_input", None) is None:
            return
        if not self.sso_checkbox.isChecked():
            self.inc_input.setEnabled(True)
            return
        alt = float(getattr(self, "altitude_input").value()) if getattr(self, "altitude_input", None) is not None else 0.0
        ecc = float(getattr(self, "ecc_input").value()) if getattr(self, "ecc_input", None) is not None else 0.0
        inc = self._compute_sso_inclination_deg(altitude_km=alt, eccentricity=ecc)
        if inc is None:
            # Leave the existing inclination editable if no solution can be computed.
            self.inc_input.setEnabled(True)
            return
        self.inc_input.blockSignals(True)
        try:
            self.inc_input.setValue(float(inc))
        finally:
            self.inc_input.blockSignals(False)
        self.inc_input.setEnabled(False)

    def _handle_sso_toggle(self, _state: int) -> None:
        self._apply_sso_if_enabled()
        self._mark_dirty()

    def _handle_orbit_size_changed(self, *_args) -> None:
        self._apply_sso_if_enabled()
        self._mark_dirty()

    def _handle_orbit_shape_changed(self, *_args) -> None:
        self._apply_sso_if_enabled()
        self._mark_dirty()

    def _build_scenario_group(self) -> QGroupBox:
        """Create date/time controls for the scenario window."""
        group = QGroupBox("Scenario Window")
        layout = QHBoxLayout(group)
        # Default scenario window (UTC)
        default_start = datetime(2028, 2, 1, 12, 0, 0, tzinfo=timezone.utc)
        from PySide6.QtWidgets import QDateTimeEdit

        self.start_datetime = QDateTimeEdit(default_start)
        self.start_datetime.setDisplayFormat("dd-MMM-yyyy HH:mm:ss")
        self.start_datetime.setCalendarPopup(True)
        self.end_datetime = QDateTimeEdit(default_start + timedelta(days=7))
        self.end_datetime.setDisplayFormat("dd-MMM-yyyy HH:mm:ss")
        self.end_datetime.setCalendarPopup(True)
        layout.addWidget(QLabel("Start (UTC):"))
        layout.addWidget(self.start_datetime)
        self.start_datetime.dateTimeChanged.connect(lambda *_: self._mark_dirty())
        layout.addWidget(QLabel("End (UTC):"))
        layout.addWidget(self.end_datetime)
        self.end_datetime.dateTimeChanged.connect(lambda *_: self._mark_dirty())
        return group

    def _build_propagation_group(self) -> QGroupBox:
        """Create controls for propagation and high-level analysis options."""
        group = QGroupBox("Analysis Configuration")
        layout = QVBoxLayout(group)
        form = QFormLayout()
        self.propagator_combo = QComboBox()
        # Brouwer-Lyddane (analytical J2-J5 + secular drag) is the default: it
        # matches the numerical propagator closely for these orbits while being
        # orders of magnitude faster. Keplerian is available for quick, coarse
        # previews with no zonal-harmonics perturbation.
        self.propagator_combo.addItems(["Brouwer-Lyddane", "Keplerian"])
        self.sample_step_input = QSpinBox()
        # Allow single-digit seconds (e.g. 1–9s) for high-resolution sampling.
        self.sample_step_input.setRange(1, 3600)
        self.sample_step_input.setValue(60)
        form.addRow("Propagator:", self.propagator_combo)
        self.propagator_combo.currentIndexChanged.connect(lambda *_: self._mark_dirty())
        form.addRow("Sample Step (s):", self.sample_step_input)
        self.sample_step_input.valueChanged.connect(lambda *_: self._mark_dirty())
        layout.addLayout(form)

        # Ground-station analysis settings and toggles
        from PySide6.QtWidgets import QCheckBox

        self.gs_access_group = QGroupBox("Ground-station Pass Settings")
        gs_form = QFormLayout(self.gs_access_group)
        self.ground_pass_checkbox = getattr(self, "ground_pass_checkbox", None)
        if self.ground_pass_checkbox is None:
            self.ground_pass_checkbox = QCheckBox("Include ground-station passes")
        # Default ON so the user immediately gets pass outputs after a run.
        self.ground_pass_checkbox.setChecked(True)
        self.ground_pass_checkbox.stateChanged.connect(
            self._handle_analysis_options_changed
        )  # type: ignore[arg-type]
        gs_form.addRow(self.ground_pass_checkbox)

        # Comms steering elevation threshold (decoupled from pass detection).
        self.contact_attitude_elevation_input = QDoubleSpinBox()
        self.contact_attitude_elevation_input.setRange(0.0, 90.0)
        self.contact_attitude_elevation_input.setDecimals(2)
        self.contact_attitude_elevation_input.setValue(1.0)
        self.contact_attitude_elevation_input.setSuffix(" °")
        self.contact_attitude_elevation_input.valueChanged.connect(
            lambda *_: self._mark_derived_outputs_stale(
                "Derived outputs stale. Refresh access from cached trajectory to apply the updated comms steering threshold."
            )
        )
        gs_form.addRow("Comms steering elevation:", self.contact_attitude_elevation_input)

        self.comms_pointing_mode_combo = getattr(self, "comms_pointing_mode_combo", None)
        if self.comms_pointing_mode_combo is None:
            self.comms_pointing_mode_combo = QComboBox()
            self.comms_pointing_mode_combo.addItems(
                ["Prograde Pointing", "Free to Roll", "Constrained AoA"]
            )
        self.comms_pointing_mode_combo.setCurrentIndex(0)
        self.comms_pointing_mode_combo.currentIndexChanged.connect(
            lambda *_: self._mark_derived_outputs_stale(
                "Derived outputs stale. Refresh access from cached trajectory to compare the updated comms pointing mode."
            )
        )  # type: ignore[arg-type]
        gs_form.addRow("Comms pointing mode:", self.comms_pointing_mode_combo)

        self.comms_pointing_aoa_limit_input = getattr(
            self, "comms_pointing_aoa_limit_input", None
        )
        if self.comms_pointing_aoa_limit_input is None:
            self.comms_pointing_aoa_limit_input = QDoubleSpinBox()
        self.comms_pointing_aoa_limit_input.setRange(0.0, 180.0)
        self.comms_pointing_aoa_limit_input.setDecimals(2)
        self.comms_pointing_aoa_limit_input.setSingleStep(1.0)
        self.comms_pointing_aoa_limit_input.setValue(5.0)
        self.comms_pointing_aoa_limit_input.setSuffix(" °")
        self.comms_pointing_aoa_limit_input.valueChanged.connect(
            lambda *_: self._mark_derived_outputs_stale(
                "Derived outputs stale. Refresh access from cached trajectory to apply the updated AoA limit."
            )
        )  # type: ignore[arg-type]
        self.comms_pointing_aoa_limit_input.setEnabled(False)
        self.comms_pointing_mode_combo.currentIndexChanged.connect(  # type: ignore[arg-type]
            lambda *_: self.comms_pointing_aoa_limit_input.setEnabled(
                self.comms_pointing_mode_combo.currentText() == "Constrained AoA"
            )
        )
        gs_form.addRow("AoA limit about prograde:", self.comms_pointing_aoa_limit_input)

        comms_pointing_hint = QLabel(
            "Prograde Pointing keeps body +X locked to prograde, Free to Roll rolls "
            "about body +X during comms, and Constrained AoA steers within the "
            "configured |AoA| envelope while keeping roll free."
        )
        comms_pointing_hint.setWordWrap(True)
        gs_form.addRow("", comms_pointing_hint)
        layout.addWidget(self.gs_access_group)

        # Keep SSO inclination updated if enabled.
        self._apply_sso_if_enabled()
        return group

    def _handle_analysis_options_changed(self, state: int) -> None:
        """React to analysis option toggles (e.g., ground-station passes)."""
        _ = state
        enabled = self.ground_pass_checkbox.isChecked()
        # Keep the checkbox active so the user can re-enable GS passes.
        _ = enabled
        self._mark_derived_outputs_stale(
            "Derived outputs stale. Refresh access from cached trajectory to apply the updated ground-pass selection."
        )

    def _handle_mission_frame_changed(self, index: int) -> None:
        """Toggle the mission globe between inertial and Earth-fixed frames."""
        mode = "ECI" if index == 0 else "ECEF"
        if mode == self._mission_frame_mode:
            return
        self._mission_frame_mode = mode
        if getattr(self, "_mission_globe_refresh_timer", None) is not None:
            self._mission_globe_refresh_timer.stop()
        self._apply_mission_control_state()
        self._refresh_mission_globe()

    def _handle_mission_window_changed(self, value: int) -> None:
        """Adjust the portion of the scenario shown on the mission globe."""
        coverage = min(max(value / 1000.0, 0.0), 1.0)
        if coverage <= 0.0:
            self._mission_window_fractions = (0.0, 0.0)
        else:
            self._mission_window_fractions = (0.0, coverage)
        self._update_mission_window_label()
        self._schedule_mission_globe_refresh()

    def _apply_mission_control_state(self) -> None:
        """Enable or disable orbit controls."""
        if self.mission_window_slider:
            self.mission_window_slider.setEnabled(True)
        self._update_mission_window_label()

    def _update_mission_window_label(self) -> None:
        """Update the scenario window label."""
        label = self.mission_window_label
        if label is None:
            return
        if self._current_config is None:
            label.setText("Scenario window")
            return
        start_frac, end_frac = self._mission_window_fractions
        start_dt = self._current_config.scenario.start_time
        end_dt = self._current_config.scenario.end_time
        total = (end_dt - start_dt).total_seconds()
        if total <= 0:
            label.setText("Scenario window")
            return
        window_start = start_dt + timedelta(seconds=total * start_frac)
        window_end = start_dt + timedelta(seconds=total * end_frac)
        label.setText(
            f"{window_start:%d-%b %H:%M UTC} → {window_end:%d-%b %H:%M UTC}"
        )

    def _refresh_mission_globe(self) -> None:
        """Re-render the mission globe track using the latest state."""
        if getattr(self, "_mission_globe_refresh_timer", None) is not None:
            self._mission_globe_refresh_timer.stop()
        if getattr(self, "mission_globe_widget", None) is None:
            return
        eph: PropagatedEphemeris | None = self._mission_ground_track
        if eph is None:
            result = getattr(self, "_last_result", None)
            eph = result.ephemeris if result is not None else None
        if eph is None or eph.ecef_pos_km is None:
            return
        self._update_mission_globe_track(eph)

    def _schedule_mission_globe_refresh(self) -> None:
        """Debounce mission globe refreshes when controls change."""
        if self._mission_globe_refresh_timer is None:
            self._refresh_mission_globe()
            return
        # Don't clear the track immediately - let the timer handle the refresh
        # This prevents flickering when the slider is moved
        self._mission_globe_refresh_timer.start()

    def _update_mission_globe_track(self, eph: PropagatedEphemeris) -> None:
        """Plot the satellite ground track on the mission globe."""
        widget = getattr(self, "mission_globe_widget", None)
        if widget is None:
            return
        if eph.ecef_pos_km is None or self._current_config is None:
            if self.mission_globe_status_label:
                self.mission_globe_status_label.setText("Run analysis to view ground track.")
            widget.update_track(None)
            widget.update_satellite_position(None)
            return
        mask = self._extract_window_mask(eph)
        if mask is not None and not np.any(mask):
            if self.mission_globe_status_label:
                slider_value = (
                    self.mission_window_slider.value()
                    if self.mission_window_slider is not None
                    else None
                )
                message = (
                    "Move the orbit slider to reveal the ground track."
                    if slider_value == 0
                    else "No ground track samples in the selected window."
                )
                self.mission_globe_status_label.setText(message)
            widget.update_track(None)
            widget.update_satellite_position(None)
            widget.update_direction_arrow(None, None)
            return

        ecef_full: np.ndarray = eph.ecef_pos_km  # type: ignore[assignment]
        ts_full: np.ndarray = eph.timestamps_unix  # type: ignore[assignment]
        ecef = ecef_full if mask is None else ecef_full[mask]
        ts_arr = ts_full if mask is None else ts_full[mask]

        ts_datetimes = [
            datetime.fromtimestamp(float(t), tz=timezone.utc) for t in ts_arr
        ]
        coords = np.array(
            [
                self._mission_transform_vector(
                    ecef_to_globe_coords(float(ecef[i, 0]), float(ecef[i, 1]), float(ecef[i, 2])),
                    ts_datetimes[i],
                )
                for i in range(len(ts_datetimes))
            ],
            dtype=float,
        )

        # Downsample if too many points for rendering performance
        MAX_RENDER_POINTS = 25000
        if coords.shape[0] > MAX_RENDER_POINTS:
            step = coords.shape[0] // MAX_RENDER_POINTS
            coords = coords[::step]

        self._update_mission_earth_rotation(ts_datetimes[0] if ts_datetimes else None)
        widget.update_track(coords)
        widget.update_satellite_position(None)
        if coords.shape[0] >= 2:
            direction = coords[-1] - coords[-2]
            widget.update_direction_arrow(tuple(coords[-1]), tuple(direction))
        else:
            widget.update_direction_arrow(None, None)
        if self.mission_globe_status_label and ts_datetimes:
            start = ts_datetimes[0].strftime("%d-%b %H:%M")
            end = ts_datetimes[-1].strftime("%d-%b %H:%M")
            self.mission_globe_status_label.setText(
                f"Showing {start} → {end} ({len(ts_datetimes)} samples)"
            )

    def _extract_window_mask(self, eph: PropagatedEphemeris) -> "np.ndarray | None":
        """Return a boolean mask for samples in the selected scenario window, or None for all."""
        if (
            eph.timestamps_unix is None
            or self._current_config is None
            or self._current_config.scenario is None
        ):
            return None
        start_frac, end_frac = self._mission_window_fractions
        start_dt = self._current_config.scenario.start_time
        end_dt = self._current_config.scenario.end_time
        total = (end_dt - start_dt).total_seconds()
        if total <= 0:
            return None
        window_start = start_dt + timedelta(seconds=total * start_frac)
        window_end = start_dt + timedelta(seconds=total * end_frac)
        if window_end <= window_start:
            return np.zeros(eph.timestamps_unix.shape[0], dtype=bool)
        w_start_unix = window_start.timestamp()
        w_end_unix = window_end.timestamp()
        return (eph.timestamps_unix >= w_start_unix) & (eph.timestamps_unix <= w_end_unix)

    def _mission_transform_vector(
        self, vector: tuple[float, float, float], timestamp: datetime
    ) -> tuple[float, float, float]:
        """Transform vectors for the mission overview globe."""
        if self._mission_frame_mode != "ECI":
            return vector
        if self._mission_reference_epoch is None:
            return vector
        delta = (timestamp - self._mission_reference_epoch).total_seconds()
        angle = EARTH_ROTATION_RATE_RAD_PER_SEC * delta
        return rotate_vector_z(vector, angle)

    def _update_mission_earth_rotation(self, timestamp: datetime | None) -> None:
        """Rotate the mission globe actors to match the selected frame."""
        if self._mission_frame_mode == "ECI":
            if timestamp is None or self._mission_reference_epoch is None:
                angle_deg = 0.0
            else:
                delta = (timestamp - self._mission_reference_epoch).total_seconds()
                angle_deg = math.degrees(EARTH_ROTATION_RATE_RAD_PER_SEC * delta)
        else:
            angle_deg = 0.0
        widget = getattr(self, "mission_globe_widget", None)
        if widget is None:
            return
        widget.set_frame_rotation(self._mission_frame_mode, angle_deg)

