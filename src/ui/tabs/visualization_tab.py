"""Visualization tab mixin."""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QSlider,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from src.models import GroundStationConfig, GroundTrackPoint, PassStatistic
from src.ui.constants import EARTH_ROTATION_RATE_RAD_PER_SEC
from src.ui.globe_math import rotate_vector_z
from src.ui.opengl import GlobeWidget

STATION_VISUAL_OFFSET_KM = 25.0
SPEED_OF_LIGHT_MPS = 299_792_458.0


def ecef_to_globe_coords(x: float, y: float, z: float) -> tuple[float, float, float]:
    """Convert ECEF/ITRF coordinates to globe plotting coordinate system.

    The globe's texture mapping uses atan2(x,y) which creates a coordinate system
    rotated 90° from standard ECEF. This function applies a +90° Z-rotation:

    Standard ECEF:      Globe Coordinates:
    +X → 0° longitude   +Y → 0° longitude (Prime Meridian)
    +Y → 90° E         -X → 90° E
    +Z → North Pole     +Z → North Pole

    Args:
        x: ECEF X coordinate (towards Prime Meridian)
        y: ECEF Y coordinate (towards 90° East)
        z: ECEF Z coordinate (towards North Pole)

    Returns:
        Tuple (globe_x, globe_y, globe_z) in globe coordinate system
    """
    return (-y, x, z)


class VisualizationTabMixin:
    """Builds and controls the visualization tab and animations."""

    def _build_visualization_tab(self) -> QWidget:
        """Create the visualization tab with pass selection and globe view."""
        tab = QWidget()
        tab_layout = QVBoxLayout(tab)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        tab_layout.addWidget(splitter)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        instructions = QLabel(
            "Select a station tab to browse its passes. Passes are sorted by max elevation."
        )
        instructions.setWordWrap(True)
        left_layout.addWidget(instructions)
        frame_toggle_row = QHBoxLayout()
        frame_toggle_row.addWidget(QLabel("Reference frame:"))
        self.visual_frame_combo = QComboBox()
        self.visual_frame_combo.addItems(["ECI (inertial)", "ECEF (earth-fixed)"])
        self.visual_frame_combo.setCurrentIndex(0)
        self.visual_frame_combo.currentIndexChanged.connect(
            self._handle_visual_frame_changed
        )  # type: ignore[attr-defined]
        frame_toggle_row.addWidget(self.visual_frame_combo, stretch=1)
        left_layout.addLayout(frame_toggle_row)
        self.visual_station_tabs = QTabWidget()
        self.visual_station_tabs.setTabPosition(QTabWidget.TabPosition.West)
        self.visual_station_tabs.setDocumentMode(True)
        left_layout.addWidget(self.visual_station_tabs, stretch=1)
        splitter.addWidget(left_panel)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        viewer_tab = QWidget()
        viewer_layout = QVBoxLayout(viewer_tab)
        viewer_layout.setContentsMargins(0, 0, 0, 0)
        globe_panel = self._build_visual_globe_panel()
        viewer_layout.addWidget(globe_panel, stretch=1)
        controls_row = QHBoxLayout()
        self.visual_play_button = QPushButton("Play")
        self.visual_play_button.setEnabled(False)
        self.visual_play_button.clicked.connect(self._toggle_visualization_playback)  # type: ignore[attr-defined]
        controls_row.addWidget(self.visual_play_button)
        controls_row.addWidget(QLabel("Speed"))
        self.visual_speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.visual_speed_slider.setRange(25, 300)
        self.visual_speed_slider.setValue(100)
        self.visual_speed_slider.setFixedWidth(120)
        self.visual_speed_slider.valueChanged.connect(
            self._handle_visual_speed_changed
        )  # type: ignore[attr-defined]
        controls_row.addWidget(self.visual_speed_slider)
        self.visual_time_slider = QSlider(Qt.Orientation.Horizontal)
        self.visual_time_slider.setRange(0, 0)
        self.visual_time_slider.setEnabled(False)
        self.visual_time_slider.sliderMoved.connect(
            self._handle_visualization_slider_moved
        )  # type: ignore[attr-defined]
        controls_row.addWidget(self.visual_time_slider, stretch=1)
        self.visual_pass_status_label = QLabel("Run an analysis to populate passes.")
        self.visual_pass_status_label.setWordWrap(True)
        controls_row.addWidget(self.visual_pass_status_label, stretch=1)
        viewer_layout.addLayout(controls_row)
        graphs_tab = self._build_visual_graph_tab()
        self.visual_view_tabs = QTabWidget()
        self.visual_view_tabs.setTabPosition(QTabWidget.TabPosition.North)
        self.visual_view_tabs.addTab(viewer_tab, "Viewer")
        self.visual_view_tabs.addTab(graphs_tab, "Graphs")
        right_layout.addWidget(self.visual_view_tabs, stretch=1)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        self._refresh_visualization_pass_tabs(None)
        return tab

    def _build_visual_globe_panel(self) -> QWidget:
        """Create the visualization globe panel showing ground tracks."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        self.visual_globe_widget = GlobeWidget()
        self.visual_globe_widget.reset_camera()
        self._update_visual_earth_rotation(None)
        layout.addWidget(self.visual_globe_widget, stretch=1)
        brightness_row = QHBoxLayout()
        brightness_row.addWidget(QLabel("Sun brightness"))
        self.visual_brightness_slider = QSlider(Qt.Orientation.Horizontal)
        self.visual_brightness_slider.setRange(20, 300)
        self.visual_brightness_slider.setValue(100)
        self.visual_brightness_slider.setSingleStep(5)
        self.visual_brightness_slider.valueChanged.connect(
            self._handle_visual_brightness_changed
        )  # type: ignore[attr-defined]
        brightness_row.addWidget(self.visual_brightness_slider, stretch=1)
        layout.addLayout(brightness_row)
        self._handle_visual_brightness_changed(self.visual_brightness_slider.value())
        return panel

    def _build_visual_graph_tab(self) -> QWidget:
        """Create the pass-metric plotting tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(4, 4, 4, 4)
        self.visual_graph_combo = QComboBox()
        self.visual_graph_combo.addItems(
            [
                "Doppler shift",
                "Doppler rate",
                "Azimuth angle",
                "Elevation angle",
                "Azimuth rate",
                "Elevation rate",
            ]
        )
        self.visual_graph_combo.setEnabled(False)
        self.visual_graph_combo.currentIndexChanged.connect(
            self._handle_visual_graph_metric_changed
        )  # type: ignore[attr-defined]
        layout.addWidget(self.visual_graph_combo)
        self.visual_graph_plot = pg.PlotWidget(title="Select a pass to populate graphs.")
        self.visual_graph_plot.setLabel("bottom", "Minutes from AOS", units="min")
        self.visual_graph_plot.showGrid(x=True, y=True, alpha=0.3)
        layout.addWidget(self.visual_graph_plot, stretch=1)
        self._visual_graph_data = None
        self._update_visual_graph_placeholder("Select a pass to populate graphs.")
        return tab

    def _update_visual_graph_placeholder(self, message: str) -> None:
        """Clear the graph panel and show a status message."""
        if getattr(self, "visual_graph_plot", None) is None:
            return
        self.visual_graph_plot.clear()
        self.visual_graph_plot.setTitle(message)
        self.visual_graph_plot.setLabel("bottom", "Minutes from AOS", units="min")
        self.visual_graph_plot.setLabel("left", "", units="")
        if getattr(self, "visual_graph_combo", None) is not None:
            self.visual_graph_combo.setEnabled(False)

    def _reset_visual_graph_panel(self, message: str | None = None) -> None:
        """Reset stored graph data and update placeholder text."""
        self._visual_graph_data = None
        text = message or "Select a pass to populate graphs."
        self._update_visual_graph_placeholder(text)
        combo = getattr(self, "visual_graph_combo", None)
        if combo is not None:
            combo.blockSignals(True)
            combo.setCurrentIndex(0)
            combo.blockSignals(False)

    def _build_visualization_placeholder_panel(self) -> QWidget:
        """Show a hint in the Mission tab pointing to the Visualization tab."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        label = QLabel(
            "Open the Visualization tab to explore animated passes on the globe."
        )
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setWordWrap(True)
        layout.addWidget(label, alignment=Qt.AlignmentFlag.AlignCenter)
        return panel

    def _refresh_visualization_pass_tabs(self, result) -> None:
        """Populate the per-station pass tabs used for visualization playback."""
        if self.visual_station_tabs is None:
            return
        self.visual_station_tabs.blockSignals(True)
        self.visual_station_tabs.clear()
        self._visual_station_lists = {}
        if result is None or not result.passes:
            placeholder = QWidget()
            placeholder_layout = QVBoxLayout(placeholder)
            label = QLabel("Run the analysis to see pass animations.")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setWordWrap(True)
            placeholder_layout.addWidget(label, alignment=Qt.AlignmentFlag.AlignCenter)
            self.visual_station_tabs.addTab(placeholder, "Stations")
            self.visual_station_tabs.blockSignals(False)
            self._clear_visualization_display("Run an analysis to populate passes.")
            self._update_visual_frame_label()
            return
        station_groups: dict[str, list[PassStatistic]] = {}
        for item in result.passes:
            name = item.station_name or "Ground Station"
            station_groups.setdefault(name, []).append(item)
        self._compute_visualization_reference_frames(result)
        station_records: list[tuple[str, float, list[PassStatistic]]] = []
        for name, items in station_groups.items():
            sorted_items = sorted(
                items, key=lambda p: p.max_elevation_deg, reverse=True
            )
            max_elev = sorted_items[0].max_elevation_deg if sorted_items else 0.0
            station_records.append((name, max_elev, sorted_items))
        station_records.sort(key=lambda entry: entry[1], reverse=True)
        for station_name, _, passes in station_records:
            tab = QWidget()
            tab_layout = QVBoxLayout(tab)
            list_widget = QListWidget()
            list_widget.setProperty("station_name", station_name)
            list_widget.itemSelectionChanged.connect(self._handle_visual_pass_selection)  # type: ignore[attr-defined]
            for idx, pass_stat in enumerate(passes, start=1):
                text = (
                    f"{idx:02d}. {pass_stat.aos:%d-%b %H:%M:%S} UTC  |  "
                    f"Max {pass_stat.max_elevation_deg:.1f}°  |  "
                    f"{pass_stat.duration_minutes:.1f} min"
                )
                item = QListWidgetItem(text)
                item.setData(Qt.ItemDataRole.UserRole, pass_stat)
                list_widget.addItem(item)
            tab_layout.addWidget(list_widget)
            self.visual_station_tabs.addTab(tab, station_name)
            self._visual_station_lists[station_name] = list_widget
        self.visual_station_tabs.blockSignals(False)
        if station_records:
            first_station = station_records[0][0]
            first_list = self._visual_station_lists.get(first_station)
            if first_list and first_list.count() > 0:
                first_list.setCurrentRow(0)
        self._update_visual_frame_label()

    def _compute_visualization_reference_frames(self, result) -> None:
        """Initialize reference epoch information for visualization transforms."""
        if self._current_config is not None:
            self._visual_reference_epoch = self._current_config.scenario.start_time
        elif result.passes:
            self._visual_reference_epoch = result.passes[0].aos
        elif result.ground_track:
            self._visual_reference_epoch = result.ground_track[0].timestamp

    def _handle_visual_pass_selection(self) -> None:
        """React to pass selection within a station tab."""
        widget = self.sender()
        if not isinstance(widget, QListWidget):
            return
        item = widget.currentItem()
        if item is None:
            return
        pass_stat = item.data(Qt.ItemDataRole.UserRole)
        if isinstance(pass_stat, PassStatistic):
            self._load_visualization_pass(pass_stat)

    def _handle_visual_frame_changed(self, index: int) -> None:
        """Switch between ECI and ECEF rendering modes."""
        mode = "ECI" if index == 0 else "ECEF"
        if mode == self._visual_frame_mode:
            return
        self._visual_frame_mode = mode
        if self._visual_animation_timer.isActive():
            self._visual_animation_timer.stop()
            if self.visual_play_button:
                self.visual_play_button.setText("Play")
        self._rebuild_visualization_scene()
        self._update_visual_frame_label()

    def _handle_visual_brightness_changed(self, value: int) -> None:
        """Update sun brightness multiplier from the UI slider."""
        if self.visual_globe_widget is None:
            return
        intensity = max(0.05, value / 100.0)
        self.visual_globe_widget.set_sun_brightness(intensity)

    def _rebuild_visualization_scene(self, *, force: bool = False) -> None:
        """Reapply the visualization assets for the current frame mode."""
        _ = force  # force is kept for backwards compatibility with older calls
        if self.visual_globe_widget is None:
            return
        self._update_visual_earth_rotation(None)
        if self._visual_pass_track is not None:
            self._update_visualization_pass_geometry(self._visual_pass_track)
            self._update_visualization_frame()

    def _update_visual_frame_label(self) -> None:
        """Show the active frame mode in the status label."""
        if not self.visual_pass_status_label:
            return
        prefix = "ECI" if self._visual_frame_mode == "ECI" else "ECEF"
        text = self.visual_pass_status_label.text()
        if " | " in text:
            text = text.split(" | ", 1)[1]
        self.visual_pass_status_label.setText(f"{prefix} | {text}")

    def _load_visualization_pass(self, pass_stat: PassStatistic) -> None:
        """Prepare the globe animation for the selected pass."""
        if self._visual_animation_timer.isActive():
            self._visual_animation_timer.stop()
        track = self._extract_pass_track(pass_stat)
        if not track:
            self._clear_visualization_display("No track samples for the selected pass.")
            return
        self._visual_selected_pass = pass_stat
        self._visual_pass_track = track
        self._visual_animation_index = 0
        self._visual_anim_fraction = 0.0
        self._visual_contact_window = (pass_stat.aos, pass_stat.los)
        self._visual_station_ecef = self._resolve_station_coordinates(pass_stat)
        first_timestamp = track[0].timestamp if track else None
        self._update_visualization_focus_point(first_timestamp)
        self._update_visualization_pass_geometry(track)
        slider_max = max(len(track) - 1, 0)
        if self.visual_time_slider:
            self.visual_time_slider.setEnabled(slider_max > 0)
            self.visual_time_slider.setRange(0, slider_max)
            self.visual_time_slider.setPageStep(max(1, slider_max // 20 or 1))
            self.visual_time_slider.setValue(0)
        if track:
            self.visual_globe_widget.set_sun_datetime(track[0].timestamp)
        if self.visual_play_button:
            self.visual_play_button.setEnabled(True)
            self.visual_play_button.setText("Play")
        if self.visual_pass_status_label:
            station_name = pass_stat.station_name or "Ground Station"
            self.visual_pass_status_label.setText(
                f"{self._visual_frame_mode} | {station_name}: {pass_stat.aos:%d-%b %H:%M:%S} → "
                f"{pass_stat.los:%H:%M:%S} UTC (max {pass_stat.max_elevation_deg:.1f}°)"
            )
        self._update_visualization_frame()
        self._prepare_visualization_graph_data(pass_stat)

    def _prepare_visualization_graph_data(self, pass_stat: PassStatistic) -> None:
        """Extract per-pass look-angle series for graphing."""
        plot_ready = getattr(self, "visual_graph_plot", None) is not None
        if not plot_ready:
            return
        series = self._extract_pass_graph_series(pass_stat)
        if series is None:
            self._reset_visual_graph_panel("No samples available for this pass.")
            return
        self._visual_graph_data = series
        combo = getattr(self, "visual_graph_combo", None)
        if combo is not None:
            combo.setEnabled(True)
        current_index = combo.currentIndex() if combo is not None else 0
        self._render_visual_graph_metric(current_index)

    def _extract_pass_track(self, pass_stat: PassStatistic) -> list[GroundTrackPoint]:
        """Return the subset of ground-track points covering the selected pass."""
        if self._last_result is None or not self._last_result.ground_track:
            return []
        pad = timedelta(minutes=2)
        start = pass_stat.aos - pad
        end = pass_stat.los + pad
        segment = [
            point
            for point in self._last_result.ground_track
            if start <= point.timestamp <= end
        ]
        return segment or list(self._last_result.ground_track)

    def _extract_pass_graph_series(
        self, pass_stat: PassStatistic
    ) -> dict[str, np.ndarray] | None:
        """Slice the per-station time series down to the requested pass."""
        result = getattr(self, "_last_result", None)
        if result is None or not getattr(result, "timeline_seconds", None):
            return None
        timeline = np.asarray(result.timeline_seconds, dtype=float)
        if timeline.size == 0:
            return None
        station_series = getattr(result, "station_elevation_series", {})
        station_name = pass_stat.station_name or next(iter(station_series), None)
        if station_name is None:
            return None
        if station_name not in station_series:
            if station_series:
                station_name = next(iter(station_series))
            else:
                return None

        scenario_start = None
        config = getattr(self, "_current_config", None)
        if config is not None and getattr(config, "scenario", None) is not None:
            scenario_start = config.scenario.start_time
        if scenario_start is None:
            scenario_start = self._visual_reference_epoch
        if scenario_start is None:
            return None

        propagation = getattr(config, "propagation", None) if config else None
        sample_step = float(getattr(propagation, "sample_step_seconds", 10.0))

        def _series_from(source: dict[str, list[float]]) -> np.ndarray:
            values = source.get(station_name)
            return (
                np.asarray(values, dtype=float)
                if values is not None
                else np.array([], dtype=float)
            )

        elev = _series_from(station_series)
        azimuth = _series_from(getattr(result, "station_azimuth_series", {}))
        az_rate = _series_from(getattr(result, "station_az_rate_series", {}))
        el_rate = _series_from(getattr(result, "station_el_rate_series", {}))
        range_rate = _series_from(getattr(result, "station_range_rate_series", {}))
        range_accel = _series_from(getattr(result, "station_range_accel_series", {}))

        expected = timeline.size
        for array in (elev, azimuth, az_rate, el_rate, range_rate, range_accel):
            if array.size != expected or expected == 0:
                return None

        aos_sec = (pass_stat.aos - scenario_start).total_seconds()
        los_sec = (pass_stat.los - scenario_start).total_seconds()
        pad_seconds = max(sample_step, 5.0)
        mask = (timeline >= aos_sec - pad_seconds) & (timeline <= los_sec + pad_seconds)
        if not np.any(mask):
            return None

        return {
            "times_minutes": (timeline[mask] - aos_sec) / 60.0,
            "azimuth_deg": azimuth[mask],
            "elevation_deg": elev[mask],
            "az_rate_deg_s": az_rate[mask],
            "el_rate_deg_s": el_rate[mask],
            "range_rate_mps": range_rate[mask],
            "range_accel_mps2": range_accel[mask],
            "station_name": station_name,
        }

    def _handle_visual_graph_metric_changed(self, index: int) -> None:
        """Update the pass-metric plot when the selected metric changes."""
        self._render_visual_graph_metric(index)

    def _render_visual_graph_metric(self, index: int) -> None:
        """Render the selected pass metric."""
        plot = getattr(self, "visual_graph_plot", None)
        data = getattr(self, "_visual_graph_data", None)
        if plot is None:
            return
        if not data:
            self._update_visual_graph_placeholder("Select a pass to populate graphs.")
            return
        times = data.get("times_minutes")
        if times is None or len(times) == 0:
            self._update_visual_graph_placeholder("No samples available for this pass.")
            return

        station_name = data.get("station_name", "Station")
        if index == 0:
            values = self._compute_doppler_shift(data.get("range_rate_mps"))
            y_label = "Doppler shift"
            units = "Hz"
            title = "Doppler Shift"
        elif index == 1:
            values = self._compute_doppler_rate(data.get("range_accel_mps2"))
            y_label = "Doppler rate"
            units = "Hz/s"
            title = "Doppler Rate"
        elif index == 2:
            values = np.asarray(data.get("azimuth_deg"))
            y_label = "Azimuth"
            units = "deg"
            title = "Azimuth Angle"
        elif index == 3:
            values = np.asarray(data.get("elevation_deg"))
            y_label = "Elevation"
            units = "deg"
            title = "Elevation Angle"
        elif index == 4:
            values = np.asarray(data.get("az_rate_deg_s"))
            y_label = "Azimuth rate"
            units = "deg/s"
            title = "Required Azimuth Rate"
        else:
            values = np.asarray(data.get("el_rate_deg_s"))
            y_label = "Elevation rate"
            units = "deg/s"
            title = "Required Elevation Rate"

        if values.size != times.size:
            self._update_visual_graph_placeholder("Incomplete samples for this metric.")
            return

        times_ds, values_ds = self._downsample_graph_series(times, values)
        plot.clear()
        plot.setTitle(f"{title} — {station_name}")
        plot.setLabel("bottom", "Minutes from AOS", units="min")
        plot.setLabel("left", y_label, units=units)
        plot.plot(times_ds, values_ds, pen=pg.mkPen("#76c7ff", width=2))
        plot.getViewBox().autoRange()

    def _compute_doppler_shift(self, range_rate: np.ndarray | None) -> np.ndarray:
        """Convert the line-of-sight range rate into Doppler shift."""
        if range_rate is None:
            return np.array([])
        freq_input = getattr(self, "lb_frequency_input", None)
        freq_ghz = float(freq_input.value()) if freq_input is not None else 8.2
        freq_hz = max(freq_ghz, 0.0) * 1e9
        if freq_hz <= 0.0:
            return np.zeros_like(range_rate)
        scale = freq_hz / SPEED_OF_LIGHT_MPS
        return -np.asarray(range_rate) * scale

    def _compute_doppler_rate(self, range_accel: np.ndarray | None) -> np.ndarray:
        """Convert line-of-sight acceleration into Doppler rate."""
        if range_accel is None:
            return np.array([])
        freq_input = getattr(self, "lb_frequency_input", None)
        freq_ghz = float(freq_input.value()) if freq_input is not None else 8.2
        freq_hz = max(freq_ghz, 0.0) * 1e9
        if freq_hz <= 0.0:
            return np.zeros_like(range_accel)
        scale = freq_hz / SPEED_OF_LIGHT_MPS
        return -np.asarray(range_accel) * scale

    def _downsample_graph_series(
        self, times: np.ndarray, values: np.ndarray, max_points: int = 2000
    ) -> tuple[np.ndarray, np.ndarray]:
        """Reduce plotted sample count to keep PyQtGraph responsive."""
        if times.size <= max_points:
            return times, values
        indices = np.linspace(0, times.size - 1, max_points, dtype=int)
        indices = np.unique(indices)
        return times[indices], values[indices]

    def _handle_visual_frequency_changed(self, _value: float) -> None:
        """Re-render Doppler plots when the link-budget frequency changes."""
        combo = getattr(self, "visual_graph_combo", None)
        if combo is None:
            return
        current = combo.currentIndex()
        if current not in (0, 1):
            return
        self._render_visual_graph_metric(current)

    def _attach_visualization_frequency_listener(self) -> None:
        """Connect the link-budget frequency input so Doppler plots stay in sync."""
        if getattr(self, "_visual_freq_listener_connected", False):
            return
        freq_input = getattr(self, "lb_frequency_input", None)
        if freq_input is None:
            return
        freq_input.valueChanged.connect(self._handle_visual_frequency_changed)  # type: ignore[attr-defined]
        self._visual_freq_listener_connected = True

    def _resolve_station_coordinates(
        self, pass_stat: PassStatistic
    ) -> tuple[float, float, float] | None:
        """Resolve the ECEF coordinates for the pass station."""
        station: GroundStationConfig | None = None
        station_name = pass_stat.station_name or ""
        if station_name:
            station = self._active_station_lookup.get(station_name)
            if station is None:
                station = next(
                    (s for s in self._station_presets if s.name == station_name),
                    None,
                )
        if station is None and self._station_presets:
            station = self._station_presets[0]
        if station is None:
            return None
        return self._station_to_ecef_km(station)

    def _station_to_ecef_km(
        self, station: GroundStationConfig
    ) -> tuple[float, float, float]:
        """Convert a ground-station lat/lon/alt to pure ECEF coordinates in kilometers."""
        lat = math.radians(station.latitude_deg)
        lon = math.radians(station.longitude_deg)
        alt_km = station.altitude_m / 1000.0
        a = 6378.137
        e2 = 6.69437999014e-3
        sin_lat = math.sin(lat)
        N = a / math.sqrt(1 - e2 * sin_lat * sin_lat)
        x = (N + alt_km) * math.cos(lat) * math.cos(lon)
        y = (N + alt_km) * math.cos(lat) * math.sin(lon)
        z = (N * (1 - e2) + alt_km) * sin_lat
        return (x, y, z)

    def _seconds_since_reference(self, timestamp: datetime) -> float:
        """Return elapsed seconds from the visualization reference epoch."""
        ref = self._visual_reference_epoch
        if ref is None:
            return 0.0
        return (timestamp - ref).total_seconds()

    def _convert_vector_to_current_frame(
        self, vector: tuple[float, float, float], timestamp: datetime
    ) -> tuple[float, float, float]:
        """Convert a globe-space vector into the currently selected frame."""
        if self._visual_frame_mode != "ECI":
            return vector
        delta = self._seconds_since_reference(timestamp)
        angle = EARTH_ROTATION_RATE_RAD_PER_SEC * delta
        return rotate_vector_z(vector, angle)

    def _get_visualization_track_coords(
        self, track: list[GroundTrackPoint]
    ) -> np.ndarray:
        """Return track coordinates transformed into the current frame."""
        coords = np.array(
            [ecef_to_globe_coords(pt.x_km, pt.y_km, pt.z_km) for pt in track],
            dtype=float,
        )
        if coords.size == 0 or self._visual_frame_mode != "ECI":
            return coords
        times = np.array(
            [self._seconds_since_reference(pt.timestamp) for pt in track], dtype=float
        )
        angles = EARTH_ROTATION_RATE_RAD_PER_SEC * times
        cos_ang = np.cos(angles)
        sin_ang = np.sin(angles)
        x = coords[:, 0].copy()
        y = coords[:, 1].copy()
        coords[:, 0] = cos_ang * x - sin_ang * y
        coords[:, 1] = sin_ang * x + cos_ang * y
        return coords

    def _update_visualization_pass_geometry(
        self, track: list[GroundTrackPoint]
    ) -> None:
        """Render the selected pass path and initialize the satellite marker."""
        if self.visual_globe_widget is None or not track:
            return
        coords = self._get_visualization_track_coords(track)
        if coords.size == 0:
            return
        first_timestamp = track[0].timestamp if track else None
        self._update_visual_earth_rotation(first_timestamp)
        if first_timestamp:
            self.visual_globe_widget.set_sun_datetime(first_timestamp)
        self.visual_globe_widget.update_track(coords)
        self.visual_globe_widget.update_satellite_position(tuple(coords[0]))
        self.visual_globe_widget.update_link_segment(None, None)
        self.visual_globe_widget.update_direction_arrow(None, None)

    def _clear_visualization_display(self, message: str | None = None) -> None:
        """Reset visualization playback state and remove temporary actors."""
        if self._visual_animation_timer.isActive():
            self._visual_animation_timer.stop()
        if self.visual_globe_widget is not None:
            self.visual_globe_widget.update_track(None)
            self.visual_globe_widget.update_satellite_position(None)
            self.visual_globe_widget.update_link_segment(None, None)
            self.visual_globe_widget.update_direction_arrow(None, None)
            self.visual_globe_widget.set_sun_datetime(None)
            self.visual_globe_widget.update_direction_arrow(None, None)
            self.visual_globe_widget.set_focus_point(None)
        self._visual_pass_track = None
        self._visual_selected_pass = None
        self._visual_station_ecef = None
        self._visual_contact_window = None
        self._visual_animation_index = 0
        self._visual_anim_fraction = 0.0
        if self.visual_time_slider:
            self.visual_time_slider.setEnabled(False)
            self.visual_time_slider.setRange(0, 0)
            self.visual_time_slider.setValue(0)
        if self.visual_play_button:
            self.visual_play_button.setEnabled(False)
            self.visual_play_button.setText("Play")
        if message and self.visual_pass_status_label:
            self.visual_pass_status_label.setText(message)
        self._reset_visual_graph_panel(message)

    def _toggle_visualization_playback(self) -> None:
        """Play or pause the current pass animation."""
        if not self._visual_pass_track or self.visual_play_button is None:
            return
        if self._visual_animation_timer.isActive():
            self._visual_animation_timer.stop()
            self.visual_play_button.setText("Play")
            return
        if self._visual_animation_index >= len(self._visual_pass_track) - 1:
            self._visual_animation_index = 0
            self._visual_anim_fraction = 0.0
            self._update_visualization_frame(render=False)
        else:
            self._visual_anim_fraction = 0.0
        self._visual_animation_timer.start()
        self.visual_play_button.setText("Pause")

    def _advance_visualization_animation(self) -> None:
        """Advance the satellite along the pass during playback."""
        if not self._visual_pass_track:
            self._stop_visualization_animation()
            return
        if self._visual_animation_index >= len(self._visual_pass_track) - 1:
            self._stop_visualization_animation(final=True)
            return
        step = self._visual_base_step * self._visual_speed_multiplier
        self._visual_anim_fraction += step
        while (
            self._visual_anim_fraction >= 1.0
            and self._visual_animation_index < len(self._visual_pass_track) - 1
        ):
            self._visual_anim_fraction -= 1.0
            self._visual_animation_index += 1
        if self._visual_animation_index >= len(self._visual_pass_track) - 1:
            self._visual_animation_index = len(self._visual_pass_track) - 1
            self._visual_anim_fraction = 0.0
            self._update_visualization_frame()
            self._stop_visualization_animation(final=True)
            return
        self._update_visualization_frame()

    def _stop_visualization_animation(self, final: bool = False) -> None:
        """Stop playback and update control labels."""
        if self._visual_animation_timer.isActive():
            self._visual_animation_timer.stop()
        if self.visual_play_button:
            self.visual_play_button.setText("Replay" if final else "Play")

    def _handle_visualization_slider_moved(self, value: int) -> None:
        """Scrub through the pass timeline."""
        if not self._visual_pass_track:
            return
        clamped = max(0, min(value, len(self._visual_pass_track) - 1))
        self._visual_animation_index = clamped
        self._visual_anim_fraction = 0.0
        if self._visual_animation_timer.isActive():
            self._visual_animation_timer.stop()
            if self.visual_play_button:
                self.visual_play_button.setText("Play")
        self._update_visualization_frame()

    def _handle_visual_speed_changed(self, value: int) -> None:
        """Adjust playback speed multiplier."""
        self._visual_speed_multiplier = max(0.1, value / 100.0)

    def _update_visualization_frame(self, render: bool = True) -> None:
        """Render the satellite position, slider, and ground link for the current frame."""
        _ = render
        if (
            not self._visual_pass_track
            or self.visual_globe_widget is None
            or self._visual_animation_index >= len(self._visual_pass_track)
        ):
            return
        point = self._visual_pass_track[self._visual_animation_index]
        base_vec = np.array(
            self._convert_vector_to_current_frame(
                ecef_to_globe_coords(point.x_km, point.y_km, point.z_km),
                point.timestamp,
            ),
            dtype=float,
        )
        timestamp = point.timestamp
        coords_vec = base_vec
        if (
            self._visual_anim_fraction > 1e-4
            and self._visual_animation_index < len(self._visual_pass_track) - 1
        ):
            next_point = self._visual_pass_track[self._visual_animation_index + 1]
            next_vec = np.array(
                self._convert_vector_to_current_frame(
                    ecef_to_globe_coords(next_point.x_km, next_point.y_km, next_point.z_km),
                    next_point.timestamp,
                ),
                dtype=float,
            )
            alpha = self._visual_anim_fraction
            coords_vec = (1.0 - alpha) * base_vec + alpha * next_vec
            dt = next_point.timestamp - point.timestamp
            timestamp = point.timestamp + alpha * dt
        coords = tuple(coords_vec.tolist())
        self._update_visual_earth_rotation(timestamp)
        self._update_visualization_focus_point(timestamp)
        if self.visual_globe_widget:
            self.visual_globe_widget.set_sun_datetime(timestamp)
        self.visual_globe_widget.update_satellite_position(coords)
        if self.visual_time_slider:
            self.visual_time_slider.blockSignals(True)
            self.visual_time_slider.setValue(self._visual_animation_index)
            self.visual_time_slider.blockSignals(False)
        if self.visual_pass_status_label and self._visual_selected_pass is not None:
            station_name = self._visual_selected_pass.station_name or "Ground Station"
            self.visual_pass_status_label.setText(
                f"{self._visual_frame_mode} | {station_name}: {timestamp:%d-%b %H:%M:%S} UTC"
            )
        self._update_visualization_link_actor(timestamp, coords)

    def _get_station_position(
        self, timestamp: datetime
    ) -> tuple[float, float, float] | None:
        """Return the station position vector in the current frame, converted to globe coordinates."""
        if self._visual_station_ecef is None:
            return None
        # First align with the globe texture coordinate system (ECEF → globe)
        globe_vec = np.array(ecef_to_globe_coords(*self._visual_station_ecef), dtype=float)
        # Nudge outward slightly so the marker/link sit above the surface for visibility
        length = np.linalg.norm(globe_vec)
        if length > 1e-6 and STATION_VISUAL_OFFSET_KM > 0.0:
            scale = (length + STATION_VISUAL_OFFSET_KM) / length
            globe_vec *= scale
        # Then rotate into the current reference frame (ECI adds the time-varying spin)
        return self._convert_vector_to_current_frame(tuple(globe_vec.tolist()), timestamp)

    def _update_visualization_focus_point(
        self, timestamp: datetime | None
    ) -> None:
        """Keep the camera focus aligned with the active ground station."""
        widget = getattr(self, "visual_globe_widget", None)
        if widget is None:
            return
        if timestamp is None:
            widget.set_focus_point(None)
            return
        station_point = self._get_station_position(timestamp)
        widget.set_focus_point(station_point if station_point is not None else None)

    def _update_visual_earth_rotation(self, timestamp: datetime | None) -> None:
        """Rotate the globe actors to match Earth orientation in the selected frame."""
        if self._visual_frame_mode != "ECI":
            angle_deg = 0.0
        elif timestamp is None:
            angle_deg = 0.0
        else:
            delta = self._seconds_since_reference(timestamp)
            angle_deg = math.degrees(EARTH_ROTATION_RATE_RAD_PER_SEC * delta)
        widget = getattr(self, "visual_globe_widget", None)
        if widget is None:
            return
        widget.set_frame_rotation(self._visual_frame_mode, angle_deg)

    def _update_visualization_link_actor(
        self, timestamp: datetime, satellite_coords: tuple[float, float, float]
    ) -> None:
        """Show or hide the green contact link for the current frame."""
        widget = getattr(self, "visual_globe_widget", None)
        if widget is None:
            return
        if self._visual_contact_window is None or self._visual_station_ecef is None:
            widget.update_link_segment(None, None)
            return
        aos, los = self._visual_contact_window
        if not (aos <= timestamp <= los):
            widget.update_link_segment(None, None)
            return
        station_point = self._get_station_position(timestamp)
        if station_point is None:
            return
        widget.update_link_segment(station_point, satellite_coords)

