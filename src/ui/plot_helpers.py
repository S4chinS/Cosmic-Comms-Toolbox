"""Plotting helpers shared across the UI tabs."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pyqtgraph as pg
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from src.models import PassStatistic, StationSummary
from src.ui.constants import HIST_BIN_OPTIONS, STATION_COLOR_PALETTE


class PlotHelpersMixin:
    """Reusable routines for timeline, histogram, and pie-chart plotting."""

    def _create_contact_plot_tabs(self) -> list[tuple[str, QWidget]]:
        """Initialize contact plots and return tab definitions."""
        pg.setConfigOptions(antialias=True)
        self.timeline_plot = pg.PlotWidget(title="Access Timeline")
        self.timeline_plot.setLabel("bottom", "Time since start", units="h")
        self.timeline_plot.hideAxis("left")
        self.timeline_plot.showGrid(x=True, y=False, alpha=0.3)
        self.timeline_plot.setMouseEnabled(x=False, y=False)
        self.timeline_plot.setMenuEnabled(False)
        self.timeline_legend = self.timeline_plot.addLegend()

        # Hover readout for cursor time.
        self._timeline_hover_label = pg.LabelItem(justify="left")
        self.timeline_plot.plotItem.layout.addItem(self._timeline_hover_label, 0, 0)
        self._timeline_hover_label.setVisible(False)
        self._timeline_hover_line = pg.InfiniteLine(
            angle=90, movable=False, pen=pg.mkPen("#d0d0d0", width=1)
        )
        self._timeline_hover_line.setVisible(False)
        self.timeline_plot.addItem(self._timeline_hover_line)
        self.timeline_plot.scene().sigMouseMoved.connect(self._handle_timeline_hover)  # type: ignore[arg-type]

        self.hist_plot = pg.PlotWidget(title="Pass Duration Histogram")
        self.hist_plot.setLabel("bottom", "Pass Duration", units="min")
        self.hist_plot.setLabel("left", "Count")
        self.hist_plot.showGrid(x=True, y=True, alpha=0.3)
        self.hist_plot.setMouseEnabled(x=False, y=False)
        self.hist_plot.setMenuEnabled(False)
        pie_figure = Figure(figsize=(4, 3), tight_layout=True)
        pie_figure.patch.set_facecolor("#0b0b10")
        self.pie_canvas = FigureCanvasQTAgg(pie_figure)
        self.pie_axes = pie_figure.add_subplot(111)
        self._configure_pie_axes()
        self._hist_hover_label = pg.LabelItem(justify="left")
        self.hist_plot.plotItem.layout.addItem(self._hist_hover_label, 0, 0)
        self._hist_hover_label.setVisible(False)
        self.hist_plot.scene().sigMouseMoved.connect(self._handle_hist_hover)  # type: ignore[arg-type]
        timeline_tab = QWidget()
        timeline_layout = QVBoxLayout(timeline_tab)
        timeline_layout.setContentsMargins(4, 4, 4, 4)

        # Timeline tab now includes:
        # - Left (75%): access timeline plot
        # - Right (25%): station enable/disable + gap statistics
        timeline_splitter = QSplitter(Qt.Orientation.Horizontal)
        timeline_layout.addWidget(timeline_splitter, stretch=1)

        timeline_left = QWidget()
        timeline_left_layout = QVBoxLayout(timeline_left)
        timeline_left_layout.setContentsMargins(0, 0, 0, 0)
        timeline_left_layout.addWidget(self.timeline_plot, stretch=1)
        timeline_splitter.addWidget(timeline_left)

        timeline_right = QWidget()
        timeline_right_layout = QVBoxLayout(timeline_right)
        timeline_right_layout.setContentsMargins(0, 0, 0, 0)

        stations_group = QGroupBox("Stations (toggle to update)")
        stations_layout = QVBoxLayout(stations_group)
        self.timeline_station_list = QListWidget()
        self.timeline_station_list.itemChanged.connect(  # type: ignore[arg-type]
            self._handle_timeline_station_toggle
        )
        stations_layout.addWidget(self.timeline_station_list, stretch=1)
        timeline_right_layout.addWidget(stations_group, stretch=2)

        stats_group = QGroupBox("Gap statistics (selected stations)")
        stats_layout = QFormLayout(stats_group)
        self._timeline_gap_stat_labels: dict[str, QLabel] = {}
        for key in (
            "Selected stations",
            "Scenario duration",
            "Total access",
            "Coverage",
            "Num gaps",
            "Mean gap",
            "Median gap",
            "P95 gap",
            "Longest no access",
        ):
            label = QLabel("—")
            label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            self._timeline_gap_stat_labels[key] = label
            stats_layout.addRow(f"{key}:", label)
        timeline_right_layout.addWidget(stats_group, stretch=1)

        timeline_splitter.addWidget(timeline_right)
        timeline_splitter.setStretchFactor(0, 3)
        timeline_splitter.setStretchFactor(1, 1)
        timeline_splitter.setSizes([3, 1])

        hist_tab = QWidget()
        hist_layout = QVBoxLayout(hist_tab)
        hist_layout.setContentsMargins(4, 4, 4, 4)
        hist_layout.addWidget(self.hist_plot, stretch=1)
        hist_button_row = QVBoxLayout()
        self.plot_config_button = QPushButton("Plot config")
        self.plot_config_button.clicked.connect(self._open_plot_config_dialog)  # type: ignore[arg-type]
        hist_button_row.addStretch(1)
        hist_button_row.addWidget(self.plot_config_button, alignment=Qt.AlignmentFlag.AlignRight)
        hist_layout.addLayout(hist_button_row)
        pie_tab = QWidget()
        pie_layout = QVBoxLayout(pie_tab)
        pie_layout.setContentsMargins(4, 4, 4, 4)
        pie_layout.addWidget(self.pie_canvas)
        self._draw_empty_plots()
        return [
            ("Timeline", timeline_tab),
            ("Histogram", hist_tab),
            ("Access Share", pie_tab),
        ]

    def _handle_timeline_station_toggle(self, _item: QListWidgetItem) -> None:
        """Re-render the timeline + gap stats when station toggles change."""
        if getattr(self, "_last_result", None) is None:
            return
        self._update_timeline_view(getattr(self, "_last_result", None))

    def _get_enabled_timeline_stations(self) -> set[str]:
        """Return the set of station names enabled in the timeline station list."""
        widget = getattr(self, "timeline_station_list", None)
        if widget is None:
            return set()
        enabled: set[str] = set()
        for idx in range(widget.count()):
            item = widget.item(idx)
            if item is None:
                continue
            if item.checkState() == Qt.CheckState.Checked:
                enabled.add(item.text())
        return enabled

    def _set_timeline_station_list(self, station_names: list[str]) -> None:
        """Populate the timeline station selector, defaulting all to enabled."""
        widget = getattr(self, "timeline_station_list", None)
        if widget is None:
            return
        prior = self._get_enabled_timeline_stations()
        widget.blockSignals(True)
        widget.clear()
        for name in station_names:
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            # Default: enabled, but preserve prior selection where possible.
            checked = name in prior if prior else True
            item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
            widget.addItem(item)
        widget.blockSignals(False)

    def _format_duration_minutes(self, minutes: float) -> str:
        if not np.isfinite(minutes):
            return "—"
        if minutes < 60.0:
            return f"{minutes:.2f} min"
        hours = minutes / 60.0
        if hours < 48.0:
            return f"{hours:.2f} h"
        days = hours / 24.0
        return f"{days:.2f} d"

    def _compute_union_intervals_seconds(
        self,
        passes: list[PassStatistic],
        *,
        scenario_start,
        scenario_end,
    ) -> list[tuple[float, float]]:
        """Return merged access intervals (seconds from scenario start)."""
        if not passes:
            return []
        intervals: list[tuple[float, float]] = []
        for p in passes:
            aos = getattr(p, "aos", None)
            los = getattr(p, "los", None)
            if aos is None or los is None:
                continue
            start = (aos - scenario_start).total_seconds()
            end = (los - scenario_start).total_seconds()
            if not np.isfinite(start) or not np.isfinite(end):
                continue
            if end <= start:
                continue
            # Clip to scenario window
            start = max(0.0, float(start))
            end = min((scenario_end - scenario_start).total_seconds(), float(end))
            if end <= start:
                continue
            intervals.append((start, end))
        if not intervals:
            return []
        intervals.sort(key=lambda t: t[0])
        merged: list[tuple[float, float]] = []
        cur_s, cur_e = intervals[0]
        for s, e in intervals[1:]:
            if s <= cur_e:
                cur_e = max(cur_e, e)
            else:
                merged.append((cur_s, cur_e))
                cur_s, cur_e = s, e
        merged.append((cur_s, cur_e))
        return merged

    def _compute_gap_durations_seconds(
        self,
        merged_intervals: list[tuple[float, float]],
        *,
        scenario_duration_s: float,
    ) -> list[float]:
        """Return gaps (seconds) covering the whole scenario window."""
        if scenario_duration_s <= 0:
            return []
        gaps: list[float] = []
        if not merged_intervals:
            return [float(scenario_duration_s)]
        # Leading gap
        first_start = float(merged_intervals[0][0])
        if first_start > 0:
            gaps.append(first_start)
        # In-between
        for (s0, e0), (s1, _e1) in zip(merged_intervals, merged_intervals[1:]):
            gap = float(s1 - e0)
            if gap > 0:
                gaps.append(gap)
        # Trailing
        last_end = float(merged_intervals[-1][1])
        tail = float(scenario_duration_s - last_end)
        if tail > 0:
            gaps.append(tail)
        return gaps

    def _update_timeline_gap_stats(
        self,
        *,
        enabled_station_names: list[str],
        merged_intervals: list[tuple[float, float]],
        scenario_duration_s: float,
    ) -> None:
        labels = getattr(self, "_timeline_gap_stat_labels", None)
        if not isinstance(labels, dict):
            return

        def _set(key: str, value: str) -> None:
            label = labels.get(key)
            if label is not None:
                label.setText(value)

        if scenario_duration_s <= 0:
            for key in labels:
                _set(key, "—")
            return

        total_access_s = float(sum(e - s for s, e in merged_intervals))
        coverage = 100.0 * total_access_s / float(scenario_duration_s) if scenario_duration_s else 0.0
        gaps_s = self._compute_gap_durations_seconds(
            merged_intervals, scenario_duration_s=float(scenario_duration_s)
        )
        gaps_min = np.asarray(gaps_s, dtype=float) / 60.0 if gaps_s else np.asarray([], dtype=float)
        longest_gap_min = float(np.max(gaps_min)) if gaps_min.size else float(scenario_duration_s / 60.0)

        def _fmt_gap(x: float) -> str:
            return self._format_duration_minutes(float(x))

        _set("Selected stations", f"{len(enabled_station_names)}")
        _set("Scenario duration", self._format_duration_minutes(float(scenario_duration_s) / 60.0))
        _set("Total access", self._format_duration_minutes(total_access_s / 60.0))
        _set("Coverage", f"{coverage:.2f}%")
        _set("Num gaps", f"{int(gaps_min.size)}")
        if gaps_min.size:
            _set("Mean gap", _fmt_gap(float(np.mean(gaps_min))))
            _set("Median gap", _fmt_gap(float(np.median(gaps_min))))
            _set("P95 gap", _fmt_gap(float(np.percentile(gaps_min, 95))))
        else:
            _set("Mean gap", "—")
            _set("Median gap", "—")
            _set("P95 gap", "—")
        _set("Longest no access", _fmt_gap(longest_gap_min))

    def _update_timeline_view(self, result) -> None:
        """Render timeline plot and gap statistics for the enabled station subset."""
        if getattr(self, "_current_config", None) is None:
            return
        if not result or not getattr(result, "passes", None):
            return
        config = self._current_config
        scenario_start = config.scenario.start_time
        scenario_end = config.scenario.end_time
        scenario_duration_s = float((scenario_end - scenario_start).total_seconds())
        if scenario_duration_s <= 0:
            return

        enabled = self._get_enabled_timeline_stations()
        passes_all = sorted(result.passes, key=lambda item: item.aos)
        if enabled:
            filtered_passes = [
                p
                for p in passes_all
                if (p.station_name or "Ground Station") in enabled
            ]
        else:
            filtered_passes = []

        # Re-render the timeline plot for the subset.
        station_groups: dict[str, list[PassStatistic]] = {}
        for item in filtered_passes:
            station_name = item.station_name or "Ground Station"
            station_groups.setdefault(station_name, []).append(item)
        station_colors = getattr(self, "_station_color_map", None) or self._build_station_color_map(
            sorted(station_groups.keys())
        )
        self.timeline_plot.clear()
        if getattr(self, "timeline_legend", None):
            self.timeline_legend.clear()
        for station_name, group in station_groups.items():
            durations = [p.duration_minutes for p in group]
            aos_hours = [(p.aos - scenario_start).total_seconds() / 3600.0 for p in group]
            widths = [d / 60.0 for d in durations]
            color = station_colors.get(station_name)
            if color is None:
                continue
            brush = pg.mkBrush(color)
            pen = pg.mkPen(color.darker(125))
            timeline_item = pg.BarGraphItem(
                x=[a + w / 2 for a, w in zip(aos_hours, widths)],
                height=[1.0] * len(aos_hours),
                width=widths,
                brush=brush,
                pen=pen,
            )
            self.timeline_plot.addItem(timeline_item)
            if getattr(self, "timeline_legend", None):
                self.timeline_legend.addItem(timeline_item, station_name)
        self.timeline_plot.setYRange(0, 1.2)
        self.timeline_plot.setTitle("Access Timeline")

        merged = self._compute_union_intervals_seconds(
            filtered_passes, scenario_start=scenario_start, scenario_end=scenario_end
        )
        self._update_timeline_gap_stats(
            enabled_station_names=sorted(enabled),
            merged_intervals=merged,
            scenario_duration_s=scenario_duration_s,
        )

    def _draw_empty_plots(self) -> None:
        """Display placeholder text when no data is available."""
        self.timeline_plot.clear()
        if getattr(self, "timeline_legend", None):
            self.timeline_legend.clear()
        self.hist_plot.clear()
        self.timeline_plot.setTitle("Access Timeline\n(Run analysis to view)")
        self.hist_plot.setTitle("Pass Duration Histogram\n(Run analysis to view)")
        self._clear_pie_chart("Access Share by Station\n(Run analysis to view)")
        self._hist_hover_label.setVisible(False)
        if getattr(self, "_timeline_hover_label", None) is not None:
            self._timeline_hover_label.setVisible(False)
        if getattr(self, "_timeline_hover_line", None) is not None:
            self._timeline_hover_line.setVisible(False)
        self._histogram_bar_data = []

    def _configure_pie_axes(self) -> None:
        """Initialize the matplotlib pie chart axes with dark styling."""
        if not getattr(self, "pie_axes", None):
            return
        self.pie_axes.clear()
        self.pie_axes.set_facecolor("#0b0b10")
        self.pie_axes.figure.set_facecolor("#0b0b10")
        self.pie_axes.tick_params(
            axis="both",
            colors="#dddddd",
            labelbottom=False,
            labelleft=False,
            bottom=False,
            left=False,
        )
        for spine in self.pie_axes.spines.values():
            spine.set_visible(False)
        self.pie_axes.set_title("Access Share by Station", color="white", pad=12)

    def _clear_pie_chart(self, title: str) -> None:
        """Reset the pie chart canvas with the provided title."""
        if not getattr(self, "pie_axes", None) or not getattr(self, "pie_canvas", None):
            return
        self._configure_pie_axes()
        self.pie_axes.set_title(title, color="white", pad=12)
        self.pie_canvas.draw_idle()

    def _update_plots(self, result) -> None:
        """Render the timeline and histogram plots."""
        if not result.passes or getattr(self, "_current_config", None) is None:
            self._draw_empty_plots()
            return
        passes = sorted(result.passes, key=lambda item: item.aos)
        start_time = self._current_config.scenario.start_time
        station_groups: dict[str, list[PassStatistic]] = {}
        for item in passes:
            station_name = item.station_name or "Ground Station"
            station_groups.setdefault(station_name, []).append(item)
        station_summaries = getattr(result, "station_summaries", [])
        ordered_station_names = (
            [entry.station_name for entry in station_summaries]
            if station_summaries
            else list(station_groups.keys())
        )
        station_colors = self._build_station_color_map(ordered_station_names)
        # Update timeline station selector (defaults all enabled).
        self._set_timeline_station_list(ordered_station_names)
        self.timeline_plot.clear()
        if getattr(self, "timeline_legend", None):
            self.timeline_legend.clear()
        for station_name, group in station_groups.items():
            durations = [p.duration_minutes for p in group]
            aos_hours = [(p.aos - start_time).total_seconds() / 3600.0 for p in group]
            widths = [d / 60.0 for d in durations]
            color = station_colors.get(station_name)
            brush = pg.mkBrush(color)
            pen = pg.mkPen(color.darker(125))
            timeline_item = pg.BarGraphItem(
                x=[a + w / 2 for a, w in zip(aos_hours, widths)],
                height=[1.0] * len(aos_hours),
                width=widths,
                brush=brush,
                pen=pen,
            )
            self.timeline_plot.addItem(timeline_item)
            if getattr(self, "timeline_legend", None):
                self.timeline_legend.addItem(timeline_item, station_name)
        self.timeline_plot.setYRange(0, 1.2)
        durations_all = [p.duration_minutes for p in passes]
        bin_width_minutes = max(self._histogram_bin_seconds / 60.0, 1 / 60.0)
        max_duration = max(durations_all)
        bins = np.arange(0, max_duration + bin_width_minutes, bin_width_minutes)
        if len(bins) < 2:
            bins = np.linspace(0, max_duration + bin_width_minutes, 5)
        counts, bin_edges = np.histogram(durations_all, bins=bins)
        centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        widths_bins = bin_edges[1:] - bin_edges[:-1]
        self.hist_plot.clear()
        hist_item = pg.BarGraphItem(
            x=centers,
            height=counts,
            width=widths_bins,
            brush=pg.mkBrush("#87CEFA"),
            pen=pg.mkPen("#1E90FF"),
        )
        self.hist_plot.addItem(hist_item)
        self._histogram_bar_data = list(zip(centers, widths_bins, counts))
        self._hist_hover_label.setVisible(False)
        summaries_for_pie = station_summaries or self._summaries_from_groups(
            station_groups
        )
        self._update_pie_chart(summaries_for_pie, station_colors)
        # Recompute timeline subset stats based on current station toggles.
        self._update_timeline_view(result)

    def _build_station_color_map(self, station_names: list[str]) -> dict[str, QColor]:
        """Assign consistent colors per station for plot rendering."""
        if not station_names:
            self._station_color_map = {}
            return {}
        palette = STATION_COLOR_PALETTE or ["#2E8B57"]
        color_map: dict[str, QColor] = {}
        for idx, name in enumerate(station_names):
            color_map[name] = QColor(palette[idx % len(palette)])
        self._station_color_map = color_map
        return color_map

    def _summaries_from_groups(
        self, station_groups: dict[str, list[PassStatistic]]
    ) -> list[StationSummary]:
        """Build StationSummary objects from grouped pass stats."""
        summaries: list[StationSummary] = []
        for name, items in station_groups.items():
            summaries.append(
                StationSummary(
                    station_name=name,
                    total_passes=len(items),
                    total_access_minutes=float(
                        sum(item.duration_minutes for item in items)
                    ),
                )
            )
        return summaries

    def _update_pie_chart(
        self,
        station_summaries: list[StationSummary],
        station_colors: dict[str, QColor],
    ) -> None:
        """Render the pie chart showing per-station contribution."""
        if not getattr(self, "pie_axes", None) or not getattr(self, "pie_canvas", None):
            return
        if not station_summaries:
            self._clear_pie_chart("Access Share by Station\n(No data)")
            return
        total_minutes = sum(entry.total_access_minutes for entry in station_summaries)
        if total_minutes <= 0:
            self._clear_pie_chart("Access Share by Station\n(No access time)")
            return
        self._configure_pie_axes()
        sizes = [entry.total_access_minutes for entry in station_summaries]
        colors = [
            station_colors.get(
                entry.station_name,
                QColor(STATION_COLOR_PALETTE[idx % len(STATION_COLOR_PALETTE)]),
            ).name()
            for idx, entry in enumerate(station_summaries)
        ]
        labels = [
            f"{entry.station_name}: {entry.total_access_minutes / total_minutes * 100:.1f}%"
            for entry in station_summaries
        ]
        self.pie_axes.pie(
            sizes,
            labels=labels,
            colors=colors,
            startangle=90,
            wedgeprops={"edgecolor": "#111", "linewidth": 1},
            textprops={"color": "white"},
        )
        self.pie_axes.axis("equal")
        self.pie_axes.set_title("Access Share by Station", color="white", pad=12)
        self.pie_canvas.draw_idle()

    def _handle_hist_hover(self, position) -> None:
        """Show annotation when hovering over histogram bars."""
        if not getattr(self, "_histogram_bar_data", None):
            self._hist_hover_label.setVisible(False)
            return
        if not self.hist_plot.sceneBoundingRect().contains(position):
            self._hist_hover_label.setVisible(False)
            return
        mouse_point = self.hist_plot.plotItem.vb.mapSceneToView(position)
        x_coord = mouse_point.x()
        hit = None
        for center, width, count in self._histogram_bar_data:
            half = width / 2.0
            if (center - half) <= x_coord <= (center + half):
                hit = (center, width, count)
                break
        if hit is None:
            self._hist_hover_label.setVisible(False)
            return
        center, _, count = hit
        text = f"{center:.2f} min | {int(count)} pass(es)"
        self._hist_hover_label.setText(text)
        self._hist_hover_label.setVisible(True)

    def _handle_timeline_hover(self, position) -> None:
        """Show the timeline x-coordinate under the mouse cursor."""
        label = getattr(self, "_timeline_hover_label", None)
        line = getattr(self, "_timeline_hover_line", None)
        if label is None or line is None or not getattr(self, "timeline_plot", None):
            return
        if not self.timeline_plot.sceneBoundingRect().contains(position):
            label.setVisible(False)
            line.setVisible(False)
            return
        mouse_point = self.timeline_plot.plotItem.vb.mapSceneToView(position)
        x_hours = float(mouse_point.x())
        if not np.isfinite(x_hours):
            label.setVisible(False)
            line.setVisible(False)
            return
        # Build label text.
        text = f"{x_hours:.3f} h"
        config = getattr(self, "_current_config", None)
        if config is not None and getattr(config, "scenario", None) is not None:
            start_time = config.scenario.start_time
            timestamp = start_time + timedelta(hours=x_hours)
            # Display as UTC (scenario times are stored as UTC in this app).
            text = f"{x_hours:.3f} h | {timestamp:%d-%b %H:%M:%S} UTC"
        label.setText(text)
        label.setVisible(True)
        line.setPos(x_hours)
        line.setVisible(True)

    def _open_plot_config_dialog(self) -> None:
        """Allow users to tweak plot settings such as histogram bin widths."""
        if getattr(self, "_current_config", None) is None:
            from PySide6.QtWidgets import QMessageBox

            QMessageBox.information(
                self,
                "Plot configuration",
                "Run the analysis first to adjust plot settings.",
            )
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("Plot configuration")
        layout = QVBoxLayout(dialog)
        info_label = QLabel("Histogram bin width (seconds).")
        layout.addWidget(info_label)
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setMinimum(0)
        slider.setMaximum(len(HIST_BIN_OPTIONS) - 1)
        slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        slider.setTickInterval(1)
        current_index = HIST_BIN_OPTIONS.index(
            min(HIST_BIN_OPTIONS, key=lambda x: abs(x - self._histogram_bin_seconds))
        )
        slider.setValue(current_index)
        layout.addWidget(slider)
        value_label = QLabel(
            self._format_bin_width_label(HIST_BIN_OPTIONS[slider.value()])
        )
        layout.addWidget(value_label)

        def _on_slider_change(value: int) -> None:
            value_label.setText(self._format_bin_width_label(HIST_BIN_OPTIONS[value]))

        slider.valueChanged.connect(_on_slider_change)  # type: ignore[arg-type]
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        layout.addWidget(buttons)

        def _accept() -> None:
            self._histogram_bin_seconds = HIST_BIN_OPTIONS[slider.value()]
            dialog.accept()

        buttons.accepted.connect(_accept)  # type: ignore[arg-type]
        buttons.rejected.connect(dialog.reject)  # type: ignore[arg-type]
        if dialog.exec() == QDialog.DialogCode.Accepted and getattr(self, "_last_result", None):
            self._update_plots(self._last_result)

    def _format_bin_width_label(self, seconds: int) -> str:
        """Return descriptive text for the slider."""
        minutes = seconds / 60.0
        return f"{seconds} s per bin ({minutes:.2f} min)"

