"""Link budget tab mixin."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src import link_budget_math
from src.itu_losses import estimate_slant_path_loss
from src.models import GroundStationConfig
from src.ui.constants import GIBIT_PER_GBIT

# Default elevation filtering bounds for the link-budget tool (and the derived
# data-volume calculations). These bounds are *only* applied in the link-budget
# pipeline; mission propagation/pass detection is intentionally ungated.
DEFAULT_LINK_BUDGET_ELEVATION_LOWER_DEG = 5.0
DEFAULT_LINK_BUDGET_ELEVATION_UPPER_DEG = 60.0


class LinkBudgetTabMixin:
    """Encapsulates link budget UI construction and logic."""

    def _build_link_budget_tab(self) -> QWidget:
        """Assemble the link budget controls, table, and plot."""
        tab = QWidget()
        layout = QHBoxLayout(tab)
        input_group = QGroupBox("Inputs")
        form = QFormLayout(input_group)
        self.link_budget_station_combo = QComboBox()
        form.addRow("Ground station:", self.link_budget_station_combo)
        self.lb_frequency_input = QDoubleSpinBox()
        self.lb_frequency_input.setRange(0.1, 100.0)
        self.lb_frequency_input.setDecimals(3)
        self.lb_frequency_input.setValue(8.2)
        self.lb_frequency_input.setSuffix(" GHz")
        form.addRow("Frequency:", self.lb_frequency_input)
        self.lb_tx_power_input = QDoubleSpinBox()
        self.lb_tx_power_input.setRange(-50.0, 50.0)
        self.lb_tx_power_input.setValue(3.0)
        self.lb_tx_power_input.setSuffix(" dBW")
        form.addRow("TX power:", self.lb_tx_power_input)
        self.lb_tx_gain_input = QDoubleSpinBox()
        self.lb_tx_gain_input.setRange(-10.0, 80.0)
        self.lb_tx_gain_input.setValue(5.41)
        self.lb_tx_gain_input.setSuffix(" dBi")
        form.addRow("TX boresight gain:", self.lb_tx_gain_input)
        self.lb_tx_losses_input = QDoubleSpinBox()
        self.lb_tx_losses_input.setRange(0.0, 20.0)
        self.lb_tx_losses_input.setValue(2.0)
        self.lb_tx_losses_input.setSuffix(" dB")
        form.addRow("TX feeder loss:", self.lb_tx_losses_input)
        self.lb_tx_backoff_input = QDoubleSpinBox()
        self.lb_tx_backoff_input.setRange(0.0, 10.0)
        self.lb_tx_backoff_input.setValue(0.0)
        self.lb_tx_backoff_input.setSuffix(" dB")
        form.addRow("TX backoff:", self.lb_tx_backoff_input)
        self.lb_antenna_gain_input = QDoubleSpinBox()
        self.lb_antenna_gain_input.setRange(-10.0, 80.0)
        self.lb_antenna_gain_input.setValue(5.41)
        self.lb_antenna_gain_input.setSuffix(" dBi")
        form.addRow("Actual antenna gain:", self.lb_antenna_gain_input)
        self.lb_rx_gt_input = QDoubleSpinBox()
        self.lb_rx_gt_input.setRange(-10.0, 80.0)
        self.lb_rx_gt_input.setValue(26.0)
        self.lb_rx_gt_input.setSuffix(" dB/K")
        form.addRow("Receiver G/T:", self.lb_rx_gt_input)
        self.lb_rx_losses_input = QDoubleSpinBox()
        self.lb_rx_losses_input.setRange(0.0, 20.0)
        self.lb_rx_losses_input.setValue(0.0)
        self.lb_rx_losses_input.setSuffix(" dB")
        form.addRow("Receiver losses:", self.lb_rx_losses_input)
        self.lb_symbol_rate_input = QDoubleSpinBox()
        self.lb_symbol_rate_input.setRange(0.01, 5000.0)
        self.lb_symbol_rate_input.setValue(300.0)
        self.lb_symbol_rate_input.setDecimals(3)
        self.lb_symbol_rate_input.setSuffix(" Msps")
        form.addRow("Symbol rate limit:", self.lb_symbol_rate_input)
        self.lb_impl_loss_input = QDoubleSpinBox()
        self.lb_impl_loss_input.setRange(0.0, 10.0)
        self.lb_impl_loss_input.setValue(1.0)
        self.lb_impl_loss_input.setSuffix(" dB")
        form.addRow("Implementation loss:", self.lb_impl_loss_input)
        self.lb_margin_input = QDoubleSpinBox()
        self.lb_margin_input.setRange(0.0, 20.0)
        self.lb_margin_input.setValue(3.0)
        self.lb_margin_input.setSuffix(" dB")
        form.addRow("Link margin target:", self.lb_margin_input)
        self.lb_sat_altitude_input = QDoubleSpinBox()
        self.lb_sat_altitude_input.setRange(150.0, 50000.0)
        self.lb_sat_altitude_input.setValue(550.0)
        self.lb_sat_altitude_input.setSuffix(" km")
        form.addRow("Satellite altitude:", self.lb_sat_altitude_input)

        # Elevation filtering bounds used by the link-budget tool + data-volume calculations.
        self.lb_elev_lower_input = QDoubleSpinBox()
        self.lb_elev_lower_input.setRange(0.0, 90.0)
        self.lb_elev_lower_input.setDecimals(2)
        self.lb_elev_lower_input.setValue(float(DEFAULT_LINK_BUDGET_ELEVATION_LOWER_DEG))
        self.lb_elev_lower_input.setSuffix(" °")
        form.addRow("Elevation lower bound:", self.lb_elev_lower_input)

        self.lb_elev_upper_input = QDoubleSpinBox()
        self.lb_elev_upper_input.setRange(0.0, 90.0)
        self.lb_elev_upper_input.setDecimals(2)
        self.lb_elev_upper_input.setValue(float(DEFAULT_LINK_BUDGET_ELEVATION_UPPER_DEG))
        self.lb_elev_upper_input.setSuffix(" °")
        form.addRow("Elevation upper bound:", self.lb_elev_upper_input)
        self.lb_gs_elevation_input = QDoubleSpinBox()
        self.lb_gs_elevation_input.setRange(0.0, 90.0)
        self.lb_gs_elevation_input.setValue(60.0)
        self.lb_gs_elevation_input.setSuffix(" °")
        form.addRow("GS elevation angle:", self.lb_gs_elevation_input)
        self.lb_unavailability_input = QDoubleSpinBox()
        self.lb_unavailability_input.setRange(0.01, 5.0)
        self.lb_unavailability_input.setValue(0.1)
        self.lb_unavailability_input.setDecimals(2)
        self.lb_unavailability_input.setSuffix(" %")
        form.addRow("Unavailability:", self.lb_unavailability_input)
        self.lb_polarization_loss_input = QDoubleSpinBox()
        self.lb_polarization_loss_input.setRange(0.0, 5.0)
        self.lb_polarization_loss_input.setDecimals(2)
        self.lb_polarization_loss_input.setValue(0.1)
        self.lb_polarization_loss_input.setSuffix(" dB")
        form.addRow("Polarization loss:", self.lb_polarization_loss_input)
        self.lb_rolloff_input = QDoubleSpinBox()
        self.lb_rolloff_input.setRange(0.05, 1.0)
        self.lb_rolloff_input.setDecimals(2)
        self.lb_rolloff_input.setValue(0.25)
        form.addRow("Roll-off factor:", self.lb_rolloff_input)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.addWidget(input_group)
        left_layout.addWidget(self._build_downlink_summary_group())
        left_layout.addStretch(1)
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        self.link_budget_tabs = QTabWidget()
        right_layout.addWidget(self.link_budget_tabs, stretch=1)
        table_tab = QWidget()
        table_layout = QVBoxLayout(table_tab)
        self.link_budget_table = QTableWidget(0, 3)
        self.link_budget_table.setHorizontalHeaderLabels(["Parameter", "Value", "Unit"])
        self.link_budget_table.verticalHeader().setVisible(False)
        self.link_budget_table.setAlternatingRowColors(True)
        self.link_budget_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.link_budget_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.link_budget_table.setWordWrap(False)
        header = self.link_budget_table.horizontalHeader()
        header.setDefaultAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        table_layout.addWidget(self.link_budget_table)
        self.link_budget_summary_label = QLabel(
            "Adjust the inputs to compute the link budget."
        )
        self.link_budget_summary_label.setWordWrap(True)
        table_layout.addWidget(self.link_budget_summary_label)
        plot_tab = QWidget()
        plot_layout = QVBoxLayout(plot_tab)
        self.link_budget_plot = pg.PlotWidget(title="VCM System Performance")
        self.link_budget_plot.setLabel("bottom", "Elevation", units="deg")
        self.link_budget_plot.setLabel("left", "Es/N0", units="dB")
        self.link_budget_plot.showGrid(x=True, y=True, alpha=0.3)
        self._link_budget_plot_legend = self.link_budget_plot.addLegend(offset=(10, 10))
        self._link_budget_plot_annotations: list = []
        plot_layout.addWidget(self.link_budget_plot)
        self.link_budget_tabs.addTab(table_tab, "Static Link Budget")
        self.link_budget_tabs.addTab(plot_tab, "Dynamic Link Budget")
        self._dynamic_link_budget_tab = plot_tab
        self._latest_link_budget_plot_data: tuple[np.ndarray, dict] | None = None
        self._loss_cache_key: tuple | None = None
        self._loss_cache_losses: np.ndarray | None = None
        self._loss_cache_contributions: dict | None = None
        self.link_budget_tabs.currentChanged.connect(self._on_link_budget_tab_changed)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([1, 1])
        layout.addWidget(splitter)
        self._register_link_budget_inputs()
        self._refresh_link_budget_station_list()
        self._link_budget_auto_enabled = True

        # Cache for ITU losses used by data-volume time-series evaluation.
        if not hasattr(self, "_timeseries_loss_cache"):
            self._timeseries_loss_cache: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}

        # Debounce expensive data-volume recomputation so sliders/spinboxes don't
        # freeze the GUI while the user is still editing values.
        if getattr(self, "_data_volume_refresh_timer", None) is None:
            self._data_volume_refresh_timer = QTimer(self)
            self._data_volume_refresh_timer.setSingleShot(True)
            self._data_volume_refresh_timer.setInterval(200)
            self._data_volume_refresh_timer.timeout.connect(  # type: ignore[attr-defined]
                self._perform_data_volume_refresh
            )

        self._trigger_link_budget_recompute()
        return tab

    def _build_downlink_summary_group(self) -> QGroupBox:
        """Create the box summarizing downlink capacity."""
        group = QGroupBox("Downlink Summary")
        layout = QVBoxLayout(group)

        # Summary labels
        form_layout = QFormLayout()
        self._downlink_total_label = QLabel("—")
        self._downlink_per_orbit_label = QLabel("—")
        form_layout.addRow("Scenario total:", self._downlink_total_label)
        form_layout.addRow("Per orbit:", self._downlink_per_orbit_label)
        layout.addLayout(form_layout)
        hint_label = QLabel(
            "Data Volume uses the full propagated elevation timeline (all passes) "
            "filtered by the Link Budget elevation bounds. "
            "The Static Link Budget table is evaluated at the single 'GS elevation angle' input."
        )
        hint_label.setWordWrap(True)
        hint_label.setStyleSheet("color: #9aa0a6; font-style: italic;")
        layout.addWidget(hint_label)
        layout.addStretch(1)

        return group

    def _build_data_volume_tab(self) -> QWidget:
        """Assemble the Data Volume tab with a selectable distribution + stats."""
        tab = QWidget()
        outer_layout = QVBoxLayout(tab)
        outer_layout.setContentsMargins(8, 8, 8, 8)
        intro_label = QLabel(
            "This histogram uses the latest analysis timeline combined with the "
            "current link-budget configuration. Select a parameter to view its distribution."
        )
        intro_label.setWordWrap(True)
        outer_layout.addWidget(intro_label)

        if not hasattr(self, "_distribution_bin_widths") or not isinstance(
            getattr(self, "_distribution_bin_widths", None), dict
        ):
            self._distribution_bin_widths: dict[str, float | None] = {}

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: distribution plot + controls
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        control_row = QHBoxLayout()
        control_row.addWidget(QLabel("Distribution:"))
        self.data_volume_distribution_combo = QComboBox()
        for dist_id, label in self._get_data_volume_distribution_options():
            self.data_volume_distribution_combo.addItem(label, dist_id)
        self.data_volume_distribution_combo.currentIndexChanged.connect(  # type: ignore[arg-type]
            lambda *_: self._update_data_volume_plots()
        )
        control_row.addWidget(self.data_volume_distribution_combo, stretch=1)
        self.data_volume_bin_label = QLabel("Bin width: —")
        control_row.addWidget(self.data_volume_bin_label)
        bin_button = QPushButton("Bin Width…")
        bin_button.clicked.connect(self._configure_distribution_bin_width)  # type: ignore[arg-type]
        control_row.addWidget(bin_button)
        left_layout.addLayout(control_row)

        self.data_volume_plot = pg.PlotWidget(title="Distribution\n(Run analysis to view)")
        self.data_volume_plot.showGrid(x=True, y=True, alpha=0.3)
        self.data_volume_plot.setMinimumHeight(520)
        left_layout.addWidget(self.data_volume_plot, stretch=1)

        splitter.addWidget(left_widget)

        # Right: statistics for the selected distribution
        stats_group = QGroupBox("Statistics")
        stats_layout = QFormLayout(stats_group)
        self._data_volume_stats_labels: dict[str, QLabel] = {}
        for name in (
            "Samples",
            "Min",
            "Max",
            "Mean",
            "Median",
            "Mode",
            "Std Dev",
            "Variance",
            "Q1 (25%)",
            "Q3 (75%)",
            "IQR",
            "P5",
            "P95",
            "Skewness",
            "Kurtosis (excess)",
            "Zero %",
        ):
            label = QLabel("—")
            label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            self._data_volume_stats_labels[name] = label
            stats_layout.addRow(f"{name}:", label)
        splitter.addWidget(stats_group)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([1, 1])
        outer_layout.addWidget(splitter, stretch=1)

        self._update_data_volume_plots()
        return tab

    def _get_selected_link_budget_station(self) -> GroundStationConfig | None:
        """Return the station referenced by the link-budget dropdown."""
        combo = self.link_budget_station_combo
        if combo is None or combo.count() == 0:
            return None
        data = combo.currentData()
        if isinstance(data, int) and 0 <= data < len(self._station_presets):
            return self._station_presets[data]
        return None

    def _refresh_link_budget_station_list(self) -> None:
        """Keep the link-budget dropdown in sync with the station list."""
        combo = self.link_budget_station_combo
        if combo is None:
            return
        current_data = combo.currentData()
        combo.blockSignals(True)
        combo.clear()
        if not self._station_presets:
            combo.addItem("No stations loaded", None)
            combo.setEnabled(False)
        else:
            for idx, station in enumerate(self._station_presets):
                combo.addItem(station.name, idx)
            combo.setEnabled(True)
            if isinstance(current_data, int) and 0 <= current_data < len(
                self._station_presets
            ):
                combo.setCurrentIndex(current_data)
        combo.blockSignals(False)
        self._trigger_link_budget_recompute()

    def _register_link_budget_inputs(self) -> None:
        """Connect all link-budget controls to automatic recomputation."""
        controls = [
            (self.link_budget_station_combo, "currentIndexChanged"),
            (self.lb_frequency_input, "valueChanged"),
            (self.lb_tx_power_input, "valueChanged"),
            (self.lb_tx_gain_input, "valueChanged"),
            (self.lb_tx_losses_input, "valueChanged"),
            (self.lb_tx_backoff_input, "valueChanged"),
            (self.lb_antenna_gain_input, "valueChanged"),
            (self.lb_rx_gt_input, "valueChanged"),
            (self.lb_rx_losses_input, "valueChanged"),
            (self.lb_symbol_rate_input, "valueChanged"),
            (self.lb_impl_loss_input, "valueChanged"),
            (self.lb_margin_input, "valueChanged"),
            (self.lb_sat_altitude_input, "valueChanged"),
            (self.lb_elev_lower_input, "valueChanged"),
            (self.lb_elev_upper_input, "valueChanged"),
            (self.lb_gs_elevation_input, "valueChanged"),
            (self.lb_unavailability_input, "valueChanged"),
            (self.lb_polarization_loss_input, "valueChanged"),
            (self.lb_rolloff_input, "valueChanged"),
        ]
        for widget, signal_name in controls:
            self._connect_link_budget_signal(widget, signal_name)

    def _connect_link_budget_signal(self, widget, signal_name: str) -> None:
        """Attach the specified signal to trigger recomputation."""
        if widget is None:
            return
        signal = getattr(widget, signal_name, None)
        if signal is None:
            return
        signal.connect(self._trigger_link_budget_recompute)  # type: ignore[attr-defined]

    def _trigger_link_budget_recompute(self, *_args) -> None:
        """Recalculate the link budget when inputs change."""
        if not self._link_budget_auto_enabled:
            return
        self._handle_link_budget_calculate()

    def _get_link_budget_elevation_bounds(self) -> tuple[float, float]:
        """Return (lower_deg, upper_deg) for link-budget-only elevation filtering."""
        lower = float(DEFAULT_LINK_BUDGET_ELEVATION_LOWER_DEG)
        upper = float(DEFAULT_LINK_BUDGET_ELEVATION_UPPER_DEG)
        lower_widget = getattr(self, "lb_elev_lower_input", None)
        upper_widget = getattr(self, "lb_elev_upper_input", None)
        if lower_widget is not None:
            try:
                lower = float(lower_widget.value())
            except Exception:
                pass
        if upper_widget is not None:
            try:
                upper = float(upper_widget.value())
            except Exception:
                pass
        lower = max(0.0, min(90.0, lower))
        upper = max(0.0, min(90.0, upper))
        if upper < lower:
            lower, upper = upper, lower
        return (lower, upper)

    def _handle_link_budget_calculate(self) -> None:
        """Compute the link budget using the math helpers."""
        if (
            self.link_budget_table is None
            or self.link_budget_summary_label is None
            or self.lb_frequency_input is None
        ):
            return
        station = self._get_selected_link_budget_station()
        if station is None:
            if self.link_budget_summary_label:
                self.link_budget_summary_label.setText(
                    "Import or select a ground station to view the link budget."
                )
            if self.link_budget_table:
                self.link_budget_table.setRowCount(0)
            self._clear_link_budget_plot()
            self._link_budget_rate_curve = None
            self._update_downlink_summary()
            self._invalidate_loss_cache()
            self._latest_link_budget_plot_data = None
            return
        frequency = self.lb_frequency_input.value()
        symbol_rate_sps = self.lb_symbol_rate_input.value() * 1e6
        elev_lower_deg, elev_upper_deg = self._get_link_budget_elevation_bounds()
        evaluation_angle = float(self.lb_gs_elevation_input.value())
        evaluation_angle = max(elev_lower_deg, min(elev_upper_deg, evaluation_angle))
        unavailability = self.lb_unavailability_input.value()
        try:
            elevations = np.linspace(float(elev_lower_deg), float(elev_upper_deg), 1000)
            # ITU-R models are not guaranteed at the horizon; evaluate losses with a
            # small floor and reuse that value for any lower elevations.
            loss_query_elev = np.clip(elevations, 0.1, 90.0)
            loss_cache_key = self._build_loss_cache_key(
                frequency=frequency,
                plot_lower_bound=float(loss_query_elev[0]) if loss_query_elev.size else 0.1,
                plot_upper_bound=float(loss_query_elev[-1]) if loss_query_elev.size else 90.0,
                unavailability=unavailability,
                station=station,
                num_samples=int(loss_query_elev.size),
            )
            use_cache = (
                self._loss_cache_key == loss_cache_key
                and self._loss_cache_losses is not None
                and self._loss_cache_losses.shape == loss_query_elev.shape
            )
            if use_cache:
                atmospheric_losses = self._loss_cache_losses
                contribution_breakdown = self._loss_cache_contributions
            else:
                loss_result = estimate_slant_path_loss(
                    frequency_GHz=frequency,
                    elevations_deg=loss_query_elev,
                    lat_deg=station.latitude_deg,
                    lon_deg=station.longitude_deg,
                    altitude_m=station.altitude_m,
                    unavailability_percent=unavailability,
                    return_contributions=True,
                )
                if isinstance(loss_result, tuple):
                    atmospheric_losses, contribution_breakdown = loss_result
                else:
                    atmospheric_losses = loss_result
                    contribution_breakdown = None
                self._loss_cache_key = loss_cache_key
                self._loss_cache_losses = atmospheric_losses
                self._loss_cache_contributions = contribution_breakdown
            results = link_budget_math.calculate_link_budget(
                elevations_deg=elevations,
                antenna_gains_dBi=np.full_like(
                    elevations, self.lb_antenna_gain_input.value()
                ),
                atmospheric_losses_dB=atmospheric_losses,
                tx_power_dBw=self.lb_tx_power_input.value(),
                tx_boresight_gain_dBi=self.lb_tx_gain_input.value(),
                tx_losses_dB=self.lb_tx_losses_input.value(),
                tx_backoff_dB=self.lb_tx_backoff_input.value(),
                frequency_GHz=frequency,
                satellite_altitude_km=self.lb_sat_altitude_input.value(),
                ground_altitude_m=station.altitude_m,
                receiver_G_T_dB_K=self.lb_rx_gt_input.value(),
                receiver_losses_dB=self.lb_rx_losses_input.value(),
                symbol_rate_sps=symbol_rate_sps,
                implementation_loss_dB=self.lb_impl_loss_input.value(),
                margin_dB=self.lb_margin_input.value(),
            )
            rows = link_budget_math.build_parameter_rows(
                elevations_deg=elevations,
                results=results,
                evaluation_elevation_deg=float(evaluation_angle),
                elevation_lower_bound_deg=float(elev_lower_deg),
                elevation_upper_bound_deg=float(elev_upper_deg),
                tx_frequency_GHz=frequency,
                tx_power_dBw=self.lb_tx_power_input.value(),
                tx_losses_dB=self.lb_tx_losses_input.value(),
                tx_boresight_gain_dBi=self.lb_tx_gain_input.value(),
                tx_backoff_dB=self.lb_tx_backoff_input.value(),
                symbol_rate_sps=symbol_rate_sps,
                receiver_G_T_dB_K=self.lb_rx_gt_input.value(),
                implementation_loss_dB=self.lb_impl_loss_input.value(),
                margin_dB=self.lb_margin_input.value(),
                rolloff=self.lb_rolloff_input.value(),
                polarization_loss_dB=self.lb_polarization_loss_input.value(),
                satellite_altitude_km=self.lb_sat_altitude_input.value(),
                atmospheric_breakdown_dB=contribution_breakdown,
            )
        except Exception as exc:
            QMessageBox.warning(self, "Link Budget Error", str(exc))
            self._clear_link_budget_plot()
            self._latest_link_budget_plot_data = None
            self._link_budget_rate_curve = None
            self._update_downlink_summary()
            return
        eval_index = int(np.argmin(np.abs(elevations - evaluation_angle)))
        summary = self._format_link_budget_summary(results, elevations, eval_index)
        self._populate_link_budget_table(rows, summary)
        self._latest_link_budget_plot_data = (elevations, results)
        if self._is_dynamic_link_budget_tab_active():
            self._update_link_budget_plot(elevations, results)
        self._cache_link_budget_curve(elevations, results)
        self._update_downlink_summary()

    def _populate_link_budget_table(
        self,
        rows: list[link_budget_math.ParameterRow],
        summary_text: str,
    ) -> None:
        """Fill the link-budget table and summary label."""
        if not self.link_budget_table or self.link_budget_summary_label is None:
            return
        table = self.link_budget_table
        table.setRowCount(len(rows))
        highlight_rows = {
            "EIRP",
            "Received Signal Power",
            "Maximum Information Rate",
            "Max. Information Rate",
            "Link Margin",
        }
        highlight_color = QColor("#1b5e20")
        for row_idx, entry in enumerate(rows):
            items = [
                QTableWidgetItem(entry.parameter),
                QTableWidgetItem(entry.value),
                QTableWidgetItem(entry.unit),
            ]
            for col, item in enumerate(items):
                if col == 0:
                    align = Qt.AlignmentFlag.AlignLeft
                elif col == 1:
                    align = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                else:
                    align = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
                item.setTextAlignment(align)
                table.setItem(row_idx, col, item)
            if entry.parameter in highlight_rows:
                for item in items:
                    font = QFont(item.font())
                    font.setBold(True)
                    item.setFont(font)
                    item.setBackground(highlight_color)
        if not rows:
            self.link_budget_summary_label.setText(
                "No valid MODCOD found for the current parameters."
            )
        else:
            self.link_budget_summary_label.setText(summary_text)

    def _format_link_budget_summary(
        self,
        results: dict,
        elevations: np.ndarray,
        index: int,
    ) -> str:
        """Generate a concise summary for the selected elevation."""
        elevation_deg = float(elevations[index])
        modcod = results["modcod_names"][index]
        data_rate = float(np.asarray(results["data_rate_mbps"])[index])
        margin = float(np.asarray(results["margin_to_required_EsN0_dB"])[index])
        modcods = np.asarray(results.get("modcod_names", []), dtype=object)
        data_rates = np.asarray(results.get("data_rate_mbps", []), dtype=float)
        closure_text = ""
        if modcods.size == elevations.size:
            linked_indices = np.where(modcods != "No Link")[0]
            if linked_indices.size == 0:
                closure_text = " No MODCOD closes at any elevation."
            else:
                first_idx = int(linked_indices[0])
                min_close = float(elevations[first_idx])
                peak_rate = float(np.nanmax(data_rates)) if data_rates.size else 0.0
                closure_text = f" Link closes above ~{min_close:.1f}° (peak {peak_rate:.2f} Mbps)."
        if modcod == "No Link" or np.isnan(margin):
            return (
                f"No MODCOD closes at {elevation_deg:.1f}°. "
                f"Available data rate: {data_rate:.2f} Mbps."
                f"{closure_text}"
            )
        return (
            f"{modcod} at {elevation_deg:.1f}° delivers {data_rate:.2f} Mbps "
            f"(margin {margin:.2f} dB to threshold)."
            f"{closure_text}"
        )

    def _clear_link_budget_plot(self) -> None:
        """Remove plot data and annotations."""
        if not getattr(self, "link_budget_plot", None):
            return
        self.link_budget_plot.clear()
        if getattr(self, "_link_budget_plot_legend", None):
            self._link_budget_plot_legend.clear()
        for item in getattr(self, "_link_budget_plot_annotations", []):
            try:
                self.link_budget_plot.removeItem(item)
            except Exception:
                pass
        self._link_budget_plot_annotations = []

    def _invalidate_loss_cache(self) -> None:
        self._loss_cache_key = None
        self._loss_cache_losses = None
        self._loss_cache_contributions = None

    def _build_loss_cache_key(
        self,
        *,
        frequency: float,
        plot_lower_bound: float,
        plot_upper_bound: float,
        unavailability: float,
        station: GroundStationConfig,
        num_samples: int,
    ) -> tuple:
        return (
            round(float(frequency), 6),
            round(float(plot_lower_bound), 4),
            round(float(plot_upper_bound), 4),
            round(float(unavailability), 6),
            round(float(station.latitude_deg), 6),
            round(float(station.longitude_deg), 6),
            round(float(station.altitude_m), 3),
            int(num_samples),
        )

    def _is_dynamic_link_budget_tab_active(self) -> bool:
        return (
            self.link_budget_tabs is not None
            and getattr(self, "_dynamic_link_budget_tab", None) is not None
            and self.link_budget_tabs.currentWidget() is self._dynamic_link_budget_tab
        )

    def _on_link_budget_tab_changed(self, index: int) -> None:
        if not self._is_dynamic_link_budget_tab_active():
            return
        if not self._latest_link_budget_plot_data:
            return
        elevations, results = self._latest_link_budget_plot_data
        self._update_link_budget_plot(elevations, results)

    def _update_link_budget_plot(
        self,
        elevations: np.ndarray,
        results: dict,
    ) -> None:
        """Render the Es/N0 envelope as a function of elevation."""
        if not getattr(self, "link_budget_plot", None):
            return
        if elevations.size == 0:
            self._clear_link_budget_plot()
            return
        es_n0 = np.asarray(results.get("es_to_n0_dB", []), dtype=float)
        required = np.asarray(results.get("required_EsN0_dB", []), dtype=float)
        if es_n0.size != elevations.size:
            self._clear_link_budget_plot()
            return
        margin_offset = self.lb_margin_input.value()

        def _build_step_curve(
            x_values: np.ndarray, y_values: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray]:
            if len(x_values) == 0 or len(y_values) == 0:
                return x_values, y_values
            repeated_x = np.repeat(x_values, 2)[1:]
            repeated_y = np.repeat(y_values, 2)[:-1]
            return repeated_x, repeated_y

        self.link_budget_plot.clear()
        if getattr(self, "_link_budget_plot_legend", None):
            self._link_budget_plot_legend.clear()
        for item in self._link_budget_plot_annotations:
            try:
                self.link_budget_plot.removeItem(item)
            except Exception:
                pass
        self._link_budget_plot_annotations = []
        self.link_budget_plot.plot(
            elevations,
            es_n0,
            pen=pg.mkPen("#ff9800", width=2),
            name="Maximum Es/N0",
        )
        modcods = results.get("modcod_names", [])
        valid_throughput = np.where(
            ~np.isnan(required), required + margin_offset, np.nan
        )
        if len(modcods) == len(valid_throughput):
            valid_mask = np.array([name != "No Link" for name in modcods], dtype=bool)
            valid_throughput = np.where(valid_mask, valid_throughput, np.nan)
        step_x, step_y = _build_step_curve(elevations, valid_throughput)
        self.link_budget_plot.plot(
            step_x,
            step_y,
            pen=pg.mkPen("#5b7dff", width=2),
            name="VCM Step",
        )
        valid_margin = es_n0 - valid_throughput
        self.link_budget_plot.plot(
            elevations,
            valid_margin,
            pen=pg.mkPen("#00c853", width=2),
            name="VCM Margin",
        )
        y_min = -5.0
        y_max = 20.0
        self.link_budget_plot.setYRange(y_min, y_max, padding=0.0)
        self.link_budget_plot.setLimits(yMin=y_min, yMax=y_max)
        modcods = results.get("modcod_names", [])
        if len(modcods) == len(elevations):
            start_idx = 0
            current_name = modcods[0]
            segments: list[tuple[int, int, str]] = []
            for idx in range(1, len(modcods)):
                if modcods[idx] != current_name:
                    segments.append((start_idx, idx - 1, current_name))
                    start_idx = idx
                    current_name = modcods[idx]
            segments.append((start_idx, len(modcods) - 1, current_name))
            for start, end, name in segments:
                if name == "No Link":
                    continue
                segment_values = valid_throughput[start : end + 1]
                mean_value = np.nanmean(segment_values)
                if np.isnan(mean_value):
                    continue
                center_x = (elevations[start] + elevations[end]) / 2.0
                text_item = pg.TextItem(name, color="#d0d0d0", anchor=(0.5, -0.3))
                text_item.setPos(center_x, mean_value)
                self.link_budget_plot.addItem(text_item)
                self._link_budget_plot_annotations.append(text_item)
        if len(modcods) == len(elevations):
            modcod_array = np.asarray(modcods, dtype=object)
            linked_indices = np.where(modcod_array != "No Link")[0]
            if linked_indices.size > 0:
                boundary_idx = int(linked_indices[0])
                if boundary_idx > 0 and np.all(
                    modcod_array[:boundary_idx] == "No Link"
                ):
                    boundary_x = float(elevations[boundary_idx])
                    shade = pg.LinearRegionItem(
                        values=(float(elevations[0]), boundary_x),
                        brush=pg.mkBrush(255, 77, 77, 60),
                        movable=False,
                    )
                    shade.setZValue(-100)
                    self.link_budget_plot.addItem(shade)
                    self._link_budget_plot_annotations.append(shade)
                    line_pen = pg.mkPen("#ff4d4d", width=2, style=Qt.PenStyle.DotLine)
                    boundary_line = pg.InfiniteLine(
                        pos=boundary_x, angle=90, pen=line_pen
                    )
                    self.link_budget_plot.addItem(boundary_line)
                    self._link_budget_plot_annotations.append(boundary_line)

    def _store_access_series(self, result) -> None:
        """Cache the elevation time series for downlink calculations."""
        if not result.timeline_seconds or not result.station_elevation_series:
            self._latest_access_series = None
            self._latest_station_rate_series = None
            self._latest_combined_rate_series = None
            return
        time_seconds = np.asarray(result.timeline_seconds, dtype=float)
        if time_seconds.ndim != 1 or time_seconds.size == 0:
            self._latest_access_series = None
            self._latest_station_rate_series = None
            self._latest_combined_rate_series = None
            return
        station_series: dict[str, np.ndarray] = {}
        for name, samples in result.station_elevation_series.items():
            series = np.asarray(samples, dtype=float)
            if series.size != time_seconds.size:
                continue
            station_series[name] = series
        if not station_series:
            self._latest_access_series = None
            self._latest_station_rate_series = None
            self._latest_combined_rate_series = None
            return
        altitude_series = np.asarray(getattr(result, "orbital_altitude_km", []), dtype=float)
        if altitude_series.size != time_seconds.size:
            altitude_series = self._infer_altitude_series_from_track(result, time_seconds.size)
        if altitude_series is not None and altitude_series.size == time_seconds.size:
            altitude_payload = altitude_series
        else:
            altitude_payload = None
        payload = {
            "time_seconds": time_seconds,
            "station_series": station_series,
            "orbit_period_s": float(result.orbit_period_seconds or 0.0),
        }
        if altitude_payload is not None:
            payload["altitude_km"] = altitude_payload
        self._latest_access_series = payload
        self._latest_station_rate_series = None
        self._latest_combined_rate_series = None

    def _infer_altitude_series_from_track(
        self, result, expected_size: int
    ) -> np.ndarray | None:
        """Derive altitude samples from the ground-track positions."""
        track = getattr(result, "ground_track", None)
        if not track or len(track) != expected_size:
            return None
        coords = np.array([[pt.x_km, pt.y_km, pt.z_km] for pt in track], dtype=float)
        if coords.size == 0:
            return None
        radii = np.linalg.norm(coords, axis=1)
        return radii - link_budget_math.EARTH_RADIUS_KM

    def _refresh_data_volume_rate_series(self) -> None:
        """Recompute cached data-rate series using propagated altitude samples."""
        try:
            result = self._calculate_station_rate_series()
        except Exception as exc:  # pragma: no cover - defensive guard for GUI stability
            print(f"[DataVolume] Failed to recompute rates: {exc}")
            result = None
        if result is None:
            self._latest_station_rate_series = None
            self._latest_combined_rate_series = None
            return
        station_rates, combined_rates = result
        self._latest_station_rate_series = station_rates
        self._latest_combined_rate_series = combined_rates

    def _calculate_station_rate_series(
        self,
    ) -> tuple[dict[str, np.ndarray], np.ndarray | None] | None:
        """Evaluate Mbps time series for each station using propagated altitudes."""
        if not self._latest_access_series:
            return None
        altitude_samples = np.asarray(
            self._latest_access_series.get("altitude_km", []), dtype=float
        )
        time_seconds = np.asarray(
            self._latest_access_series.get("time_seconds", []), dtype=float
        )
        station_series: dict[str, np.ndarray] = self._latest_access_series.get(
            "station_series", {}
        )
        if time_seconds.size == 0 or not station_series:
            return None
        if altitude_samples.size != time_seconds.size:
            return None
        controls = [
            self.lb_frequency_input,
            self.lb_tx_power_input,
            self.lb_tx_gain_input,
            self.lb_tx_losses_input,
            self.lb_tx_backoff_input,
            self.lb_antenna_gain_input,
            self.lb_rx_gt_input,
            self.lb_rx_losses_input,
            self.lb_symbol_rate_input,
            self.lb_impl_loss_input,
            self.lb_margin_input,
            self.lb_unavailability_input,
        ]
        if any(widget is None for widget in controls):
            return None
        frequency = self.lb_frequency_input.value()
        if frequency <= 0.0:
            return None
        symbol_rate_sps = self.lb_symbol_rate_input.value() * 1e6
        if symbol_rate_sps <= 0.0:
            return None
        params = {
            "frequency_GHz": frequency,
            "symbol_rate_sps": symbol_rate_sps,
            "tx_power_dBw": self.lb_tx_power_input.value(),
            "tx_gain_dBi": self.lb_tx_gain_input.value(),
            "tx_losses_dB": self.lb_tx_losses_input.value(),
            "tx_backoff_dB": self.lb_tx_backoff_input.value(),
            "antenna_gain_dBi": self.lb_antenna_gain_input.value(),
            "rx_gt_dB": self.lb_rx_gt_input.value(),
            "rx_losses_dB": self.lb_rx_losses_input.value(),
            "implementation_loss_dB": self.lb_impl_loss_input.value(),
            "margin_dB": self.lb_margin_input.value(),
            "unavailability_percent": self.lb_unavailability_input.value(),
        }
        finite_altitude = np.nan_to_num(
            altitude_samples,
            nan=float(np.nanmedian(altitude_samples)) if np.any(np.isfinite(altitude_samples)) else 0.0,
        )
        station_rates: dict[str, np.ndarray] = {}
        combined_rates: np.ndarray | None = None
        for name, elevations in station_series.items():
            elevation_array = np.asarray(elevations, dtype=float)
            if elevation_array.size != finite_altitude.size:
                continue
            station_cfg = self._resolve_station_config(name)
            if station_cfg is None:
                continue
            rates = self._evaluate_station_data_rate(
                station=station_cfg,
                elevations_deg=elevation_array,
                altitude_km=finite_altitude,
                **params,
            )
            if rates is None or rates.size != finite_altitude.size:
                continue
            station_rates[name] = rates
            combined_rates = rates if combined_rates is None else combined_rates + rates
        if not station_rates:
            return None
        return station_rates, combined_rates

    def _evaluate_station_data_rate(
        self,
        *,
        station: GroundStationConfig,
        elevations_deg: np.ndarray,
        altitude_km: np.ndarray,
        frequency_GHz: float,
        symbol_rate_sps: float,
        tx_power_dBw: float,
        tx_gain_dBi: float,
        tx_losses_dB: float,
        tx_backoff_dB: float,
        antenna_gain_dBi: float,
        rx_gt_dB: float,
        rx_losses_dB: float,
        implementation_loss_dB: float,
        margin_dB: float,
        unavailability_percent: float,
    ) -> np.ndarray | None:
        """Run the link-budget math for a specific station timeline."""
        if elevations_deg.size == 0 or altitude_km.size != elevations_deg.size:
            return None
        elev_lower_deg, elev_upper_deg = self._get_link_budget_elevation_bounds()
        raw_elev = np.asarray(elevations_deg, dtype=float)
        raw_elev = np.nan_to_num(raw_elev, nan=-90.0)
        sanitized_elev = np.clip(raw_elev, 0.0, 90.0)
        if sanitized_elev.size == 0:
            return None
        curve = self._get_timeseries_atmospheric_loss_curve(
            station=station,
            frequency_GHz=frequency_GHz,
            unavailability_percent=unavailability_percent,
            min_elev_deg=max(0.1, float(elev_lower_deg)),
            max_elev_deg=max(max(0.1, float(elev_lower_deg)), float(elev_upper_deg)),
        )
        if curve is None:
            return None
        loss_grid, loss_values = curve
        losses = np.interp(
            np.maximum(sanitized_elev, float(loss_grid[0])),
            loss_grid,
            loss_values,
            left=float(loss_values[0]),
            right=float(loss_values[-1]),
        )
        antenna_gains = np.full_like(elevations_deg, float(antenna_gain_dBi), dtype=float)
        try:
            results = link_budget_math.calculate_link_budget(
                elevations_deg=sanitized_elev,
                antenna_gains_dBi=antenna_gains,
                atmospheric_losses_dB=losses,
                tx_power_dBw=tx_power_dBw,
                tx_boresight_gain_dBi=tx_gain_dBi,
                tx_losses_dB=tx_losses_dB,
                tx_backoff_dB=tx_backoff_dB,
                frequency_GHz=frequency_GHz,
                satellite_altitude_km=altitude_km,
                ground_altitude_m=station.altitude_m,
                receiver_G_T_dB_K=rx_gt_dB,
                receiver_losses_dB=rx_losses_dB,
                symbol_rate_sps=symbol_rate_sps,
                implementation_loss_dB=implementation_loss_dB,
                margin_dB=margin_dB,
            )
        except Exception:
            return None
        rates = np.asarray(results.get("data_rate_mbps", []), dtype=float)
        if rates.size != elevations_deg.size:
            return None
        rates = np.nan_to_num(rates, nan=0.0)
        rates[~np.isfinite(rates)] = 0.0
        outside_mask = (raw_elev < float(elev_lower_deg)) | (raw_elev > float(elev_upper_deg))
        if np.any(outside_mask):
            rates[outside_mask] = 0.0
        return rates

    def _get_timeseries_atmospheric_loss_curve(
        self,
        *,
        station: GroundStationConfig,
        frequency_GHz: float,
        unavailability_percent: float,
        min_elev_deg: float = 0.1,
        max_elev_deg: float = 90.0,
        num_samples: int = 361,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Return (elevation_grid, loss_dB_grid) for fast interpolation in time-series.

        This keeps ITU model calls limited to changes that actually affect atmospheric loss
        (frequency, unavailability, station location/height), so tweaks like implementation
        loss/margin don't re-run ITU and freeze the GUI.
        """
        if frequency_GHz <= 0.0:
            return None
        if num_samples < 10:
            num_samples = 10
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
        cache = getattr(self, "_timeseries_loss_cache", None)
        if isinstance(cache, dict) and key in cache:
            cached_grid, cached_losses = cache[key]
            if (
                isinstance(cached_grid, np.ndarray)
                and isinstance(cached_losses, np.ndarray)
                and cached_grid.shape == grid.shape
                and cached_losses.shape == grid.shape
            ):
                return cached_grid, cached_losses
        try:
            losses = estimate_slant_path_loss(
                frequency_GHz=frequency_GHz,
                elevations_deg=grid,
                lat_deg=station.latitude_deg,
                lon_deg=station.longitude_deg,
                altitude_m=station.altitude_m,
                unavailability_percent=unavailability_percent,
            )
        except Exception:
            return None
        losses = np.asarray(losses, dtype=float)
        losses = np.nan_to_num(losses, nan=0.0)
        losses[~np.isfinite(losses)] = 0.0
        if isinstance(cache, dict):
            cache[key] = (grid, losses)
        return grid, losses

    def _resolve_station_config(self, station_name: str) -> GroundStationConfig | None:
        """Find the active station configuration by name."""
        if not station_name:
            return None
        station = self._active_station_lookup.get(station_name)
        if station is None:
            station = next(
                (cfg for cfg in getattr(self, "_station_presets", []) if cfg.name == station_name),
                None,
            )
        return station

    def _cache_link_budget_curve(self, elevations: np.ndarray, results: dict) -> None:
        """Store the latest throughput curve for interpolation."""
        rates = np.asarray(results.get("data_rate_mbps", []), dtype=float)
        if elevations.size and rates.size == elevations.size:
            self._link_budget_rate_curve = (np.asarray(elevations, dtype=float), rates)
        else:
            self._link_budget_rate_curve = None

    def _update_downlink_summary(self) -> None:
        """Refresh the downlink summary labels and data volume plots."""
        if not self._downlink_total_label or not self._downlink_per_orbit_label:
            return
        # Debounce the heavy recomputation so UI edits don't freeze the GUI.
        self._schedule_data_volume_refresh()

        # Keep showing the last computed values (or placeholders) until the refresh completes.
        metrics = self._compute_downlink_metrics()
        if metrics is None:
            self._downlink_total_label.setText("—")
            self._downlink_per_orbit_label.setText("—")
            return
        total_gibit, per_orbit_gibit = metrics
        self._downlink_total_label.setText(f"{total_gibit:.2f} Gibit")
        if per_orbit_gibit is None:
            self._downlink_per_orbit_label.setText("—")
        else:
            self._downlink_per_orbit_label.setText(f"{per_orbit_gibit:.2f} Gibit/orbit")

    def _schedule_data_volume_refresh(self) -> None:
        timer = getattr(self, "_data_volume_refresh_timer", None)
        if timer is None:
            return
        timer.start()

    def _perform_data_volume_refresh(self) -> None:
        """Compute time-series rates + refresh summary labels + plots (debounced)."""
        if not self._downlink_total_label or not self._downlink_per_orbit_label:
            return
        self._refresh_data_volume_rate_series()
        metrics = self._compute_downlink_metrics()
        if metrics is None:
            self._downlink_total_label.setText("—")
            self._downlink_per_orbit_label.setText("—")
        else:
            total_gibit, per_orbit_gibit = metrics
            self._downlink_total_label.setText(f"{total_gibit:.2f} Gibit")
            if per_orbit_gibit is None:
                self._downlink_per_orbit_label.setText("—")
            else:
                self._downlink_per_orbit_label.setText(f"{per_orbit_gibit:.2f} Gibit/orbit")
        self._update_data_volume_plots()

    def _compute_downlink_metrics(self) -> tuple[float, float | None] | None:
        """Compute total and per-orbit Gibit using cached data."""
        if (
            not self._latest_access_series
            or not getattr(self, "_latest_station_rate_series", None)
        ):
            return None
        time_seconds = np.asarray(
            self._latest_access_series.get("time_seconds", []), dtype=float
        )
        station_series: dict[str, np.ndarray] = self._latest_access_series.get(
            "station_series", {}
        )
        orbit_period_s = float(self._latest_access_series.get("orbit_period_s", 0.0))
        if time_seconds.size < 2 or not station_series:
            return None
        station_rate_lookup = getattr(self, "_latest_station_rate_series", None) or {}
        combined_rates = getattr(self, "_latest_combined_rate_series", None)
        total_gbit = 0.0
        for name, series in station_series.items():
            if series.size != time_seconds.size:
                continue
            rates_array = station_rate_lookup.get(name)
            if rates_array is None or rates_array.size != time_seconds.size:
                continue
            if not np.any(rates_array):
                continue
            gbit = link_budget_math.integrate_data_volume_gb(
                time_seconds, rates_array
            )
            total_gbit += max(0.0, float(gbit))
        if total_gbit <= 0.0 and combined_rates is not None:
            if combined_rates.size == time_seconds.size and np.any(combined_rates):
                gbit = link_budget_math.integrate_data_volume_gb(
                    time_seconds, combined_rates
                )
                total_gbit = max(0.0, float(gbit))
        if total_gbit <= 0.0:
            return (0.0, None)
        total_gibit = total_gbit * GIBIT_PER_GBIT
        duration_s = float(time_seconds[-1] - time_seconds[0])
        per_orbit = None
        if orbit_period_s > 0.0:
            num_orbits = duration_s / orbit_period_s
            if num_orbits > 0:
                per_orbit = total_gibit / num_orbits
        return (total_gibit, per_orbit)

    def _compute_daily_downlink_volumes(self) -> dict[int, float] | None:
        """Compute data volume per day over the scenario window.

        Returns a dictionary mapping day number (0-indexed from start) to
        data volume in Gibit for that day.
        """
        if (
            not self._latest_access_series
            or not getattr(self, "_latest_station_rate_series", None)
        ):
            return None
        if not hasattr(self, "_current_config") or self._current_config is None:
            return None

        time_seconds = np.asarray(
            self._latest_access_series.get("time_seconds", []), dtype=float
        )
        station_series: dict[str, np.ndarray] = self._latest_access_series.get(
            "station_series", {}
        )
        if time_seconds.size < 2 or not station_series:
            return None

        combined_rates = getattr(self, "_latest_combined_rate_series", None)
        if combined_rates is None:
            station_rates = getattr(self, "_latest_station_rate_series", None) or {}
            accum = np.zeros_like(time_seconds, dtype=float)
            any_rates = False
            for rates_array in station_rates.values():
                if rates_array.size != time_seconds.size:
                    continue
                accum += rates_array
                any_rates = True
            combined_rates = accum if any_rates else None
        if combined_rates is None or combined_rates.size != time_seconds.size:
            return None

        # Get scenario start time
        start_time = self._current_config.scenario.start_time

        # Compute combined data rate at each time sample
        combined_rates = np.nan_to_num(combined_rates, nan=0.0)
        # Group by day and integrate
        from datetime import timedelta

        daily_volumes: dict[int, float] = {}

        for i in range(len(time_seconds) - 1):
            current_time = start_time + timedelta(seconds=float(time_seconds[i]))
            next_time = start_time + timedelta(seconds=float(time_seconds[i + 1]))

            # Calculate day number from scenario start
            current_day = (current_time.date() - start_time.date()).days
            next_day = (next_time.date() - start_time.date()).days

            # Time interval in seconds
            dt = time_seconds[i + 1] - time_seconds[i]

            # Average data rate over the interval (Mbps)
            avg_rate = (combined_rates[i] + combined_rates[i + 1]) / 2.0

            # Data volume in Mb
            data_megabits = avg_rate * dt

            if current_day == next_day:
                # Interval is within the same day
                daily_volumes[current_day] = (
                    daily_volumes.get(current_day, 0.0) + data_megabits
                )
            else:
                # Interval spans multiple days - need to split
                # For simplicity, we'll use a linear approximation
                # More sophisticated handling could be added if needed

                # Calculate seconds from current_time to end of current day
                from datetime import datetime

                end_of_current_day = current_time.replace(
                    hour=23, minute=59, second=59, microsecond=999999
                )
                seconds_in_current_day = (
                    end_of_current_day - current_time
                ).total_seconds()

                # Fraction of interval in current day
                fraction_current = min(seconds_in_current_day / dt, 1.0)

                # Split the data volume proportionally
                daily_volumes[current_day] = (
                    daily_volumes.get(current_day, 0.0)
                    + data_megabits * fraction_current
                )

                # Remaining goes to next day(s)
                remaining_data = data_megabits * (1.0 - fraction_current)
                days_spanned = next_day - current_day

                if days_spanned == 1:
                    daily_volumes[next_day] = (
                        daily_volumes.get(next_day, 0.0) + remaining_data
                    )
                else:
                    # Multiple days spanned - distribute evenly
                    # This is a rare edge case for very long time steps
                    per_day = remaining_data / days_spanned
                    for day in range(current_day + 1, next_day + 1):
                        daily_volumes[day] = daily_volumes.get(day, 0.0) + per_day

        # Convert from Mb to Gibit
        for day in daily_volumes:
            daily_volumes[day] = daily_volumes[day] / 1000.0 * GIBIT_PER_GBIT

        return daily_volumes

    def _get_data_volume_distribution_options(self) -> list[tuple[str, str]]:
        """Return (id, label) options for the distribution selector."""
        return [
            ("gibit_per_pass", "Gibit per pass"),
            (
                "downlink_rate_mbps",
                "Downlink rate (Mbps) — within elevation bounds (all samples)",
            ),
            (
                "downlink_rate_mbps_nonzero",
                "Downlink rate (Mbps) — within elevation bounds (nonzero)",
            ),
            ("pass_duration_min", "Pass time in elevation bounds (min)"),
            ("pass_in_bounds_pct", "Pass time in elevation bounds (%)"),
            ("pass_peak_elevation_in_bounds_deg", "Pass peak elevation within bounds (deg)"),
            ("gibit_per_day", "Gibit per day"),
        ]

    def _get_selected_distribution_id(self) -> str:
        combo = getattr(self, "data_volume_distribution_combo", None)
        if combo is None or combo.count() == 0:
            return "gibit_per_pass"
        data = combo.currentData()
        if data is None:
            return "gibit_per_pass"
        return str(data)

    def _get_combined_rate_series(self) -> np.ndarray | None:
        """Return combined Mbps time-series, if available."""
        combined = getattr(self, "_latest_combined_rate_series", None)
        if combined is not None:
            try:
                arr = np.asarray(combined, dtype=float)
                if arr.ndim == 1 and arr.size:
                    return arr
            except Exception:
                pass
        station_rates = getattr(self, "_latest_station_rate_series", None) or {}
        if not isinstance(station_rates, dict) or not station_rates:
            return None
        combined_series = None
        for series in station_rates.values():
            if series is None:
                continue
            arr = np.asarray(series, dtype=float)
            combined_series = arr if combined_series is None else combined_series + arr
        return combined_series

    def _get_any_station_in_bounds_mask(self) -> np.ndarray | None:
        """Return mask for samples where any station elevation is within bounds."""
        if not self._latest_access_series:
            return None
        time_seconds = np.asarray(
            self._latest_access_series.get("time_seconds", []), dtype=float
        )
        if time_seconds.ndim != 1 or time_seconds.size == 0:
            return None
        station_series: dict[str, np.ndarray] = self._latest_access_series.get(
            "station_series", {}
        )
        if not isinstance(station_series, dict) or not station_series:
            return None
        lower, upper = self._get_link_budget_elevation_bounds()
        mask_any = np.zeros(time_seconds.size, dtype=bool)
        for elevations in station_series.values():
            elev = np.asarray(elevations, dtype=float)
            if elev.size != time_seconds.size:
                continue
            elev = np.nan_to_num(elev, nan=-90.0)
            mask_any |= (elev >= float(lower)) & (elev <= float(upper))
        return mask_any

    def _integrate_indicator_interval(
        self,
        time_axis: np.ndarray,
        indicator: np.ndarray,
        start_sec: float,
        end_sec: float,
    ) -> float:
        """Integrate a 0/1 indicator over an interval (returns seconds)."""
        if end_sec <= start_sec:
            return 0.0
        if time_axis.size < 2 or indicator.size != time_axis.size:
            return 0.0
        y = np.asarray(indicator, dtype=float)
        mask = (time_axis > start_sec) & (time_axis < end_sec)
        interval_times = time_axis[mask]
        interval_vals = y[mask]
        start_val = float(np.interp(start_sec, time_axis, y, left=0.0, right=0.0))
        end_val = float(np.interp(end_sec, time_axis, y, left=0.0, right=0.0))
        times = np.concatenate(([start_sec], interval_times, [end_sec]))
        vals = np.concatenate(([start_val], interval_vals, [end_val]))
        return float(np.trapezoid(vals, times))

    def _compute_pass_time_in_bounds_minutes(self) -> list[float] | None:
        """Return per-pass duration spent inside the link-budget elevation bounds."""
        if (
            not getattr(self, "_last_result", None)
            or not self._latest_access_series
            or not getattr(self, "_current_config", None)
        ):
            return None
        passes = getattr(self._last_result, "passes", None)
        if not passes:
            return None
        time_seconds = np.asarray(
            self._latest_access_series.get("time_seconds", []), dtype=float
        )
        if time_seconds.size < 2:
            return None
        station_series: dict[str, np.ndarray] = self._latest_access_series.get(
            "station_series", {}
        )
        lower, upper = self._get_link_budget_elevation_bounds()
        start_time = getattr(self._current_config.scenario, "start_time", None)
        if start_time is None:
            return None
        start_axis = float(time_seconds[0])
        end_axis = float(time_seconds[-1])

        # Fallback union mask when a pass has no station name.
        union_mask = self._get_any_station_in_bounds_mask()
        if union_mask is None:
            union_mask = np.zeros(time_seconds.size, dtype=bool)

        durations_min: list[float] = []
        for entry in passes:
            station_name = getattr(entry, "station_name", None)
            elev = None
            if station_name and station_name in station_series:
                elev = np.asarray(station_series[station_name], dtype=float)
            if elev is None or elev.size != time_seconds.size:
                in_bounds = union_mask
            else:
                elev = np.nan_to_num(elev, nan=-90.0)
                in_bounds = (elev >= float(lower)) & (elev <= float(upper))

            aos_seconds = (entry.aos - start_time).total_seconds()
            los_seconds = (entry.los - start_time).total_seconds()
            pass_start = max(start_axis, float(aos_seconds))
            pass_end = min(end_axis, float(los_seconds))
            if pass_end <= pass_start:
                durations_min.append(0.0)
                continue
            seconds_in_bounds = self._integrate_indicator_interval(
                time_axis=time_seconds,
                indicator=in_bounds.astype(float),
                start_sec=pass_start,
                end_sec=pass_end,
            )
            durations_min.append(max(0.0, seconds_in_bounds / 60.0))
        return durations_min or None

    def _compute_distribution_samples(
        self, distribution_id: str
    ) -> tuple[np.ndarray | None, str, str, str, str]:
        """Return (samples, x_label, x_units, empty_title, full_title)."""
        distribution_id = str(distribution_id or "gibit_per_pass")

        if distribution_id == "gibit_per_pass":
            samples = self._compute_pass_downlink_volumes()
            array = (
                np.asarray(samples, dtype=float)
                if samples is not None and len(samples) > 0
                else None
            )
            return (
                array,
                "Data volume",
                "Gibit/pass",
                "Gibit per pass\n(Run analysis to view)",
                "Gibit per pass distribution",
            )

        if distribution_id in ("downlink_rate_mbps", "downlink_rate_mbps_nonzero"):
            series = self._get_combined_rate_series()
            if series is None or series.size == 0:
                return (
                    None,
                    "Downlink rate",
                    "Mbps",
                    "Downlink rate\n(Run analysis to view)",
                    "Downlink rate distribution",
                )
            values = np.asarray(series, dtype=float)
            mask_any = self._get_any_station_in_bounds_mask()
            if mask_any is not None and mask_any.size == values.size:
                values = values[mask_any]
            if distribution_id.endswith("_nonzero"):
                values = values[values > 0.0]
            return (
                values,
                "Downlink rate",
                "Mbps",
                "Downlink rate\n(Run analysis to view)",
                "Downlink rate distribution",
            )

        if distribution_id == "pass_duration_min":
            durations = self._compute_pass_time_in_bounds_minutes()
            if not durations:
                return (
                    None,
                    "Time in bounds",
                    "min",
                    "Pass time in bounds\n(Run analysis to view)",
                    "Pass time in elevation bounds distribution",
                )
            return (
                np.asarray(durations, dtype=float),
                "Time in bounds",
                "min",
                "Pass time in bounds\n(Run analysis to view)",
                "Pass time in elevation bounds distribution",
            )

        if distribution_id == "pass_in_bounds_pct":
            durations = self._compute_pass_time_in_bounds_minutes()
            passes = getattr(getattr(self, "_last_result", None), "passes", None)
            if not durations or not passes:
                return (
                    None,
                    "Time in bounds",
                    "%",
                    "Pass time in bounds\n(Run analysis to view)",
                    "Pass time in elevation bounds distribution",
                )
            total_minutes = np.asarray(
                [float(getattr(p, "duration_minutes", 0.0)) for p in passes], dtype=float
            )
            in_bounds = np.asarray(durations, dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                pct = np.where(total_minutes > 0.0, 100.0 * in_bounds / total_minutes, np.nan)
            pct = np.nan_to_num(pct, nan=0.0, posinf=0.0, neginf=0.0)
            return (
                pct,
                "Time in bounds",
                "%",
                "Pass time in bounds\n(Run analysis to view)",
                "Pass time in elevation bounds distribution",
            )

        if distribution_id == "pass_peak_elevation_in_bounds_deg":
            passes = getattr(getattr(self, "_last_result", None), "passes", None)
            if not passes:
                return (
                    None,
                    "Peak elevation",
                    "deg",
                    "Pass peak elevation\n(Run analysis to view)",
                    "Pass peak elevation within bounds distribution",
                )
            lower, upper = self._get_link_budget_elevation_bounds()
            raw = np.asarray(
                [float(getattr(p, "max_elevation_deg", 0.0)) for p in passes],
                dtype=float,
            )
            # Only passes that reach at least the lower bound are relevant; for the
            # upper bound, clamp so the metric reflects the selected window.
            values = np.where(raw >= float(lower), np.minimum(raw, float(upper)), np.nan)
            return (
                values,
                "Peak elevation",
                "deg",
                "Pass peak elevation\n(Run analysis to view)",
                "Pass peak elevation within bounds distribution",
            )

        if distribution_id == "gibit_per_day":
            daily = self._compute_daily_downlink_volumes()
            values = (
                np.asarray(list(daily.values()), dtype=float)
                if daily
                else None
            )
            return (
                values,
                "Data volume",
                "Gibit/day",
                "Gibit per day\n(Run analysis to view)",
                "Gibit per day distribution",
            )

        # Unknown id fallback.
        return (
            None,
            "Value",
            "",
            "Distribution\n(Run analysis to view)",
            "Distribution",
        )

    def _update_data_volume_plots(self) -> None:
        """Refresh the selected distribution plot + statistics."""
        plot = getattr(self, "data_volume_plot", None)
        if plot is None:
            return
        distribution_id = self._get_selected_distribution_id()
        samples, x_label, x_units, empty_title, full_title = self._compute_distribution_samples(
            distribution_id
        )
        array = (
            np.asarray(samples, dtype=float)
            if samples is not None and np.asarray(samples).size > 0
            else None
        )
        self._last_distribution_samples = array
        self._last_distribution_units = x_units
        manual_width = None
        widths = getattr(self, "_distribution_bin_widths", None)
        if isinstance(widths, dict):
            manual_width = widths.get(distribution_id)
        self._render_volume_histogram(
            plot_widget=plot,
            samples=array,
            manual_width=manual_width,
            bin_label=getattr(self, "data_volume_bin_label", None),
            empty_title=empty_title,
            full_title=full_title,
            x_label=x_label,
            x_units=x_units,
            y_label="Count",
        )
        self._update_distribution_stats(array, x_units, manual_width)

    def _update_distribution_stats(
        self, samples: np.ndarray | None, units: str, manual_width: float | None
    ) -> None:
        labels = getattr(self, "_data_volume_stats_labels", None)
        if not isinstance(labels, dict):
            return

        def _set(key: str, value: str) -> None:
            label = labels.get(key)
            if label is not None:
                label.setText(value)

        # Clear
        if samples is None or samples.size == 0:
            for key in labels:
                _set(key, "—")
            return

        values = np.asarray(samples, dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            for key in labels:
                _set(key, "—")
            return

        # Most distributions are non-negative; keep consistent with the histogram.
        values = values[values >= 0.0]
        if values.size == 0:
            for key in labels:
                _set(key, "—")
            return

        n = int(values.size)
        vmin = float(np.min(values))
        vmax = float(np.max(values))
        mean = float(np.mean(values))
        median = float(np.median(values))
        std = float(np.std(values, ddof=1)) if n > 1 else 0.0
        var = float(np.var(values, ddof=1)) if n > 1 else 0.0
        q1, q3 = np.percentile(values, [25, 75]).astype(float)
        p5, p95 = np.percentile(values, [5, 95]).astype(float)
        iqr = float(q3 - q1)

        # Histogram-mode for continuous distributions.
        mode_val: float | None = None
        if n == 1 or abs(vmax - vmin) < 1e-12:
            mode_val = float(values[0])
        else:
            bw = (
                float(manual_width)
                if manual_width is not None and manual_width > 0
                else float(self._suggest_bin_width(values))
            )
            bw = max(bw, 1e-6)
            bins = np.arange(0.0, vmax + bw * 1.5, bw)
            if bins.size >= 2:
                counts, edges = np.histogram(values, bins=bins)
                if counts.size:
                    idx = int(np.argmax(counts))
                    mode_val = float((edges[idx] + edges[idx + 1]) / 2.0)

        # Shape stats (population moments).
        if std > 0.0 and np.isfinite(std):
            centered = values - mean
            m3 = float(np.mean(centered**3))
            m4 = float(np.mean(centered**4))
            skew = m3 / (std**3) if std > 0 else 0.0
            kurt_excess = m4 / (std**4) - 3.0 if std > 0 else 0.0
        else:
            skew = 0.0
            kurt_excess = 0.0

        zeros = int(np.sum(values == 0.0))
        zero_pct = 100.0 * zeros / n if n > 0 else 0.0

        def _fmt(x: float | None) -> str:
            if x is None or not np.isfinite(float(x)):
                return "—"
            suffix = f" {units}" if units else ""
            return f"{float(x):.4g}{suffix}"

        _set("Samples", f"{n}")
        _set("Min", _fmt(vmin))
        _set("Max", _fmt(vmax))
        _set("Mean", _fmt(mean))
        _set("Median", _fmt(median))
        _set("Mode", _fmt(mode_val))
        _set("Std Dev", _fmt(std))
        _set("Variance", _fmt(var))
        _set("Q1 (25%)", _fmt(float(q1)))
        _set("Q3 (75%)", _fmt(float(q3)))
        _set("IQR", _fmt(iqr))
        _set("P5", _fmt(float(p5)))
        _set("P95", _fmt(float(p95)))
        _set("Skewness", f"{skew:.4g}")
        _set("Kurtosis (excess)", f"{kurt_excess:.4g}")
        _set("Zero %", f"{zero_pct:.2f}%")

    def _render_volume_histogram(
        self,
        *,
        plot_widget,
        samples: np.ndarray | None,
        manual_width: float | None,
        bin_label,
        empty_title: str,
        full_title: str,
        x_label: str,
        x_units: str,
        y_label: str,
    ) -> None:
        """Generic renderer for histogram bar charts."""
        if plot_widget is None:
            return
        plot_widget.clear()
        plot_widget.setLabel("bottom", x_label, units=x_units)
        plot_widget.setLabel("left", y_label, units="count")
        if samples is None or samples.size == 0:
            plot_widget.setTitle(empty_title)
            if bin_label is not None:
                bin_label.setText("Bin width: —")
            return
        values = np.asarray(samples, dtype=float)
        values = values[np.isfinite(values)]
        values = values[values >= 0.0]
        if values.size == 0:
            plot_widget.setTitle(empty_title)
            if bin_label is not None:
                bin_label.setText("Bin width: —")
            return
        bin_width = (
            manual_width if manual_width and manual_width > 0 else self._suggest_bin_width(values)
        )
        bin_width = max(bin_width, 0.01)
        max_value = float(np.max(values))
        stop = max_value + bin_width
        bins = np.arange(0.0, stop + bin_width * 0.5, bin_width)
        if bins.size < 2:
            bins = np.array([0.0, bin_width])
        counts, bin_edges = np.histogram(values, bins=bins)
        centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        widths = (bin_edges[1:] - bin_edges[:-1]) * 0.9
        if centers.size:
            bar_item = pg.BarGraphItem(
                x=centers,
                height=counts,
                width=widths,
                brush=pg.mkBrush("#87CEFA"),
                pen=pg.mkPen("#1E90FF", width=1),
            )
            plot_widget.addItem(bar_item)
        plot_widget.setTitle(full_title)
        x_max = max(max_value * 1.1, bin_width)
        y_max = max(int(np.max(counts)) * 1.1 if counts.size else 0.0, 1.0)
        plot_widget.setXRange(0.0, x_max, padding=0.0)
        plot_widget.setYRange(0.0, y_max, padding=0.0)
        if bin_label is not None:
            suffix = f" {x_units}" if x_units else ""
            bin_label.setText(f"Bin width: {bin_width:.4g}{suffix}")

    def _configure_distribution_bin_width(self) -> None:
        """Prompt the user to set/reset the bin width for the selected distribution."""
        distribution_id = self._get_selected_distribution_id()
        samples = getattr(self, "_last_distribution_samples", None)
        units = str(getattr(self, "_last_distribution_units", "") or "")
        if samples is None:
            QMessageBox.information(
                self,
                "No data",
                "Run an analysis to compute distributions before adjusting bin widths.",
            )
            return
        sample_array = np.asarray(samples, dtype=float)
        sample_array = sample_array[np.isfinite(sample_array)]
        sample_array = sample_array[sample_array >= 0.0]
        if sample_array.size == 0:
            QMessageBox.information(
                self,
                "No data",
                "Run an analysis to compute distributions before adjusting bin widths.",
            )
            return
        widths = getattr(self, "_distribution_bin_widths", None)
        current = widths.get(distribution_id) if isinstance(widths, dict) else None
        suggested = (
            float(current)
            if current is not None and current > 0
            else float(self._suggest_bin_width(sample_array))
        )
        dialog = QDialog(self)
        dialog.setWindowTitle("Configure Histogram Bin Width")
        layout = QVBoxLayout(dialog)
        unit_suffix = f" ({units})" if units else ""
        layout.addWidget(QLabel(f"Set histogram bin width{unit_suffix}."))
        spin = QDoubleSpinBox()
        spin.setDecimals(6)
        upper = max(1000.0, float(np.max(sample_array)) * 2.0 + suggested)
        spin.setRange(1e-6, upper)
        spin.setSingleStep(max(suggested / 10.0, 0.01))
        spin.setValue(max(suggested, 1e-6))
        layout.addWidget(spin)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        auto_button = buttons.addButton("Auto", QDialogButtonBox.ButtonRole.ResetRole)  # type: ignore[arg-type]
        layout.addWidget(buttons)

        def _apply(value: float | None) -> None:
            widths_map = getattr(self, "_distribution_bin_widths", None)
            if not isinstance(widths_map, dict):
                self._distribution_bin_widths = {}
                widths_map = self._distribution_bin_widths
            widths_map[distribution_id] = value
            self._update_data_volume_plots()

        def _save() -> None:
            _apply(max(float(spin.value()), 1e-6))
            dialog.accept()

        def _auto() -> None:
            _apply(None)
            dialog.accept()

        buttons.accepted.connect(_save)  # type: ignore[arg-type]
        buttons.rejected.connect(dialog.reject)  # type: ignore[arg-type]
        auto_button.clicked.connect(_auto)
        dialog.exec()

    def _suggest_bin_width(self, values: np.ndarray, target_bins: int = 15) -> float:
        """Return a 'nice' bin width for the provided data vector."""
        finite_vals = values[np.isfinite(values)]
        if finite_vals.size == 0:
            return 1.0
        max_value = float(np.max(finite_vals))
        min_value = float(np.min(finite_vals))
        data_range = max_value - min_value
        if data_range < 1e-6:
            return max(0.1, max_value * 0.1 or 0.1)
        raw_width = data_range / max(target_bins, 1)
        magnitude = 10 ** np.floor(np.log10(raw_width))
        normalized = raw_width / magnitude
        if normalized < 1.5:
            return float(magnitude)
        if normalized < 3.5:
            return float(2 * magnitude)
        if normalized < 7.5:
            return float(5 * magnitude)
        return float(10 * magnitude)

    def _compute_pass_downlink_volumes(self) -> list[float] | None:
        """Integrate data volume for every pass in the latest analysis."""
        if (
            not getattr(self, "_last_result", None)
            or not self._latest_access_series
            or not getattr(self, "_latest_station_rate_series", None)
            or not getattr(self, "_current_config", None)
        ):
            return None
        passes = getattr(self._last_result, "passes", None)
        if not passes:
            return None
        time_seconds = np.asarray(
            self._latest_access_series.get("time_seconds", []), dtype=float
        )
        if time_seconds.size < 2:
            return None
        station_rate_lookup: dict[str, np.ndarray] = (
            getattr(self, "_latest_station_rate_series", None) or {}
        )
        if not station_rate_lookup:
            return None
        combined_rates = getattr(self, "_latest_combined_rate_series", None)
        if combined_rates is None and station_rate_lookup:
            for value in station_rate_lookup.values():
                if value.size != time_seconds.size:
                    continue
                combined_rates = (
                    value if combined_rates is None else combined_rates + value
                )
        start_time = getattr(self._current_config.scenario, "start_time", None)
        if start_time is None:
            return None
        start_axis = float(time_seconds[0])
        end_axis = float(time_seconds[-1])
        volumes: list[float] = []
        for entry in passes:
            station_name = getattr(entry, "station_name", None)
            rates_array = None
            if station_name and station_name in station_rate_lookup:
                rates_array = station_rate_lookup[station_name]
            elif combined_rates is not None:
                rates_array = combined_rates
            if rates_array is None:
                continue
            if rates_array.size != time_seconds.size:
                continue
            aos_seconds = (entry.aos - start_time).total_seconds()
            los_seconds = (entry.los - start_time).total_seconds()
            pass_start = max(start_axis, float(aos_seconds))
            pass_end = min(end_axis, float(los_seconds))
            if pass_end <= pass_start:
                continue
            volume_gb = self._integrate_data_volume_interval(
                time_seconds, rates_array, pass_start, pass_end
            )
            if volume_gb <= 0:
                continue
            volumes.append(volume_gb * GIBIT_PER_GBIT)
        return volumes or None

    def _integrate_data_volume_interval(
        self,
        time_axis: np.ndarray,
        data_rates: np.ndarray,
        start_sec: float,
        end_sec: float,
    ) -> float:
        """Integrate Mbps samples over the requested interval and return Gigabits."""
        if end_sec <= start_sec:
            return 0.0
        mask = (time_axis > start_sec) & (time_axis < end_sec)
        interval_times = time_axis[mask]
        interval_rates = data_rates[mask]
        start_rate = float(np.interp(start_sec, time_axis, data_rates, left=0.0, right=0.0))
        end_rate = float(np.interp(end_sec, time_axis, data_rates, left=0.0, right=0.0))
        times = np.concatenate(([start_sec], interval_times, [end_sec]))
        rates = np.concatenate(([start_rate], interval_rates, [end_rate]))
        return link_budget_math.integrate_data_volume_gb(times, rates)

