"""Link budget tab mixin."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from cosmic_toolbox import link_budget_math
from cosmic_toolbox.itu_losses import estimate_slant_path_loss
from cosmic_toolbox.link_budget_defaults import load_link_budget_defaults
from cosmic_toolbox.models import GroundStationConfig
from cosmic_toolbox.services import antenna_pattern

_LB_DEFAULTS = load_link_budget_defaults()

# Default elevation filtering bounds for the link-budget tool (and the derived
# data-volume calculations). These bounds are *only* applied in the link-budget
# pipeline; mission propagation/pass detection is intentionally ungated.
DEFAULT_LINK_BUDGET_ELEVATION_LOWER_DEG = _LB_DEFAULTS.min_elevation_deg
DEFAULT_LINK_BUDGET_ELEVATION_UPPER_DEG = 60.0
STATIC_MODE_VCM_INDEX = 0
STATIC_MODE_ECSS_QPSK_TM_INDEX = 1
STATIC_MODE_QPSK78_INDEX = 2
STATIC_MODE_FIXED_MODCOD_INDEX = 3
DYNAMIC_MODE_VCM_INDEX = 0
DYNAMIC_MODE_CCM_INDEX = 1
DYNAMIC_MODE_ECSS_QPSK_TM_INDEX = 2
DYNAMIC_MODE_QPSK78_INDEX = 3
DYNAMIC_MODE_FIXED_MODCOD_INDEX = 4


class LinkBudgetTabMixin:
    """Encapsulates link budget UI construction and logic."""

    _BIT_DISPLAY_UNITS: tuple[tuple[str, float], ...] = (
        ("bit", 1.0),
        ("kbit", 1e3),
        ("Mbit", 1e6),
        ("Gbit", 1e9),
        ("Tbit", 1e12),
    )

    def _select_data_unit(self, bits_value: float) -> tuple[str, float]:
        """Return the most appropriate decimal bit unit for a value in bits."""
        if not np.isfinite(bits_value):
            raise ValueError("Bit-unit selection requires a finite value.")
        magnitude = abs(float(bits_value))
        unit_name = self._BIT_DISPLAY_UNITS[0][0]
        unit_scale = self._BIT_DISPLAY_UNITS[0][1]
        for candidate_name, candidate_scale in self._BIT_DISPLAY_UNITS:
            if magnitude >= candidate_scale:
                unit_name = candidate_name
                unit_scale = candidate_scale
        return (unit_name, unit_scale)

    def _format_data_quantity(self, value_gbit: float, *, suffix: str = "") -> str:
        """Format a data quantity stored in Gbit using decimal bit units."""
        if not np.isfinite(value_gbit):
            raise ValueError("Data quantity must be finite.")
        unit_name, unit_scale = self._select_data_unit(float(value_gbit) * 1e9)
        scaled = float(value_gbit) * 1e9 / unit_scale
        unit_text = f"{unit_name}{suffix}"
        return f"{scaled:.4g} {unit_text}"

    def _format_rate_quantity(self, value_mbps: float) -> str:
        """Format a rate stored in Mbps using decimal bit/s units."""
        if not np.isfinite(value_mbps):
            raise ValueError("Rate must be finite.")
        unit_name, unit_scale = self._select_data_unit(float(value_mbps) * 1e6)
        scaled = float(value_mbps) * 1e6 / unit_scale
        return f"{scaled:.4g} {unit_name}/s"

    def _build_static_link_budget_tab(self) -> QWidget:
        """Assemble the standalone static link-budget tool."""
        tab = QWidget()
        layout = QHBoxLayout(tab)

        # ── Common Inputs ─────────────────────────────────────────────────
        common_group = QGroupBox("Common Inputs")
        common_form = QFormLayout(common_group)
        self.static_link_budget_station_combo = QComboBox()
        common_form.addRow("Ground station:", self.static_link_budget_station_combo)
        self.slb_sat_altitude_input = QDoubleSpinBox()
        self.slb_sat_altitude_input.setRange(150.0, 50000.0)
        self.slb_sat_altitude_input.setValue(_LB_DEFAULTS.satellite_altitude_km)
        self.slb_sat_altitude_input.setSuffix(" km")
        common_form.addRow("Satellite altitude:", self.slb_sat_altitude_input)
        self.slb_gs_elevation_input = QDoubleSpinBox()
        self.slb_gs_elevation_input.setRange(0.0, 90.0)
        self.slb_gs_elevation_input.setValue(30.0)
        self.slb_gs_elevation_input.setSuffix(" °")
        common_form.addRow("Elevation:", self.slb_gs_elevation_input)
        self.slb_common_antenna_gain_input = QDoubleSpinBox()
        self.slb_common_antenna_gain_input.setRange(-10.0, 80.0)
        self.slb_common_antenna_gain_input.setDecimals(1)
        self.slb_common_antenna_gain_input.setValue(6.5)
        self.slb_common_antenna_gain_input.setSuffix(" dBi")
        common_form.addRow("Common antenna gain:", self.slb_common_antenna_gain_input)
        self.slb_unavailability_input = QDoubleSpinBox()
        self.slb_unavailability_input.setRange(0.01, 5.0)
        self.slb_unavailability_input.setValue(_LB_DEFAULTS.unavailability_percent)
        self.slb_unavailability_input.setDecimals(2)
        self.slb_unavailability_input.setSuffix(" %")
        common_form.addRow("Unavailability:", self.slb_unavailability_input)
        self.slb_margin_input = QDoubleSpinBox()
        self.slb_margin_input.setRange(0.0, 20.0)
        self.slb_margin_input.setValue(_LB_DEFAULTS.margin_dB)
        self.slb_margin_input.setSuffix(" dB")
        common_form.addRow("Link margin target:", self.slb_margin_input)
        self.slb_polarization_loss_input = QDoubleSpinBox()
        self.slb_polarization_loss_input.setRange(0.0, 5.0)
        self.slb_polarization_loss_input.setDecimals(2)
        self.slb_polarization_loss_input.setValue(_LB_DEFAULTS.polarization_loss_dB)
        self.slb_polarization_loss_input.setSuffix(" dB")
        common_form.addRow("Polarization loss:", self.slb_polarization_loss_input)
        self.slb_rolloff_input = QDoubleSpinBox()
        self.slb_rolloff_input.setRange(0.05, 1.0)
        self.slb_rolloff_input.setDecimals(2)
        self.slb_rolloff_input.setValue(_LB_DEFAULTS.rolloff)
        common_form.addRow("Roll-off factor:", self.slb_rolloff_input)

        # ── Uplink Inputs (Ground → Spacecraft) ───────────────────────────
        uplink_group = QGroupBox("Uplink Inputs (Ground → Spacecraft)")
        ul_form = QFormLayout(uplink_group)
        self.slb_ul_frequency_input = QDoubleSpinBox()
        self.slb_ul_frequency_input.setRange(1.0, 99000.0)
        self.slb_ul_frequency_input.setDecimals(1)
        self.slb_ul_frequency_input.setValue(_LB_DEFAULTS.uplink_frequency_MHz)
        self.slb_ul_frequency_input.setSuffix(" MHz")
        ul_form.addRow("UL frequency:", self.slb_ul_frequency_input)
        self.slb_ul_gs_eirp_input = QDoubleSpinBox()
        self.slb_ul_gs_eirp_input.setRange(-30.0, 100.0)
        self.slb_ul_gs_eirp_input.setDecimals(1)
        self.slb_ul_gs_eirp_input.setValue(45.0)
        self.slb_ul_gs_eirp_input.setSuffix(" dBW")
        ul_form.addRow("Ground station EIRP:", self.slb_ul_gs_eirp_input)
        self.slb_ul_symbol_rate_input = QDoubleSpinBox()
        self.slb_ul_symbol_rate_input.setRange(0.001, 5000.0)
        self.slb_ul_symbol_rate_input.setDecimals(3)
        self.slb_ul_symbol_rate_input.setValue(0.5)
        self.slb_ul_symbol_rate_input.setSuffix(" Msps")
        ul_form.addRow("Symbol rate:", self.slb_ul_symbol_rate_input)
        self.slb_ul_rx_nf_input = QDoubleSpinBox()
        self.slb_ul_rx_nf_input.setRange(0.0, 30.0)
        self.slb_ul_rx_nf_input.setDecimals(1)
        self.slb_ul_rx_nf_input.setValue(6.5)
        self.slb_ul_rx_nf_input.setSuffix(" dB")
        ul_form.addRow("SC receiver NF:", self.slb_ul_rx_nf_input)
        self.slb_ul_sky_bg_temp_input = QDoubleSpinBox()
        self.slb_ul_sky_bg_temp_input.setRange(3.0, 500.0)
        self.slb_ul_sky_bg_temp_input.setDecimals(0)
        self.slb_ul_sky_bg_temp_input.setValue(255.0)
        self.slb_ul_sky_bg_temp_input.setSuffix(" K")
        ul_form.addRow("SC sky background temp.:", self.slb_ul_sky_bg_temp_input)
        self.slb_ul_rx_losses_input = QDoubleSpinBox()
        self.slb_ul_rx_losses_input.setRange(0.0, 20.0)
        self.slb_ul_rx_losses_input.setDecimals(2)
        self.slb_ul_rx_losses_input.setValue(0.0)
        self.slb_ul_rx_losses_input.setSuffix(" dB")
        ul_form.addRow("SC receiver losses:", self.slb_ul_rx_losses_input)
        self.slb_ul_impl_loss_input = QDoubleSpinBox()
        self.slb_ul_impl_loss_input.setRange(0.0, 10.0)
        self.slb_ul_impl_loss_input.setDecimals(1)
        self.slb_ul_impl_loss_input.setValue(1.5)
        self.slb_ul_impl_loss_input.setSuffix(" dB")
        ul_form.addRow("Implementation loss:", self.slb_ul_impl_loss_input)
        _ul_mode_label = QLabel("ECSS QPSK for TM")
        _ul_mode_label.setStyleSheet("color: grey;")
        ul_form.addRow("Operating mode:", _ul_mode_label)

        # ── Downlink Inputs (Spacecraft → Ground) ─────────────────────────
        downlink_group = QGroupBox("Downlink Inputs (Spacecraft → Ground)")
        dl_form = QFormLayout(downlink_group)
        self.slb_dl_frequency_input = QDoubleSpinBox()
        self.slb_dl_frequency_input.setRange(1.0, 99000.0)
        self.slb_dl_frequency_input.setDecimals(1)
        self.slb_dl_frequency_input.setValue(_LB_DEFAULTS.frequency_MHz)
        self.slb_dl_frequency_input.setSuffix(" MHz")
        dl_form.addRow("DL frequency:", self.slb_dl_frequency_input)
        self.slb_tx_power_input = QDoubleSpinBox()
        self.slb_tx_power_input.setRange(-50.0, 50.0)
        self.slb_tx_power_input.setValue(_LB_DEFAULTS.tx_power_dBw)
        self.slb_tx_power_input.setSuffix(" dBW")
        dl_form.addRow("TX power:", self.slb_tx_power_input)
        self.slb_tx_losses_input = QDoubleSpinBox()
        self.slb_tx_losses_input.setRange(0.0, 20.0)
        self.slb_tx_losses_input.setDecimals(2)
        self.slb_tx_losses_input.setValue(_LB_DEFAULTS.tx_losses_dB)
        self.slb_tx_losses_input.setSuffix(" dB")
        dl_form.addRow("TX feeder loss:", self.slb_tx_losses_input)
        self.slb_tx_backoff_input = QDoubleSpinBox()
        self.slb_tx_backoff_input.setRange(0.0, 30.0)
        self.slb_tx_backoff_input.setValue(0.0)
        self.slb_tx_backoff_input.setDecimals(2)
        self.slb_tx_backoff_input.setSuffix(" dB")
        self.slb_pfd_compliance_checkbox = QCheckBox("Back-off for PFD Compliance")
        self.slb_pfd_compliance_checkbox.setChecked(False)
        self.slb_pfd_compliance_checkbox.toggled.connect(
            lambda checked: self.slb_tx_backoff_input.setEnabled(not checked)
        )
        _backoff_row = QWidget()
        _backoff_layout = QHBoxLayout(_backoff_row)
        _backoff_layout.setContentsMargins(0, 0, 0, 0)
        _backoff_layout.setSpacing(8)
        _backoff_layout.addWidget(self.slb_tx_backoff_input)
        _backoff_layout.addWidget(self.slb_pfd_compliance_checkbox)
        dl_form.addRow("TX backoff:", _backoff_row)
        self.slb_mispointing_loss_input = QDoubleSpinBox()
        self.slb_mispointing_loss_input.setRange(0.0, 30.0)
        self.slb_mispointing_loss_input.setDecimals(2)
        self.slb_mispointing_loss_input.setValue(0.0)
        self.slb_mispointing_loss_input.setSuffix(" dB")
        dl_form.addRow("Mispointing loss:", self.slb_mispointing_loss_input)
        self.slb_rx_antenna_gain_input = QDoubleSpinBox()
        self.slb_rx_antenna_gain_input.setRange(-10.0, 80.0)
        self.slb_rx_antenna_gain_input.setValue(_LB_DEFAULTS.rx_antenna_gain_dBi)
        self.slb_rx_antenna_gain_input.setSuffix(" dBi")
        dl_form.addRow("GS RX antenna gain:", self.slb_rx_antenna_gain_input)
        self.slb_rx_nf_input = QDoubleSpinBox()
        self.slb_rx_nf_input.setRange(0.0, 30.0)
        self.slb_rx_nf_input.setDecimals(1)
        self.slb_rx_nf_input.setValue(_LB_DEFAULTS.receiver_noise_figure_dB)
        self.slb_rx_nf_input.setSuffix(" dB")
        dl_form.addRow("GS receiver NF:", self.slb_rx_nf_input)
        self.slb_sky_bg_temp_input = QDoubleSpinBox()
        self.slb_sky_bg_temp_input.setRange(3.0, 500.0)
        self.slb_sky_bg_temp_input.setDecimals(0)
        self.slb_sky_bg_temp_input.setValue(_LB_DEFAULTS.sky_background_temperature_K)
        self.slb_sky_bg_temp_input.setSuffix(" K")
        dl_form.addRow("GS sky background temp.:", self.slb_sky_bg_temp_input)
        self.slb_rx_losses_input = QDoubleSpinBox()
        self.slb_rx_losses_input.setRange(0.0, 20.0)
        self.slb_rx_losses_input.setValue(_LB_DEFAULTS.receiver_losses_dB)
        self.slb_rx_losses_input.setSuffix(" dB")
        dl_form.addRow("GS receiver losses:", self.slb_rx_losses_input)
        self.slb_symbol_rate_input = QDoubleSpinBox()
        self.slb_symbol_rate_input.setRange(0.001, 5000.0)
        self.slb_symbol_rate_input.setValue(_LB_DEFAULTS.symbol_rate_Msps)
        self.slb_symbol_rate_input.setDecimals(3)
        self.slb_symbol_rate_input.setSuffix(" Msps")
        dl_form.addRow("Symbol rate:", self.slb_symbol_rate_input)
        self.slb_impl_loss_input = QDoubleSpinBox()
        self.slb_impl_loss_input.setRange(0.0, 10.0)
        self.slb_impl_loss_input.setDecimals(1)
        self.slb_impl_loss_input.setValue(_LB_DEFAULTS.implementation_loss_dB)
        self.slb_impl_loss_input.setSuffix(" dB")
        dl_form.addRow("Implementation loss:", self.slb_impl_loss_input)
        self.static_link_budget_mode_combo = QComboBox()
        self.static_link_budget_mode_combo.addItems(
            [
                "VCM (default)",
                link_budget_math.QPSK_VITERBI_MODE_NAME,
                link_budget_math.QPSK_78_MODE_NAME,
                "Fixed MODCOD",
            ]
        )
        self.static_link_budget_mode_combo.setCurrentIndex(STATIC_MODE_ECSS_QPSK_TM_INDEX)
        self.static_link_budget_mode_combo.currentIndexChanged.connect(
            self._on_static_link_budget_mode_changed
        )
        dl_form.addRow("Operating mode:", self.static_link_budget_mode_combo)
        self.static_link_budget_fixed_modcod_combo = QComboBox()
        for modcod_name, _bits, _thresh in link_budget_math.MODCOD_TABLE:
            self.static_link_budget_fixed_modcod_combo.addItem(modcod_name)
        self.static_link_budget_fixed_modcod_combo.setCurrentIndex(
            max(0, self.static_link_budget_fixed_modcod_combo.findText(_LB_DEFAULTS.fixed_modcod_name))
        )
        self.static_link_budget_fixed_modcod_combo.setEnabled(False)
        dl_form.addRow("Fixed MODCOD:", self.static_link_budget_fixed_modcod_combo)

        self.static_link_budget_calculate_button = QPushButton("Calculate")
        self.static_link_budget_calculate_button.clicked.connect(
            self._handle_static_link_budget_calculate
        )
        self.static_link_budget_export_button = QPushButton("Export Link Budget to XLSX")
        self.static_link_budget_export_button.setToolTip(
            "Write the uplink and downlink link budget to a styled .xlsx workbook."
        )
        self.static_link_budget_export_button.clicked.connect(
            self._handle_export_static_link_budget_xlsx
        )

        # ── Left panel (scrollable inputs) ────────────────────────────────
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.addWidget(common_group)
        left_layout.addWidget(uplink_group)
        left_layout.addWidget(downlink_group)
        left_layout.addWidget(self.static_link_budget_calculate_button)
        left_layout.addWidget(self.static_link_budget_export_button)
        left_layout.addStretch(1)
        left_widget.setMinimumWidth(0)
        left_widget.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_scroll.setWidget(left_widget)
        left_scroll.setMinimumWidth(0)
        left_scroll.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding
        )

        # ── Right panel (UL + DL tables + summary) ────────────────────────
        def _make_budget_table() -> QTableWidget:
            t = QTableWidget(0, 4)
            t.setHorizontalHeaderLabels(["Parameter", "Input", "Calcs", "Unit"])
            t.verticalHeader().setVisible(False)
            t.setAlternatingRowColors(True)
            t.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
            t.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
            t.setWordWrap(False)
            t.setMinimumWidth(0)
            t.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
            hdr = t.horizontalHeader()
            hdr.setDefaultAlignment(Qt.AlignmentFlag.AlignCenter)
            hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
            hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
            hdr.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
            hdr.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
            return t

        self.static_link_budget_uplink_table = _make_budget_table()
        self.static_link_budget_table = _make_budget_table()

        _bold_font = QFont()
        _bold_font.setBold(True)
        ul_table_label = QLabel("Uplink")
        ul_table_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ul_table_label.setFont(_bold_font)
        dl_table_label = QLabel("Downlink")
        dl_table_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        dl_table_label.setFont(_bold_font)

        ul_pane = QWidget()
        ul_pane.setMinimumWidth(0)
        ul_pane.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        ul_pane_layout = QVBoxLayout(ul_pane)
        ul_pane_layout.setContentsMargins(0, 0, 4, 0)
        ul_pane_layout.addWidget(ul_table_label)
        ul_pane_layout.addWidget(self.static_link_budget_uplink_table)

        dl_pane = QWidget()
        dl_pane.setMinimumWidth(0)
        dl_pane.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        dl_pane_layout = QVBoxLayout(dl_pane)
        dl_pane_layout.setContentsMargins(4, 0, 0, 0)
        dl_pane_layout.addWidget(dl_table_label)
        dl_pane_layout.addWidget(self.static_link_budget_table)

        tables_splitter = QSplitter(Qt.Orientation.Horizontal)
        tables_splitter.setChildrenCollapsible(True)
        tables_splitter.addWidget(ul_pane)
        tables_splitter.addWidget(dl_pane)
        tables_splitter.setStretchFactor(0, 1)
        tables_splitter.setStretchFactor(1, 1)

        self.static_link_budget_summary_label = QLabel(
            "Configure the inputs and press Calculate."
        )
        self.static_link_budget_summary_label.setWordWrap(True)
        self.static_link_budget_plot = None
        self._static_link_budget_plot_legend = None
        self._static_link_budget_plot_annotations = []

        right_widget = QWidget()
        right_widget.setMinimumWidth(0)
        right_widget.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding
        )
        right_layout = QVBoxLayout(right_widget)
        right_layout.addWidget(tables_splitter, stretch=1)
        right_layout.addWidget(self.static_link_budget_summary_label)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(True)
        splitter.addWidget(left_scroll)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([360, 900])
        layout.addWidget(splitter)

        self._latest_static_link_budget_plot_data: tuple[np.ndarray, dict] | None = None
        self._static_loss_cache_key: tuple | None = None
        self._static_loss_cache_losses: np.ndarray | None = None
        self._static_loss_cache_contributions: dict | None = None
        self._static_ul_loss_cache_key: tuple | None = None
        self._static_ul_loss_cache_losses: np.ndarray | None = None
        self._static_ul_loss_cache_contributions: dict | None = None
        self._refresh_link_budget_station_list()
        return tab

    def _build_link_budget_tab(self) -> QWidget:
        """Assemble the dynamic link-budget tool."""
        self.link_budget_table = None
        self.link_budget_summary_label = None
        self.link_budget_plot = None
        self.link_budget_tabs = None
        self._latest_link_budget_plot_data = None
        self._link_budget_plot_legend = None
        self._link_budget_plot_annotations = []
        tab = QWidget()
        layout = QHBoxLayout(tab)
        input_group = QGroupBox("Inputs")
        form = QFormLayout(input_group)
        self.link_budget_station_combo = QComboBox()
        form.addRow("Ground station:", self.link_budget_station_combo)
        self.lb_frequency_input = QDoubleSpinBox()
        self.lb_frequency_input.setRange(0.1, 100.0)
        self.lb_frequency_input.setDecimals(3)
        self.lb_frequency_input.setValue(_LB_DEFAULTS.frequency_GHz)
        self.lb_frequency_input.setSuffix(" GHz")
        form.addRow("Frequency:", self.lb_frequency_input)
        self.lb_tx_power_input = QDoubleSpinBox()
        self.lb_tx_power_input.setRange(-50.0, 50.0)
        self.lb_tx_power_input.setValue(_LB_DEFAULTS.tx_power_dBw)
        self.lb_tx_power_input.setSuffix(" dBW")
        form.addRow("TX power:", self.lb_tx_power_input)
        self.lb_tx_gain_input = QDoubleSpinBox()
        self.lb_tx_gain_input.setRange(-10.0, 80.0)
        self.lb_tx_gain_input.setValue(_LB_DEFAULTS.tx_boresight_gain_dBi)
        self.lb_tx_gain_input.setSuffix(" dBi")
        self.lb_tx_losses_input = QDoubleSpinBox()
        self.lb_tx_losses_input.setRange(0.0, 20.0)
        self.lb_tx_losses_input.setValue(_LB_DEFAULTS.tx_losses_dB)
        self.lb_tx_losses_input.setSuffix(" dB")
        form.addRow("TX feeder loss:", self.lb_tx_losses_input)
        self.lb_tx_backoff_input = QDoubleSpinBox()
        self.lb_tx_backoff_input.setRange(0.0, 10.0)
        self.lb_tx_backoff_input.setValue(0.0)
        self.lb_tx_backoff_input.setSuffix(" dB")
        form.addRow("TX backoff:", self.lb_tx_backoff_input)
        self.lb_antenna_gain_input = QDoubleSpinBox()
        self.lb_antenna_gain_input.setRange(-10.0, 80.0)
        self.lb_antenna_gain_input.setValue(_LB_DEFAULTS.tx_boresight_gain_dBi)
        self.lb_antenna_gain_input.setSuffix(" dBi")
        self.lb_antenna_mode_combo = QComboBox()
        self.lb_antenna_mode_combo.addItems(
            ["Fixed gain", "LUT-driven (+Z boresight)"]
        )
        self.lb_antenna_mode_combo.setCurrentIndex(1)
        self.lb_antenna_mode_combo.currentIndexChanged.connect(
            self._on_antenna_mode_changed
        )
        default_lut_path = antenna_pattern.default_synthesized_lut_path()
        self._antenna_lut_path = (
            str(default_lut_path) if default_lut_path.exists() else ""
        )
        self.lb_antenna_lut_path_label = QLabel()
        self.lb_antenna_lut_path_label.setWordWrap(True)
        form.addRow("Antenna LUT:", self.lb_antenna_lut_path_label)
        self.lb_antenna_lut_browse_button = QPushButton("Select LUT...")
        self.lb_antenna_lut_browse_button.clicked.connect(
            self._select_antenna_lut_path
        )
        form.addRow("", self.lb_antenna_lut_browse_button)
        self.lb_antenna_mount_label = QLabel("Mounting: body +Z boresight (nadir face)")
        self.lb_antenna_mount_label.setWordWrap(True)
        form.addRow("", self.lb_antenna_mount_label)
        self.lb_antenna_peak_gain_label = QLabel()
        self.lb_antenna_peak_gain_label.setWordWrap(True)
        form.addRow("Derived boresight gain:", self.lb_antenna_peak_gain_label)
        self.lb_gs_gt_input = QDoubleSpinBox()
        self.lb_gs_gt_input.setRange(-30.0, 30.0)
        self.lb_gs_gt_input.setDecimals(1)
        self.lb_gs_gt_input.setValue(_LB_DEFAULTS.gs_gt_dBK)
        self.lb_gs_gt_input.setSuffix(" dB/K")
        form.addRow("Ground station G/T:", self.lb_gs_gt_input)
        self.lb_rx_losses_input = QDoubleSpinBox()
        self.lb_rx_losses_input.setRange(0.0, 20.0)
        self.lb_rx_losses_input.setValue(_LB_DEFAULTS.receiver_losses_dB)
        self.lb_rx_losses_input.setSuffix(" dB")
        form.addRow("Receiver losses:", self.lb_rx_losses_input)
        self.lb_symbol_rate_input = QDoubleSpinBox()
        self.lb_symbol_rate_input.setRange(0.01, 5000.0)
        self.lb_symbol_rate_input.setValue(_LB_DEFAULTS.symbol_rate_Msps)
        self.lb_symbol_rate_input.setDecimals(3)
        self.lb_symbol_rate_input.setSuffix(" Msps")
        form.addRow("Symbol rate limit:", self.lb_symbol_rate_input)
        self.lb_impl_loss_input = QDoubleSpinBox()
        self.lb_impl_loss_input.setRange(0.0, 10.0)
        self.lb_impl_loss_input.setValue(_LB_DEFAULTS.implementation_loss_dB)
        self.lb_impl_loss_input.setSuffix(" dB")
        form.addRow("Implementation loss:", self.lb_impl_loss_input)
        self.lb_margin_input = QDoubleSpinBox()
        self.lb_margin_input.setRange(0.0, 20.0)
        self.lb_margin_input.setValue(_LB_DEFAULTS.margin_dB)
        self.lb_margin_input.setSuffix(" dB")
        form.addRow("Link margin target:", self.lb_margin_input)
        self.lb_sat_altitude_input = QDoubleSpinBox()
        self.lb_sat_altitude_input.setRange(150.0, 50000.0)
        self.lb_sat_altitude_input.setValue(_LB_DEFAULTS.satellite_altitude_km)
        self.lb_sat_altitude_input.setSuffix(" km")

        # Elevation filtering bounds used by the link-budget tool + data-volume calculations.
        self.lb_elev_lower_input = QDoubleSpinBox()
        self.lb_elev_lower_input.setRange(0.0, 90.0)
        self.lb_elev_lower_input.setDecimals(2)
        self.lb_elev_lower_input.setValue(float(DEFAULT_LINK_BUDGET_ELEVATION_LOWER_DEG))
        self.lb_elev_lower_input.setSuffix(" °")

        self.lb_elev_upper_input = QDoubleSpinBox()
        self.lb_elev_upper_input.setRange(0.0, 90.0)
        self.lb_elev_upper_input.setDecimals(2)
        self.lb_elev_upper_input.setValue(float(DEFAULT_LINK_BUDGET_ELEVATION_UPPER_DEG))
        self.lb_elev_upper_input.setSuffix(" °")
        self.lb_dynamic_filter_label = QLabel(
            "Dynamic filtering uses the Contact Statistics elevation slider and propagated orbit state."
        )
        self.lb_dynamic_filter_label.setWordWrap(True)
        form.addRow("", self.lb_dynamic_filter_label)

        # Operating mode: VCM (default), Dynamic CCM Optimal (per pass), or Fixed MODCOD.
        self.link_budget_mode_combo = QComboBox()
        self.link_budget_mode_combo.addItems(
            [
                "VCM (default)",
                "Dynamic CCM Optimal (per pass)",
                link_budget_math.QPSK_VITERBI_MODE_NAME,
                link_budget_math.QPSK_78_MODE_NAME,
                "Fixed MODCOD",
            ]
        )
        self.link_budget_mode_combo.setCurrentIndex(2)
        self.link_budget_mode_combo.currentIndexChanged.connect(
            self._on_link_budget_mode_changed
        )
        form.addRow("Operating mode:", self.link_budget_mode_combo)

        self.link_budget_fixed_modcod_combo = QComboBox()
        for modcod_name, _bits, _thresh in link_budget_math.MODCOD_TABLE:
            self.link_budget_fixed_modcod_combo.addItem(modcod_name)
        self.link_budget_fixed_modcod_combo.setCurrentIndex(
            max(0, self.link_budget_fixed_modcod_combo.findText(_LB_DEFAULTS.fixed_modcod_name))
        )
        self.link_budget_fixed_modcod_combo.setEnabled(False)
        form.addRow("Fixed MODCOD:", self.link_budget_fixed_modcod_combo)

        # CCM optimization: step and max offset (seconds from AOS/LOS).
        self.lb_ccm_offset_step_input = QDoubleSpinBox()
        self.lb_ccm_offset_step_input.setRange(1.0, 120.0)
        self.lb_ccm_offset_step_input.setValue(10.0)
        self.lb_ccm_offset_step_input.setDecimals(1)
        self.lb_ccm_offset_step_input.setSuffix(" s")
        form.addRow("CCM offset step:", self.lb_ccm_offset_step_input)
        self.lb_ccm_max_offset_input = QDoubleSpinBox()
        self.lb_ccm_max_offset_input.setRange(0.0, 600.0)
        self.lb_ccm_max_offset_input.setValue(300.0)
        self.lb_ccm_max_offset_input.setDecimals(0)
        self.lb_ccm_max_offset_input.setSuffix(" s")
        form.addRow("CCM max offset:", self.lb_ccm_max_offset_input)

        self.lb_unavailability_input = QDoubleSpinBox()
        self.lb_unavailability_input.setRange(0.01, 5.0)
        self.lb_unavailability_input.setValue(_LB_DEFAULTS.unavailability_percent)
        self.lb_unavailability_input.setDecimals(2)
        self.lb_unavailability_input.setSuffix(" %")
        form.addRow("Unavailability:", self.lb_unavailability_input)
        self.lb_polarization_loss_input = QDoubleSpinBox()
        self.lb_polarization_loss_input.setRange(0.0, 5.0)
        self.lb_polarization_loss_input.setDecimals(2)
        self.lb_polarization_loss_input.setValue(_LB_DEFAULTS.polarization_loss_dB)
        self.lb_polarization_loss_input.setSuffix(" dB")
        form.addRow("Polarization loss:", self.lb_polarization_loss_input)
        self.lb_rolloff_input = QDoubleSpinBox()
        self.lb_rolloff_input.setRange(0.05, 1.0)
        self.lb_rolloff_input.setDecimals(2)
        self.lb_rolloff_input.setValue(_LB_DEFAULTS.rolloff)
        form.addRow("Roll-off factor:", self.lb_rolloff_input)
        self.compute_link_budget_button = QPushButton("Compute Link Budget")
        self.compute_link_budget_button.clicked.connect(  # type: ignore[attr-defined]
            self._trigger_link_budget_recompute
        )
        form.addRow("", self.compute_link_budget_button)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.addWidget(input_group)
        left_layout.addWidget(self._build_downlink_summary_group())
        left_layout.addStretch(1)
        left_widget.setMinimumWidth(0)
        left_widget.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_scroll.setWidget(left_widget)
        left_scroll.setMinimumWidth(0)
        left_scroll.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding
        )
        right_widget = QWidget()
        right_widget.setMinimumWidth(0)
        right_widget.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding
        )
        right_layout = QVBoxLayout(right_widget)
        self.dynamic_link_budget_tabs = QTabWidget()
        self.dynamic_link_budget_tabs.setUsesScrollButtons(True)
        self.dynamic_link_budget_tabs.tabBar().setElideMode(Qt.TextElideMode.ElideRight)
        self.dynamic_link_budget_tabs.tabBar().setExpanding(False)
        right_layout.addWidget(self.dynamic_link_budget_tabs, stretch=1)
        plot_tab = QWidget()
        plot_layout = QVBoxLayout(plot_tab)
        self.dynamic_link_budget_plot = pg.PlotWidget(
            title="Dynamic Link Budget System Performance"
        )
        self.dynamic_link_budget_plot.setLabel("bottom", "Time", units="min")
        self.dynamic_link_budget_plot.setLabel("left", "Metric", units="dB")
        self.dynamic_link_budget_plot.showGrid(x=True, y=True, alpha=0.3)
        self._dynamic_link_budget_plot_legend = self.dynamic_link_budget_plot.addLegend(
            offset=(10, 10)
        )
        self._dynamic_link_budget_plot_annotations: list = []
        plot_layout.addWidget(self.dynamic_link_budget_plot)
        self.dynamic_link_budget_tabs.addTab(plot_tab, "Dynamic Link Budget")
        self.dynamic_link_budget_tabs.addTab(self._build_data_volume_tab(), "Data Volume")
        self._dynamic_link_budget_tab = plot_tab
        self._loss_cache_key: tuple | None = None
        self._loss_cache_losses: np.ndarray | None = None
        self._loss_cache_contributions: dict | None = None
        self.dynamic_link_budget_tabs.currentChanged.connect(
            self._on_link_budget_tab_changed
        )
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(True)
        splitter.addWidget(left_scroll)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([440, 920])
        layout.addWidget(splitter)
        self._register_link_budget_inputs()
        self._refresh_link_budget_station_list()
        self._link_budget_auto_enabled = True

        # Cache for ITU losses used by data-volume time-series evaluation.
        if not hasattr(self, "_timeseries_loss_cache"):
            self._timeseries_loss_cache: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
        if not hasattr(self, "_antenna_lut_cache"):
            self._antenna_lut_cache: dict[str, antenna_pattern.SphericalGainLut] = {}
        self._latest_dynamic_link_budget_plot_data: dict | None = None

        # Debounce expensive data-volume recomputation so sliders/spinboxes don't
        # freeze the GUI while the user is still editing values.
        if getattr(self, "_data_volume_refresh_timer", None) is None:
            self._data_volume_refresh_timer = QTimer(self)
            self._data_volume_refresh_timer.setSingleShot(True)
            self._data_volume_refresh_timer.setInterval(200)
            self._data_volume_refresh_timer.timeout.connect(  # type: ignore[attr-defined]
                self._perform_data_volume_refresh
            )

        self._sync_antenna_mode_ui()
        self._trigger_link_budget_recompute()
        return tab

    def _build_downlink_summary_group(self) -> QGroupBox:
        """Create the box summarizing downlink capacity."""
        group = QGroupBox("Downlink Summary")
        layout = QVBoxLayout(group)

        # Summary labels
        form_layout = QFormLayout()
        self._downlink_mode_label = QLabel("—")
        self._downlink_total_label = QLabel("—")
        self._downlink_per_orbit_label = QLabel("—")
        form_layout.addRow("Mode:", self._downlink_mode_label)
        form_layout.addRow("Scenario total:", self._downlink_total_label)
        form_layout.addRow("Per orbit:", self._downlink_per_orbit_label)
        layout.addLayout(form_layout)
        hint_label = QLabel(
            "Data Volume uses the full propagated timeline inside the Contact Statistics pass filter. "
            "The dynamic tool auto-loads the antenna LUT and derives boresight gain from its peak. "
            "The separate Static Link Budget Tool is a standalone calculator and does not use simulation outputs."
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
        left_widget.setMinimumWidth(0)
        left_widget.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding
        )
        stats_group.setMinimumWidth(0)
        stats_group.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        splitter.setChildrenCollapsible(True)

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

    def _refresh_link_budget_station_list(self, trigger_auto_recompute: bool = True) -> None:
        """Keep the link-budget station dropdowns in sync with the station list."""

        def _sync_combo(combo: QComboBox | None) -> None:
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

        _sync_combo(getattr(self, "link_budget_station_combo", None))
        _sync_combo(getattr(self, "static_link_budget_station_combo", None))
        if trigger_auto_recompute and getattr(self, "_link_budget_auto_enabled", False):
            self._trigger_link_budget_recompute()

    def _on_link_budget_mode_changed(self, index: int) -> None:
        """Enable fixed MODCOD combo only when mode is Fixed MODCOD."""
        if getattr(self, "link_budget_fixed_modcod_combo", None) is None:
            return
        self.link_budget_fixed_modcod_combo.setEnabled(
            index == DYNAMIC_MODE_FIXED_MODCOD_INDEX
        )

    def _on_static_link_budget_mode_changed(self, index: int) -> None:
        fixed = getattr(self, "static_link_budget_fixed_modcod_combo", None)
        if fixed is not None:
            fixed.setEnabled(index == STATIC_MODE_FIXED_MODCOD_INDEX)

    def _is_lut_antenna_mode_enabled(self) -> bool:
        return True

    def _sync_antenna_mode_ui(self) -> None:
        if getattr(self, "lb_antenna_lut_path_label", None) is not None:
            path_text = self._antenna_lut_path.strip()
            self.lb_antenna_lut_path_label.setText(path_text or "No LUT selected")
        lut = None
        peak_text = "Unavailable"
        try:
            lut = self._get_active_antenna_lut()
        except Exception:
            lut = None
        if lut is not None:
            peak_text = f"{float(np.max(lut.gain_dbi_grid)):.2f} dBi"
        if getattr(self, "lb_antenna_peak_gain_label", None) is not None:
            self.lb_antenna_peak_gain_label.setText(peak_text)

    def _on_antenna_mode_changed(self, _index: int) -> None:
        self._sync_antenna_mode_ui()

    def _select_antenna_lut_path(self) -> None:
        current_path = self._antenna_lut_path.strip() or str(
            antenna_pattern.default_synthesized_lut_path()
        )
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Select Antenna LUT",
            str(Path(current_path).parent),
            "NPZ files (*.npz)",
        )
        if not selected:
            return
        self._antenna_lut_path = str(Path(selected).expanduser().resolve())
        self._sync_antenna_mode_ui()

    def _get_active_antenna_lut(self) -> antenna_pattern.SphericalGainLut:
        default_path = antenna_pattern.default_synthesized_lut_path()
        path_text = self._antenna_lut_path.strip() or str(default_path)
        if not path_text:
            raise ValueError("Antenna LUT mode is enabled but no NPZ path is configured")
        lut_path = str(Path(path_text).expanduser().resolve())
        self._antenna_lut_path = lut_path
        cache = getattr(self, "_antenna_lut_cache", None)
        if not isinstance(cache, dict):
            cache = {}
            self._antenna_lut_cache = cache
        lut = cache.get(lut_path)
        if lut is None:
            lut = antenna_pattern.load_spherical_gain_lut(lut_path)
            cache[lut_path] = lut
        return lut

    def _get_link_budget_mode_label(self) -> str:
        """Return a short label for the current link budget operating mode."""
        mode_combo = getattr(self, "link_budget_mode_combo", None)
        if mode_combo is None:
            return "VCM"
        idx = mode_combo.currentIndex()
        if idx == DYNAMIC_MODE_VCM_INDEX:
            return "VCM"
        if idx == DYNAMIC_MODE_CCM_INDEX:
            return "Dynamic CCM Optimal"
        if idx == DYNAMIC_MODE_ECSS_QPSK_TM_INDEX:
            return link_budget_math.QPSK_VITERBI_MODE_NAME
        if idx == DYNAMIC_MODE_QPSK78_INDEX:
            return link_budget_math.QPSK_78_MODE_NAME
        if idx == DYNAMIC_MODE_FIXED_MODCOD_INDEX:
            fixed = getattr(self, "link_budget_fixed_modcod_combo", None)
            modcod = fixed.currentText().strip() if fixed else ""
            return f"Fixed MODCOD ({modcod})" if modcod else "Fixed MODCOD"
        return "VCM"

    def _register_link_budget_inputs(self) -> None:
        """Register dynamic link-budget controls.

        Auto-recomputation on value change is intentionally disabled — the user
        must press the Compute Link Budget button to apply changed inputs.
        """

    def _connect_link_budget_signal(self, widget, signal_name: str) -> None:
        """Attach the specified signal to trigger recomputation."""
        if widget is None:
            return
        signal = getattr(widget, signal_name, None)
        if signal is None:
            return
        signal.connect(self._trigger_link_budget_recompute)  # type: ignore[attr-defined]

    def _trigger_link_budget_recompute(self, *_args) -> None:
        """Recalculate dynamic link-budget outputs when inputs change."""
        if not self._link_budget_auto_enabled:
            return
        if not self._latest_access_series:
            self._latest_dynamic_link_budget_plot_data = None
            self._clear_dynamic_link_budget_plot()
        self._update_downlink_summary()

    def _get_selected_static_link_budget_station(self) -> GroundStationConfig | None:
        combo = getattr(self, "static_link_budget_station_combo", None)
        if combo is None or combo.count() == 0:
            return None
        data = combo.currentData()
        if isinstance(data, int) and 0 <= data < len(self._station_presets):
            return self._station_presets[data]
        return None

    def _get_static_link_budget_mode_label(self) -> str:
        combo = getattr(self, "static_link_budget_mode_combo", None)
        if combo is None:
            return "VCM"
        if combo.currentIndex() == STATIC_MODE_VCM_INDEX:
            return "VCM"
        if combo.currentIndex() == STATIC_MODE_ECSS_QPSK_TM_INDEX:
            return link_budget_math.QPSK_VITERBI_MODE_NAME
        if combo.currentIndex() == STATIC_MODE_QPSK78_INDEX:
            return link_budget_math.QPSK_78_MODE_NAME
        fixed = getattr(self, "static_link_budget_fixed_modcod_combo", None)
        modcod = fixed.currentText().strip() if fixed is not None else ""
        return f"Fixed MODCOD ({modcod})" if modcod else "Fixed MODCOD"

    def _compute_pfd_compliance_backoff(
        self,
        *,
        elevations: np.ndarray,
        sat_altitude_km: float,
        ground_altitude_m: float,
        tx_power_dBw: float,
        tx_losses_dB: float,
        antenna_gain_dBi: float,
        frequency_GHz: float,
        symbol_rate_sps: float,
        rolloff: float,
    ) -> float:
        """Return the TX back-off (dB) that makes worst-case PFD meet the ITU mask."""

        from cosmic_toolbox.analyses.pfd import compliance_backoff_db

        return compliance_backoff_db(
            elevations_deg=elevations,
            sat_altitude_km=sat_altitude_km,
            ground_altitude_m=ground_altitude_m,
            tx_power_dBw=tx_power_dBw,
            tx_losses_dB=tx_losses_dB,
            antenna_gain_dBi=antenna_gain_dBi,
            frequency_GHz=frequency_GHz,
            symbol_rate_sps=symbol_rate_sps,
            rolloff=rolloff,
        )

    def _handle_static_link_budget_calculate(self) -> None:
        """Compute the standalone static link budget (uplink and downlink)."""
        station = self._get_selected_static_link_budget_station()
        if station is None:
            self.static_link_budget_summary_label.setText(
                "Import or select a ground station to view the static link budget."
            )
            self.static_link_budget_table.setRowCount(0)
            ul_table = getattr(self, "static_link_budget_uplink_table", None)
            if ul_table is not None:
                ul_table.setRowCount(0)
            self._clear_static_link_budget_plot()
            self._latest_static_link_budget_plot_data = None
            return
        dl_frequency_GHz = float(self.slb_dl_frequency_input.value()) / 1000.0
        ul_frequency_GHz = float(self.slb_ul_frequency_input.value()) / 1000.0
        dl_symbol_rate_sps = float(self.slb_symbol_rate_input.value()) * 1e6
        ul_symbol_rate_sps = float(self.slb_ul_symbol_rate_input.value()) * 1e6
        evaluation_angle = float(self.slb_gs_elevation_input.value())
        evaluation_angle = max(0.0, min(90.0, evaluation_angle))
        unavailability = float(self.slb_unavailability_input.value())
        common_antenna_gain_dBi = float(self.slb_common_antenna_gain_input.value())
        mispointing_loss_dB = float(self.slb_mispointing_loss_input.value())
        dl_actual_antenna_gain_dBi = common_antenna_gain_dBi - mispointing_loss_dB
        try:
            elevations = np.array([evaluation_angle], dtype=float)
            loss_query_elev = np.clip(elevations, 0.1, 90.0)

            # ── DL PFD compliance back-off ─────────────────────────────────
            pfd_compliance_checked = (
                getattr(self, "slb_pfd_compliance_checkbox", None) is not None
                and self.slb_pfd_compliance_checkbox.isChecked()
            )
            if pfd_compliance_checked:
                tx_backoff_dB = self._compute_pfd_compliance_backoff(
                    elevations=elevations,
                    sat_altitude_km=float(self.slb_sat_altitude_input.value()),
                    ground_altitude_m=station.altitude_m,
                    tx_power_dBw=float(self.slb_tx_power_input.value()),
                    tx_losses_dB=float(self.slb_tx_losses_input.value()),
                    antenna_gain_dBi=dl_actual_antenna_gain_dBi,
                    frequency_GHz=dl_frequency_GHz,
                    symbol_rate_sps=dl_symbol_rate_sps,
                    rolloff=float(self.slb_rolloff_input.value()),
                )
                self.slb_tx_backoff_input.setValue(tx_backoff_dB)
            else:
                tx_backoff_dB = float(self.slb_tx_backoff_input.value())

            # ── DL atmospheric losses ──────────────────────────────────────
            dl_loss_cache_key = self._build_loss_cache_key(
                frequency=dl_frequency_GHz,
                plot_lower_bound=float(loss_query_elev[0]) if loss_query_elev.size else 0.1,
                plot_upper_bound=float(loss_query_elev[-1]) if loss_query_elev.size else 90.0,
                unavailability=unavailability,
                station=station,
                num_samples=int(loss_query_elev.size),
            )
            if (
                self._static_loss_cache_key == dl_loss_cache_key
                and self._static_loss_cache_losses is not None
                and self._static_loss_cache_losses.shape == loss_query_elev.shape
            ):
                dl_atmospheric_losses = self._static_loss_cache_losses
                dl_contribution_breakdown = self._static_loss_cache_contributions
            else:
                dl_loss_result = estimate_slant_path_loss(
                    frequency_GHz=dl_frequency_GHz,
                    elevations_deg=loss_query_elev,
                    lat_deg=station.latitude_deg,
                    lon_deg=station.longitude_deg,
                    altitude_m=station.altitude_m,
                    unavailability_percent=unavailability,
                    return_contributions=True,
                )
                if isinstance(dl_loss_result, tuple):
                    dl_atmospheric_losses, dl_contribution_breakdown = dl_loss_result
                else:
                    dl_atmospheric_losses = dl_loss_result
                    dl_contribution_breakdown = None
                self._static_loss_cache_key = dl_loss_cache_key
                self._static_loss_cache_losses = dl_atmospheric_losses
                self._static_loss_cache_contributions = dl_contribution_breakdown

            # ── UL atmospheric losses ──────────────────────────────────────
            ul_loss_cache_key = self._build_loss_cache_key(
                frequency=ul_frequency_GHz,
                plot_lower_bound=float(loss_query_elev[0]) if loss_query_elev.size else 0.1,
                plot_upper_bound=float(loss_query_elev[-1]) if loss_query_elev.size else 90.0,
                unavailability=unavailability,
                station=station,
                num_samples=int(loss_query_elev.size),
            )
            if (
                self._static_ul_loss_cache_key == ul_loss_cache_key
                and self._static_ul_loss_cache_losses is not None
                and self._static_ul_loss_cache_losses.shape == loss_query_elev.shape
            ):
                ul_atmospheric_losses = self._static_ul_loss_cache_losses
                ul_contribution_breakdown = self._static_ul_loss_cache_contributions
            else:
                ul_loss_result = estimate_slant_path_loss(
                    frequency_GHz=ul_frequency_GHz,
                    elevations_deg=loss_query_elev,
                    lat_deg=station.latitude_deg,
                    lon_deg=station.longitude_deg,
                    altitude_m=station.altitude_m,
                    unavailability_percent=unavailability,
                    return_contributions=True,
                )
                if isinstance(ul_loss_result, tuple):
                    ul_atmospheric_losses, ul_contribution_breakdown = ul_loss_result
                else:
                    ul_atmospheric_losses = ul_loss_result
                    ul_contribution_breakdown = None
                self._static_ul_loss_cache_key = ul_loss_cache_key
                self._static_ul_loss_cache_losses = ul_atmospheric_losses
                self._static_ul_loss_cache_contributions = ul_contribution_breakdown

            # ── DL operating mode ──────────────────────────────────────────
            dl_fixed_modcod_name = None
            mode_index = int(self.static_link_budget_mode_combo.currentIndex())
            if mode_index == STATIC_MODE_ECSS_QPSK_TM_INDEX:
                dl_fixed_modcod_name = link_budget_math.QPSK_VITERBI_MODE_NAME
            elif mode_index == STATIC_MODE_QPSK78_INDEX:
                dl_fixed_modcod_name = link_budget_math.QPSK_78_MODE_NAME
            elif mode_index == STATIC_MODE_FIXED_MODCOD_INDEX:
                dl_fixed_modcod_name = (
                    self.static_link_budget_fixed_modcod_combo.currentText().strip() or None
                )

            # ── DL link budget ─────────────────────────────────────────────
            dl_results = link_budget_math.calculate_link_budget(
                elevations_deg=elevations,
                antenna_gains_dBi=np.full_like(elevations, dl_actual_antenna_gain_dBi),
                atmospheric_losses_dB=dl_atmospheric_losses,
                tx_power_dBw=float(self.slb_tx_power_input.value()),
                tx_boresight_gain_dBi=common_antenna_gain_dBi,
                tx_losses_dB=float(self.slb_tx_losses_input.value()),
                tx_backoff_dB=tx_backoff_dB,
                frequency_GHz=dl_frequency_GHz,
                satellite_altitude_km=float(self.slb_sat_altitude_input.value()),
                ground_altitude_m=station.altitude_m,
                rx_antenna_gain_dBi=float(self.slb_rx_antenna_gain_input.value()),
                receiver_noise_figure_dB=float(self.slb_rx_nf_input.value()),
                sky_background_temperature_K=float(self.slb_sky_bg_temp_input.value()),
                receiver_losses_dB=float(self.slb_rx_losses_input.value()),
                polarization_loss_dB=float(self.slb_polarization_loss_input.value()),
                symbol_rate_sps=dl_symbol_rate_sps,
                implementation_loss_dB=float(self.slb_impl_loss_input.value()),
                margin_dB=float(self.slb_margin_input.value()),
                fixed_modcod_name=dl_fixed_modcod_name,
            )
            dl_rows = link_budget_math.build_parameter_rows(
                elevations_deg=elevations,
                results=dl_results,
                evaluation_elevation_deg=float(evaluation_angle),
                elevation_lower_bound_deg=float(evaluation_angle),
                elevation_upper_bound_deg=float(evaluation_angle),
                tx_frequency_GHz=dl_frequency_GHz,
                tx_power_dBw=float(self.slb_tx_power_input.value()),
                tx_losses_dB=float(self.slb_tx_losses_input.value()),
                tx_boresight_gain_dBi=common_antenna_gain_dBi,
                tx_backoff_dB=float(self.slb_tx_backoff_input.value()),
                symbol_rate_sps=dl_symbol_rate_sps,
                rx_antenna_gain_dBi=float(self.slb_rx_antenna_gain_input.value()),
                receiver_noise_figure_dB=float(self.slb_rx_nf_input.value()),
                sky_background_temperature_K=float(self.slb_sky_bg_temp_input.value()),
                implementation_loss_dB=float(self.slb_impl_loss_input.value()),
                margin_dB=float(self.slb_margin_input.value()),
                rolloff=float(self.slb_rolloff_input.value()),
                polarization_loss_dB=float(self.slb_polarization_loss_input.value()),
                satellite_altitude_km=float(self.slb_sat_altitude_input.value()),
                atmospheric_breakdown_dB=dl_contribution_breakdown,
            )
            dl_rows = [
                row
                for row in dl_rows
                if row.parameter not in {"Elevation Lower Bound", "Elevation Upper Bound"}
            ]

            # ── UL link budget (ground station EIRP as tx_power, no separate gain) ──
            ul_gs_eirp_dBw = float(self.slb_ul_gs_eirp_input.value())
            ul_results = link_budget_math.calculate_link_budget(
                elevations_deg=elevations,
                antenna_gains_dBi=np.zeros_like(elevations),
                atmospheric_losses_dB=ul_atmospheric_losses,
                tx_power_dBw=ul_gs_eirp_dBw,
                tx_boresight_gain_dBi=0.0,
                tx_losses_dB=0.0,
                tx_backoff_dB=0.0,
                frequency_GHz=ul_frequency_GHz,
                satellite_altitude_km=float(self.slb_sat_altitude_input.value()),
                ground_altitude_m=station.altitude_m,
                rx_antenna_gain_dBi=common_antenna_gain_dBi,
                receiver_noise_figure_dB=float(self.slb_ul_rx_nf_input.value()),
                sky_background_temperature_K=float(self.slb_ul_sky_bg_temp_input.value()),
                receiver_losses_dB=float(self.slb_ul_rx_losses_input.value()),
                polarization_loss_dB=float(self.slb_polarization_loss_input.value()),
                symbol_rate_sps=ul_symbol_rate_sps,
                implementation_loss_dB=float(self.slb_ul_impl_loss_input.value()),
                margin_dB=float(self.slb_margin_input.value()),
                fixed_modcod_name=link_budget_math.QPSK_VITERBI_MODE_NAME,
            )
            ul_rows = link_budget_math.build_parameter_rows(
                elevations_deg=elevations,
                results=ul_results,
                evaluation_elevation_deg=float(evaluation_angle),
                elevation_lower_bound_deg=float(evaluation_angle),
                elevation_upper_bound_deg=float(evaluation_angle),
                tx_frequency_GHz=ul_frequency_GHz,
                tx_power_dBw=ul_gs_eirp_dBw,
                tx_losses_dB=0.0,
                tx_boresight_gain_dBi=0.0,
                tx_backoff_dB=0.0,
                symbol_rate_sps=ul_symbol_rate_sps,
                rx_antenna_gain_dBi=common_antenna_gain_dBi,
                receiver_noise_figure_dB=float(self.slb_ul_rx_nf_input.value()),
                sky_background_temperature_K=float(self.slb_ul_sky_bg_temp_input.value()),
                implementation_loss_dB=float(self.slb_ul_impl_loss_input.value()),
                margin_dB=float(self.slb_margin_input.value()),
                rolloff=float(self.slb_rolloff_input.value()),
                polarization_loss_dB=float(self.slb_polarization_loss_input.value()),
                satellite_altitude_km=float(self.slb_sat_altitude_input.value()),
                atmospheric_breakdown_dB=ul_contribution_breakdown,
            )
            ul_rows = [
                row
                for row in ul_rows
                if row.parameter
                not in {
                    "Elevation Lower Bound",
                    "Elevation Upper Bound",
                    "PFD",
                    "PFD Limit",
                    "PFD Margin",
                }
            ]
        except Exception as exc:
            QMessageBox.warning(self, "Static Link Budget Error", str(exc))
            self._clear_static_link_budget_plot()
            self._latest_static_link_budget_plot_data = None
            raise
        eval_index = int(np.argmin(np.abs(elevations - evaluation_angle)))
        dl_summary = self._format_static_link_budget_summary(dl_results, elevations, eval_index)
        ul_summary = self._format_static_link_budget_summary(ul_results, elevations, eval_index)
        combined_summary = f"UL: {ul_summary}  |  DL: {dl_summary}"
        self._populate_static_link_budget_table(dl_rows, combined_summary)
        ul_table = getattr(self, "static_link_budget_uplink_table", None)
        if ul_table is not None:
            self._fill_static_budget_table(ul_rows, ul_table)
        self._latest_static_link_budget_plot_data = (elevations, dl_results)
        self._update_static_link_budget_plot(elevations, dl_results)

    def _static_mode_modulation_params(
        self, mode_name: str
    ) -> tuple[str, int, float, float, float]:
        """Resolve (modulation type, order, coding rate, bits/sym, required Eb/N0).

        ``mode_name`` is a fixed link mode from
        ``link_budget_math.FIXED_LINK_MODE_TABLE`` (e.g. the QPSK Viterbi mode or
        a DVB-S2 MODCOD).  Required Eb/N0 is derived from the table's Es/N0
        threshold via ``Eb/N0 = Es/N0 - 10*log10(bits_per_symbol)``.
        """
        try:
            required_esn0 = link_budget_math.required_esn0_for_mode(mode_name)
            idx = link_budget_math._modcod_index_by_name(mode_name)
            bits = float(link_budget_math.FIXED_LINK_MODE_TABLE[idx][1])
        except Exception:
            required_esn0, bits = 4.5, 1.0
        order_map = {"BPSK": 1, "QPSK": 2, "8PSK": 3, "16APSK": 4, "32APSK": 5}
        token = mode_name.split()[0] if mode_name else "QPSK"
        mod_type = token if token in order_map else "QPSK"
        order = order_map.get(mod_type, 2)
        coding_rate = bits / order if order else 1.0
        required_ebn0 = required_esn0 - 10.0 * float(np.log10(bits)) if bits > 0 else required_esn0
        return mod_type, order, coding_rate, bits, required_ebn0

    def _two_point_atmospheric_loss(
        self,
        *,
        frequency_GHz: float,
        station: GroundStationConfig,
        slant_elevation_deg: float,
        unavailability_percent: float,
    ) -> tuple[float, float]:
        """Return (slant, nadir) one-way atmospheric loss in dB via the ITU model."""
        elevations = np.array(
            [max(0.1, min(90.0, slant_elevation_deg)), 90.0], dtype=float
        )
        losses = estimate_slant_path_loss(
            frequency_GHz=frequency_GHz,
            elevations_deg=elevations,
            lat_deg=station.latitude_deg,
            lon_deg=station.longitude_deg,
            altitude_m=station.altitude_m,
            unavailability_percent=unavailability_percent,
        )
        losses = np.asarray(losses, dtype=float)
        return float(losses[0]), float(losses[1])

    def _build_static_link_budget_export_config(
        self, station: GroundStationConfig
    ):
        """Assemble a LinkBudgetExportConfig from the static-tab inputs."""
        from cosmic_toolbox.services.link_budget_xlsx import LinkBudgetExportConfig

        sat_alt_km = float(self.slb_sat_altitude_input.value())
        elevation_deg = max(0.0, min(90.0, float(self.slb_gs_elevation_input.value())))
        unavailability = float(self.slb_unavailability_input.value())
        common_gain = float(self.slb_common_antenna_gain_input.value())
        polarization = float(self.slb_polarization_loss_input.value())
        rolloff = float(self.slb_rolloff_input.value())
        margin = float(self.slb_margin_input.value())

        dl_freq_ghz = float(self.slb_dl_frequency_input.value()) / 1000.0
        ul_freq_ghz = float(self.slb_ul_frequency_input.value()) / 1000.0

        # Atmospheric losses (slant + nadir) from the ITU model — matches the
        # values the on-screen tables use.
        ul_atm_slant, ul_atm_nadir = self._two_point_atmospheric_loss(
            frequency_GHz=ul_freq_ghz,
            station=station,
            slant_elevation_deg=elevation_deg,
            unavailability_percent=unavailability,
        )
        dl_atm_slant, dl_atm_nadir = self._two_point_atmospheric_loss(
            frequency_GHz=dl_freq_ghz,
            station=station,
            slant_elevation_deg=elevation_deg,
            unavailability_percent=unavailability,
        )

        # Downlink operating mode → modulation/coding + required Eb/N0.
        dl_mode = self._get_static_link_budget_mode_label()
        # Strip the "Fixed MODCOD (...)" wrapper to the bare modcod name.
        if dl_mode.startswith("Fixed MODCOD (") and dl_mode.endswith(")"):
            dl_mode = dl_mode[len("Fixed MODCOD (") : -1]
        if dl_mode in {"VCM", "Fixed MODCOD"}:
            dl_mode = link_budget_math.QPSK_VITERBI_MODE_NAME
        dl_type, dl_order, dl_coding, _dl_bits, dl_req_ebn0 = (
            self._static_mode_modulation_params(dl_mode)
        )

        # Uplink is modelled as ECSS QPSK for TM (QPSK Viterbi k=7, r=1/2).
        ul_mode = link_budget_math.QPSK_VITERBI_MODE_NAME
        ul_type, _ul_order, ul_coding, ul_bits, ul_req_ebn0 = (
            self._static_mode_modulation_params(ul_mode)
        )

        # Downlink system noise temperature from the GS receiver NF + sky temp.
        gs_nf = float(self.slb_rx_nf_input.value())
        gs_sky = float(self.slb_sky_bg_temp_input.value())
        dl_sys_noise_K = float(
            link_budget_math.system_noise_temperature_K(
                np.array([dl_atm_slant], dtype=float),
                receiver_noise_figure_dB=gs_nf,
                sky_background_temperature_K=gs_sky,
            )[0]
        )

        tx_backoff = float(self.slb_tx_backoff_input.value())

        return LinkBudgetExportConfig(
            ground_station_name=station.name,
            satellite_altitude_km=sat_alt_km,
            slant_elevation_deg=elevation_deg,
            ground_altitude_km=station.altitude_m / 1000.0,
            latitude_deg=station.latitude_deg,
            longitude_deg=station.longitude_deg,
            unavailability_percent=unavailability,
            system_margin_dB=margin,
            # ── Uplink ──
            ul_frequency_MHz=float(self.slb_ul_frequency_input.value()),
            ul_modulation_type=ul_type,
            ul_modulation_order=max(1, int(round(ul_bits))),
            ul_coding_label=f"{ul_mode} coding rate",
            ul_coding_rate=ul_coding,
            ul_rolloff=rolloff,
            ul_symbol_rate_ksps=float(self.slb_ul_symbol_rate_input.value()) * 1000.0,
            ul_gs_eirp_dBW=float(self.slb_ul_gs_eirp_input.value()),
            ul_pointing_loss_dB=0.0,
            ul_atmospheric_loss_slant_dB=ul_atm_slant,
            ul_atmospheric_loss_nadir_dB=ul_atm_nadir,
            ul_background_noise_temp_K=float(self.slb_ul_sky_bg_temp_input.value()),
            ul_sc_rx_antenna_gain_dBi=common_gain,
            ul_polarization_loss_dB=polarization,
            ul_sc_harness_loss_dB=float(self.slb_ul_rx_losses_input.value()),
            ul_sc_receiver_noise_figure_dB=float(self.slb_ul_rx_nf_input.value()),
            ul_required_ebn0_dB=ul_req_ebn0,
            # ── Downlink ──
            dl_frequency_MHz=float(self.slb_dl_frequency_input.value()),
            dl_modulation_type=dl_type,
            dl_modulation_order=dl_order,
            dl_coding_label=f"{dl_mode} coding rate",
            dl_coding_rate=dl_coding,
            dl_rolloff=rolloff,
            dl_symbol_rate_ksps=float(self.slb_symbol_rate_input.value()) * 1000.0,
            dl_tx_power_dBW=float(self.slb_tx_power_input.value()) - tx_backoff,
            dl_tx_harness_loss_dB=float(self.slb_tx_losses_input.value()),
            dl_tx_antenna_gain_slant_dBi=common_gain,
            dl_tx_antenna_gain_nadir_dBi=common_gain,
            dl_pointing_loss_dB=0.0,
            dl_implementation_loss_dB=float(self.slb_impl_loss_input.value()),
            dl_atmospheric_loss_slant_dB=dl_atm_slant,
            dl_atmospheric_loss_nadir_dB=dl_atm_nadir,
            dl_background_noise_temp_K=gs_sky,
            dl_rx_antenna_gain_dBi=float(self.slb_rx_antenna_gain_input.value()),
            dl_polarization_loss_dB=polarization,
            dl_mispointing_loss_dB=float(self.slb_mispointing_loss_input.value()),
            dl_system_noise_temperature_K=dl_sys_noise_K,
            dl_required_ebn0_dB=dl_req_ebn0,
            include_pfd=True,
        )

    def _handle_export_static_link_budget_xlsx(self) -> None:
        """Export the static uplink/downlink link budget to a styled .xlsx file."""
        station = self._get_selected_static_link_budget_station()
        if station is None:
            QMessageBox.information(
                self,
                "Export Link Budget",
                "Select a ground station before exporting the link budget.",
            )
            return
        default_name = f"link_budget_{station.name.replace(',', '').replace(' ', '_')}.xlsx"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Link Budget to XLSX",
            default_name,
            "Excel Workbook (*.xlsx)",
        )
        if not path:
            return
        try:
            from cosmic_toolbox.services.link_budget_xlsx import write_link_budget_xlsx

            cfg = self._build_static_link_budget_export_config(station)
            out_path = write_link_budget_xlsx(cfg, path)
        except Exception as exc:  # surface any failure to the user
            QMessageBox.warning(self, "Export Link Budget Error", str(exc))
            return
        QMessageBox.information(
            self,
            "Export Link Budget",
            f"Link budget workbook written to:\n{out_path}",
        )

    def _fill_static_budget_table(
        self,
        rows: list[link_budget_math.ParameterRow],
        table: QTableWidget,
    ) -> None:
        """Populate a link-budget QTableWidget from a list of ParameterRows."""
        highlight_rows = {
            "EIRP",
            "Received Signal Power",
            "Maximum Information Rate",
            "Max. Information Rate",
            "Link Margin",
        }
        highlight_color = QColor("#1b5e20")
        table.setRowCount(len(rows))
        for row_idx, entry in enumerate(rows):
            input_text = "" if entry.is_calc else entry.value
            calc_text = entry.value if entry.is_calc else ""
            items = [
                QTableWidgetItem(entry.parameter),
                QTableWidgetItem(input_text),
                QTableWidgetItem(calc_text),
                QTableWidgetItem(entry.unit),
            ]
            for col, item in enumerate(items):
                if col == 0:
                    align = Qt.AlignmentFlag.AlignLeft
                elif col in (1, 2):
                    align = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                else:
                    align = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
                item.setTextAlignment(align)
                table.setItem(row_idx, col, item)
            if entry.color is not None:
                row_color = QColor(entry.color)
                for item in items:
                    font = QFont(item.font())
                    font.setBold(True)
                    item.setFont(font)
                    item.setBackground(row_color)
            elif entry.parameter in highlight_rows:
                for item in items:
                    font = QFont(item.font())
                    font.setBold(True)
                    item.setFont(font)
                    item.setBackground(highlight_color)

    def _populate_static_link_budget_table(
        self,
        rows: list[link_budget_math.ParameterRow],
        summary_text: str,
    ) -> None:
        table = getattr(self, "static_link_budget_table", None)
        label = getattr(self, "static_link_budget_summary_label", None)
        if table is None or label is None:
            return
        self._fill_static_budget_table(rows, table)
        label.setText(summary_text if rows else "No valid MODCOD found for the current parameters.")

    def _format_static_link_budget_summary(
        self,
        results: dict,
        elevations: np.ndarray,
        index: int,
    ) -> str:
        elevation_deg = float(elevations[index])
        modcod = results["modcod_names"][index]
        data_rate = float(np.asarray(results["data_rate_mbps"])[index])
        margin = float(np.asarray(results["margin_to_required_EsN0_dB"])[index])
        if modcod == "No Link" or np.isnan(margin):
            return (
                f"No MODCOD closes at {elevation_deg:.1f}°. "
                f"Available data rate: {self._format_rate_quantity(data_rate)}."
            )
        return (
            f"{modcod} at {elevation_deg:.1f}° delivers {self._format_rate_quantity(data_rate)} "
            f"(margin {margin:.2f} dB to threshold)."
        )

    def _clear_static_link_budget_plot(self) -> None:
        plot = getattr(self, "static_link_budget_plot", None)
        if plot is None:
            return
        plot.clear()
        legend = getattr(self, "_static_link_budget_plot_legend", None)
        if legend is not None:
            legend.clear()
        for item in getattr(self, "_static_link_budget_plot_annotations", []):
            plot.removeItem(item)
        self._static_link_budget_plot_annotations = []

    def _update_static_link_budget_plot(
        self,
        elevations: np.ndarray,
        results: dict,
    ) -> None:
        plot = getattr(self, "static_link_budget_plot", None)
        if plot is None:
            return
        if elevations.size == 0:
            self._clear_static_link_budget_plot()
            return
        es_n0 = np.asarray(results.get("es_to_n0_dB", []), dtype=float)
        required = np.asarray(results.get("required_EsN0_dB", []), dtype=float)
        if es_n0.size != elevations.size:
            self._clear_static_link_budget_plot()
            return
        margin_offset = float(self.slb_margin_input.value())

        def _build_step_curve(
            x_values: np.ndarray, y_values: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray]:
            if len(x_values) == 0 or len(y_values) == 0:
                return x_values, y_values
            repeated_x = np.repeat(x_values, 2)[1:]
            repeated_y = np.repeat(y_values, 2)[:-1]
            return repeated_x, repeated_y

        plot.clear()
        legend = getattr(self, "_static_link_budget_plot_legend", None)
        if legend is not None:
            legend.clear()
        for item in self._static_link_budget_plot_annotations:
            plot.removeItem(item)
        self._static_link_budget_plot_annotations = []
        plot.setTitle(f"{self._get_static_link_budget_mode_label()} — System Performance")
        plot.plot(elevations, es_n0, pen=pg.mkPen("#ff9800", width=2), name="Maximum Es/N0")
        modcods = results.get("modcod_names", [])
        valid_throughput = np.where(~np.isnan(required), required + margin_offset, np.nan)
        if len(modcods) == len(valid_throughput):
            valid_mask = np.array([name != "No Link" for name in modcods], dtype=bool)
            valid_throughput = np.where(valid_mask, valid_throughput, np.nan)
        step_x, step_y = _build_step_curve(elevations, valid_throughput)
        plot.plot(step_x, step_y, pen=pg.mkPen("#5b7dff", width=2), name="Threshold")
        plot.plot(
            elevations,
            es_n0 - valid_throughput,
            pen=pg.mkPen("#00c853", width=2),
            name="Margin",
        )
        plot.setYRange(-5.0, 20.0, padding=0.0)
        plot.setLimits(yMin=-5.0, yMax=20.0)

    def _get_link_budget_elevation_bounds(self) -> tuple[float, float]:
        """Return (lower_deg, upper_deg) for link-budget-only elevation filtering."""
        lower = float(DEFAULT_LINK_BUDGET_ELEVATION_LOWER_DEG)
        upper = float(DEFAULT_LINK_BUDGET_ELEVATION_UPPER_DEG)
        lower_widget = getattr(self, "lb_elev_lower_input", None)
        upper_widget = getattr(self, "lb_elev_upper_input", None)
        if lower_widget is not None:
            lower = float(lower_widget.value())
        if upper_widget is not None:
            upper = float(upper_widget.value())
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
                gs_gt_dBK=self.lb_gs_gt_input.value(),
                receiver_losses_dB=self.lb_rx_losses_input.value(),
                polarization_loss_dB=self.lb_polarization_loss_input.value(),
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
                gs_gt_dBK=self.lb_gs_gt_input.value(),
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
            raise
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
            if entry.color is not None:
                row_color = QColor(entry.color)
                for item in items:
                    font = QFont(item.font())
                    font.setBold(True)
                    item.setFont(font)
                    item.setBackground(row_color)
            elif entry.parameter in highlight_rows:
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
                closure_text = (
                    f" Link closes above ~{min_close:.1f}° "
                    f"(peak {self._format_rate_quantity(peak_rate)})."
                )
        if modcod == "No Link" or np.isnan(margin):
            return (
                f"No MODCOD closes at {elevation_deg:.1f}°. "
                f"Available data rate: {self._format_rate_quantity(data_rate)}."
                f"{closure_text}"
            )
        return (
            f"{modcod} at {elevation_deg:.1f}° delivers {self._format_rate_quantity(data_rate)} "
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
            self.link_budget_plot.removeItem(item)
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
            getattr(self, "dynamic_link_budget_tabs", None) is not None
            and getattr(self, "_dynamic_link_budget_tab", None) is not None
            and self.dynamic_link_budget_tabs.currentWidget() is self._dynamic_link_budget_tab
        )

    def _on_link_budget_tab_changed(self, index: int) -> None:
        if not self._is_dynamic_link_budget_tab_active():
            return
        dynamic_results = getattr(self, "_latest_dynamic_link_budget_plot_data", None)
        if dynamic_results:
            self._update_dynamic_link_budget_plot(dynamic_results)
        else:
            self._clear_dynamic_link_budget_plot()

    def _clear_dynamic_link_budget_plot(self) -> None:
        plot = getattr(self, "dynamic_link_budget_plot", None)
        if plot is None:
            return
        plot.clear()
        legend = getattr(self, "_dynamic_link_budget_plot_legend", None)
        if legend is not None:
            legend.clear()
        for item in getattr(self, "_dynamic_link_budget_plot_annotations", []):
            plot.removeItem(item)
        self._dynamic_link_budget_plot_annotations = []

    def _update_dynamic_link_budget_plot(self, results: dict) -> None:
        """Render time-varying gain and margin for the selected station."""
        plot = getattr(self, "dynamic_link_budget_plot", None)
        if plot is None:
            return
        time_seconds = np.asarray(results.get("time_seconds", []), dtype=float)
        if time_seconds.size == 0:
            self._clear_dynamic_link_budget_plot()
            return
        es_n0 = np.asarray(results.get("es_to_n0_dB", []), dtype=float)
        required = np.asarray(results.get("required_EsN0_dB", []), dtype=float)
        margin = np.asarray(results.get("margin_to_required_EsN0_dB", []), dtype=float)
        antenna_gains = np.asarray(results.get("antenna_gains_dBi", []), dtype=float)
        if (
            es_n0.size != time_seconds.size
            or required.size != time_seconds.size
            or margin.size != time_seconds.size
            or antenna_gains.size != time_seconds.size
        ):
            self._clear_dynamic_link_budget_plot()
            return

        station = self._get_selected_link_budget_station()
        station_name = station.name if station is not None else "Selected station"
        mode_label = self._get_link_budget_mode_label()
        time_minutes = (time_seconds - float(time_seconds[0])) / 60.0

        plot.clear()
        if getattr(self, "_dynamic_link_budget_plot_legend", None):
            self._dynamic_link_budget_plot_legend.clear()
        for item in self._dynamic_link_budget_plot_annotations:
            plot.removeItem(item)
        self._dynamic_link_budget_plot_annotations = []
        plot.setTitle(
            f"{mode_label} — Dynamic Link Budget ({station_name})"
        )
        plot.setLabel("bottom", "Time", units="min")
        plot.setLabel("left", "Metric", units="dB")
        plot.plot(
            time_minutes,
            es_n0,
            pen=pg.mkPen("#ff9800", width=2),
            name="Es/N0",
        )
        plot.plot(
            time_minutes,
            required,
            pen=pg.mkPen("#5b7dff", width=2),
            name="Required Es/N0",
        )
        plot.plot(
            time_minutes,
            margin,
            pen=pg.mkPen("#00c853", width=2),
            name="Margin",
        )
        plot.plot(
            time_minutes,
            antenna_gains,
            pen=pg.mkPen("#d500f9", width=2),
            name="Antenna Gain",
        )

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
        mode_label = self._get_link_budget_mode_label()
        self.link_budget_plot.setTitle(f"{mode_label} — System Performance")
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
            self.link_budget_plot.removeItem(item)
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
        step_legend = "VCM Step" if mode_label == "VCM" else "Threshold"
        margin_legend = "VCM Margin" if mode_label == "VCM" else "Margin"
        step_x, step_y = _build_step_curve(elevations, valid_throughput)
        self.link_budget_plot.plot(
            step_x,
            step_y,
            pen=pg.mkPen("#5b7dff", width=2),
            name=step_legend,
        )
        valid_margin = es_n0 - valid_throughput
        self.link_budget_plot.plot(
            elevations,
            valid_margin,
            pen=pg.mkPen("#00c853", width=2),
            name=margin_legend,
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
        if not result.timeline_seconds.size or not result.station_elevation_series:
            self._latest_access_series = None
            self._latest_station_rate_series = None
            self._latest_combined_rate_series = None
            self._latest_dynamic_link_budget_plot_data = None
            return
        time_seconds = np.asarray(result.timeline_seconds, dtype=float)
        if time_seconds.ndim != 1 or time_seconds.size == 0:
            self._latest_access_series = None
            self._latest_station_rate_series = None
            self._latest_combined_rate_series = None
            self._latest_dynamic_link_budget_plot_data = None
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
            self._latest_dynamic_link_budget_plot_data = None
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
        eph = getattr(result, "ephemeris", None)
        if (
            eph is not None
            and eph.ecef_pos_km is not None
            and eph.ecef_pos_km.shape[0] == time_seconds.size
        ):
            # Use pre-built arrays from CachedRecomputeWorker when available.
            prebuilt = getattr(self, "_prebuilt_sv_arrays", None)
            if (
                prebuilt is not None
                and isinstance(prebuilt.get("sat_ecef_m"), np.ndarray)
                and prebuilt["sat_ecef_m"].shape == (time_seconds.size, 3)
            ):
                sat_ecef_m = prebuilt["sat_ecef_m"]
                body_x_ecef = prebuilt["body_x_ecef"]
                body_y_ecef = prebuilt["body_y_ecef"]
                body_z_ecef = prebuilt["body_z_ecef"]
            else:
                sat_ecef_m = eph.ecef_pos_km * 1000.0
                body_x_ecef = eph.body_x_ecef if eph.body_x_ecef is not None else np.full((time_seconds.size, 3), np.nan)
                body_y_ecef = eph.body_y_ecef if eph.body_y_ecef is not None else np.full((time_seconds.size, 3), np.nan)
                body_z_ecef = eph.body_z_ecef if eph.body_z_ecef is not None else np.full((time_seconds.size, 3), np.nan)
            payload["sat_ecef_m"] = sat_ecef_m
            payload["body_x_ecef"] = body_x_ecef
            payload["body_y_ecef"] = body_y_ecef
            payload["body_z_ecef"] = body_z_ecef
        above_horizon_src: dict = getattr(result, "station_above_horizon_series", {}) or {}
        station_above_mask: dict[str, np.ndarray] = {}
        for name, mask in above_horizon_src.items():
            arr = np.asarray(mask, dtype=bool)
            if arr.size == time_seconds.size:
                station_above_mask[name] = arr
        if station_above_mask:
            payload["station_above_mask"] = station_above_mask
        self._latest_access_series = payload
        self._latest_station_rate_series = None
        self._latest_combined_rate_series = None
        self._latest_dynamic_link_budget_plot_data = None

    def _infer_altitude_series_from_track(
        self, result, expected_size: int
    ) -> np.ndarray | None:
        """Derive altitude samples from the ground-track positions."""
        eph = getattr(result, "ephemeris", None)
        if eph is None or eph.ecef_pos_km is None or eph.ecef_pos_km.shape[0] != expected_size:
            return None
        radii = np.linalg.norm(eph.ecef_pos_km, axis=1)
        return radii - link_budget_math.EARTH_RADIUS_KM

    def _refresh_data_volume_rate_series(self) -> None:
        """Recompute cached data-rate series using propagated altitude samples."""
        result = self._calculate_station_rate_series()
        if result is None:
            self._latest_station_rate_series = None
            self._latest_combined_rate_series = None
            self._latest_dynamic_link_budget_plot_data = None
            return
        station_rates, combined_rates = result
        self._latest_station_rate_series = station_rates
        self._latest_combined_rate_series = combined_rates
        self._refresh_dynamic_link_budget_plot_data()

    def _get_dynamic_tx_boresight_gain_dBi(self) -> float:
        """Return the boresight gain implied by the auto-loaded antenna LUT."""
        lut = self._get_active_antenna_lut()
        peak_gain_dBi = float(np.max(lut.gain_dbi_grid))
        if not np.isfinite(peak_gain_dBi):
            raise ValueError("Antenna LUT peak gain is non-finite")
        return peak_gain_dBi

    def _get_comms_pointing_mode(self) -> str:
        cfg = getattr(self, "_current_config", None)
        propagation = getattr(cfg, "propagation", None) if cfg is not None else None
        mode = str(getattr(propagation, "comms_pointing_mode", "prograde_pointing") or "")
        mode = mode.strip().lower()
        if mode not in {"prograde_pointing", "free_to_roll", "constrained_aoa"}:
            raise ValueError(f"Unsupported comms pointing mode: {mode!r}")
        return mode

    def _get_comms_pointing_aoa_limit_deg(self) -> float:
        cfg = getattr(self, "_current_config", None)
        propagation = getattr(cfg, "propagation", None) if cfg is not None else None
        aoa_limit = float(getattr(propagation, "comms_pointing_aoa_limit_deg", 0.0))
        if not np.isfinite(aoa_limit) or aoa_limit < 0.0 or aoa_limit > 180.0:
            raise ValueError(f"Invalid comms AoA limit: {aoa_limit!r}")
        return aoa_limit

    def _get_comms_pointing_active_mask(self, station_name: str, sample_size: int) -> np.ndarray:
        if not self._latest_access_series:
            raise ValueError("Comms pointing requires cached access-series data")
        station_series: dict[str, np.ndarray] = self._latest_access_series.get("station_series", {})
        elevations = np.asarray(station_series.get(station_name, []), dtype=float)
        if elevations.shape != (sample_size,):
            raise ValueError(
                f"Missing station elevation series for comms pointing: {station_name!r}"
            )
        cfg = getattr(self, "_current_config", None)
        propagation = getattr(cfg, "propagation", None) if cfg is not None else None
        threshold_deg = float(getattr(propagation, "contact_elevation_deg", 10.0))
        return np.isfinite(elevations) & (elevations >= threshold_deg)

    def _get_station_antenna_gain_series(
        self,
        *,
        station: GroundStationConfig,
        sample_size: int,
    ) -> np.ndarray:
        """Return per-sample antenna gain for a station."""
        if not self._latest_access_series:
            raise ValueError("Antenna LUT mode requires cached access-series geometry")

        sat_ecef_m = np.asarray(self._latest_access_series.get("sat_ecef_m", []), dtype=float)
        body_x_ecef = np.asarray(
            self._latest_access_series.get("body_x_ecef", []), dtype=float
        )
        body_y_ecef = np.asarray(
            self._latest_access_series.get("body_y_ecef", []), dtype=float
        )
        body_z_ecef = np.asarray(
            self._latest_access_series.get("body_z_ecef", []), dtype=float
        )
        if (
            sat_ecef_m.shape != (sample_size, 3)
            or body_x_ecef.shape != (sample_size, 3)
            or body_y_ecef.shape != (sample_size, 3)
            or body_z_ecef.shape != (sample_size, 3)
        ):
            raise ValueError(
                "Antenna LUT mode requires ECEF state vectors and body axes for every sample"
            )

        lut = self._get_active_antenna_lut()
        pointing_mode = self._get_comms_pointing_mode()
        gains_dbi, _az_deg, _el_deg, _roll_deg = antenna_pattern.evaluate_station_gain_series(
            lut=lut,
            station=station,
            sat_ecef_m=sat_ecef_m,
            body_x_ecef=body_x_ecef,
            body_y_ecef=body_y_ecef,
            body_z_ecef=body_z_ecef,
            pointing_mode=pointing_mode,
            max_aoa_deg=self._get_comms_pointing_aoa_limit_deg(),
            steering_active_mask=self._get_comms_pointing_active_mask(
                station.name, sample_size
            ),
        )
        return gains_dbi

    def _refresh_dynamic_link_budget_plot_data(self) -> None:
        """Refresh the dynamic plot payload for the selected station."""
        self._latest_dynamic_link_budget_plot_data = None
        if not self._latest_access_series:
            return
        station = self._get_selected_link_budget_station()
        if station is None:
            return

        time_seconds = np.asarray(
            self._latest_access_series.get("time_seconds", []), dtype=float
        )
        altitude_samples = np.asarray(
            self._latest_access_series.get("altitude_km", []), dtype=float
        )
        station_series: dict[str, np.ndarray] = self._latest_access_series.get(
            "station_series", {}
        )
        elevations = np.asarray(station_series.get(station.name, []), dtype=float)
        if (
            time_seconds.ndim != 1
            or altitude_samples.ndim != 1
            or elevations.ndim != 1
            or time_seconds.size == 0
            or time_seconds.size != altitude_samples.size
            or time_seconds.size != elevations.size
        ):
            return

        finite_altitude = np.nan_to_num(
            altitude_samples,
            nan=float(np.nanmedian(altitude_samples))
            if np.any(np.isfinite(altitude_samples))
            else 0.0,
        )
        mode_combo = getattr(self, "link_budget_mode_combo", None)
        mode_index = int(mode_combo.currentIndex()) if mode_combo is not None else 0
        fixed_modcod_name: str | None = None
        if mode_index == DYNAMIC_MODE_ECSS_QPSK_TM_INDEX:
            fixed_modcod_name = link_budget_math.QPSK_VITERBI_MODE_NAME
        elif mode_index == DYNAMIC_MODE_QPSK78_INDEX:
            fixed_modcod_name = link_budget_math.QPSK_78_MODE_NAME
        elif mode_index == DYNAMIC_MODE_FIXED_MODCOD_INDEX:
            fixed_combo = getattr(self, "link_budget_fixed_modcod_combo", None)
            if fixed_combo is not None:
                fixed_modcod_name = fixed_combo.currentText().strip() or None
        tx_boresight_gain_dBi = self._get_dynamic_tx_boresight_gain_dBi()
        antenna_gains = self._get_station_antenna_gain_series(
            station=station,
            sample_size=time_seconds.size,
        )
        base_results = self._evaluate_station_link_budget_results(
            station=station,
            elevations_deg=elevations,
            altitude_km=finite_altitude,
            frequency_GHz=float(self.lb_frequency_input.value()),
            symbol_rate_sps=float(self.lb_symbol_rate_input.value()) * 1e6,
            tx_power_dBw=float(self.lb_tx_power_input.value()),
            tx_gain_dBi=tx_boresight_gain_dBi,
            tx_losses_dB=float(self.lb_tx_losses_input.value()),
            tx_backoff_dB=float(self.lb_tx_backoff_input.value()),
            antenna_gains_dBi=antenna_gains,
            gs_gt_dBK=float(self.lb_gs_gt_input.value()),
            rx_losses_dB=float(self.lb_rx_losses_input.value()),
            polarization_loss_dB=float(self.lb_polarization_loss_input.value()),
            implementation_loss_dB=float(self.lb_impl_loss_input.value()),
            margin_dB=float(self.lb_margin_input.value()),
            unavailability_percent=float(self.lb_unavailability_input.value()),
            fixed_modcod_name=fixed_modcod_name,
        )
        if base_results is None:
            return
        results = dict(base_results)
        if mode_index == DYNAMIC_MODE_CCM_INDEX:
            results = self._build_dynamic_ccm_plot_results(
                time_seconds=time_seconds,
                station_name=station.name,
                base_results=base_results,
                symbol_rate_sps=float(self.lb_symbol_rate_input.value()) * 1e6,
                margin_dB=float(self.lb_margin_input.value()),
            )
        results["time_seconds"] = np.asarray(time_seconds, dtype=float)
        results["elevations_deg"] = np.asarray(elevations, dtype=float)
        self._latest_dynamic_link_budget_plot_data = results

    def _modcod_threshold_dB(self, modcod_name: str) -> float:
        """Return the required Es/N0 threshold for a MODCOD name."""
        return link_budget_math.required_esn0_for_mode(modcod_name)

    def _build_dynamic_ccm_plot_results(
        self,
        *,
        time_seconds: np.ndarray,
        station_name: str,
        base_results: dict,
        symbol_rate_sps: float,
        margin_dB: float,
    ) -> dict:
        """Build mode-specific dynamic plot results for CCM optimization."""
        es_n0 = np.asarray(base_results.get("es_to_n0_dB", []), dtype=float)
        antenna_gains = np.asarray(base_results.get("antenna_gains_dBi", []), dtype=float)
        if es_n0.size != time_seconds.size:
            raise ValueError("CCM plot build requires Es/N0 series matching time axis")
        required = np.full(time_seconds.size, np.nan, dtype=float)
        margin_series = np.full(time_seconds.size, np.nan, dtype=float)
        data_rate_mbps = np.zeros(time_seconds.size, dtype=float)
        modcod_names = np.full(time_seconds.size, "No Link", dtype=object)
        result = self._get_contact_result_for_data_volume()
        if result is None or not getattr(self, "_current_config", None):
            raise ValueError("CCM plot build requires filtered contact results and current config")
        passes = getattr(result, "passes", None) or []
        start_time = self._current_config.scenario.start_time
        offset_step_s = float(
            getattr(self, "lb_ccm_offset_step_input", None)
            and self.lb_ccm_offset_step_input.value()
            or 10.0
        )
        max_offset_s = float(
            getattr(self, "lb_ccm_max_offset_input", None)
            and self.lb_ccm_max_offset_input.value()
            or 300.0
        )
        station_passes = [
            p for p in passes if getattr(p, "station_name", None) == station_name
        ]
        for entry in station_passes:
            aos_seconds = (entry.aos - start_time).total_seconds()
            los_seconds = (entry.los - start_time).total_seconds()
            if not np.isfinite(aos_seconds) or not np.isfinite(los_seconds) or los_seconds <= aos_seconds:
                continue
            pass_mask = (time_seconds >= float(aos_seconds)) & (time_seconds <= float(los_seconds))
            if not np.any(pass_mask):
                continue
            t_pass = time_seconds[pass_mask]
            es_n0_pass = es_n0[pass_mask]
            ccm_result = link_budget_math.optimize_ccm_per_pass(
                time_seconds=t_pass,
                es_n0_dB=es_n0_pass,
                margin_dB=margin_dB,
                symbol_rate_sps=symbol_rate_sps,
                offset_step_s=offset_step_s,
                max_offset_s=max_offset_s,
            )
            active_mask = np.asarray(ccm_result.rate_mbps, dtype=float) > 0.0
            required_pass = np.full(t_pass.size, np.nan, dtype=float)
            margin_pass = np.full(t_pass.size, np.nan, dtype=float)
            modcod_pass = np.full(t_pass.size, "No Link", dtype=object)
            if np.any(active_mask):
                threshold = self._modcod_threshold_dB(ccm_result.modcod_name)
                required_pass[active_mask] = threshold
                margin_pass[active_mask] = es_n0_pass[active_mask] - threshold
                modcod_pass[active_mask] = ccm_result.modcod_name
            required[pass_mask] = required_pass
            margin_series[pass_mask] = margin_pass
            data_rate_mbps[pass_mask] = np.asarray(ccm_result.rate_mbps, dtype=float)
            modcod_names[pass_mask] = modcod_pass
        return {
            **base_results,
            "required_EsN0_dB": required,
            "margin_to_required_EsN0_dB": margin_series,
            "data_rate_mbps": data_rate_mbps,
            "modcod_names": modcod_names.tolist(),
            "antenna_gains_dBi": antenna_gains,
        }

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
            self.lb_tx_losses_input,
            self.lb_tx_backoff_input,
            self.lb_gs_gt_input,
            self.lb_rx_losses_input,
            self.lb_symbol_rate_input,
            self.lb_impl_loss_input,
            self.lb_margin_input,
            self.lb_unavailability_input,
            self.lb_polarization_loss_input,
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
            "tx_gain_dBi": self._get_dynamic_tx_boresight_gain_dBi(),
            "tx_losses_dB": self.lb_tx_losses_input.value(),
            "tx_backoff_dB": self.lb_tx_backoff_input.value(),
            "gs_gt_dBK": self.lb_gs_gt_input.value(),
            "rx_losses_dB": self.lb_rx_losses_input.value(),
            "polarization_loss_dB": self.lb_polarization_loss_input.value(),
            "implementation_loss_dB": self.lb_impl_loss_input.value(),
            "margin_dB": self.lb_margin_input.value(),
            "unavailability_percent": self.lb_unavailability_input.value(),
        }
        finite_altitude = np.nan_to_num(
            altitude_samples,
            nan=float(np.nanmedian(altitude_samples)) if np.any(np.isfinite(altitude_samples)) else 0.0,
        )
        mode_index = getattr(self, "link_budget_mode_combo", None)
        mode_index = int(mode_index.currentIndex()) if mode_index is not None else 0
        fixed_modcod_name: str | None = None
        if mode_index == DYNAMIC_MODE_ECSS_QPSK_TM_INDEX:
            fixed_modcod_name = link_budget_math.QPSK_VITERBI_MODE_NAME
        elif mode_index == DYNAMIC_MODE_QPSK78_INDEX:
            fixed_modcod_name = link_budget_math.QPSK_78_MODE_NAME
        elif mode_index == DYNAMIC_MODE_FIXED_MODCOD_INDEX:
            fixed_combo = getattr(self, "link_budget_fixed_modcod_combo", None)
            if fixed_combo is not None:
                fixed_modcod_name = fixed_combo.currentText().strip() or None

        station_rates: dict[str, np.ndarray] = {}
        combined_rates: np.ndarray | None = None
        station_above_mask: dict[str, np.ndarray] = self._latest_access_series.get(
            "station_above_mask", {}
        )

        if mode_index == DYNAMIC_MODE_CCM_INDEX:
            # Dynamic CCM Optimal (per pass): optimize start/end offset + MODCOD per pass.
            result = self._get_contact_result_for_data_volume()
            if result is None or not getattr(self, "_current_config", None):
                return None
            passes = getattr(result, "passes", None) or []
            start_time = self._current_config.scenario.start_time
            offset_step_s = float(getattr(self, "lb_ccm_offset_step_input", None) and self.lb_ccm_offset_step_input.value() or 10.0)
            max_offset_s = float(getattr(self, "lb_ccm_max_offset_input", None) and self.lb_ccm_max_offset_input.value() or 300.0)
            for name, elevations in station_series.items():
                elevation_array = np.asarray(elevations, dtype=float)
                if elevation_array.size != finite_altitude.size:
                    continue
                station_cfg = self._resolve_station_config(name)
                if station_cfg is None:
                    continue
                antenna_gain_series = self._get_station_antenna_gain_series(
                    station=station_cfg,
                    sample_size=time_seconds.size,
                )
                rates = np.zeros(time_seconds.size, dtype=float)
                station_passes = [p for p in passes if getattr(p, "station_name", None) == name]
                for entry in station_passes:
                    aos_seconds = (entry.aos - start_time).total_seconds()
                    los_seconds = (entry.los - start_time).total_seconds()
                    if not np.isfinite(aos_seconds) or not np.isfinite(los_seconds) or los_seconds <= aos_seconds:
                        continue
                    pass_mask = (time_seconds >= float(aos_seconds)) & (time_seconds <= float(los_seconds))
                    if not np.any(pass_mask):
                        continue
                    t_pass = time_seconds[pass_mask]
                    elev_pass = elevation_array[pass_mask]
                    alt_pass = finite_altitude[pass_mask]
                    lb_results = self._evaluate_station_link_budget_results(
                        station=station_cfg,
                        elevations_deg=elev_pass,
                        altitude_km=alt_pass,
                        antenna_gains_dBi=antenna_gain_series[pass_mask],
                        **params,
                    )
                    if lb_results is None:
                        continue
                    es_n0_dB = np.asarray(lb_results.get("es_to_n0_dB", []), dtype=float)
                    if es_n0_dB.size != t_pass.size:
                        continue
                    ccm_result = link_budget_math.optimize_ccm_per_pass(
                        time_seconds=t_pass,
                        es_n0_dB=es_n0_dB,
                        margin_dB=params["margin_dB"],
                        symbol_rate_sps=symbol_rate_sps,
                        offset_step_s=offset_step_s,
                        max_offset_s=max_offset_s,
                    )
                    rates[pass_mask] = ccm_result.rate_mbps
                above = station_above_mask.get(name)
                if above is not None and above.size == rates.size:
                    rates = np.where(above, rates, 0.0)
                station_rates[name] = rates
                combined_rates = rates if combined_rates is None else combined_rates + rates
        else:
            # VCM, ECSS fixed mode, or Fixed MODCOD.
            for name, elevations in station_series.items():
                elevation_array = np.asarray(elevations, dtype=float)
                if elevation_array.size != finite_altitude.size:
                    continue
                station_cfg = self._resolve_station_config(name)
                if station_cfg is None:
                    continue
                antenna_gain_series = self._get_station_antenna_gain_series(
                    station=station_cfg,
                    sample_size=time_seconds.size,
                )
                rates = self._evaluate_station_data_rate(
                    station=station_cfg,
                    elevations_deg=elevation_array,
                    altitude_km=finite_altitude,
                    antenna_gains_dBi=antenna_gain_series,
                    **params,
                    fixed_modcod_name=fixed_modcod_name,
                )
                if rates is None or rates.size != finite_altitude.size:
                    continue
                above = station_above_mask.get(name)
                if above is not None and above.size == rates.size:
                    rates = np.where(above, rates, 0.0)
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
        gs_gt_dBK: float,
        rx_losses_dB: float,
        implementation_loss_dB: float,
        margin_dB: float,
        unavailability_percent: float,
        polarization_loss_dB: float,
        antenna_gains_dBi: np.ndarray | None = None,
        antenna_gain_dBi: float | None = None,
        fixed_modcod_name: str | None = None,
    ) -> np.ndarray | None:
        """Run the link-budget math for a specific station timeline."""
        if elevations_deg.size == 0 or altitude_km.size != elevations_deg.size:
            return None
        raw_elev = np.asarray(elevations_deg, dtype=float)
        raw_elev = np.nan_to_num(raw_elev, nan=-90.0)
        sanitized_elev = np.clip(raw_elev, 0.0, 90.0)
        if sanitized_elev.size == 0:
            raise ValueError("No elevation samples available for station rate series computation")
        curve = self._get_timeseries_atmospheric_loss_curve(
            station=station,
            frequency_GHz=frequency_GHz,
            unavailability_percent=unavailability_percent,
            min_elev_deg=0.1,
            max_elev_deg=90.0,
        )
        if curve is None:
            raise RuntimeError("Atmospheric loss curve unavailable for time-series computation")
        loss_grid, loss_values = curve
        losses = np.interp(
            np.maximum(sanitized_elev, float(loss_grid[0])),
            loss_grid,
            loss_values,
            left=float(loss_values[0]),
            right=float(loss_values[-1]),
        )
        if antenna_gains_dBi is None:
            if antenna_gain_dBi is None:
                raise ValueError("Fixed antenna gain is required when no gain series is provided")
            antenna_gains = np.full_like(
                elevations_deg, float(antenna_gain_dBi), dtype=float
            )
        else:
            antenna_gains = np.asarray(antenna_gains_dBi, dtype=float)
            if antenna_gains.shape != elevations_deg.shape:
                raise ValueError(
                    "Antenna gain series must match elevation series shape for station rate evaluation"
                )
            if not np.all(np.isfinite(antenna_gains)):
                raise ValueError("Antenna gain series contains non-finite values")
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
            gs_gt_dBK=gs_gt_dBK,
            receiver_losses_dB=rx_losses_dB,
            polarization_loss_dB=polarization_loss_dB,
            symbol_rate_sps=symbol_rate_sps,
            implementation_loss_dB=implementation_loss_dB,
            margin_dB=margin_dB,
            fixed_modcod_name=fixed_modcod_name,
        )
        results["antenna_gains_dBi"] = np.asarray(antenna_gains, dtype=float)
        rates = np.asarray(results.get("data_rate_mbps", []), dtype=float)
        if rates.size != elevations_deg.size:
            raise ValueError(
                "Link-budget results did not return a rate series matching input size "
                f"(rates.size={int(rates.size)}, elevations.size={int(elevations_deg.size)})"
            )
        if not np.all(np.isfinite(rates)):
            raise ValueError("Non-finite data rate produced by link budget math")
        no_contact_mask = raw_elev <= 0.0
        if np.any(no_contact_mask):
            rates[no_contact_mask] = 0.0
        return rates

    def _evaluate_station_link_budget_results(
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
        gs_gt_dBK: float,
        rx_losses_dB: float,
        implementation_loss_dB: float,
        margin_dB: float,
        unavailability_percent: float,
        polarization_loss_dB: float,
        antenna_gains_dBi: np.ndarray | None = None,
        antenna_gain_dBi: float | None = None,
        fixed_modcod_name: str | None = None,
    ) -> dict | None:
        """Return full link-budget results dict (incl. es_to_n0_dB) for a station timeline."""
        if elevations_deg.size == 0 or altitude_km.size != elevations_deg.size:
            return None
        raw_elev = np.asarray(elevations_deg, dtype=float)
        sanitized_elev = np.clip(np.nan_to_num(raw_elev, nan=-90.0), 0.0, 90.0)
        if sanitized_elev.size == 0:
            return None
        curve = self._get_timeseries_atmospheric_loss_curve(
            station=station,
            frequency_GHz=frequency_GHz,
            unavailability_percent=unavailability_percent,
            min_elev_deg=0.1,
            max_elev_deg=90.0,
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
        if antenna_gains_dBi is None:
            if antenna_gain_dBi is None:
                raise ValueError("Fixed antenna gain is required when no gain series is provided")
            antenna_gains = np.full_like(
                elevations_deg, float(antenna_gain_dBi), dtype=float
            )
        else:
            antenna_gains = np.asarray(antenna_gains_dBi, dtype=float)
            if antenna_gains.shape != elevations_deg.shape:
                raise ValueError(
                    "Antenna gain series must match elevation series shape for link-budget evaluation"
                )
            if not np.all(np.isfinite(antenna_gains)):
                raise ValueError("Antenna gain series contains non-finite values")
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
            gs_gt_dBK=gs_gt_dBK,
            receiver_losses_dB=rx_losses_dB,
            polarization_loss_dB=polarization_loss_dB,
            symbol_rate_sps=symbol_rate_sps,
            implementation_loss_dB=implementation_loss_dB,
            margin_dB=margin_dB,
            fixed_modcod_name=fixed_modcod_name,
        )
        results["antenna_gains_dBi"] = np.asarray(antenna_gains, dtype=float)
        return results

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
            raise ValueError(f"Invalid frequency_GHz: {frequency_GHz!r}")
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
        losses = estimate_slant_path_loss(
            frequency_GHz=frequency_GHz,
            elevations_deg=grid,
            lat_deg=station.latitude_deg,
            lon_deg=station.longitude_deg,
            altitude_m=station.altitude_m,
            unavailability_percent=unavailability_percent,
        )
        losses = np.asarray(losses, dtype=float)
        if not np.all(np.isfinite(losses)):
            raise ValueError("Non-finite atmospheric loss values returned by ITU-R model")
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
        if getattr(self, "_downlink_mode_label", None) is not None:
            self._downlink_mode_label.setText(self._get_link_budget_mode_label())
        # Debounce the heavy recomputation so UI edits don't freeze the GUI.
        self._schedule_data_volume_refresh()

        # Keep showing the last computed values (or placeholders) until the refresh completes.
        metrics = self._compute_downlink_metrics()
        if metrics is None:
            self._downlink_total_label.setText("—")
            self._downlink_per_orbit_label.setText("—")
            return
        total_gbit, per_orbit_gbit = metrics
        self._downlink_total_label.setText(self._format_data_quantity(total_gbit))
        if per_orbit_gbit is None:
            self._downlink_per_orbit_label.setText("—")
        else:
            self._downlink_per_orbit_label.setText(
                self._format_data_quantity(per_orbit_gbit, suffix="/orbit")
            )

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
        use_link_close_windows = getattr(
            self, "_use_link_close_for_contact_windows", None
        )
        refresh_contact_view = getattr(self, "_refresh_contact_statistics_view", None)
        build_filtered_result = getattr(self, "_build_filtered_contact_result", None)
        last_result = getattr(self, "_last_result", None)
        if (
            callable(use_link_close_windows)
            and use_link_close_windows()
            and callable(refresh_contact_view)
            and callable(build_filtered_result)
            and last_result is not None
        ):
            filtered_result = build_filtered_result(last_result)
            refresh_contact_view(filtered_result, update_downlink_summary=False)
        else:
            # Refresh just the results table so the Data Volume column reflects
            # the latest rate series without rebuilding the full contact view.
            populate_table = getattr(self, "_populate_results_table", None)
            filtered_contact_result = getattr(self, "_filtered_contact_result", None)
            if callable(populate_table) and filtered_contact_result is not None:
                populate_table(filtered_contact_result)
        metrics = self._compute_downlink_metrics()
        if metrics is None:
            self._downlink_total_label.setText("—")
            self._downlink_per_orbit_label.setText("—")
        else:
            total_gbit, per_orbit_gbit = metrics
            self._downlink_total_label.setText(self._format_data_quantity(total_gbit))
            if per_orbit_gbit is None:
                self._downlink_per_orbit_label.setText("—")
            else:
                self._downlink_per_orbit_label.setText(
                    self._format_data_quantity(per_orbit_gbit, suffix="/orbit")
                )
        self._update_data_volume_plots()
        if self._is_dynamic_link_budget_tab_active():
            dynamic_results = getattr(self, "_latest_dynamic_link_budget_plot_data", None)
            if dynamic_results:
                self._update_dynamic_link_budget_plot(dynamic_results)
            else:
                self._clear_dynamic_link_budget_plot()

    def _compute_downlink_metrics(self) -> tuple[float, float | None] | None:
        """Compute total and per-orbit Gbit using cached data."""
        if (
            not self._latest_access_series
            or not getattr(self, "_latest_station_rate_series", None)
        ):
            return None
        result = self._get_contact_result_for_data_volume()
        if result is None:
            return None
        passes = getattr(result, "passes", None)
        if not passes:
            return (0.0, None)
        time_seconds = np.asarray(
            self._latest_access_series.get("time_seconds", []), dtype=float
        )
        station_series: dict[str, np.ndarray] = self._latest_access_series.get(
            "station_series", {}
        )
        orbit_period_s = float(self._latest_access_series.get("orbit_period_s", 0.0))
        if time_seconds.size < 2 or not station_series:
            return None
        pass_volumes = self._compute_pass_downlink_volumes()
        if not pass_volumes:
            return (0.0, None)
        total_gbit = float(np.sum(pass_volumes))
        duration_s = float(time_seconds[-1] - time_seconds[0])
        per_orbit = None
        if orbit_period_s > 0.0:
            num_orbits = duration_s / orbit_period_s
            if num_orbits > 0:
                per_orbit = total_gbit / num_orbits
        return (total_gbit, per_orbit)

    def _compute_daily_downlink_volumes(self) -> dict[int, float] | None:
        """Compute data volume per day over the scenario window.

        Returns a dictionary mapping day number (0-indexed from start) to
        data volume in Gbit for that day.
        """
        if (
            not self._latest_access_series
            or not getattr(self, "_latest_station_rate_series", None)
        ):
            return None
        result = self._get_contact_result_for_data_volume()
        if result is None:
            return None
        passes = getattr(result, "passes", None)
        if not passes:
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
        passes_mask = self._get_contact_pass_mask(time_seconds)
        if passes_mask is None:
            return None
        combined_rates = np.where(passes_mask, combined_rates, 0.0)

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

        # Convert from Mb to Gbit
        for day in daily_volumes:
            daily_volumes[day] = daily_volumes[day] / 1000.0

        return daily_volumes

    def _get_data_volume_distribution_options(self) -> list[tuple[str, str]]:
        """Return (id, label) options for the distribution selector."""
        return [
            ("gbit_per_pass", "Data volume per pass"),
            (
                "downlink_rate_mbps",
                "Downlink rate — within filtered contact passes (all samples)",
            ),
            (
                "downlink_rate_mbps_nonzero",
                "Downlink rate — within filtered contact passes (nonzero)",
            ),
            ("pass_duration_min", "Pass time in filtered contacts (min)"),
            ("pass_in_bounds_pct", "Pass time in filtered contacts (%)"),
            ("pass_peak_elevation_in_bounds_deg", "Pass peak elevation within bounds (deg)"),
            ("gbit_per_day", "Data volume per day"),
        ]

    def _get_selected_distribution_id(self) -> str:
        combo = getattr(self, "data_volume_distribution_combo", None)
        if combo is None or combo.count() == 0:
            return "gbit_per_pass"
        data = combo.currentData()
        if data is None:
            return "gbit_per_pass"
        return str(data)

    def _get_combined_rate_series(self) -> np.ndarray | None:
        """Return combined Mbps time-series, if available."""
        combined = getattr(self, "_latest_combined_rate_series", None)
        if combined is not None:
            arr = np.asarray(combined, dtype=float)
            if arr.ndim != 1 or not arr.size:
                raise ValueError("Cached combined rate series is invalid or empty")
            return arr
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
        """Return mask for samples inside the filtered contact-pass set."""
        if not self._latest_access_series:
            return None
        time_seconds = np.asarray(
            self._latest_access_series.get("time_seconds", []), dtype=float
        )
        if time_seconds.ndim != 1 or time_seconds.size == 0:
            return None
        passes_mask = self._get_contact_pass_mask(time_seconds)
        if passes_mask is None:
            return None
        return passes_mask

    def _get_contact_result_for_data_volume(self):
        """Return the contact statistics result used by the Data Volume tab.

        Uses the same fully-filtered result as the Contact Statistics panel
        (elevation filter + optional link-close trim), so all Data Volume
        distributions reflect the same pass set visible in statistics.
        """
        getter = getattr(self, "_get_contact_statistics_result", None)
        if not callable(getter):
            raise ValueError("Contact statistics accessor is unavailable.")
        return getter()

    def _get_contact_pass_mask(self, time_seconds: np.ndarray) -> np.ndarray | None:
        """Return mask of timeline samples within filtered contact passes."""
        result = self._get_contact_result_for_data_volume()
        if result is None:
            return None
        passes = getattr(result, "passes", None)
        if not passes:
            return np.zeros(time_seconds.size, dtype=bool)
        if not getattr(self, "_current_config", None):
            raise ValueError("Missing scenario configuration for pass mask.")
        start_time = self._current_config.scenario.start_time
        mask = np.zeros(time_seconds.size, dtype=bool)
        for entry in passes:
            aos_seconds = (entry.aos - start_time).total_seconds()
            los_seconds = (entry.los - start_time).total_seconds()
            if not np.isfinite(aos_seconds) or not np.isfinite(los_seconds):
                continue
            if los_seconds <= aos_seconds:
                continue
            mask |= (time_seconds >= float(aos_seconds)) & (
                time_seconds <= float(los_seconds)
            )
        return mask

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
        """Return per-pass duration inside the filtered contact-pass set."""
        if (
            self._get_contact_result_for_data_volume() is None
            or not self._latest_access_series
            or not getattr(self, "_current_config", None)
        ):
            return None
        result = self._get_contact_result_for_data_volume()
        passes = getattr(result, "passes", None) if result is not None else None
        if not passes:
            return None
        time_seconds = np.asarray(
            self._latest_access_series.get("time_seconds", []), dtype=float
        )
        if time_seconds.size < 2:
            return None
        start_time = getattr(self._current_config.scenario, "start_time", None)
        if start_time is None:
            return None
        start_axis = float(time_seconds[0])
        end_axis = float(time_seconds[-1])

        union_mask = self._get_any_station_in_bounds_mask()
        if union_mask is None:
            union_mask = np.zeros(time_seconds.size, dtype=bool)

        durations_min: list[float] = []
        for entry in passes:
            aos_seconds = (entry.aos - start_time).total_seconds()
            los_seconds = (entry.los - start_time).total_seconds()
            pass_start = max(start_axis, float(aos_seconds))
            pass_end = min(end_axis, float(los_seconds))
            if pass_end <= pass_start:
                durations_min.append(0.0)
                continue
            seconds_in_bounds = self._integrate_indicator_interval(
                time_axis=time_seconds,
                indicator=union_mask.astype(float),
                start_sec=pass_start,
                end_sec=pass_end,
            )
            durations_min.append(max(0.0, seconds_in_bounds / 60.0))
        return durations_min or None

    def _compute_distribution_samples(
        self, distribution_id: str
    ) -> tuple[np.ndarray | None, str, str, str, str]:
        """Return (samples, x_label, x_units, empty_title, full_title)."""
        distribution_id = str(distribution_id or "gbit_per_pass")

        if distribution_id == "gbit_per_pass":
            samples = self._compute_pass_downlink_volumes()
            array = (
                np.asarray(samples, dtype=float)
                if samples is not None and len(samples) > 0
                else None
            )
            return (
                array,
                "Data volume",
                "Gbit/pass",
                "Data volume per pass\n(Run analysis to view)",
                "Data volume per pass distribution",
            )

        if distribution_id in ("downlink_rate_mbps", "downlink_rate_mbps_nonzero"):
            series = self._get_combined_rate_series()
            if series is None or series.size == 0:
                return (
                    None,
                    "Downlink rate",
                    "Mbit/s",
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
                "Mbit/s",
                "Downlink rate\n(Run analysis to view)",
                "Downlink rate distribution",
            )

        if distribution_id == "pass_duration_min":
            durations = self._compute_pass_time_in_bounds_minutes()
            if not durations:
                return (
                    None,
                    "Filtered contact time",
                    "min",
                    "Filtered contact time\n(Run analysis to view)",
                    "Filtered contact time distribution",
                )
            return (
                np.asarray(durations, dtype=float),
                "Filtered contact time",
                "min",
                "Filtered contact time\n(Run analysis to view)",
                "Filtered contact time distribution",
            )

        if distribution_id == "pass_in_bounds_pct":
            durations = self._compute_pass_time_in_bounds_minutes()
            result = self._get_contact_result_for_data_volume()
            passes = getattr(result, "passes", None) if result is not None else None
            if not durations or not passes:
                return (
                    None,
                    "Filtered contact time",
                    "%",
                    "Filtered contact time\n(Run analysis to view)",
                    "Filtered contact time distribution",
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
                "Filtered contact time",
                "%",
                "Filtered contact time\n(Run analysis to view)",
                "Filtered contact time distribution",
            )

        if distribution_id == "pass_peak_elevation_in_bounds_deg":
            result = self._get_contact_result_for_data_volume()
            passes = getattr(result, "passes", None) if result is not None else None
            if not passes:
                return (
                    None,
                    "Peak elevation",
                    "deg",
                    "Pass peak elevation\n(Run analysis to view)",
                    "Pass peak elevation distribution",
                )
            values = np.asarray(
                [float(getattr(p, "max_elevation_deg", 0.0)) for p in passes],
                dtype=float,
            )
            return (
                values,
                "Peak elevation",
                "deg",
                "Pass peak elevation\n(Run analysis to view)",
                "Pass peak elevation distribution",
            )

        if distribution_id == "gbit_per_day":
            daily = self._compute_daily_downlink_volumes()
            values = (
                np.asarray(list(daily.values()), dtype=float)
                if daily
                else None
            )
            return (
                values,
                "Data volume",
                "Gbit/day",
                "Data volume per day\n(Run analysis to view)",
                "Data volume per day distribution",
            )

        # Unknown id fallback.
        return (
            None,
            "Value",
            "",
            "Distribution\n(Run analysis to view)",
            "Distribution",
        )

    def _get_distribution_display_scaling(
        self, samples: np.ndarray | None, units: str
    ) -> tuple[np.ndarray | None, str, float]:
        """Return samples rescaled for display plus the chosen display units."""
        if samples is None:
            return (None, units, 1.0)
        values = np.asarray(samples, dtype=float)
        unit_text = str(units or "")
        if unit_text.startswith("Mbit/s"):
            finite = values[np.isfinite(values)]
            finite = finite[finite >= 0.0]
            bits_per_mbit = float(1e6)
            if finite.size == 0:
                return (values * bits_per_mbit, "bit/s", bits_per_mbit)
            max_bits = float(np.max(finite)) * bits_per_mbit
            unit_name, unit_scale = self._select_data_unit(max_bits)
            scale = bits_per_mbit / unit_scale
            return (values * scale, f"{unit_name}/s", scale)
        if not unit_text.startswith("Gbit/"):
            return (values, unit_text, 1.0)
        suffix = unit_text[len("Gbit") :]
        bits_per_gbit = float(1e9)
        finite = values[np.isfinite(values)]
        finite = finite[finite >= 0.0]
        if finite.size == 0:
            display_unit = f"bit{suffix}"
            return (values * bits_per_gbit, display_unit, bits_per_gbit)
        max_bits = float(np.max(finite)) * bits_per_gbit
        if max_bits <= 0.0:
            display_unit = f"bit{suffix}"
            return (values * bits_per_gbit, display_unit, bits_per_gbit)
        unit_name = self._BIT_DISPLAY_UNITS[0][0]
        unit_scale = self._BIT_DISPLAY_UNITS[0][1]
        for candidate_name, candidate_scale in self._BIT_DISPLAY_UNITS:
            if max_bits >= candidate_scale:
                unit_name = candidate_name
                unit_scale = candidate_scale
        display_unit = f"{unit_name}{suffix}"
        scale = bits_per_gbit / unit_scale
        return (values * scale, display_unit, scale)

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
        display_array, display_units, display_scale = self._get_distribution_display_scaling(
            array, x_units
        )
        self._last_distribution_units = display_units
        self._last_distribution_scale = float(display_scale)
        manual_width = None
        widths = getattr(self, "_distribution_bin_widths", None)
        if isinstance(widths, dict):
            manual_width = widths.get(distribution_id)
        display_manual_width = (
            float(manual_width) * float(display_scale)
            if manual_width is not None
            else None
        )
        self._render_volume_histogram(
            plot_widget=plot,
            samples=display_array,
            manual_width=display_manual_width,
            bin_label=getattr(self, "data_volume_bin_label", None),
            empty_title=empty_title,
            full_title=full_title,
            x_label=x_label,
            x_units=display_units,
            y_label="Count",
        )
        self._update_distribution_stats(display_array, display_units, display_manual_width)

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
        bottom_label = f"{x_label} ({x_units})" if x_units else x_label
        plot_widget.setLabel("bottom", bottom_label)
        plot_widget.setLabel("left", y_label)
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
        display_scale = float(getattr(self, "_last_distribution_scale", 1.0) or 1.0)
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
        suggested_raw = (
            float(current)
            if current is not None and current > 0
            else float(self._suggest_bin_width(sample_array))
        )
        suggested = suggested_raw * display_scale
        dialog = QDialog(self)
        dialog.setWindowTitle("Configure Histogram Bin Width")
        layout = QVBoxLayout(dialog)
        unit_suffix = f" ({units})" if units else ""
        layout.addWidget(QLabel(f"Set histogram bin width{unit_suffix}."))
        spin = QDoubleSpinBox()
        spin.setDecimals(6)
        upper = max(1000.0, float(np.max(sample_array)) * display_scale * 2.0 + suggested)
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
            _apply(max(float(spin.value()) / display_scale, 1e-6))
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
            self._get_contact_result_for_data_volume() is None
            or not self._latest_access_series
            or not getattr(self, "_latest_station_rate_series", None)
            or not getattr(self, "_current_config", None)
        ):
            return None
        result = self._get_contact_result_for_data_volume()
        passes = getattr(result, "passes", None) if result is not None else None
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
            volumes.append(volume_gb)
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

