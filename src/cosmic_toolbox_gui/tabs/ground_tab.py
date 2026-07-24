"""Ground station tab mixin."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtGui import QColor
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.ticker import FixedLocator

from cosmic_toolbox.models import GroundStationConfig
from cosmic_toolbox.services.station_importer import (
    StationImportError,
    load_ground_stations_from_file,
)
from cosmic_toolbox_gui.constants import (
    DEFAULT_MANUAL_STATION,
    DEFAULT_STATION_DIR,
    MAP_COASTLINE_COLOR,
    MAP_GRID_COLOR,
    MAP_LAND_COLOR,
    MAP_OCEAN_COLOR,
)


class GroundTabMixin:
    """Encapsulates the ground station tab logic."""

    def _build_ground_station_tab(self) -> QWidget:
        """Create the tab containing station import tools and manual configuration."""
        tab = QWidget()
        tab_layout = QVBoxLayout(tab)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.addWidget(self._build_import_group())
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
        map_widget = self._build_import_visual_panel()
        map_widget.setMinimumWidth(0)
        map_widget.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding
        )
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(True)
        splitter.addWidget(left_scroll)
        splitter.addWidget(map_widget)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        splitter.setSizes([420, 840])
        tab_layout.addWidget(splitter)
        return tab

    def _build_import_group(self) -> QGroupBox:
        """Create controls for importing and selecting multiple stations."""
        group = QGroupBox("Station List")
        layout = QVBoxLayout(group)
        method_row = QHBoxLayout()
        method_row.addWidget(QLabel("Input Method:"))
        combo = getattr(self, "station_method_combo", None)
        if combo is None:
            combo = self._create_station_method_combo()
        self.station_method_combo = combo
        method_row.addWidget(combo, stretch=1)
        layout.addLayout(method_row)
        self.import_button = QPushButton("Import stations…")
        self.import_button.clicked.connect(self._handle_import_stations)  # type: ignore[arg-type]
        layout.addWidget(self.import_button)
        self.manual_form = self._build_manual_station_form()
        layout.addWidget(self.manual_form)
        self.manual_form.setVisible(False)
        button_row = QHBoxLayout()
        self.select_all_button = QPushButton("Select all")
        self.select_all_button.clicked.connect(
            lambda: self._set_station_table_checks(True)
        )  # type: ignore[arg-type]
        button_row.addWidget(self.select_all_button)
        self.clear_selection_button = QPushButton("Clear selection")
        self.clear_selection_button.clicked.connect(
            lambda: self._set_station_table_checks(False)
        )  # type: ignore[arg-type]
        button_row.addWidget(self.clear_selection_button)
        layout.addLayout(button_row)
        self.station_table = QTableWidget(0, 7)
        self.station_table.setHorizontalHeaderLabels(
            ["Use", "Name", "Latitude", "Longitude", "Altitude (m)", "Supplier", "Mask"]
        )
        self.station_table.verticalHeader().setVisible(False)
        self.station_table.setAlternatingRowColors(True)
        self.station_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.station_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self.station_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        header = self.station_table.horizontalHeader()
        header.setStretchLastSection(True)
        header.setDefaultSectionSize(110)
        self.station_table.itemChanged.connect(self._handle_station_table_item_changed)  # type: ignore[arg-type]
        self.station_table.itemSelectionChanged.connect(
            self._update_remove_button_state
        )
        self.station_table.itemDoubleClicked.connect(
            self._handle_station_table_double_clicked
        )  # type: ignore[arg-type]
        self.station_table.setMinimumWidth(0)
        self.station_table.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding
        )
        layout.addWidget(self.station_table, stretch=1)
        remove_row = QHBoxLayout()
        remove_row.addStretch(1)
        self.remove_station_button = QPushButton("Remove selected")
        self.remove_station_button.setEnabled(False)
        self.remove_station_button.clicked.connect(self._handle_remove_selected_station)  # type: ignore[arg-type]
        remove_row.addWidget(self.remove_station_button)
        layout.addLayout(remove_row)
        self.ground_station_recompute_hint_label = QLabel(
            "After a cached run, use the button below to recompute accesses for the current station selection without a full rerun."
        )
        self.ground_station_recompute_hint_label.setWordWrap(True)
        layout.addWidget(self.ground_station_recompute_hint_label)
        self.ground_station_recompute_button = QPushButton("Recompute Accesses")
        self.ground_station_recompute_button.setEnabled(False)
        self.ground_station_recompute_button.clicked.connect(  # type: ignore[attr-defined]
            lambda: self._refresh_cached_access_from_inputs(silent=False)
        )
        layout.addWidget(self.ground_station_recompute_button)
        self._update_station_controls_enabled_state()
        return group

    def _create_station_method_combo(self):
        combo = getattr(self, "station_method_combo", None)
        if combo is None:
            combo = QComboBox()
            combo.addItems(["Import CSV/XLSX", "Manual entry"])
            combo.setCurrentIndex(0)
            combo.currentIndexChanged.connect(self._handle_station_method_changed)  # type: ignore[arg-type]
        return combo

    def _build_manual_station_form(self) -> QWidget:
        """Return the manual entry form and add button."""
        form_widget = QWidget()
        form_layout = QFormLayout(form_widget)
        self.manual_name_input = QLineEdit()
        self.manual_lat_input = QDoubleSpinBox()
        self.manual_lat_input.setRange(-90.0, 90.0)
        self.manual_lat_input.setDecimals(6)
        self.manual_lon_input = QDoubleSpinBox()
        self.manual_lon_input.setRange(-180.0, 180.0)
        self.manual_lon_input.setDecimals(6)
        self.manual_alt_input = QDoubleSpinBox()
        self.manual_alt_input.setRange(-500.0, 10000.0)
        self.manual_alt_input.setSuffix(" m")
        add_button = QPushButton("Add manual station")
        add_button.clicked.connect(self._handle_add_manual_station)  # type: ignore[arg-type]
        form_layout.addRow("Name:", self.manual_name_input)
        form_layout.addRow("Latitude (deg):", self.manual_lat_input)
        form_layout.addRow("Longitude (deg):", self.manual_lon_input)
        form_layout.addRow("Altitude:", self.manual_alt_input)
        form_layout.addRow(add_button)
        self._prefill_manual_inputs()
        return form_widget

    def _prefill_manual_inputs(self) -> None:
        """Populate manual entry controls with the default station values."""
        if not hasattr(self, "manual_name_input"):
            return
        self.manual_name_input.setText(DEFAULT_MANUAL_STATION.name)
        self.manual_lat_input.setValue(DEFAULT_MANUAL_STATION.latitude_deg)
        self.manual_lon_input.setValue(DEFAULT_MANUAL_STATION.longitude_deg)
        self.manual_alt_input.setValue(DEFAULT_MANUAL_STATION.altitude_m)

    def _build_import_visual_panel(self) -> QWidget:
        """Create the world-map visualization panel for imported stations."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        figure = Figure(figsize=(5, 4), tight_layout=True)
        figure.patch.set_facecolor(MAP_OCEAN_COLOR)
        self.map_canvas = FigureCanvasQTAgg(figure)
        self.map_canvas.setStyleSheet("background-color: #000000;")
        self.map_canvas.setMinimumWidth(0)
        self.map_axes = figure.add_subplot(111, projection=self.map_projection)
        self._map_station_artists = []
        self._configure_map_axes(self.map_axes)
        layout.addWidget(self.map_canvas, stretch=1)
        self.map_status_label = QLabel("No stations imported.")
        self.map_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.map_status_label)
        return panel

    def _configure_map_axes(self, axes) -> None:
        """Render the static map background on the provided matplotlib axes."""
        if axes is None:
            return
        ax = axes
        ax.clear()
        ax.set_facecolor(MAP_OCEAN_COLOR)
        ax.patch.set_edgecolor(MAP_COASTLINE_COLOR)
        ax.set_global()
        from cartopy import feature as cfeature

        land = cfeature.NaturalEarthFeature(
            "physical",
            "land",
            "110m",
            facecolor=MAP_LAND_COLOR,
            edgecolor=MAP_COASTLINE_COLOR,
        )
        ocean = cfeature.NaturalEarthFeature(
            "physical",
            "ocean",
            "110m",
            facecolor=MAP_OCEAN_COLOR,
            edgecolor=MAP_OCEAN_COLOR,
        )
        ax.add_feature(ocean, zorder=0)
        ax.add_feature(land, zorder=1)
        ax.add_feature(
            cfeature.COASTLINE.with_scale("110m"),
            edgecolor=MAP_COASTLINE_COLOR,
            linewidth=0.6,
            zorder=2,
        )
        ax.add_feature(
            cfeature.BORDERS.with_scale("110m"),
            edgecolor=MAP_COASTLINE_COLOR,
            linewidth=0.3,
            alpha=0.4,
            zorder=2,
        )
        ax.set_extent([-180, 180, -90, 90], crs=self.map_projection)
        gridlines = ax.gridlines(
            draw_labels=False,
            linewidth=0.3,
            color=MAP_GRID_COLOR,
            alpha=0.2,
            linestyle="--",
        )
        gridlines.xlocator = FixedLocator(np.arange(-180, 181, 60))  # type: ignore[name-defined]
        gridlines.ylocator = FixedLocator(np.arange(-90, 91, 30))  # type: ignore[name-defined]
        if self.map_canvas:
            self.map_canvas.draw_idle()

    def _clear_map_station_artists(self) -> None:
        """Remove any previously drawn station markers."""
        for artist in getattr(self, "_map_station_artists", []):
            artist.remove()
        self._map_station_artists = []

    def _update_station_map(self) -> None:
        """Plot imported stations on the world map."""
        if getattr(self, "map_axes", None) is None or getattr(self, "map_canvas", None) is None:
            return
        self._clear_map_station_artists()
        total = len(self._station_presets)
        if total == 0:
            if getattr(self, "map_status_label", None):
                self.map_status_label.setText("No stations imported.")
            self.map_canvas.draw_idle()
            return
        enabled_indices = set(self._get_enabled_station_indices())
        for idx, station in enumerate(self._station_presets):
            is_enabled = idx in enabled_indices
            color = "red" if is_enabled else "#777777"
            marker = self.map_axes.plot(
                station.longitude_deg,
                station.latitude_deg,
                "o",
                color=color,
                markersize=6,
                transform=self.map_projection,
            )[0]
            label = self.map_axes.text(
                station.longitude_deg + 2.0,
                station.latitude_deg + 2.0,
                station.name,
                fontsize=8,
                color="white",
                ha="left",
                va="bottom",
                alpha=0.9,
                transform=self.map_projection,
            )
            self._map_station_artists.extend([marker, label])
        if getattr(self, "map_status_label", None):
            self.map_status_label.setText(
                f"{total} station(s) loaded | {len(enabled_indices)} active"
            )
        self.map_canvas.draw_idle()

    def _update_station_controls_enabled_state(self) -> None:
        """Enable or disable station selection controls based on context."""
        has_rows = bool(self.station_table and self.station_table.rowCount() > 0)
        controls_enabled = has_rows
        if self.station_table:
            self.station_table.setEnabled(controls_enabled)
        for button in (self.select_all_button, self.clear_selection_button):
            if button:
                button.setEnabled(has_rows)
        self._update_remove_button_state()

    def _set_station_table_checks(self, checked: bool) -> None:
        """Select or clear all station rows."""
        if not self.station_table:
            return
        self.station_table.blockSignals(True)
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        for row in range(self.station_table.rowCount()):
            item = self.station_table.item(row, 0)
            if item is not None:
                item.setCheckState(state)
        self.station_table.blockSignals(False)
        self._update_station_map()
        self._mark_derived_outputs_stale(
            "Derived outputs stale. Refresh access from cached trajectory to apply the updated station selection."
        )

    def _handle_station_table_item_changed(self, item: QTableWidgetItem) -> None:
        """React to checkbox toggles inside the station table."""
        if item.column() != 0:
            return
        self._update_station_map()
        self._update_remove_button_state()
        self._mark_derived_outputs_stale(
            "Derived outputs stale. Refresh access from cached trajectory to apply the updated station selection."
        )

    def _handle_station_table_double_clicked(self, item: QTableWidgetItem) -> None:
        """Open the editor dialog for the double-clicked station entry."""
        if item is None:
            return
        self._edit_station_table_row(item.row())

    def _edit_station_table_row(self, row: int) -> None:
        """Launch a dialog to edit the selected station details."""
        if not (0 <= row < len(self._station_presets)):
            return
        station = self._station_presets[row]
        updated = self._open_station_editor_dialog(station)
        if updated is None:
            return
        self._station_presets[row] = updated
        self._populate_station_table(self._station_presets)
        if self.station_table:
            self.station_table.selectRow(row)
        self._update_station_map()
        self._update_station_controls_enabled_state()
        self._mark_derived_outputs_stale(
            "Derived outputs stale. Refresh access from cached trajectory to apply the edited station definition."
        )

    def _open_station_editor_dialog(
        self, station: GroundStationConfig
    ) -> GroundStationConfig | None:
        """Display a modal dialog for editing station parameters."""
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Edit Station – {station.name}")
        layout = QVBoxLayout(dialog)
        form_layout = QFormLayout()
        name_input = QLineEdit(station.name)
        lat_input = QDoubleSpinBox()
        lat_input.setRange(-90.0, 90.0)
        lat_input.setDecimals(6)
        lat_input.setValue(station.latitude_deg)
        lon_input = QDoubleSpinBox()
        lon_input.setRange(-180.0, 180.0)
        lon_input.setDecimals(6)
        lon_input.setValue(station.longitude_deg)
        alt_input = QDoubleSpinBox()
        alt_input.setRange(-500.0, 10000.0)
        alt_input.setDecimals(2)
        alt_input.setSuffix(" m")
        alt_input.setValue(station.altitude_m)

        # Horizon mask picker
        mask_path_state: list[str | None] = [
            getattr(station, "horizon_mask_path", None)
        ]
        mask_label = QLabel(
            Path(mask_path_state[0]).name if mask_path_state[0] else "None"
        )
        mask_label.setWordWrap(True)

        mask_browse = QPushButton("Browse…")
        mask_clear = QPushButton("Clear")

        def _browse_mask() -> None:
            resources = (
                Path(__file__).resolve().parents[2]
                / "cosmic_toolbox"
                / "resources"
                / "horizon_masks"
            )
            start = str(resources) if resources.exists() else str(Path.home())
            fname, _ = QFileDialog.getOpenFileName(
                dialog, "Select Horizon Mask CSV", start, "CSV files (*.csv);;All files (*.*)"
            )
            if fname:
                mask_path_state[0] = fname
                mask_label.setText(Path(fname).name)

        def _clear_mask() -> None:
            mask_path_state[0] = None
            mask_label.setText("None")

        mask_browse.clicked.connect(_browse_mask)  # type: ignore[arg-type]
        mask_clear.clicked.connect(_clear_mask)    # type: ignore[arg-type]
        mask_row = QHBoxLayout()
        mask_row.addWidget(mask_label, stretch=1)
        mask_row.addWidget(mask_browse)
        mask_row.addWidget(mask_clear)

        form_layout.addRow("Name:", name_input)
        form_layout.addRow("Latitude (deg):", lat_input)
        form_layout.addRow("Longitude (deg):", lon_input)
        form_layout.addRow("Altitude:", alt_input)
        form_layout.addRow("Horizon mask:", mask_row)
        layout.addLayout(form_layout)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        result = dialog.exec()
        if result != QDialog.DialogCode.Accepted:
            return None
        return GroundStationConfig(
            name=name_input.text().strip() or station.name,
            latitude_deg=float(lat_input.value()),
            longitude_deg=float(lon_input.value()),
            altitude_m=float(alt_input.value()),
            horizon_mask_path=mask_path_state[0],
            supplier=getattr(station, "supplier", ""),
        )

    def _populate_station_table(self, stations: list[GroundStationConfig]) -> None:
        """Fill the station table with imported presets."""
        if not self.station_table:
            return
        self.station_table.blockSignals(True)
        self.station_table.setRowCount(len(stations))
        for row, station in enumerate(stations):
            use_item = QTableWidgetItem()
            use_item.setFlags(
                Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled
            )
            use_item.setCheckState(Qt.CheckState.Checked)
            self.station_table.setItem(row, 0, use_item)
            name_item = QTableWidgetItem(station.name)
            name_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            self.station_table.setItem(row, 1, name_item)
            lat_item = QTableWidgetItem(f"{station.latitude_deg:.2f}")
            lat_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            lat_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            self.station_table.setItem(row, 2, lat_item)
            lon_item = QTableWidgetItem(f"{station.longitude_deg:.2f}")
            lon_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            lon_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            self.station_table.setItem(row, 3, lon_item)
            alt_item = QTableWidgetItem(f"{station.altitude_m:.0f}")
            alt_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            alt_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            self.station_table.setItem(row, 4, alt_item)
            supplier_item = QTableWidgetItem(getattr(station, "supplier", ""))
            supplier_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            self.station_table.setItem(row, 5, supplier_item)
            mask_path = getattr(station, "horizon_mask_path", None)
            has_mask = bool(mask_path)
            mask_item = QTableWidgetItem("✓" if has_mask else "✗")
            mask_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            mask_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            mask_item.setForeground(
                QColor(80, 200, 80) if has_mask else QColor(220, 60, 60)
            )
            self.station_table.setItem(row, 6, mask_item)
        self.station_table.blockSignals(False)
        self._update_station_controls_enabled_state()

    def _get_enabled_station_indices(self) -> list[int]:
        """Return the table row indices that are checked for analysis."""
        if not self.station_table:
            return []
        indices: list[int] = []
        for row in range(
            min(self.station_table.rowCount(), len(self._station_presets))
        ):
            item = self.station_table.item(row, 0)
            if item and item.checkState() == Qt.CheckState.Checked:
                indices.append(row)
        return indices

    def _update_remove_button_state(self) -> None:
        """Enable/disable the remove button based on selection."""
        if not getattr(self, "remove_station_button", None):
            return
        has_rows = bool(self.station_table and self.station_table.rowCount() > 0)
        has_selection = bool(self.station_table and self.station_table.selectedItems())
        self.remove_station_button.setEnabled(has_rows and has_selection)

    def _handle_remove_selected_station(self) -> None:
        """Remove the currently selected station from the table/presets."""
        if not self.station_table:
            return
        row = self.station_table.currentRow()
        if row < 0:
            return
        remove_index = row
        if 0 <= remove_index < len(self._station_presets):
            self._station_presets.pop(remove_index)
        self.station_table.blockSignals(True)
        self.station_table.removeRow(row)
        self.station_table.blockSignals(False)
        self._update_station_controls_enabled_state()
        self._update_station_map()
        self._refresh_link_budget_station_list(trigger_auto_recompute=False)
        self._mark_derived_outputs_stale(
            "Derived outputs stale. Refresh access from cached trajectory to apply the removed station."
        )

    def _collect_active_stations_for_run(self) -> list[GroundStationConfig]:
        """Resolve which ground stations should be included in the analysis run."""
        if not self._station_presets:
            QMessageBox.warning(
                self,
                "No Stations Available",
                "Import stations or add them manually before running the analysis.",
            )
            return []
        indices = self._get_enabled_station_indices()
        if not indices:
            QMessageBox.warning(
                self,
                "No Stations Enabled",
                "Enable at least one station in the list (Use column) to run the analysis.",
            )
            return []
        return [
            self._station_presets[i]
            for i in indices
            if 0 <= i < len(self._station_presets)
        ]

    def _handle_import_stations(self) -> None:
        """Let the user pick a CSV/Excel file and load presets."""
        default_dir = (
            DEFAULT_STATION_DIR if DEFAULT_STATION_DIR.exists() else Path.home()
        )
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select Ground Station File",
            str(default_dir),
            "CSV/Excel Files (*.csv *.xlsx *.xls)",
        )
        if not filename:
            return
        try:
            stations = load_ground_stations_from_file(filename)
        except StationImportError as exc:
            QMessageBox.warning(self, "Import Failed", str(exc))
            return
        existing_count = len(self._station_presets)
        enabled_indices = set(self._get_enabled_station_indices())
        self._station_presets.extend(stations)
        self._populate_station_table(self._station_presets)
        if self.station_table and existing_count > 0:
            self.station_table.blockSignals(True)
            for row in range(existing_count):
                if row not in enabled_indices:
                    item = self.station_table.item(row, 0)
                    if item is not None:
                        item.setCheckState(Qt.CheckState.Unchecked)
            self.station_table.blockSignals(False)
        self._update_station_map()
        self._refresh_link_budget_station_list(trigger_auto_recompute=False)
        self._mark_derived_outputs_stale(
            "Derived outputs stale. Refresh access from cached trajectory to apply the imported station set."
        )
        QMessageBox.information(
            self,
            "Stations Imported",
            (
                f"Added {len(stations)} station(s) from {Path(filename).name}. "
                f"Total loaded: {len(self._station_presets)}."
            ),
        )

    def _handle_station_method_changed(self, index: int) -> None:
        """Update UI controls based on the selected station input method."""
        use_manual = index == 1
        if getattr(self, "import_button", None):
            self.import_button.setEnabled(not use_manual)
        if getattr(self, "manual_form", None):
            self.manual_form.setVisible(use_manual)
        if use_manual:
            self._prefill_manual_inputs()
        self._update_station_controls_enabled_state()

    def _handle_add_manual_station(self) -> None:
        """Add a manually entered station into the presets/table."""
        name = self.manual_name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Invalid station", "Please enter a station name.")
            return
        station = GroundStationConfig(
            name=name,
            latitude_deg=self.manual_lat_input.value(),
            longitude_deg=self.manual_lon_input.value(),
            altitude_m=self.manual_alt_input.value(),
        )
        self._station_presets.append(station)
        self._populate_station_table(self._station_presets)
        self._prefill_manual_inputs()
        self._refresh_link_budget_station_list(trigger_auto_recompute=False)
        self._mark_derived_outputs_stale(
            "Derived outputs stale. Refresh access from cached trajectory to apply the added station."
        )

