"""Tests covering the lightweight link-budget GUI wiring."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest
from PySide6.QtWidgets import QApplication, QComboBox, QLabel, QTableWidget

pytest.importorskip("cartopy")
pytest.importorskip("moderngl")
pytest.importorskip("orekit")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models import GroundStationConfig
from ui.main_window import GroundStationApp
import link_budget_math


class _MinimalGroundStationApp(GroundStationApp):
    """Override UI creation so tests can inject only the needed widgets."""

    def _build_ui(self) -> None:  # pragma: no cover - intentionally empty for tests
        pass


@pytest.fixture(scope="session", autouse=True)
def _qt_app():
    """Ensure a QApplication exists for all GUI-centric tests."""

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_station_combo_syncs_with_presets():
    app = _MinimalGroundStationApp()
    combo = QComboBox()
    app.link_budget_station_combo = combo
    app._station_presets = [
        GroundStationConfig("A", 10.0, 20.0, 100.0),
        GroundStationConfig("B", -5.0, 42.0, 50.0),
    ]

    app._refresh_link_budget_station_list()

    assert combo.count() == 2
    assert combo.itemText(0) == "A"
    assert combo.itemData(1) == 1


def test_populate_link_budget_table_updates_summary():
    app = _MinimalGroundStationApp()
    app.link_budget_table = QTableWidget()
    app.link_budget_summary_label = QLabel()

    rows = [
        link_budget_math.ParameterRow("Parameter A", "1.0", "unit"),
        link_budget_math.ParameterRow("Parameter B", "2.0", "unit"),
    ]

    app._populate_link_budget_table(rows, "Summary text")

    assert app.link_budget_table.rowCount() == 2
    assert app.link_budget_summary_label.text() == "Summary text"
    assert app.link_budget_table.item(0, 0).text() == "Parameter A"
