"""Tests covering the lightweight link-budget GUI wiring."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest
from PySide6.QtWidgets import QApplication, QComboBox, QLabel, QTableWidget

# Repo layout: <repo_root>/src, <repo_root>/tests

from cosmic_toolbox.models import GroundStationConfig
from cosmic_toolbox_gui.tabs.link_budget_tab import LinkBudgetTabMixin
from cosmic_toolbox import link_budget_math


class _MinimalLinkBudgetApp(LinkBudgetTabMixin):
    """Only the LinkBudgetTabMixin methods used by these tests."""

    def __init__(self):
        # Attributes normally created by the full UI.
        self.link_budget_station_combo = None
        self._station_presets = []
        self.link_budget_table = None
        self.link_budget_summary_label = None

    def _trigger_link_budget_recompute(self, *_args) -> None:  # pragma: no cover
        """No-op for tests (full recompute requires many UI inputs)."""
        return None


@pytest.fixture(scope="session", autouse=True)
def _qt_app():
    """Ensure a QApplication exists for all GUI-centric tests."""

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_station_combo_syncs_with_presets():
    app = _MinimalLinkBudgetApp()
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
    app = _MinimalLinkBudgetApp()
    app.link_budget_table = QTableWidget()
    app.link_budget_table.setColumnCount(3)
    app.link_budget_summary_label = QLabel()

    rows = [
        link_budget_math.ParameterRow("Parameter A", "1.0", "unit"),
        link_budget_math.ParameterRow("Parameter B", "2.0", "unit"),
    ]

    app._populate_link_budget_table(rows, "Summary text")

    assert app.link_budget_table.rowCount() == 2
    assert app.link_budget_summary_label.text() == "Summary text"
    assert app.link_budget_table.item(0, 0).text() == "Parameter A"
