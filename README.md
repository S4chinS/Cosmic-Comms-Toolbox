# Cosmic Comms Toolbox

Orekit-backed satellite ground-station access, link-budget, and data-volume
analysis. The repository ships two packages from a single source tree:

- `cosmic_toolbox` — pure analysis core (no Qt, no plotting).
- `cosmic_toolbox_gui` — PySide6 desktop app built on top of the toolbox.

The `ToolboxFacade` (`cosmic_toolbox.facade`) is the recommended stable
surface — it exposes access analysis, the cached fast path, link budget /
data volume, orbit summaries, and the shared inputs (canonical link-budget
defaults, ground stations, antenna LUTs, ITU losses). You can run analyses
without installing the GUI.

## Project Layout

```
.
├── pyproject.toml
├── configs/                          # Example YAML scenario inputs
├── tests/
│   ├── toolbox/                      # Headless analysis tests
│   └── gui/                          # PySide6 / GroundStationApp tests (auto-skip without PySide6)
└── src/
    ├── cosmic_toolbox/               # Analysis core
    │   ├── facade.py                 # ToolboxFacade — the public surface
    │   ├── models/                   # configs.py, results.py, …
    │   ├── services/                 # access_analysis, antenna_pattern,
    │   │                             #   cached_access_recompute, scenario_*,
    │   │                             #   pass_geometry, station_importer,
    │   │                             #   progress.py, cancel.py
    │   ├── analyses/                 # Pure helpers lifted from the GUI
    │   ├── tools/                    # link_budget_profile, orbit_summary_npz
    │   ├── link_budget_defaults.py   # Canonical RF defaults loader (one source)
    │   ├── orbit_utils.py            # SSO inclination + Keplerian relations
    │   ├── resources/                # Bundled station CSVs + link budget defaults
    │   ├── paths.py                  # resources root, repo root, outputs helpers
    │   └── cli/                      # `cosmic-toolbox` console subcommands
    └── cosmic_toolbox_gui/           # PySide6 adapter
        ├── app.py                    # `cosmic-toolbox-gui` entry point
        ├── main_window.py            # GroundStationApp (slimmed)
        ├── tabs/                     # Tab mixins (Ground/Mission/Vis/LB/OrbitSummary)
        ├── widgets/                  # RangeSlider, …
        ├── workers/                  # AnalysisWorker (QThread → ToolboxFacade)
        ├── opengl/                   # ModernGL globe renderer
        └── resources/                # GUI-only textures and icons
```

See [`docs/architecture.md`](docs/architecture.md) for a deeper tour of the
ports-and-adapters layout.

## Installation

Headless install (analysis only — no Qt, cartopy, or moderngl):

```bash
pip install -e .
```

Full install (GUI + plotting):

```bash
pip install -e ".[gui]"
```

Tests, ruff, and mypy:

```bash
pip install -e ".[dev]"
```

The `orekitdata` package (Orekit's bundled IERS / EOP data) is not on PyPI;
install it directly once per environment:

```bash
pip install git+https://gitlab.orekit.org/orekit-labs/python-package.git#subdirectory=orekitdata
```

## Running

GUI:

```bash
cosmic-toolbox-gui          # after pip install -e ".[gui]"
# or
python -m cosmic_toolbox_gui.app
```

CLI:

```bash
cosmic-toolbox --help
cosmic-toolbox link-budget-profile
cosmic-toolbox export-link-budget configs/link_budget_example.yaml -o link_budget.xlsx
```

Calling the toolbox from a script or notebook:

```python
from cosmic_toolbox import ToolboxFacade, AnalysisConfig, GroundStationConfig

facade = ToolboxFacade()
result = facade.run_access(config, stations=[GroundStationConfig(...)])
```

## Coding Standards

- **PEP 8** for formatting, imports, and naming.
- **Google-style docstrings** for all public modules, classes, and functions.
- `cosmic_toolbox` must never import PySide6, pyqtgraph, matplotlib, cartopy,
  or moderngl — those live exclusively in `cosmic_toolbox_gui`.

## Ground Station Importer

- Use the **Import stations...** button inside the *Ground Station* group to
  load presets from any CSV or Excel file containing `name`, `latitude`,
  `longitude`, and `altitude` columns.
- A sample CSV ships inside the package:
  `cosmic_toolbox/resources/groundstation_list/ground_stations.csv`.

## Visualization

- The right-side panel uses PyQtGraph for interactive plots:
  - A Gantt-style bar view of each access window along the scenario timeline.
  - A pass-duration histogram whose bin width equals twice the configured
    sample time.

## ModernGL Globe

- The Mission and Visualization tabs render their globes with the ModernGL-based
  `GlobeWidget` at `cosmic_toolbox_gui/opengl/globe_widget.py`.
- Ensure the host GPU exposes at least OpenGL 3.3. When running headless (CI),
  set `QT_QPA_PLATFORM=offscreen` to let Qt create an off-screen context.

## License

This project is licensed under the MIT License — see the LICENSE file for details.
