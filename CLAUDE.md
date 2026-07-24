# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install (headless — no Qt, no plotting)
pip install -e .

# Install with GUI dependencies
pip install -e ".[gui]"

# Install with dev tools (pytest, ruff, mypy)
pip install -e ".[dev]"

# Orekit Earth data — must be installed separately
pip install git+https://gitlab.orekit.org/orekit-labs/python-package.git#subdirectory=orekitdata

# Run tests
pytest

# Run a single test
pytest tests/toolbox/test_facade.py::test_cancel_token_round_trip

# Lint / format
ruff check src/ tests/
ruff format src/ tests/

# Type check
mypy src/

# Launch GUI
cosmic-toolbox-gui
# or
python -m cosmic_toolbox_gui.app

# CLI (benchmark / link-budget profiling)
cosmic-toolbox --help
```

## Architecture

### Two packages, one source tree

`cosmic_toolbox` — pure analysis core. No Qt, no matplotlib, no cartopy, no moderngl. This constraint is enforced: adding any GUI import here breaks the headless install.

`cosmic_toolbox_gui` — PySide6 desktop application that wraps the core via `ToolboxFacade`.

### Data flow

```
AnalysisConfig
    │
    ▼
access_analysis.run_access_analysis()   ← Orekit propagation (JVM, expensive)
    │
    ├── PropagatedEphemeris             ← orbit-only output; reusable
    │       state_vectors, ground_track, orbital element time series
    │
    └── AnalysisResult                  ← adds station-dependent outputs
            passes, summaries, per-station elevation/azimuth/range series
                │
                ▼
        scenario_data_volume             ← link budget per pass
                │
                ▼
        ArchitectureResult               ← data volume, SNR, modcod per pass
```

**The fast path** — if a `PropagatedEphemeris` is already cached (loaded from an NPZ package), `cached_access_recompute.derive_access_results_from_ephemeris()` skips Orekit entirely and recomputes all station geometry using vectorised NumPy. Changing stations, elevation masks, or link-budget params never needs to re-propagate.

**The slow path** — `access_analysis._sample_elevation_time_series()` runs a per-timestep Orekit integration loop, which is the dominant runtime cost (~85–95% of a fresh run).

### ToolboxFacade

`src/cosmic_toolbox/facade.py` is the single public API. All long-running methods accept `progress: ProgressReporter` and `cancel: CancelToken`:

- `run_access()` — full fresh propagation
- `derive_access_from_ephemeris()` — fast cached path
- `run_link_budget()` / `run_data_volume()`
- `build_orbit_summary()`

### ProgressReporter / CancelToken

Defined in `services/progress.py` and `services/cancel.py`. These are ports: the core knows only the `ProgressReporter` protocol (`report(fraction, message)`). The GUI wires `QtProgressReporter` (emits Qt signals); the CLI uses `NullProgress`. `CancelToken.check()` raises `AnalysisCancelled` — propagation loops call it once per timestep.

### Orekit JVM

`services/orekit_vm.ensure_orekit_vm_started()` must be called once before any Orekit Java class is imported. It is idempotent after the first call. Worker threads must call `orekit.getVMEnv().attachCurrentThread()` before touching Orekit objects. Extra JARs must be passed to `ensure_orekit_vm_started(extra_jars=...)` — they cannot be added after JVM startup.

### Scenario package format (cached trajectories)

A saved scenario is two files: `<name>.npz` + `<name>.toolbox.json`.

- **NPZ**: compressed NumPy arrays — `eci_pos_km`, `eci_vel_km_s`, `ecef_pos_km`, `ecef_vel_km_s`, attitude body axes (`body_{x,y,z}_ecef`), ground track lat/lon, orbital element time series.
- **JSON sidecar** (`schema_version: 2`): `settings` (AnalysisConfig), `stations` (list of GroundStationConfig), `enabled_station_names`.

Read/write via `scenario_package_io._read_ephemeris_npz` / `_write_ephemeris_npz`. Schema version 1 (OEM + fat JSON) is read-only legacy.

### Tests

`tests/toolbox/` — headless tests, always run.  
`tests/gui/` — PySide6 tests, auto-skipped if PySide6 is not installed.

The `conftest.py` in each directory handles Orekit VM startup and shared fixtures.
