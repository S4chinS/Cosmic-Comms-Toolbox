# Cosmic Comms Toolbox - Codebase Guide

> Audience: an autonomous agent (or new engineer) that needs to understand what
> this toolbox does, where each piece lives, and the work patterns to follow
> when extending it. This document is intentionally low-level. Pair it with the
> higher-level [`docs/architecture.md`](architecture.md).

---

## 1. What this toolbox does

The Cosmic Comms Toolbox is a satellite communications mission-analysis tool. It
answers questions like:

- When can a satellite see a set of ground stations, and for how long? (access /
  contact analysis)
- How good is the RF link during each contact, and which modulation/coding
  (MODCOD) closes? (link budget)
- How much data can be downlinked across a scenario? (data volume)
- How do the orbital elements drift over the scenario? (orbit summary)
- Does the downlink comply with the ITU surface power-flux-density (PFD) mask?

It is built around **Orekit** (a Java orbital-mechanics library accessed from
Python through JCC/`orekit`), **NumPy** vectorised geometry, and the **ITU-R**
atmospheric propagation models (`itur`).

The same analysis core is consumed by two frontends:

| Frontend | Package | Tech | Status |
| --- | --- | --- | --- |
| Desktop GUI | `cosmic_toolbox_gui` | PySide6 + pyqtgraph + matplotlib + ModernGL | Primary, full-featured |
| CLI | `cosmic_toolbox.cli` | argparse | Batch / benchmarking |

---

## 2. Repository layout (top level)

```
.
├── pyproject.toml            # packaging, deps, entry points, pytest config
├── README.md
├── CLAUDE.md                 # agent guidance (commands + architecture summary)
├── configs/                  # example YAML scenario inputs
├── data/                     # gitignored scratch/output
├── docs/                     # architecture.md, this guide, drag/globe notes
├── tests/                    # toolbox/ (headless), gui/ (PySide6)
└── src/
    ├── cosmic_toolbox/       # pure analysis core (NO Qt / plotting deps)
    └── cosmic_toolbox_gui/   # PySide6 desktop app
```

### The golden import rule

`cosmic_toolbox` **must never** import `PySide6`, `pyqtgraph`, `matplotlib`,
`cartopy`, or `moderngl`. Those belong only to the `_gui` package. CI proves
this by doing a headless `pip install -e .` (no GUI extras). If you add a GUI
import to the core, the headless install breaks. Put pure compute in the
core; put adapter code in the frontend.

### Installation extras (`pyproject.toml`)

- base: `numpy, scipy, pandas, openpyxl, orekit, itur, astropy, PyYAML, tqdm, Pillow`
- `[gui]`: `PySide6, pyqtgraph, matplotlib, cartopy, moderngl, imageio, imageio-ffmpeg`
- `[dev]`: `pytest, ruff, mypy`
- Orekit Earth data is separate: `pip install orekitdata` from its Git mirror.

### Console entry points

| Command | Target |
| --- | --- |
| `cosmic-toolbox` | `cosmic_toolbox.cli:main` (subcommands `link-budget-profile`, `export-link-budget`) |
| `cosmic-toolbox-gui` | `cosmic_toolbox_gui.app:main` |

---

## 3. The two execution paths (most important mental model)

Everything in the toolbox is structured around a **slow path** and a **fast
path**. Understanding this split explains most of the architecture.

```
                       AnalysisConfig + [GroundStationConfig]
                                     │
            ┌────────────────────────┴────────────────────────┐
            │ SLOW PATH (fresh run)                            │ FAST PATH (cached)
            ▼                                                  ▼
  access_analysis.run_access_analysis()          (already have a PropagatedEphemeris,
   - starts Orekit JVM                            e.g. loaded from an NPZ package)
   - per-timestep propagation loop (JVM)                      │
   - collects ONLY state vectors,                             │
     ground track, orbital elements                          │
            │                                                  │
            ▼                                                  │
     PropagatedEphemeris  ───────────────────────────────────►│
     (orbit-only, reusable cache)                              │
            │                                                  ▼
            └──────────────►  cached_access_recompute.derive_access_results_from_ephemeris()
                              - PURE NumPy, no JVM
                              - per-station elevation/azimuth/range/rates
                              - pass (AOS/LOS) detection
                                     │
                                     ▼
                              AnalysisResult  (passes, summaries, series, + ephemeris)
                                     │
                          ┌──────────┼─────────────┬───────────────┐
                          ▼          ▼             ▼               ▼
                  scenario_data_volume  orbit_summary   link budget (per pass)   PFD
```

Key consequence: **the expensive Orekit propagation runs only when the orbit or
scenario changes.** Changing stations, elevation masks, link-budget parameters,
or comms pointing only re-runs the vectorised NumPy `cached_access_recompute`
path. The GUI tracks this with two dirty flags (`_is_dirty` =>
re-propagate; `_derived_outputs_stale` => recompute cached only).

The slow path itself *also* calls `derive_access_results_from_ephemeris` after
propagation, so pass geometry is computed in exactly one place (no duplicate
in-loop sampling).

---

## 4. Core package: `cosmic_toolbox`

### 4.1 Public surface: `ToolboxFacade` (`facade.py`)

`facade.py` is the single, stable, GUI-agnostic entry point. It is intentionally
thin: every method lazily imports the relevant service and forwards to it, while
threading a `ProgressReporter` and `CancelToken` through. Methods:

| Method | Delegates to | Returns |
| --- | --- | --- |
| `run_access(config, stations, *, progress, cancel)` | `services.access_analysis.run_access_analysis` | `AnalysisResult` (slow path) |
| `derive_access_from_ephemeris(*, ephemeris, config, stations)` | `services.cached_access_recompute.derive_access_results_from_ephemeris` | `DerivedAccessResult` (fast path) |
| `run_link_budget(inputs)` | `tools.link_budget_profile.run_pipeline` | timing dict |
| `run_data_volume(*, result, architecture, stations, scenario_start_time, ...)` | `services.scenario_data_volume.ScenarioDataVolumeEvaluator` | `ArchitectureResult` |
| `build_orbit_summary(*, times_s, instantaneous)` | `analyses.orbit_summary` | `{averaged, meta, stats}` |
| `pfd_compliance_backoff_db(**kwargs)` | `analyses.pfd.compliance_backoff_db` | `float` |

`cosmic_toolbox/__init__.py` re-exports the facade plus all config/result
dataclasses and the progress/cancel ports, so callers do
`from cosmic_toolbox import ToolboxFacade, AnalysisConfig, ...`.

### 4.2 Ports: progress and cancellation

These are the abstraction boundary between the core and any caller. The core
only knows the protocols; each frontend supplies an adapter.

- `services/progress.py`
  - `ProgressReporter` (`@runtime_checkable` Protocol): `report(fraction: float, message: str | None)` where `fraction` is `[0, 1]`.
  - `NullProgress`: no-op default.
  - `progress_from_legacy_callback` / `legacy_callback_from_progress`: adapters
    to/from the legacy `progress_callback(percent: float)` (0-100) style still
    accepted by `run_access_analysis`.
- `services/cancel.py`
  - `CancelToken`: thread-safe flag with `cancel()`, `.cancelled`, and
    `check()` (raises `AnalysisCancelled`). Propagation loops call
    `cancel.check()` once per timestep.

Adapters:
- GUI: `cosmic_toolbox_gui.workers.QtProgressReporter` (wraps a Qt signal).

### 4.3 Models (`models/`)

`models/configs.py` (inputs):
- `GroundStationConfig` — `name, latitude_deg, longitude_deg, altitude_m,
  horizon_mask_path (optional CSV), supplier`.
- `OrbitConfig` — classical Keplerian elements (`semi_major_axis_km,
  eccentricity, inclination_deg, raan_deg, arg_perigee_deg, mean_anomaly_deg`).
- `PropagationConfig` — `propagator_type ("keplerian"|"brouwer_lyddane"),
  min_elevation_deg, sample_step_seconds, attitude_mode ("prograde"|"nadir"|
  "zenith"), contact_elevation_deg, enable_contact_attitude_switching,
  comms_pointing_mode, comms_pointing_aoa_limit_deg, sensor_fov_cone_total_deg,
  max_off_nadir_slew_deg, drag_coefficient (BrouwerLyddane M2 secular drag
  term), apply_mean_orbit_correction (BrouwerLyddane closed-form mean-orbit
  SMA correction)`.
- `AnalysisOptions` — `compute_ground_station_passes: bool`.
- `ScenarioConfig` — `start_time, end_time` (datetimes).
- `AnalysisConfig` — aggregates `ground_station, orbit, propagation, scenario,
  options`.

`models/results.py` (outputs):
- `StateVectorSample`, `GroundTrackPoint` — legacy per-sample dataclasses (kept
  as constructor shims).
- `PassStatistic` — one access window: `index, aos, los, duration_minutes,
  max_elevation_deg, max_sc_slew_rate_deg_s, station_name`.
- `AnalysisSummary` — totals (`total_passes, total_access_minutes,
  coverage_percent, avg/min/max_duration_minutes`).
- `StationSummary` — per-station totals.
- `PropagatedEphemeris` — **the orbit-only reusable cache**. Stores everything
  columnar in NumPy: `timeline_seconds, timestamps_unix, eci_pos_km/eci_vel_km_s,
  ecef_pos_km/ecef_vel_km_s, body_{x,y,z}_ecef` (attitude axes), `gt_lat_deg/
  gt_lon_deg`, and orbital-element time series (`orbital_altitude_km,
  semi_major_axis_km, perigee/apogee_altitude_km, eccentricity, inclination_deg,
  argument_of_perigee_deg, orbital_period_series_s, true_anomaly_deg,
  angle_of_attack_deg`). `__post_init__` converts legacy list-of-dataclass
  inputs into the columnar arrays.
- `DerivedAccessResult` — station-dependent outputs (passes, summaries, and the
  per-station series dicts: elevation, azimuth, az/el rate, range rate, range
  accel, above-horizon mask).
- `AnalysisResult` — the UI-facing payload: flattens `DerivedAccessResult` +
  `PropagatedEphemeris` into one object, and also keeps references to both
  (`.ephemeris`, `.derived_access`). Helper `analysis_result_sample_count`.

Pattern: arrays are columnar NumPy for speed; per-sample dataclasses exist only
as backward-compatibility constructors. Series are aligned 1-D arrays indexed by
the shared `timeline_seconds`; out-of-pass detail samples are `NaN`.

### 4.4 Services (`services/`)

#### `access_analysis.py` (the slow path, ~1040 lines)

The only module that talks to Orekit propagators. Notable details:

- **Lazy JVM imports**: All `java.*` / `org.orekit.*` / `org.hipparchus.*`
  classes are module-level `None` until `ensure_orekit_bootstrapped()` runs
  `_import_orekit_symbols()` and binds them into module globals. This is because
  JCC classes are only importable after `orekit.initVM()`.
- `ensure_orekit_bootstrapped()` — starts the JVM (via `orekit_vm`), imports
  symbols, and registers `orekitdata` with a `DirectoryCrawler`. Idempotent.
- `@_attach_thread` decorator on `run_access_analysis` — ensures the calling
  thread is attached to the JVM (needed for worker threads).
- `run_access_analysis(config, stations, *, progress, cancel, progress_callback)`:
  1. Bootstraps Orekit, builds UTC/start/end `AbsoluteDate`s in EME2000.
  2. Optional **closed-form mean-orbit correction** (BrouwerLyddane +
     `apply_mean_orbit_correction`): `_brouwer_lyddane_mean_orbit_correction_km`
     computes the J2 short-period semi-major-axis offset so the requested
     *mean* SMA is preserved once fed to `BrouwerLyddanePropagator` as an
     osculating element. No iterative solver — a single closed-form formula.
  3. Builds the initial `KeplerianOrbit` from `OrbitConfig` (using the
     corrected osculating SMA when applicable).
  4. Builds the propagator (`_build_propagator`): `KeplerianPropagator` or
     `BrouwerLyddanePropagator` (fixed J2-J5 zonal harmonics -- Orekit's
     implementation always evaluates the (5, 0) term internally, so the
     gravity provider is always built at degree 5 -- plus optional
     `drag_coefficient` secular drag M2 term). Attitude provider set per
     `attitude_mode`.
  5. Attaches **contact-attitude switching** for the primary station
     (`_enable_station_pointing_during_contact`): swaps body +Z between nadir
     (prograde default) and station line-of-sight during contact, triggered by
     an `ElevationDetector` (optionally AND-combined with a custom off-nadir
     slew-envelope `PythonAbstractDetector`).
     **Not supported with `propagator_type="brouwer_lyddane"`** — raises
     `ValueError` up front instead of failing deep in the propagation loop.
     Every event-triggered state reset (i.e. every attitude switch) makes
     `BrouwerLyddanePropagator.resetIntermediateState()` re-run its internal
     osculating<->mean fixed-point conversion at a fixed tolerance/iteration
     cap (1e-13, 200) with no public override, and that conversion is prone to
     non-convergence for near-circular orbits. Use `propagator_type="keplerian"`
     for scripts needing contact-attitude switching (see
     `analyze_first_pass_rssi_with_spin.py`).
  6. `_propagate_and_collect_state_vectors()` — the hot loop. Steps the
     propagator by `sample_step_seconds`, collecting ground track (geodetic
     lat/lon + ECEF), ECI/ECEF state vectors, body axes in ECEF (from the
     attitude rotation), angle-of-attack, and osculating Keplerian elements.
     Calls `cancel.check()` + progress each step. Stops at a re-entry cutoff of
     80 km altitude. **No per-station geometry here.**
  7. Builds the `PropagatedEphemeris` from the columnar samples.
  8. Calls `derive_access_results_from_ephemeris` then
     `analysis_result_from_components` to produce the `AnalysisResult`.

Body-frame convention used throughout: **+X forward, +Z down/nadir (Earth-pointing
modes), +Y completes right-handed set.**

#### `cached_access_recompute.py` (the fast path, ~500 lines)

Pure NumPy. Given a `PropagatedEphemeris`, a config, and stations, produces a
`DerivedAccessResult`. Key functions:

- `_station_topocentric_basis(station)` — east/north/up unit vectors at the
  station.
- `_compute_station_geometry(...)` — **two-pass** per station:
  - Pass 1 (full timeline): line-of-sight vectors, range, elevation.
  - Horizon threshold: per-azimuth terrain cutoff from a horizon-mask CSV
    (`horizon_mask_io`) if configured, else a scalar `min_elevation_deg`.
  - Pass 2 (in-pass samples only): azimuth, az/el rates (via `np.gradient`),
    range rate, range accel. Out-of-pass entries are `NaN`.
  - Runs stations concurrently under `ThreadPoolExecutor` (NumPy releases the
    GIL); up to 8 workers.
- `_extract_pass_statistics_from_series(...)` — detects passes from the
  above-horizon boolean mask. Interpolates exact AOS/LOS crossing times,
  computes max elevation and max spacecraft slew rate
  (`sqrt(el_rate^2 + (az_rate*cos(el))^2)`). Uses `np.searchsorted` for O(log n)
  pass slicing.
- `_build_summary(...)` — totals; coverage uses the **union** of access
  intervals (`_union_interval_seconds`) to avoid double-counting overlapping
  multi-station contacts.
- `analysis_result_from_components(*, ephemeris, derived_access)` — flattens the
  two caches into `AnalysisResult`.
- `DEFAULT_MIN_ELEVATION_DEG = 5.0` — used only when no horizon mask and the
  config supplies no `min_elevation_deg`.

#### `scenario_data_volume.py` (data volume, ~600 lines)

Batch downlink data-volume evaluation. Key types:

- `AccessSeries` — strictly validated payload built by `build_access_series(result)`
  (timeline, per-station elevation series, orbit period, altitude, satellite
  ECEF, body axes). Raises on any shape mismatch or non-finite value (no silent
  fallbacks).
- `ArchitectureConfig` — one link architecture (band, frequency, symbol rate,
  operating mode `VCM`|`Fixed MODCOD`, margins, antenna LUT path, TX/RX params,
  unavailability).
- `ArchitectureResult` — totals (`total_gbit, gbit_per_orbit, num_filtered_passes`).
- `ScenarioDataVolumeEvaluator` — orchestrates per-station rate evaluation with
  caches (atmospheric loss curve, antenna LUT, boresight gain, gain series).
  `evaluate_architecture(architecture, filtered_passes)`:
  - Computes a per-station antenna gain series via `antenna_pattern`
    (only on in-view samples for speed).
  - Builds an atmospheric loss-vs-elevation curve (`itu_losses`) once per
    (freq, unavailability, station) and interpolates per sample.
  - Calls `link_budget_math.calculate_link_budget` to get Mbps per sample.
  - Integrates each filtered pass's Mbps over [AOS, LOS]
    (`compute_pass_downlink_volumes_gbit` -> `integrate_data_volume_interval`).
  - `compute_total_and_per_orbit_gbit` aggregates total + Gbit/orbit.
- `filter_passes_by_max_elevation(passes, lower, upper)` — same semantics as the
  UI elevation slider.

#### `antenna_pattern.py` (~370 lines)

Spacecraft body-frame antenna gain lookup.

- `SphericalGainLut` — regular Az/El grid (`az_deg 0..360`, `el_deg -90..90`,
  `gain_dbi_grid`) with bilinear interpolation (`gain_dbi(az, el)`).
- `load_spherical_gain_lut(path)` — loads + strictly validates an NPZ LUT
  (uniform spacing, full sphere coverage, finite values).
- `default_synthesized_lut_path()` — the shipped Anywaves S-band TTC LUT.
- `station_ecef_m(station)` — WGS84 geodetic -> ECEF meters.
- `body_vectors_to_az_el`, `project_los_to_body_frame` — geometry helpers.
- Comms pointing-mode steering (applied to the line-of-sight in body frame):
  - `prograde_pointing` — no steering (use fixed body axes).
  - `free_to_roll` — `apply_roll_toward_station_about_x` (roll about +X to put
    station in X/Z plane).
  - `constrained_aoa` — `apply_constrained_aoa_toward_station(max_aoa_deg)`
    (steer toward station but cap angle-of-attack of +X off prograde).
- `evaluate_station_gain_series(...)` — top-level: LOS in ECEF -> body frame ->
  apply pointing mode -> Az/El -> LUT gain. Returns `(gains_dbi, az, el, roll)`.

#### `itu_losses.py`

Thin wrapper around `itur` (ITU-R P.618/676 etc.). `estimate_slant_path_loss(
frequency_GHz, elevations_deg, lat, lon, altitude_m, unavailability_percent,
antenna_diameter_m, return_contributions)` returns total atmospheric attenuation
(dB) per elevation, optionally with the gaseous/cloud/rain/scintillation
breakdown. Clamps elevation to >= 0.5 deg (scintillation diverges at the
horizon) and silences known spurious `itur` numpy warnings.

#### `pfd_math.py`

ITU surface power-flux-density compliance. `PfdMaskSpec` per band (S 1525-2300,
X 8025-8500 with 4 kHz ref BW; Ka 25500-27000 with 1 MHz ref BW).
`itu_surface_pfd_mask_spec(freq_MHz)`, `itu_surface_pfd_limit_dBW_per_m2(
elevation, freq)` (piecewise: flat low limit <=5 deg, linear ramp 5-25 deg, flat
high limit >=25 deg), `pfd_at_reference_bandwidth_dBW_per_m2(...)`,
`directional_eirp_dBW(...)`, `occupied_bandwidth_hz(symbol_rate, rolloff)`.

#### `link_budget_xlsx.py` (styled link-budget export)

Renders an uplink/downlink link budget to a styled `.xlsx` workbook that
reproduces the layout/styling of the reference "N-STAR" spreadsheet (sheets
`Uplink`, `Downlink`, and a combined `Link Budget`): bold headers, thin-border
grid, hidden gridlines, merged Slant/Nadir cells, units column, right-hand
summary block. Writes **static computed values** (not live Excel formulas) by
porting the spreadsheet's closed-form relations into NumPy/`math`. Imports only
`openpyxl` (+ lazily `itu_losses`), so it stays headless-safe.

- `LinkBudgetExportConfig` — fully parameterises both links (geometry,
  modulation/coding, RF chain, required Eb/N0, system margin). Defaults
  reproduce the N-STAR scenario. `from_dict`/`to_dict`; `load_config(path)`
  reads YAML or JSON.
- `compute_uplink(cfg)` / `compute_downlink(cfg)` — return every displayed
  quantity at slant (the configured elevation) and nadir (90 deg). Atmospheric
  loss is taken from the config, or computed via `itu_losses` when station
  coordinates are supplied and the explicit values are `None`.
- `build_workbook(cfg)` -> `openpyxl.Workbook`; `write_link_budget_xlsx(cfg, path)`
  writes it. Backs both the CLI `export-link-budget` subcommand and the GUI
  Static Link Budget Tool's "Export Link Budget to XLSX" button.

#### `scenario_package_io.py` (NPZ scenario packages)

Save/load cached trajectory packages. **Schema v2** = two files:
`<name>.npz` (compressed columnar arrays) + `<name>.toolbox.json` (lean
sidecar: `schema_version, orbit_period_seconds, mean_ic_report, settings,
stations, enabled_station_names`).
- `export_cached_trajectory_package(*, ephemeris, settings, stations,
  enabled_station_names, output_path)` -> `(npz_path, sidecar_path)`.
- `import_cached_trajectory_package(path)` -> `ImportedScenarioPackage`. Accepts
  `.npz`, `.toolbox.json`, or legacy `.oem`. **Schema v1** (OEM + fat JSON) is
  read-only legacy via `_legacy_import`. `timeline_seconds` is not stored in v2
  (re-derived from `timestamps_unix`).

#### `station_importer.py`

`load_ground_stations_from_file(path)` — reads CSV/XLSX into
`GroundStationConfig`s. Two formats: **name-only** (resolved against a master
`resources/ground_station_database.csv`, not bundled by default — supply your
own, which supplies lat/lon/alt + supplier + horizon mask) and **full**
(`name, latitude, longitude, altitude`, enriched from the DB when names
match, or usable standalone with no DB at all). Raises `StationImportError` on
problems.

#### `horizon_mask_io.py`

Per-azimuth terrain horizon masks. `load_horizon_mask(path)` -> `(360,)` array
(min elevation deg per integer azimuth), parsing both a simple and a
header-block CSV format. `horizon_elevations_at_azimuths(az, mask_path)` —
vectorised interpolated lookup with 360-deg wrap (LRU-cached).
`builtin_mask_path(name)` — fuzzy lookup in `resources/horizon_masks/` (empty
unless you drop your own mask CSVs there).

#### `ephemeris_helpers.py`

Indexing helpers over the columnar ephemeris: `ephemeris_time_slice`,
`ephemeris_pass_mask`, `ecef_to_geodetic_deg` (spherical), `body_axes_at_index`.

#### `orekit_vm.py`

`ensure_orekit_vm_started(*, extra_jars=())` — starts the JVM exactly once
(`orekit.initVM()`), appending extra jars to the JCC classpath **before**
startup (cannot be added after), and attaches the current thread. All Orekit
usage goes through this.

### 4.5 Analyses (`analyses/`)

Pure-compute helpers lifted out of the GUI so they can be reused/tested headless:
- `orbit_summary.py` — `compute_orbit_averaged_orbit_summary(times_s, inst)`:
  orbit-averaged (secular) element series using a moving-average window sized to
  one orbital period; angle series averaged via sin/cos to avoid wrap. Returns
  `(averaged_dict, meta)`. `compute_series_stats` gives min/max/mean/std per
  series. This matches the GUI's "secular" plots exactly.
- `pfd.py` — `compliance_backoff_db(...)`: TX back-off (dB) required so worst-case
  surface PFD over an elevation grid meets the ITU mask.
- `visualization.py` — `ecef_to_globe_coords(x, y, z) -> (-y, x, z)`: ECEF/ITRF
  to globe plotting frame (+90 deg about Z). Used by GUI and (conceptually) web.

### 4.6 Tools (`tools/`)

- `link_budget_profile.py` — `LinkBudgetInputs`, `LossCache`, `run_pipeline`,
  and a `main()` timing benchmark over parameter perturbations. Backs
  `ToolboxFacade.run_link_budget` and `cosmic-toolbox link-budget-profile`.
- `orbit_summary_npz.py` — `save_orbit_summary_npz` / `load_orbit_summary_npz`
  (flattened-key NPZ with `instantaneous__*` / `orbit_averaged__*` + JSON meta).

### 4.7 Link budget math (`link_budget_math.py`, ~850 lines)

The heart of the RF analysis. Pure functions, every parameter explicit.

- Constants: `EARTH_RADIUS_KM`, `BOLTZMANN_CONSTANT_DB = 228.6`.
- `MODCOD_TABLE` — DVB-S2 subset `(name, bits_per_symbol, required_EsN0_dB)`.
  `FIXED_LINK_MODE_TABLE` adds custom QPSK Viterbi and QPSK 7/8 modes.
- Geometry/noise: `slant_range_km`, `free_space_path_loss_dB`,
  `system_noise_temperature_K` (couples sky brightness to atmospheric loss),
  `system_g_over_t_dB_K`.
- MODCOD selection: `select_modcod` (VCM — best mode per Es/N0 after margin),
  `select_modcod_fixed` (single MODCOD, link closes only above threshold+margin),
  `required_esn0_for_mode`.
- `calculate_link_budget(...)` — **the main routine.** Vectorised over elevation
  samples. Computes EIRP (with pointing loss from boresight vs actual gain),
  FSPL, received power, G/T (or a fixed `gs_gt_dBK`), C/N0, Es/N0, then per-sample
  MODCOD + data rate (Mbps). Returns a dict of all intermediate terms.
- `optimize_ccm_per_pass(...)` — grid search over (start offset, end offset,
  MODCOD) maximizing data volume for a single pass (`CcmOptimizerResult`).
- `integrate_data_volume_gb(time_s, rate_mbps)` — trapezoidal integration -> Gbit.
- `build_parameter_rows(...)` — builds the GUI link-budget table
  (`ParameterRow` list) at one evaluation elevation, including PFD/PFD-limit/
  PFD-margin rows (PFD is "N/A" outside supported bands).

### 4.8 Resources (`resources/`)

Bundled static data, resolved via `paths.package_resources_root()`
(`importlib.resources` with a filesystem fallback for editable installs):
- `groundstation_list/ground_stations.csv` — a small example station preset
  (`name, latitude, longitude, altitude`).
- `link_budget_defaults.yaml` — canonical RF link-budget defaults (§4.1).

An antenna gain LUT, a full station database (`ground_station_database.csv`),
and horizon masks are **not bundled** — supply your own NPZ LUT / CSV database
and pass their paths explicitly; `station_importer`'s by-name lookups and
`antenna_pattern.default_synthesized_lut_path()` need one to be present.

### 4.9 CLI (`cli/`)

`cli/__init__.py:main` is an argparse dispatcher (using `parse_known_args` so
subcommands can own their remaining args) with subcommands:
- `link-budget-profile` -> `tools.link_budget_profile.main`.
- `export-link-budget` -> `cli/export_link_budget.py:main`: loads a YAML/JSON
  `LinkBudgetExportConfig` (positional, optional — falls back to built-in
  defaults), supports repeatable `--set key=value` overrides (JSON-parsed), and
  writes the workbook via `services.link_budget_xlsx.write_link_budget_xlsx`.
  See `configs/link_budget_example.yaml` for the full key set.

---

## 5. GUI package: `cosmic_toolbox_gui`

PySide6 desktop app. The file map in §5.2 is the authoritative entry point;
start from `main_window.py` and the per-tab mixins under `tabs/`.

### 5.1 Composition pattern

There is **one window**, `GroundStationApp(QMainWindow)` in `main_window.py`,
built by **mixin composition**. Each tab is a mixin contributing `_build_*_tab()`
and handlers; shared state lives on the `GroundStationApp` instance.

```python
class GroundStationApp(
    GroundTabMixin, MissionTabMixin, VisualizationTabMixin,
    LinkBudgetTabMixin, OrbitSummaryTabMixin, PlotHelpersMixin, QMainWindow,
):
```

### 5.2 File map

| File | Role |
| --- | --- |
| `app.py` | Entry point: logging, `QApplication`, splash, texture preload, `GroundStationApp().showMaximized()` |
| `main_window.py` (~1640 lines) | Orchestrator: state, analysis lifecycle, contact filtering, scenario import/export, tab assembly, central `_apply_analysis_result()` fan-out |
| `constants.py` | Palettes, histogram bins, texture paths, default station |
| `globe_math.py` | Pure helpers: `gmst_rad_utc`, `julian_date_utc`, `rotate_vector_z` |
| `plot_helpers.py` (~930 lines) | `PlotHelpersMixin`: timeline/access-share/elevation-distribution plots |
| `workers/analysis_worker.py` | `AnalysisWorker` (QThread -> `ToolboxFacade.run_access`), `QtProgressReporter`, unused `CachedRecomputeWorker` |
| `tabs/ground_tab.py` | `GroundTabMixin`: station import/edit/select + Cartopy world map |
| `tabs/mission_tab.py` | `MissionTabMixin`: orbit/scenario/propagation inputs, Run/Stop, cached I/O, mission globe |
| `tabs/visualization_tab.py` (~2450 lines) | `VisualizationTabMixin`: per-pass animation on globe, antenna lobe, video/series export, metric graphs |
| `tabs/link_budget_tab.py` (~3790 lines) | `LinkBudgetTabMixin`: Static + Dynamic link budget tools, data volume, downlink summary, static-budget XLSX export |
| `tabs/orbit_summary_tab.py` (~1095 lines) | `OrbitSummaryTabMixin`: secular + instantaneous element plots, ephemeris/summary export |
| `widgets/range_slider.py` | `RangeSlider`: dual-handle slider for the elevation filter |
| `opengl/globe_widget.py` (~2090 lines) | `GlobeWidget` (ModernGL): wireframe/textured Earth, ground track, satellite mesh, body triad, link line, sensor cone, terminator |

### 5.3 Tab functionality

| Tab | What the user does |
| --- | --- |
| Ground Station Selection | Import/edit/toggle stations; view on world map; feeds enabled stations into analysis |
| Mission Configuration | Set orbit (with SSO inclination helper), scenario times, propagation (sample step, comms pointing); Run/Stop; cached trajectory refresh/export/import; mission globe ground track |
| Contact Analysis > Statistics | Filter passes by max elevation (RangeSlider); 9-column pass table; NPZ/XLSX export |
| Contact Analysis > Timeline / Access Share / Elevation Distribution | Gantt timeline (pyqtgraph), pie share (matplotlib), PDF/CDF elevation (matplotlib) |
| Contact Analysis > Pass Visualization | Animate a pass on the globe (ECI/ECEF, play/scrub/speed), antenna lobe overlay, export MP4/series, metric graphs (Doppler, rates, PFD) |
| Contact Analysis > Static Link Budget Tool | Standalone what-if link budget at fixed elevation (UL+DL tables, elevation sweep); "Export Link Budget to XLSX" button -> `services.link_budget_xlsx` |
| Contact Analysis > Dynamic Link Budget Tool | Time-varying link budget over real passes + data-volume histogram + downlink summary |
| Orbit Summary | Secular + osculating element time series; export ephemeris (ECI/ECEF) and orbit summary NPZ |

### 5.4 Threading and the central fan-out

- **Run**: `_handle_run_clicked` -> `_start_analysis_worker` creates an
  `AnalysisWorker`, `moveToThread`, connects `progress`/`finished`/`error`
  signals. `worker.run()` calls `ToolboxFacade.run_access(...)`. Stop calls
  `worker.request_cancel()` -> `CancelToken.cancel()`.
- **Cached recompute** runs **synchronously on the main thread** via
  `_refresh_cached_access_from_inputs` (calls
  `derive_access_results_from_ephemeris` + `analysis_result_from_components`).
- `_apply_analysis_result()` is the single fan-out after any run/import: stores
  config/result/ephemeris, clears dirty flags, then pushes data into the contact
  statistics view, mission globe, visualization tabs, link budget cache, and
  orbit summary.
- Dirty model: `_mark_dirty()` (orbit/scenario change => re-propagate) vs
  `_mark_derived_outputs_stale()` (station/comms/LB change => cached recompute).

### 5.5 ModernGL globe

`GlobeWidget` (OpenGL 3.3 Core via moderngl inside a `QOpenGLWidget`). Two
instances: `mission_globe_widget` (full ground track, wireframe) and
`visual_globe_widget` (per-pass animation with sun/terminator, link line,
antenna sensor cone). Geometry is procedural (sphere, graticule, Natural Earth
coastlines, simplified spacecraft). Three GLSL programs (`textured`, `color`,
`color_vertex`). Textures preloaded during the splash. Convert ECEF km ->
globe frame with `analyses.visualization.ecef_to_globe_coords` before calling
update methods.

---

## 6. Tests (`tests/`)

`pyproject.toml` sets `testpaths = ["tests"]` and `pythonpath = ["src"]`.

- `tests/toolbox/` — headless core tests, always run: `test_facade.py`,
  `test_link_budget_math.py`, `test_link_budget_xlsx.py`, `test_itu_losses.py`,
  `test_pfd_math.py`, `test_antenna_pattern.py`, `test_scenario_data_volume.py`,
  `test_scenario_package_io.py`, `test_shared_defaults.py`.
- `tests/gui/` — PySide6 tests, auto-skipped when PySide6 is missing
  (`conftest.py`). Includes `test_cached_access_recompute.py`,
  `test_link_budget_ui.py`, `test_visualization_free_to_roll.py`.

Each test directory's `conftest.py` handles Orekit VM startup and shared
fixtures.

Run: `pytest`; single: `pytest tests/toolbox/test_facade.py::test_name`.
Lint/format/type: `ruff check src/ tests/`, `ruff format ...`, `mypy src/`.

---

## 7. Work patterns / conventions

1. **Where new code goes**
   - Pure compute (no Qt): `cosmic_toolbox/analyses/*` or `services/*`.
   - Long-running orchestrated runs: `cosmic_toolbox/services/*`.
   - Static data: `cosmic_toolbox/resources/...`.
   - Console scripts: `cosmic_toolbox/cli/*`.
   - Qt threading: `cosmic_toolbox_gui/workers/*`. Tab UI:
     `cosmic_toolbox_gui/tabs/*`. Globe: `cosmic_toolbox_gui/opengl/*`.

2. **Always go through `ToolboxFacade`** from a frontend/script for the
   high-level operations; don't reach into services directly unless you need a
   lower-level helper not exposed by the facade.

3. **Respect the ports**: long-running operations accept `progress:
   ProgressReporter` and `cancel: CancelToken`. New worker threads must attach to
   the Orekit JVM (`ensure_orekit_vm_started()`) before touching Orekit classes.

4. **Slow vs fast path**: never re-propagate when only station/link parameters
   change — use the cached `PropagatedEphemeris` + `derive_access_from_ephemeris`.

5. **No silent fallbacks** (matches the repo's stated preference): validate
   inputs and raise explicit errors rather than substituting hardcoded defaults.
   `build_access_series`, `calculate_link_budget`, and the LUT loaders are good
   examples — they raise on shape/finiteness violations. Use ASCII-only output
   in CLI/print paths (no emoji / non-ASCII that breaks Windows consoles).

6. **Columnar NumPy everywhere** for time series; arrays are aligned to a shared
   `timeline_seconds`; out-of-pass detail samples are `NaN`.

7. **Orekit JVM is process-wide and single-start**: `orekit.initVM()` happens
   once via `orekit_vm`; classpath jars must be added before that call; worker
   threads must `attachCurrentThread()`. The GUI keeps one analysis running at
   a time because of this.

8. **Body-frame convention**: +X forward, +Z nadir/down (Earth-pointing), +Y
   right-handed. Comms pointing modes: `prograde_pointing`, `free_to_roll`,
   `constrained_aoa`.

9. **Scenario persistence**: schema v2 NPZ + `.toolbox.json` sidecar; v1 OEM is
   read-only legacy.

---

## 8. Quick reference: where is X?

| I want to... | Look at |
| --- | --- |
| Run a fresh propagation + access | `services/access_analysis.py:run_access_analysis` |
| Recompute access without Orekit | `services/cached_access_recompute.py:derive_access_results_from_ephemeris` |
| Compute a link budget | `link_budget_math.py:calculate_link_budget` |
| Export a link budget to XLSX | `services/link_budget_xlsx.py:write_link_budget_xlsx` (CLI `export-link-budget`) |
| Pick a MODCOD / optimize CCM | `link_budget_math.py:select_modcod`, `optimize_ccm_per_pass` |
| Atmospheric loss | `itu_losses.py:estimate_slant_path_loss` |
| PFD compliance | `pfd_math.py`, `analyses/pfd.py` |
| Antenna gain to a station | `services/antenna_pattern.py:evaluate_station_gain_series` |
| Scenario data volume | `services/scenario_data_volume.py:ScenarioDataVolumeEvaluator` |
| Orbit-averaged elements | `analyses/orbit_summary.py` |
| Mean-orbit SMA correction | `services/access_analysis.py:_brouwer_lyddane_mean_orbit_correction_km` |
| Save/load a scenario package | `services/scenario_package_io.py` |
| Import ground stations | `services/station_importer.py` |
| Horizon masks | `services/horizon_mask_io.py` |
| The public API | `facade.py` / `__init__.py` |
| GUI window + orchestration | `cosmic_toolbox_gui/main_window.py` |
| The 3D globe | `cosmic_toolbox_gui/opengl/globe_widget.py` |
