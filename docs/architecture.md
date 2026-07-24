# Architecture

The repository is split into two cooperating Python packages that share a
single source tree.

```
                ┌──────────────────────────────────────┐
                │  cosmic_toolbox  (no Qt, no plotting)│
                │                                      │
  scripts/  ──► │   facade.ToolboxFacade               │ ◄── notebooks
  CLI tools ──► │      .run_access(...)                │ ◄── tests
                │      .run_link_budget(...)           │
                │      .run_data_volume(...)           │
                │      .build_orbit_summary(...)       │
                │      .pfd_compliance_backoff_db(...) │
                │   models/  services/  analyses/      │
                │   tools/   resources/                │
                └──────────────────────────────────────┘
                                  ▲
                                  │ depends on
                                  │
                ┌──────────────────────────────────────┐
                │ cosmic_toolbox_gui (PySide6 adapter) │
                │   workers/   tabs/   widgets/        │
                │   QThread wrapper around the facade  │
                └──────────────────────────────────────┘
```

## Ports / adapters

The boundary between the toolbox and any caller is mediated by two tiny
abstractions:

- `cosmic_toolbox.services.progress.ProgressReporter` — a `Protocol` with one
  method, `report(fraction: float, message: str | None)`. The default
  `NullProgress` is a no-op. The GUI ships
  `cosmic_toolbox_gui.workers.QtProgressReporter`, which forwards to a Qt
  `Signal[float]`.
- `cosmic_toolbox.services.cancel.CancelToken` — a thread-safe cooperative
  cancellation flag with `cancel()`, `cancelled`, and `check()` (raises
  `AnalysisCancelled`). The GUI's stop button calls
  `AnalysisWorker.request_cancel()`, which calls `CancelToken.cancel()`.

`run_access_analysis` accepts both abstractions through its keyword arguments
and still honours the legacy `progress_callback: Callable[[float], None]`
overload for backwards compatibility.

## Public surface

`ToolboxFacade` is the only entry point you should reach for from a script,
notebook, or the GUI worker:

```python
from cosmic_toolbox import (
    ToolboxFacade,
    AnalysisConfig,
    GroundStationConfig,
    CancelToken,
    NullProgress,
)

facade = ToolboxFacade(progress=NullProgress(), cancel=CancelToken())
result = facade.run_access(config, [GroundStationConfig(...)])
```

The facade exposes:

- `run_access(config, stations, *, progress, cancel)`
- `derive_access_from_ephemeris(*, ephemeris, config, stations)`
- `run_link_budget(inputs)`
- `run_data_volume(*, result, architecture, stations, scenario_start_time, ...)`
- `build_orbit_summary(*, times_s, instantaneous)`
- `pfd_compliance_backoff_db(...)`

## Where to put new code

| Concern                          | Goes in                                        |
| -------------------------------- | ----------------------------------------------- |
| Pure compute (no Qt)             | `cosmic_toolbox/analyses/*.py` or `services/`  |
| Long-running orchestrated runs   | `cosmic_toolbox/services/*.py`                 |
| Static data (LUTs, CSVs)         | `cosmic_toolbox/resources/...`                 |
| Console scripts / batch entry    | `cosmic_toolbox/cli/*.py`                      |
| Qt worker / threading            | `cosmic_toolbox_gui/workers/*.py`              |
| Tab UI (widgets, layouts, paint) | `cosmic_toolbox_gui/tabs/*.py`                 |
| Globe rendering / textures       | `cosmic_toolbox_gui/opengl/*.py`               |
| Icons / splashes                 | `cosmic_toolbox_gui/resources/...`             |

`cosmic_toolbox` must never import `PySide6`, `pyqtgraph`, `matplotlib`,
`cartopy`, or `moderngl`. CI's headless `pip install -e .` install proves
this rule by omitting those packages entirely.

## Resource resolution

Bundled assets live under `cosmic_toolbox/resources/` and are looked up via
`cosmic_toolbox.paths.package_resources_root()`, which prefers
`importlib.resources.files("cosmic_toolbox.resources")` and falls back to a
filesystem path for editable installs.

## Process model

```
GUI thread                                Worker thread (QThread)
─────────                                 ───────────────────────

GroundStationApp.run_button → click       AnalysisWorker.run()
   │                                          │
   │  build AnalysisWorker(config, stations)  │
   │──────────────────────────────────────►   │
   │                                          │
   │  worker.moveToThread(thread)             │
   │  thread.start()                          │
   │                                          ▼
   │                                      ToolboxFacade.run_access(
   │                                          progress=QtProgressReporter,
   │                                          cancel=CancelToken,
   │                                      )
   │                                          │
   │                                          ▼
   │                                      services.access_analysis
   │                                      .run_access_analysis(...)
   │                                          │
   │ progress.emit(percent) ◄─────────────────┤   QtProgressReporter.report(f)
   │                                          │
   │ AnalysisCancelled ◄──────────────────────┤   CancelToken.check()
   │                                          ▼
   │ finished.emit(AnalysisResult) ◄────── return AnalysisResult
```

## JCC / Orekit JVM

`orekit.initVM()` may only be called once per process. The toolbox
encapsulates this behind
`cosmic_toolbox.services.orekit_vm.ensure_orekit_vm_started()`, which is
called lazily by `run_access_analysis` and the IC finder. Scripts and the GUI
both go through the same bootstrap, so there's nothing extra to wire up.
