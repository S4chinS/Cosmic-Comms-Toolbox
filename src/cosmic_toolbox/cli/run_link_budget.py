"""Run the link-budget profiler pipeline (timing benchmarks)."""

from __future__ import annotations

from cosmic_toolbox.tools.link_budget_profile import main as _profile_main


def main() -> int:
    _profile_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
