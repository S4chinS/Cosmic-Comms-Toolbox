"""CLI: export an uplink/downlink link budget to a styled ``.xlsx`` workbook.

Usage::

    cosmic-toolbox export-link-budget CONFIG.yaml -o link_budget.xlsx

``CONFIG`` may be YAML or JSON; see ``configs/link_budget_example.yaml`` for the
full set of supported keys.  Individual values may be overridden on the command
line with ``--set key=value`` (values are parsed as JSON, falling back to a raw
string), which is handy for quick one-off sweeps.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from cosmic_toolbox.services.link_budget_xlsx import (
    LinkBudgetExportConfig,
    load_config,
    write_link_budget_xlsx,
)


def _coerce(value: str):
    """Parse a CLI override value as JSON, falling back to the raw string."""
    try:
        return json.loads(value)
    except (json.JSONDecodeError, ValueError):
        return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cosmic-toolbox export-link-budget",
        description="Export an uplink/downlink link budget to a styled .xlsx workbook.",
    )
    parser.add_argument(
        "config",
        nargs="?",
        default=None,
        help="Path to a YAML or JSON link-budget config. Omit to use built-in defaults.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="link_budget.xlsx",
        help="Output .xlsx path (default: link_budget.xlsx).",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a single config value (repeatable). Value is parsed as JSON.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)

    if args.config:
        cfg = load_config(args.config)
    else:
        cfg = LinkBudgetExportConfig()

    if args.overrides:
        data = cfg.to_dict()
        for item in args.overrides:
            if "=" not in item:
                raise SystemExit(f"Invalid --set value (expected KEY=VALUE): {item!r}")
            key, _, raw = item.partition("=")
            data[key.strip()] = _coerce(raw.strip())
        cfg = LinkBudgetExportConfig.from_dict(data)

    out_path = write_link_budget_xlsx(cfg, Path(args.output))
    print(f"Wrote link budget workbook: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
