"""Console-script entry points for ``cosmic_toolbox``."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch to a CLI subcommand. Run ``cosmic-toolbox --help`` for options."""

    parser = argparse.ArgumentParser(prog="cosmic-toolbox")
    sub = parser.add_subparsers(dest="command", required=False)

    sub.add_parser("link-budget-profile", help="Profile the link-budget pipeline")
    # Export subcommand: parses its own args, so accept everything after the name.
    sub.add_parser(
        "export-link-budget",
        help="Export an uplink/downlink link budget to a styled .xlsx workbook",
        add_help=False,
    )

    args, rest = parser.parse_known_args(list(argv) if argv is not None else None)

    if args.command == "link-budget-profile":
        from cosmic_toolbox.tools.link_budget_profile import main as _main

        _main()
        return 0
    if args.command == "export-link-budget":
        from cosmic_toolbox.cli.export_link_budget import main as _main

        return int(_main(rest) or 0)

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
