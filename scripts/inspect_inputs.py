"""CLI: report the environment and the datasets that were discovered.

Usage:
    python scripts/inspect_inputs.py [--config configs/default.json]

Prints the environment report and the resolved input locations (with explicit
"MISSING" guidance when a required dataset/column is not present).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from foodmacros.config import load_config  # noqa: E402
from foodmacros.data.discovery import resolve_inputs  # noqa: E402
from foodmacros.env_checks import format_report, get_env_report  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Inspect environment and discovered inputs.")
    parser.add_argument("--config", default=None, help="Path to a config JSON")
    args = parser.parse_args(argv)

    config = load_config(args.config) if args.config else load_config()

    print(format_report(get_env_report()))
    print()
    resolved = resolve_inputs(config)
    print(resolved.summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
