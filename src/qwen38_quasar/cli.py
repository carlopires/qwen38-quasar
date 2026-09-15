"""Command-line surface for qwen38-quasar.

Deliberately small (see handoff section 44). Experiment settings live in TOML
under ``configs/``; CLI flags may override config values.

Only ``inspect-source`` has a first implementation. The remaining commands are
declared so the surface is stable and scriptable, and raise ``NotImplementedError``
until implemented in rung 0.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

DESCRIPTION = "QUASAR-style NVFP4 quantization-aware distillation for Qwen3.8-derived models."

COMMANDS = (
    "inspect-source",
    "prepare-data",
    "train",
    "export",
    "verify-export",
    "compare",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qwen38-quasar",
        description=DESCRIPTION,
    )
    parser.add_argument("--version", action="store_true", help="print version and exit")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    for name in COMMANDS:
        cmd = sub.add_parser(name, help=f"{name} (see docs/)")
        cmd.add_argument(
            "--config",
            type=str,
            default=None,
            help="path to a TOML config file",
        )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.version:
        from qwen38_quasar import __version__

        print(__version__)
        return 0

    if not args.command:
        parser.print_help()
        return 1

    raise NotImplementedError(
        f"'{args.command}' is not implemented yet; see drafts/execution-plan.md (rung 0)."
    )


if __name__ == "__main__":
    sys.exit(main())
