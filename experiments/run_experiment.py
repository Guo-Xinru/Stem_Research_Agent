"""Compatibility wrapper for the current QASPER-mini CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from stem_research.cli import main as cli_main


def main(argv: Sequence[str] | None = None) -> Path:
    args = list(argv or [])
    if not args or args[0] not in {"run-qasper", "prepare-qasper-mini", "evaluate", "run-ai-demo"}:
        args = ["run-qasper", *args]
    exit_code = cli_main(args)
    if exit_code != 0:
        raise SystemExit(exit_code)
    output_dir = Path("outputs")
    if "--output-dir" in args:
        output_dir = Path(args[args.index("--output-dir") + 1])
    return output_dir / "qasper_metrics.json"


if __name__ == "__main__":
    main()
