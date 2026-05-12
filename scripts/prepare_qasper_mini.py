"""Generate local QASPER-mini JSONL files."""

from __future__ import annotations

import argparse
from pathlib import Path

from stem_research.prepare_qasper_mini import prepare_qasper_mini


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare deterministic QASPER-mini JSONL files.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/qasper_mini"))
    parser.add_argument("--train-size", type=int, default=30)
    parser.add_argument("--eval-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    prepare_qasper_mini(
        output_dir=args.output_dir,
        train_size=args.train_size,
        eval_size=args.eval_size,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
