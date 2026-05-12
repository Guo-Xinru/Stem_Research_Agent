"""Compatibility wrapper for `python -m stemresearch.cli`."""

from __future__ import annotations

from stem_research.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
