"""Minimal OpenAI API permission diagnostic calls.

This script loads .env, sends tiny OpenAI API probes, and prints the API
error reason without printing the API key. It cannot list every scope on
the key; it infers permissions from endpoint success/failure.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Callable

from stem_research.llm import DEFAULT_OPENAI_MODEL


def main() -> None:
    args = _parse_args()
    _load_dotenv(Path(".env"))
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    model = os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL).strip() or DEFAULT_OPENAI_MODEL

    print(f"OPENAI_API_KEY detected: {'yes' if api_key else 'no'}")
    print(f"OPENAI_MODEL: {model}")

    if not api_key:
        print("Cannot call OpenAI: OPENAI_API_KEY is missing.")
        raise SystemExit(1)

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        probes: dict[str, Callable[[], None]] = {
            "models": lambda: _probe_models(client),
            "responses": lambda: _probe_responses(client, model),
        }
        selected = probes if args.probe == "all" else {args.probe: probes[args.probe]}
        failed = False
        for name, probe in selected.items():
            print(f"\nProbe: {name}")
            try:
                probe()
                print("Result: allowed")
            except Exception as exc:
                failed = True
                _print_openai_error(exc)
        if failed:
            raise SystemExit(1)
    except Exception as exc:
        if isinstance(exc, SystemExit):
            raise
        _print_openai_error(exc)
        raise SystemExit(1) from exc


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe OpenAI API key permissions.")
    parser.add_argument(
        "--probe",
        choices=("all", "models", "responses"),
        default="all",
        help="Which minimal permission probe to run.",
    )
    return parser.parse_args()


def _probe_models(client) -> None:
    models = client.models.list()
    first_model = next(iter(models.data), None)
    print("models.list succeeded.")
    if first_model is not None:
        print(f"Example visible model: {first_model.id}")


def _probe_responses(client, model: str) -> None:
    response = client.responses.create(
        model=model,
        input="Reply with one short sentence: API key works.",
        max_output_tokens=20,
    )
    print("responses.create succeeded.")
    print(response.output_text)


def _print_openai_error(exc: Exception) -> None:
    print("Result: denied or failed")
    print(f"Exception class: {exc.__class__.__name__}")
    status_code = getattr(exc, "status_code", None)
    if status_code is not None:
        print(f"Status code: {status_code}")
    print(f"Message: {exc}")
    body = getattr(exc, "body", None)
    if body:
        print(f"Error body: {body}")


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", maxsplit=1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


if __name__ == "__main__":
    main()
