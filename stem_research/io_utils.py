"""Small JSON and path helpers for reproducible CLI experiments."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def load_json(path: str | Path) -> Any:
    json_path = Path(path)
    if not json_path.exists():
        raise FileNotFoundError(f"Required JSON file not found: {json_path}")
    try:
        with json_path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON in {json_path}: {exc}") from exc


def write_json(path: str | Path, data: Any) -> Path:
    json_path = Path(path)
    ensure_dir(json_path.parent)
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(_to_jsonable(data), file, indent=2, sort_keys=True)
        file.write("\n")
    return json_path


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    jsonl_path = Path(path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Required JSONL file not found: {jsonl_path}")
    records: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSONL in {jsonl_path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"JSONL record must be an object in {jsonl_path}:{line_number}")
            records.append(record)
    return records


def write_jsonl(path: str | Path, records: list[Any]) -> Path:
    jsonl_path = Path(path)
    ensure_dir(jsonl_path.parent)
    with jsonl_path.open("w", encoding="utf-8") as file:
        for record in records:
            json.dump(_to_jsonable(record), file, sort_keys=True)
            file.write("\n")
    return jsonl_path


def timestamped_run_path(
    prefix: str = "run",
    suffix: str = ".json",
    output_dir: str | Path = "runs",
) -> Path:
    ensure_dir(output_dir)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path(output_dir) / f"{prefix}_{stamp}{suffix}"


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    return value
