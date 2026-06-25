"""Small utility functions shared by the pipeline."""

from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dictionaries."""

    records: list[dict[str, Any]] = []
    file_path = Path(path)
    if not file_path.exists():
        return records
    with file_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {file_path} line {line_no}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"JSONL line must be an object: {file_path} line {line_no}")
            records.append(obj)
    return records


def write_jsonl(records: Iterable[dict[str, Any]], path: str | Path) -> Path:
    """Write dictionaries to a JSONL file."""

    file_path = Path(path)
    ensure_dir(file_path.parent)
    with file_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return file_path


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file."""

    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def write_json(data: dict[str, Any], path: str | Path) -> Path:
    """Write a dictionary as pretty JSON."""

    file_path = Path(path)
    ensure_dir(file_path.parent)
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return file_path


def write_csv(rows: list[dict[str, Any]], path: str | Path, fieldnames: list[str] | None = None) -> Path:
    """Write rows to CSV, creating an empty header-only file when rows are empty."""

    file_path = Path(path)
    ensure_dir(file_path.parent)
    if fieldnames is None:
        keys: set[str] = set()
        for row in rows:
            keys.update(row.keys())
        fieldnames = sorted(keys)
    with file_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    return file_path


def utc_now_string() -> str:
    """Return an ISO-like UTC timestamp used in reports."""

    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def count_values(items: Iterable[Any]) -> dict[str, int]:
    """Count values and return a normal dict with string keys."""

    counter = Counter(str(item) for item in items)
    return dict(counter)


def safe_divide(numerator: float, denominator: float) -> float:
    """Return numerator / denominator with zero protection."""

    return numerator / denominator if denominator else 0.0


def truncate_text(text: str, max_chars: int = 220) -> str:
    """Compact text for Markdown examples."""

    compact = " ".join(str(text).split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3] + "..."


def resolve_project_root(config_path: str | Path | None = None) -> Path:
    """Infer the project root from a config path or the current directory."""

    if config_path:
        path = Path(config_path).resolve()
        if path.parent.name == "configs":
            return path.parent.parent
    return Path.cwd().resolve()


def resolve_path(path_value: str | Path, base_dir: str | Path) -> Path:
    """Resolve a possibly relative path against a base directory."""

    path = Path(path_value)
    if path.is_absolute():
        return path
    return Path(base_dir) / path
