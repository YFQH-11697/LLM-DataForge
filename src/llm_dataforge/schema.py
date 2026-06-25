"""Unified JSONL schema helpers for LLM-DataForge."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


VALID_SOURCES = {"web", "document", "code", "instruction"}
VALID_LANGUAGES = {"zh", "en", "mixed", "code", "unknown"}


@dataclass
class DataRecord:
    """Canonical record used across the data pipeline."""

    id: str
    text: str
    source: str = "unknown"
    language: str = "unknown"
    domain: str = "unknown"
    token_count: int = 0
    quality_score: float = 0.0
    dedup_signature: str = ""
    safety_flags: list[str] = field(default_factory=list)
    filtered: bool = False
    filter_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dictionary suitable for JSONL serialization."""

        return asdict(self)


def make_record_id(index: int, prefix: str = "doc") -> str:
    """Create a stable zero-padded record id."""

    return f"{prefix}_{index:06d}"


def _default_metadata(metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    base = {
        "url": "",
        "title": "",
        "parser": "",
        "version": "clean_v1",
    }
    if metadata:
        base.update(metadata)
    return base


def normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    """Normalize arbitrary input into the canonical schema.

    Extra fields are intentionally preserved. This makes the schema strict
    enough for pipeline contracts while still allowing diagnostics such as
    ``quality_bucket`` to be added later.
    """

    normalized = dict(record)
    normalized["id"] = str(normalized.get("id") or "")
    normalized["text"] = str(normalized.get("text") or "")
    source = normalized.get("source") or "unknown"
    normalized["source"] = source if source in VALID_SOURCES else "unknown"
    language = normalized.get("language") or "unknown"
    normalized["language"] = language if language in VALID_LANGUAGES else "unknown"
    normalized["domain"] = str(normalized.get("domain") or "unknown")
    normalized["token_count"] = int(normalized.get("token_count") or 0)
    normalized["quality_score"] = float(normalized.get("quality_score") or 0.0)
    normalized["dedup_signature"] = str(normalized.get("dedup_signature") or "")
    safety_flags = normalized.get("safety_flags") or []
    if isinstance(safety_flags, str):
        safety_flags = [safety_flags]
    normalized["safety_flags"] = list(safety_flags)
    normalized["filtered"] = bool(normalized.get("filtered", False))
    normalized["filter_reason"] = str(normalized.get("filter_reason") or "")
    normalized["metadata"] = _default_metadata(normalized.get("metadata") or {})
    return normalized


def validate_record(record: dict[str, Any]) -> bool:
    """Validate that a record satisfies the minimum canonical schema."""

    required = {
        "id",
        "text",
        "source",
        "language",
        "domain",
        "token_count",
        "quality_score",
        "dedup_signature",
        "safety_flags",
        "filtered",
        "filter_reason",
        "metadata",
    }
    if not required.issubset(record.keys()):
        return False
    if not isinstance(record["id"], str) or not isinstance(record["text"], str):
        return False
    if record["source"] not in VALID_SOURCES | {"unknown"}:
        return False
    if record["language"] not in VALID_LANGUAGES:
        return False
    if not isinstance(record["token_count"], int):
        return False
    if not isinstance(record["quality_score"], (int, float)):
        return False
    if not isinstance(record["safety_flags"], list):
        return False
    if not isinstance(record["filtered"], bool):
        return False
    if not isinstance(record["metadata"], dict):
        return False
    return True
