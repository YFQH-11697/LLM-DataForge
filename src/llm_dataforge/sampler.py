"""Sampling and train JSONL export."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from .utils import write_jsonl


def sample_by_quality(records: list[dict[str, Any]], min_score: float) -> list[dict[str, Any]]:
    """Keep non-filtered records whose quality score is above min_score."""

    return [
        record
        for record in records
        if not record.get("filtered", False) and float(record.get("quality_score") or 0.0) >= float(min_score)
    ]


def sample_by_source_ratio(records: list[dict[str, Any]], ratios: dict[str, float], max_records: int) -> list[dict[str, Any]]:
    """Sample records according to source ratios with deterministic ranking."""

    if max_records <= 0:
        return []
    candidates_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        candidates_by_source[str(record.get("source", "unknown"))].append(record)

    for source in candidates_by_source:
        candidates_by_source[source] = sorted(
            candidates_by_source[source],
            key=lambda item: (float(item.get("quality_score") or 0.0), int(item.get("token_count") or 0)),
            reverse=True,
        )

    if not ratios:
        return sorted(
            records,
            key=lambda item: (float(item.get("quality_score") or 0.0), int(item.get("token_count") or 0)),
            reverse=True,
        )[:max_records]

    total_ratio = sum(float(value) for value in ratios.values()) or 1.0
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    for source, ratio in ratios.items():
        target = int(round(max_records * (float(ratio) / total_ratio)))
        if ratio > 0 and candidates_by_source.get(source):
            target = max(1, target)
        for record in candidates_by_source.get(source, [])[:target]:
            if len(selected) >= max_records:
                break
            selected.append(record)
            selected_ids.add(str(record.get("id", "")))

    if len(selected) < max_records:
        remaining = [
            record
            for record in sorted(
                records,
                key=lambda item: (float(item.get("quality_score") or 0.0), int(item.get("token_count") or 0)),
                reverse=True,
            )
            if str(record.get("id", "")) not in selected_ids
        ]
        selected.extend(remaining[: max_records - len(selected)])

    return selected[:max_records]


def export_train_jsonl(records: list[dict[str, Any]], output_path: str | Path) -> Path:
    """Export minimal train records to JSONL."""

    train_records: list[dict[str, Any]] = []
    for record in records:
        train_records.append(
            {
                "text": record.get("text", ""),
                "source": record.get("source", "unknown"),
                "token_count": int(record.get("token_count") or 0),
                "quality_score": float(record.get("quality_score") or 0.0),
                "metadata": record.get("metadata", {}),
            }
        )
    return write_jsonl(train_records, output_path)
