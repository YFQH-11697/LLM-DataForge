"""Rule-based quality feature extraction and scoring."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from .clean import CODE_SYMBOLS, URL_PATTERN, detect_pii


def _repeat_char_ratio(text: str) -> float:
    if not text:
        return 0.0
    max_run = 1
    current = 1
    previous = text[0]
    for char in text[1:]:
        if char == previous and not char.isspace():
            current += 1
            max_run = max(max_run, current)
        else:
            current = 1
            previous = char
    return max_run / max(len(text), 1)


def extract_features(record: dict[str, Any]) -> dict[str, Any]:
    """Extract transparent quality features from one record."""

    text = str(record.get("text") or "")
    length = len(text)
    visible = [ch for ch in text if not ch.isspace()]
    visible_len = len(visible)
    code_symbols = sum(1 for ch in visible if ch in CODE_SYMBOLS)
    alpha_digits = sum(1 for ch in visible if ch.isalnum() or "\u4e00" <= ch <= "\u9fff")
    safety_flags = record.get("safety_flags") or []
    pii_flag = bool(safety_flags) or bool(detect_pii(text))

    return {
        "token_count": int(record.get("token_count") or 0),
        "text_length": length,
        "url_count": len(URL_PATTERN.findall(text)),
        "repeat_char_ratio": _repeat_char_ratio(text),
        "pii_flag": pii_flag,
        "language": record.get("language", "unknown"),
        "source": record.get("source", "unknown"),
        "code_symbol_ratio": code_symbols / visible_len if visible_len else 0.0,
        "alpha_digit_ratio": alpha_digits / visible_len if visible_len else 0.0,
        "has_instruction_structure": "Instruction:" in text and "Output:" in text,
    }


def compute_quality_score(features: dict[str, Any], config: dict[str, Any] | None) -> float:
    """Compute a bounded rule-based quality score in [0, 1]."""

    score = 1.0
    token_count = int(features.get("token_count") or 0)
    source = str(features.get("source") or "unknown")
    language = str(features.get("language") or "unknown")
    url_count = int(features.get("url_count") or 0)
    repeat_ratio = float(features.get("repeat_char_ratio") or 0.0)
    code_symbol_ratio = float(features.get("code_symbol_ratio") or 0.0)
    alpha_digit_ratio = float(features.get("alpha_digit_ratio") or 0.0)

    if token_count <= 0:
        score -= 0.6
    elif token_count < 15:
        score -= 0.3
    elif token_count < 32:
        score -= 0.12

    if url_count > 10:
        score -= 0.35
    elif url_count > 3:
        score -= 0.15

    if repeat_ratio > 0.25:
        score -= 0.35
    elif repeat_ratio > 0.12:
        score -= 0.15

    if bool(features.get("pii_flag")):
        score -= 0.55

    if language == "unknown":
        score -= 0.15

    if source == "code":
        if code_symbol_ratio > 0.55:
            score -= 0.1
    else:
        if code_symbol_ratio > 0.3:
            score -= 0.18
        if alpha_digit_ratio < 0.35:
            score -= 0.18

    if source == "instruction":
        if features.get("has_instruction_structure"):
            score += 0.08
        else:
            score -= 0.12

    return round(max(0.0, min(1.0, score)), 4)


def _quality_bucket(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.45:
        return "medium"
    return "low"


def add_quality_scores(records: list[dict[str, Any]], config: dict[str, Any] | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Add quality_score and quality_bucket to records."""

    output: list[dict[str, Any]] = []
    scores: list[float] = []
    bucket_counter: Counter[str] = Counter()
    source_bucket_counter: Counter[str] = Counter()

    for record in records:
        item = dict(record)
        features = extract_features(item)
        score = compute_quality_score(features, config)
        bucket = _quality_bucket(score)
        item["quality_score"] = score
        item["quality_bucket"] = bucket
        item.setdefault("metadata", {})
        item["metadata"]["quality_bucket"] = bucket
        output.append(item)
        scores.append(score)
        bucket_counter[bucket] += 1
        source_bucket_counter[f"{item.get('source', 'unknown')}:{bucket}"] += 1

    stats = {
        "quality_summary": {
            "avg_score": round(sum(scores) / len(scores), 4) if scores else 0,
            "min_score": min(scores) if scores else 0,
            "max_score": max(scores) if scores else 0,
        },
        "quality_distribution": dict(bucket_counter),
        "quality_by_source_bucket": dict(source_bucket_counter),
    }
    return output, stats
