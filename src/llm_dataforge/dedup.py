"""Exact and small-scale approximate deduplication utilities."""

from __future__ import annotations

import hashlib
import re
from typing import Any


def normalize_for_hash(text: str) -> str:
    """Normalize text before hash-based exact deduplication."""

    value = str(text or "").lower().strip()
    value = re.sub(r"\s+", " ", value)
    return value


def md5_text(text: str) -> str:
    """Return MD5 hex digest for text."""

    return hashlib.md5(str(text).encode("utf-8")).hexdigest()


def sha1_text(text: str) -> str:
    """Return SHA1 hex digest for text."""

    return hashlib.sha1(str(text).encode("utf-8")).hexdigest()


def exact_dedup(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Remove exact duplicates using MD5 of normalized text."""

    seen: set[str] = set()
    kept: list[dict[str, Any]] = []
    duplicate_ids: list[str] = []

    for record in records:
        signature = md5_text(normalize_for_hash(record.get("text", "")))
        if signature in seen:
            duplicate_ids.append(str(record.get("id", "")))
            continue
        seen.add(signature)
        item = dict(record)
        item["dedup_signature"] = signature
        kept.append(item)

    stats = {
        "method": "md5_exact",
        "input_records": len(records),
        "output_records": len(kept),
        "duplicates_removed": len(records) - len(kept),
        "duplicate_ids": duplicate_ids,
    }
    return kept, stats


def get_char_ngrams(text: str, n: int = 5) -> set[str]:
    """Return character n-grams from normalized text."""

    value = normalize_for_hash(text)
    value = re.sub(r"\s+", "", value)
    if not value:
        return set()
    if len(value) <= n:
        return {value}
    return {value[i : i + n] for i in range(len(value) - n + 1)}


def jaccard_similarity(a: set[str], b: set[str]) -> float:
    """Compute Jaccard similarity for two sets."""

    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def approximate_dedup_small(
    records: list[dict[str, Any]], threshold: float = 0.85, ngram_size: int = 5
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Remove near duplicates using all-pairs Jaccard similarity.

    This is intentionally only for small demos. Production corpora should use
    scalable methods such as MinHash/SimHash with LSH instead of all-pairs
    comparison.
    """

    kept: list[dict[str, Any]] = []
    kept_ngrams: list[set[str]] = []
    duplicate_pairs: list[dict[str, Any]] = []
    comparisons = 0

    for record in records:
        grams = get_char_ngrams(record.get("text", ""), n=ngram_size)
        duplicate_of = None
        duplicate_score = 0.0
        for kept_record, kept_grams in zip(kept, kept_ngrams):
            comparisons += 1
            score = jaccard_similarity(grams, kept_grams)
            if score >= threshold:
                duplicate_of = kept_record
                duplicate_score = score
                break
        if duplicate_of is not None:
            duplicate_pairs.append(
                {
                    "record_id": record.get("id", ""),
                    "duplicate_of": duplicate_of.get("id", ""),
                    "similarity": round(duplicate_score, 4),
                }
            )
            continue
        kept.append(dict(record))
        kept_ngrams.append(grams)

    stats = {
        "method": "char_ngram_jaccard_small",
        "input_records": len(records),
        "output_records": len(kept),
        "duplicates_removed": len(records) - len(kept),
        "threshold": threshold,
        "ngram_size": ngram_size,
        "comparisons": comparisons,
        "duplicate_pairs": duplicate_pairs,
    }
    return kept, stats


def simple_simhash(text: str, bits: int = 64) -> int:
    """Compute a simple SimHash integer for demonstration."""

    weights = [0] * bits
    for token in re.findall(r"\w+|[^\w\s]", normalize_for_hash(text)):
        digest = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
        for i in range(bits):
            weights[i] += 1 if digest & (1 << i) else -1
    value = 0
    for i, weight in enumerate(weights):
        if weight >= 0:
            value |= 1 << i
    return value


def hamming_distance(a: int, b: int) -> int:
    """Return Hamming distance between two integers."""

    return (a ^ b).bit_count()
