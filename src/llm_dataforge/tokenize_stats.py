"""Tokenizer wrappers and token statistics."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any


TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]|[A-Za-z_]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?|[^\s]")


class SimpleTokenizer:
    """Fallback tokenizer that works offline.

    Chinese text is counted roughly by character, English by words, and code by
    words plus symbols. It is an approximation for engineering diagnostics, not
    a replacement for the actual model tokenizer used in training.
    """

    name = "simple_fallback_tokenizer"

    def tokenize(self, text: str) -> list[str]:
        """Tokenize text with simple regex rules."""

        return TOKEN_PATTERN.findall(str(text or ""))

    def count(self, text: str) -> int:
        """Count approximate tokens."""

        return len(self.tokenize(text))


class HFTokenizerWrapper:
    """Hugging Face tokenizer wrapper with automatic fallback."""

    def __init__(self, model_name: str, use_hf_tokenizer: bool = False):
        self.model_name = model_name
        self.backend_name = SimpleTokenizer.name
        self.error = ""
        self.tokenizer: Any = SimpleTokenizer()

        if use_hf_tokenizer:
            try:
                from transformers import AutoTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                self.backend_name = f"hf:{model_name}"
            except Exception as exc:  # pragma: no cover - depends on local/network environment
                self.tokenizer = SimpleTokenizer()
                self.backend_name = SimpleTokenizer.name
                self.error = str(exc)

    def tokenize(self, text: str) -> list[Any]:
        """Tokenize text with HF if available, otherwise fallback."""

        if isinstance(self.tokenizer, SimpleTokenizer):
            return self.tokenizer.tokenize(text)
        return self.tokenizer.tokenize(str(text or ""))

    def count(self, text: str) -> int:
        """Count tokens with the active backend."""

        if isinstance(self.tokenizer, SimpleTokenizer):
            return self.tokenizer.count(text)
        return len(self.tokenizer.encode(str(text or ""), add_special_tokens=False))


def count_tokens(text: str, tokenizer: Any) -> int:
    """Count tokens with any tokenizer object exposing count or tokenize."""

    if hasattr(tokenizer, "count"):
        return int(tokenizer.count(text))
    return len(tokenizer.tokenize(text))


def _length_bucket(token_count: int) -> str:
    if token_count <= 32:
        return "0-32"
    if token_count <= 128:
        return "33-128"
    if token_count <= 512:
        return "129-512"
    if token_count <= 2048:
        return "513-2048"
    return "2049+"


def add_token_counts(records: list[dict[str, Any]], config: dict[str, Any] | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Add token_count to records and return aggregate token statistics."""

    tokenizer_cfg = (config or {}).get("tokenizer", config or {})
    tokenizer = HFTokenizerWrapper(
        model_name=tokenizer_cfg.get("model_name", "Qwen/Qwen2.5-0.5B"),
        use_hf_tokenizer=bool(tokenizer_cfg.get("use_hf_tokenizer", False)),
    )

    output: list[dict[str, Any]] = []
    token_counts: list[int] = []
    token_by_source: dict[str, int] = defaultdict(int)
    length_distribution: dict[str, int] = defaultdict(int)

    for record in records:
        item = dict(record)
        token_count = count_tokens(item.get("text", ""), tokenizer)
        item["token_count"] = token_count
        output.append(item)
        token_counts.append(token_count)
        token_by_source[str(item.get("source", "unknown"))] += token_count
        length_distribution[_length_bucket(token_count)] += 1

    total_tokens = sum(token_counts)
    stats = {
        "tokenizer": tokenizer.backend_name,
        "tokenizer_error": tokenizer.error,
        "total_tokens": total_tokens,
        "avg_tokens": round(total_tokens / len(token_counts), 2) if token_counts else 0,
        "min_tokens": min(token_counts) if token_counts else 0,
        "max_tokens": max(token_counts) if token_counts else 0,
        "token_count_by_source": dict(token_by_source),
        "length_distribution": dict(length_distribution),
    }
    return output, stats
