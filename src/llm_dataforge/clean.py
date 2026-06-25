"""Text normalization, language detection, PII detection, and quality filters."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from .schema import normalize_record


URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_PATTERN = re.compile(r"(?<!\d)(?:\+?\d{1,3}[-\s]?)?(?:1[3-9]\d{9}|\d{3}[-.\s]?\d{3}[-.\s]?\d{4})(?!\d)")
ID_CARD_PATTERN = re.compile(r"(?<!\d)(?:\d{15}|\d{17}[\dXx])(?!\d)")
API_KEY_PATTERN = re.compile(
    r"(?i)(?:api[_-]?key|access[_-]?key|secret|token|password)\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{8,}|sk-[A-Za-z0-9_\-]{10,}"
)
SENSITIVE_KEYWORD_PATTERN = re.compile(r"(?i)\b(password|secret|api[_-]?key|access[_-]?token|bearer\s+[A-Za-z0-9_\-.]+)\b")
CODE_SYMBOLS = set("{}[]();=<>+-*/_%:\\`|&^~#$")


def _cleaning_config(config: dict[str, Any] | None) -> dict[str, Any]:
    if not config:
        return {}
    return config.get("cleaning", config)


def normalize_text(text: str) -> str:
    """Normalize text without destroying useful document/code structure.

    Data cleaning must balance quality and diversity: aggressive rules can
    remove valuable long-tail language, domain terms, or code formatting.
    This function only removes control characters and obvious whitespace noise.
    """

    if text is None:
        return ""
    value = str(text).replace("\r\n", "\n").replace("\r", "\n")
    chars: list[str] = []
    for char in value:
        if char in {"\n", "\t"}:
            chars.append(char)
            continue
        if unicodedata.category(char).startswith("C"):
            continue
        chars.append(char)
    value = "".join(chars)
    value = re.sub(r"[ \t\f\v]+", " ", value)
    value = re.sub(r" *\n *", "\n", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def detect_language(text: str) -> str:
    """Detect coarse language category using transparent character ratios."""

    value = str(text or "")
    visible = [ch for ch in value if not ch.isspace()]
    if not visible:
        return "unknown"

    total = len(visible)
    zh = sum(1 for ch in visible if "\u4e00" <= ch <= "\u9fff")
    en = sum(1 for ch in visible if ch.isascii() and ch.isalpha())
    digits = sum(1 for ch in visible if ch.isdigit())
    code_symbols = sum(1 for ch in visible if ch in CODE_SYMBOLS)
    code_keywords = len(re.findall(r"\b(def|class|import|return|for|while|if|else|try|except|function|const|let|var)\b", value))

    zh_ratio = zh / total
    en_ratio = en / total
    code_ratio = code_symbols / total

    if (code_ratio > 0.16 and en + digits > 8) or code_keywords >= 2:
        return "code"
    if zh_ratio > 0.25 and en_ratio > 0.18:
        return "mixed"
    if zh_ratio > 0.3:
        return "zh"
    if en_ratio > 0.5:
        return "en"
    if zh_ratio > 0.08 and en_ratio > 0.08:
        return "mixed"
    return "unknown"


def detect_pii(text: str) -> list[str]:
    """Detect simple PII and secret-like patterns."""

    value = str(text or "")
    flags: list[str] = []
    if EMAIL_PATTERN.search(value):
        flags.append("email")
    if PHONE_PATTERN.search(value):
        flags.append("phone")
    if ID_CARD_PATTERN.search(value):
        flags.append("id_card")
    if API_KEY_PATTERN.search(value):
        flags.append("api_key")
    if SENSITIVE_KEYWORD_PATTERN.search(value):
        flags.append("sensitive_keyword")
    return list(dict.fromkeys(flags))


def _max_repeated_run(text: str) -> int:
    if not text:
        return 0
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
    return max_run


def _information_ratio(text: str) -> float:
    visible = [ch for ch in text if not ch.isspace()]
    if not visible:
        return 0.0
    informative = sum(1 for ch in visible if ch.isalnum() or "\u4e00" <= ch <= "\u9fff")
    return informative / len(visible)


def basic_quality_filter(text: str, config: dict[str, Any] | None) -> tuple[bool, str]:
    """Return whether text should be filtered and the first matched reason."""

    cfg = _cleaning_config(config)
    value = str(text or "")
    min_chars = int(cfg.get("min_chars", 50))
    max_chars = int(cfg.get("max_chars", 200_000))
    max_urls = int(cfg.get("max_urls", 10))
    repeated_char_threshold = int(cfg.get("repeated_char_threshold", 20))
    enable_pii_filter = bool(cfg.get("enable_pii_filter", True))

    if not value.strip():
        return True, "empty_text"
    if len(value) < min_chars:
        return True, "too_short"
    if len(value) > max_chars:
        return True, "too_long"
    if len(URL_PATTERN.findall(value)) > max_urls:
        return True, "too_many_urls"
    if _max_repeated_run(value) >= repeated_char_threshold:
        return True, "repeated_chars"
    if enable_pii_filter and detect_pii(value):
        return True, "pii_detected"

    visible = [ch for ch in value if not ch.isspace()]
    unique_ratio = len(set(visible)) / len(visible) if visible else 0.0
    if len(value) >= min_chars and (_information_ratio(value) < 0.2 or unique_ratio < 0.04):
        return True, "low_information"

    return False, ""


def clean_record(record: dict[str, Any], config: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize text, detect language/PII, and set filtering metadata."""

    cleaned = normalize_record(record)
    cleaned["text"] = normalize_text(cleaned.get("text", ""))
    detected_language = detect_language(cleaned["text"])
    cleaned["language"] = "code" if cleaned.get("source") == "code" else detected_language

    pii_flags = detect_pii(cleaned["text"])
    existing_flags = list(cleaned.get("safety_flags") or [])
    cleaned["safety_flags"] = list(dict.fromkeys(existing_flags + pii_flags))

    filtered, reason = basic_quality_filter(cleaned["text"], config)
    cleaned["filtered"] = filtered
    cleaned["filter_reason"] = reason
    cleaned.setdefault("metadata", {})
    cleaned["metadata"]["version"] = cleaned["metadata"].get("version", "clean_v1")
    return cleaned
