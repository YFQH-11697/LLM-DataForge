"""Parsing and source-specific normalization for raw records."""

from __future__ import annotations

import re
from typing import Any

from bs4 import BeautifulSoup

from .schema import normalize_record


def _compact_whitespace(text: str) -> str:
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_html_to_text(html: str) -> str:
    """Extract readable text from an HTML document.

    The parser intentionally removes obvious boilerplate regions such as
    script, style, nav and footer. It is a lightweight demo parser, not a
    replacement for production readability extraction.
    """

    if not html:
        return ""
    soup = BeautifulSoup(str(html), "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "noscript", "header"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return _compact_whitespace("\n".join(lines))


def parse_instruction_record(record: dict[str, Any]) -> str:
    """Convert instruction/input/output style data into a single text field."""

    instruction = record.get("instruction") or record.get("prompt") or record.get("query") or ""
    input_text = record.get("input") or record.get("context") or ""
    output = record.get("output") or record.get("response") or record.get("answer") or ""

    if not instruction and not output:
        return str(record.get("text") or "")

    parts = [f"Instruction: {str(instruction).strip()}"]
    if str(input_text).strip():
        parts.append(f"Input: {str(input_text).strip()}")
    parts.append(f"Output: {str(output).strip()}")
    return "\n".join(parts)


def _infer_domain(source: str, record: dict[str, Any]) -> str:
    domain = record.get("domain")
    if domain:
        return str(domain)
    if source == "code":
        return "code"
    if source == "document":
        return "education"
    if source == "web":
        return "technology"
    return "general"


def parse_raw_record(record: dict[str, Any]) -> dict[str, Any]:
    """Parse one raw record into the unified schema."""

    source = str(record.get("source") or "unknown")
    text = str(record.get("text") or "")
    parser_name = "plain_text"

    if source == "web":
        html = record.get("html")
        if html is not None:
            text = parse_html_to_text(str(html))
            parser_name = "beautifulsoup_html"
        elif "<html" in text.lower() or "<p" in text.lower():
            text = parse_html_to_text(text)
            parser_name = "beautifulsoup_html"
    elif source == "document":
        text = str(record.get("content") or record.get("body") or text)
        parser_name = "document_plain_text"
    elif source == "code":
        text = str(record.get("code") or text)
        parser_name = "code_plain_text"
    elif source == "instruction":
        text = parse_instruction_record(record)
        parser_name = "instruction_template"

    metadata = dict(record.get("metadata") or {})
    metadata.update(
        {
            "url": record.get("url") or metadata.get("url", ""),
            "title": record.get("title") or metadata.get("title", ""),
            "parser": parser_name,
            "version": metadata.get("version", "clean_v1"),
        }
    )

    normalized = normalize_record(
        {
            "id": str(record.get("id") or ""),
            "text": text,
            "source": source,
            "language": record.get("language", "unknown"),
            "domain": _infer_domain(source, record),
            "metadata": metadata,
        }
    )
    return normalized
