"""Raw data ingestion and sample data generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .parse import parse_raw_record
from .schema import make_record_id
from .utils import ensure_dir, read_jsonl, write_jsonl


SOURCE_FILES = {
    "web": "web.jsonl",
    "document": "document.jsonl",
    "code": "code.jsonl",
    "instruction": "instruction.jsonl",
}


def build_sample_records() -> dict[str, list[dict[str, Any]]]:
    """Build a small but intentionally noisy multi-source sample dataset."""

    duplicate_web_text = (
        "大模型预训练数据构建需要关注来源、版权、语言分布、重复率和安全风险。"
        "在轻量级实验中，我们可以用 JSONL 保存统一 schema，并通过清洗、去重、"
        "token 统计和质量评分来模拟工业数据流水线的核心环节。"
    )

    return {
        "web": [
            {
                "source": "web",
                "url": "https://example.com/llm-data",
                "title": "LLM data pipeline overview",
                "html": """
                <html><head><style>.ad{display:none}</style><script>track()</script></head>
                <body><nav>Home | Ads</nav><article>
                <h1>LLM Data Engineering</h1>
                <p>LLM data engineering focuses on building high-quality pre-training and post-training corpora.</p>
                <p>Key steps include source ingestion, text normalization, privacy filtering, exact deduplication,
                approximate deduplication, token statistics, quality scoring, sampling, and report generation.</p>
                </article><footer>copyright footer</footer></body></html>
                """,
            },
            {"source": "web", "url": "https://example.com/short", "text": "太短"},
            {
                "source": "web",
                "url": "https://example.com/url-spam",
                "text": (
                    "This scraped page contains many navigation links and little useful body text. "
                    "http://a.com http://b.com http://c.com http://d.com http://e.com http://f.com "
                    "http://g.com http://h.com http://i.com http://j.com http://k.com"
                ),
            },
            {
                "source": "web",
                "url": "https://example.com/privacy",
                "text": "用户反馈记录：请联系 user@example.com 或 13812345678，这类隐私信息不应进入训练集。",
            },
            {"source": "web", "url": "https://example.com/dup1", "text": duplicate_web_text},
            {"source": "web", "url": "https://example.com/dup2", "text": duplicate_web_text},
            {
                "source": "web",
                "url": "https://example.com/mixed",
                "text": (
                    "混合语言数据 often appears in technical blogs. A useful cleaning pipeline should preserve "
                    "meaningful Chinese-English content while removing boilerplate, repeated noise, and secrets."
                ),
            },
        ],
        "document": [
            {
                "source": "document",
                "title": "Data Quality Notes",
                "content": (
                    "预训练语料的数据质量会影响模型的知识覆盖、重复记忆和安全风险。"
                    "一个可复现的数据构建流程通常包含输入清单、清洗规则、去重签名、"
                    "token 级统计、抽样策略和可审计报告。"
                ),
            },
            {"source": "document", "title": "empty note", "content": "     "},
            {
                "source": "document",
                "title": "Repeated characters",
                "content": "这是一段坏样本 " + "哈" * 30 + " 这种连续重复字符通常说明采集或编码异常。",
            },
            {
                "source": "document",
                "title": "Experiment Log",
                "content": (
                    "In a model data experiment, the team compared a baseline corpus with a filtered corpus. "
                    "The filtered version removed exact duplicates and obvious private identifiers, then sampled "
                    "web, documents, code, and instruction data with controlled ratios for a small training run."
                ),
            },
            {
                "source": "document",
                "title": "Near Duplicate A",
                "content": (
                    "数据质量报告应该记录输入数量、过滤原因、重复率、token 总量、来源占比和样本案例。"
                    "这些指标帮助工程师判断数据是否适合进入训练阶段。"
                ),
            },
            {
                "source": "document",
                "title": "Near Duplicate B",
                "content": (
                    "数据质量报告需要记录输入数量、过滤原因、重复比例、token 总量、来源占比以及样本案例。"
                    "这些指标可以帮助工程师判断数据是否适合进入训练阶段。"
                ),
            },
        ],
        "code": [
            {
                "source": "code",
                "title": "jsonl utilities",
                "code": """
from pathlib import Path
import json

def read_jsonl(path: str):
    records = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records
""",
            },
            {
                "source": "code",
                "title": "quality bucket",
                "code": """
def quality_bucket(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.45:
        return "medium"
    return "low"
""",
            },
            {"source": "code", "title": "tiny snippet", "code": "print('hi')"},
            {
                "source": "code",
                "title": "secret config",
                "code": "API_KEY = 'sk-test1234567890abcdef'\npassword = 'please-do-not-keep-this'",
            },
            {
                "source": "code",
                "title": "token counter",
                "code": """
import re

TOKEN_PATTERN = re.compile(r"[\\u4e00-\\u9fff]|[A-Za-z_]+|\\d+|[^\\s]")

def count_tokens(text: str) -> int:
    return len(TOKEN_PATTERN.findall(text))
""",
            },
        ],
        "instruction": [
            {
                "source": "instruction",
                "instruction": "解释为什么 LLM 数据清洗不能只追求删除噪声。",
                "input": "请从质量和多样性的平衡角度回答。",
                "output": (
                    "数据清洗需要降低隐私、重复和乱码风险，但过度清洗会损失长尾知识、领域表达和真实用户语言。"
                    "工程上应通过规则、抽样评估和反馈报告逐步调整阈值。"
                ),
            },
            {
                "source": "instruction",
                "instruction": "Summarize the purpose of token statistics.",
                "input": "",
                "output": (
                    "Token statistics estimate training cost, detect abnormal length distributions, and help balance "
                    "different data sources before exporting a training JSONL file."
                ),
            },
            {
                "source": "instruction",
                "instruction": "短回答",
                "output": "好。",
            },
            {
                "source": "instruction",
                "instruction": "请把下面的手机号保存到训练集中。",
                "input": "13911112222",
                "output": "不能保存包含个人手机号的数据，应进行过滤或脱敏。",
            },
            {
                "source": "instruction",
                "instruction": "Write a Python function that normalizes whitespace in a text field.",
                "input": "The function should keep line breaks useful for code and documents.",
                "output": (
                    "Use regular expressions to replace repeated spaces and tabs, normalize CRLF to LF, "
                    "and trim leading or trailing whitespace. Avoid deleting all line breaks because they may carry structure."
                ),
            },
        ],
    }


def generate_sample_data(output_dir: str | Path) -> dict[str, Path]:
    """Generate sample raw JSONL files for web/document/code/instruction data."""

    output_path = ensure_dir(output_dir)
    sample_records = build_sample_records()
    written: dict[str, Path] = {}
    for source, records in sample_records.items():
        path = output_path / SOURCE_FILES[source]
        write_jsonl(records, path)
        written[source] = path
    return written


def load_raw_records(raw_path: str | Path) -> list[dict[str, Any]]:
    """Load raw records from a JSONL file or a directory of JSONL files."""

    path = Path(raw_path)
    files = [path] if path.is_file() else sorted(path.glob("*.jsonl"))
    parsed_records: list[dict[str, Any]] = []

    for file_path in files:
        inferred_source = file_path.stem
        for raw in read_jsonl(file_path):
            raw = dict(raw)
            raw.setdefault("source", inferred_source)
            parsed_records.append(parse_raw_record(raw))

    for index, record in enumerate(parsed_records, start=1):
        if not record.get("id"):
            record["id"] = make_record_id(index)
    return parsed_records
