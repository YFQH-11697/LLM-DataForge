"""Markdown, JSON, CSV, and optional chart report generation."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .utils import ensure_dir, safe_divide, truncate_text, utc_now_string, write_csv, write_json


def _bucket(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.45:
        return "medium"
    return "low"


def _compute_report_tables(
    records: list[dict[str, Any]], all_records: list[dict[str, Any]] | None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    source_counts: Counter[str] = Counter()
    source_tokens: Counter[str] = Counter()
    quality_counts: Counter[str] = Counter()
    filtered_reasons: Counter[str] = Counter()
    safety_flags: Counter[str] = Counter()

    for record in records:
        source = str(record.get("source", "unknown"))
        token_count = int(record.get("token_count") or 0)
        source_counts[source] += 1
        source_tokens[source] += token_count
        quality_counts[str(record.get("quality_bucket") or _bucket(float(record.get("quality_score") or 0.0)))] += 1

    for record in all_records or []:
        if record.get("filtered"):
            filtered_reasons[str(record.get("filter_reason") or "unknown")] += 1
        for flag in record.get("safety_flags") or []:
            safety_flags[str(flag)] += 1

    total_tokens = sum(source_tokens.values())
    source_rows = [
        {
            "source": source,
            "record_count": source_counts[source],
            "token_count": source_tokens[source],
            "token_share": round(safe_divide(source_tokens[source], total_tokens), 4),
        }
        for source in sorted(source_counts)
    ]
    filtered_rows = [
        {"filter_reason": reason, "count": count}
        for reason, count in sorted(filtered_reasons.items(), key=lambda item: (-item[1], item[0]))
    ]
    quality_rows = [
        {"quality_bucket": bucket, "count": quality_counts.get(bucket, 0)}
        for bucket in ["high", "medium", "low"]
    ]
    summary = {
        "source_counts": dict(source_counts),
        "source_tokens": dict(source_tokens),
        "quality_distribution": dict(quality_counts),
        "filtered_reasons": dict(filtered_reasons),
        "safety_flags": dict(safety_flags),
    }
    return source_rows, filtered_rows, quality_rows, summary


def _write_optional_charts(output_dir: Path, source_rows: list[dict[str, Any]], quality_rows: list[dict[str, Any]]) -> list[str]:
    generated: list[str] = []
    try:
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - optional dependency/environment
        return generated

    if source_rows:
        plt.figure(figsize=(7, 4))
        plt.bar([row["source"] for row in source_rows], [row["token_count"] for row in source_rows])
        plt.title("Token Count by Source")
        plt.tight_layout()
        path = output_dir / "source_token_distribution.png"
        plt.savefig(path)
        plt.close()
        generated.append(str(path))

    if quality_rows:
        plt.figure(figsize=(6, 4))
        plt.bar([row["quality_bucket"] for row in quality_rows], [row["count"] for row in quality_rows])
        plt.title("Quality Distribution")
        plt.tight_layout()
        path = output_dir / "quality_distribution.png"
        plt.savefig(path)
        plt.close()
        generated.append(str(path))

    return generated


def generate_report(
    records: list[dict[str, Any]],
    output_dir: str | Path,
    metrics: dict[str, Any] | None = None,
    train_path: str | Path | None = None,
    all_records: list[dict[str, Any]] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Generate Markdown report plus JSON/CSV metric artifacts."""

    out_dir = ensure_dir(output_dir)
    metrics = dict(metrics or {})
    project_cfg = (config or {}).get("project", {})
    project_name = project_cfg.get("name") or metrics.get("project", {}).get("name") or "LLM-DataForge"

    source_rows, filtered_rows, quality_rows, table_summary = _compute_report_tables(records, all_records)
    chart_paths = _write_optional_charts(out_dir, source_rows, quality_rows)

    raw_total = metrics.get("ingest", {}).get("raw_records", len(all_records or records))
    cleaning = metrics.get("cleaning", {})
    filtered_count = cleaning.get("filtered_records", sum(row["count"] for row in filtered_rows))
    kept_after_clean = cleaning.get("kept_records", raw_total - filtered_count)
    dedup_before = metrics.get("dedup", {}).get("input_records", kept_after_clean)
    dedup_after = metrics.get("dedup", {}).get("output_records", len(records))
    duplicate_rate = metrics.get("dedup", {}).get("duplicate_rate", safe_divide(dedup_before - dedup_after, dedup_before))
    token_stats = metrics.get("tokenizer", {})
    total_tokens = token_stats.get("total_tokens", sum(row["token_count"] for row in source_rows))
    output_train_path = str(train_path or metrics.get("sampling", {}).get("train_path", ""))

    high_examples = sorted(records, key=lambda item: float(item.get("quality_score") or 0.0), reverse=True)[:3]
    filtered_examples = [record for record in all_records or [] if record.get("filtered")][:5]

    report_lines = [
        f"# {project_name} Data Quality Report",
        "",
        f"- Run time: {utc_now_string()}",
        f"- Output train file: `{output_train_path}`",
        "",
        "## 1. Data Processing Pipeline",
        "",
        "`raw data -> ingest/parse -> clean -> dedup -> tokenize -> quality score -> sample -> export train.jsonl -> generate report`",
        "",
        "## 2. Core Metrics",
        "",
        f"- Input raw records: {raw_total}",
        f"- Records kept after cleaning: {kept_after_clean}",
        f"- Filtered records: {filtered_count}",
        f"- Dedup records before/after: {dedup_before} -> {dedup_after}",
        f"- Duplicate rate: {duplicate_rate:.2%}",
        f"- Train records: {len(records)}",
        f"- Total tokens: {total_tokens}",
        "",
        "## 3. Filter Reason Distribution",
        "",
    ]

    if filtered_rows:
        for row in filtered_rows:
            report_lines.append(f"- {row['filter_reason']}: {row['count']}")
    else:
        report_lines.append("- No filtered reason data available.")

    report_lines.extend(["", "## 4. Token Share by Source", ""])
    if source_rows:
        for row in source_rows:
            report_lines.append(
                f"- {row['source']}: records={row['record_count']}, tokens={row['token_count']}, share={row['token_share']:.2%}"
            )
    else:
        report_lines.append("- No source distribution data available.")

    report_lines.extend(["", "## 5. Quality Score Distribution", ""])
    for row in quality_rows:
        report_lines.append(f"- {row['quality_bucket']}: {row['count']}")

    report_lines.extend(["", "## 6. Safety Flags", ""])
    safety_flags = table_summary["safety_flags"]
    if safety_flags:
        for flag, count in sorted(safety_flags.items(), key=lambda item: (-item[1], item[0])):
            report_lines.append(f"- {flag}: {count}")
    else:
        report_lines.append("- No safety flags in the exported set or no full cleaning records were provided.")

    report_lines.extend(["", "## 7. Typical High-Quality Samples", ""])
    if high_examples:
        for idx, record in enumerate(high_examples, start=1):
            report_lines.append(
                f"{idx}. source={record.get('source')}, score={record.get('quality_score')}: {truncate_text(record.get('text', ''))}"
            )
    else:
        report_lines.append("- No high-quality examples available.")

    report_lines.extend(["", "## 8. Typical Filtered Samples", ""])
    if filtered_examples:
        for idx, record in enumerate(filtered_examples, start=1):
            report_lines.append(
                f"{idx}. reason={record.get('filter_reason')}, source={record.get('source')}: {truncate_text(record.get('text', ''))}"
            )
    else:
        report_lines.append("- Filtered sample details are unavailable when the report command only receives train.jsonl.")

    report_lines.extend(
        [
            "",
            "## 9. Limitations",
            "",
            "- This project is a lightweight teaching/interview pipeline, not a trillion-token production system.",
            "- Approximate deduplication uses all-pairs Jaccard similarity only for small demo datasets.",
            "- Rule-based quality_score is interpretable but cannot replace model-based quality classifiers or training feedback.",
            "- PII detection uses regular expressions and does not cover every sensitive-data category.",
            "- The pipeline is single-machine Python; production systems should use distributed processing and versioned storage.",
            "",
            "## 10. Next Steps",
            "",
            "- Replace small-scale approximate deduplication with MinHash/SimHash plus LSH.",
            "- Store large corpora in Parquet and process them with Spark or Ray.",
            "- Connect a real model tokenizer, perplexity model, quality classifier, and training/evaluation feedback.",
            "- Add data version manifests, manual audit UI, and source-level governance metadata.",
        ]
    )

    if chart_paths:
        report_lines.extend(["", "## 11. Optional Charts", ""])
        for chart_path in chart_paths:
            report_lines.append(f"- `{chart_path}`")

    report_path = out_dir / "data_quality_report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    metrics_output = {
        **metrics,
        "report_generated_at": utc_now_string(),
        "tables": table_summary,
        "output_train_path": output_train_path,
    }
    metrics_path = write_json(metrics_output, out_dir / "metrics.json")
    source_csv = write_csv(source_rows, out_dir / "source_distribution.csv", ["source", "record_count", "token_count", "token_share"])
    filtered_csv = write_csv(filtered_rows, out_dir / "filtered_reasons.csv", ["filter_reason", "count"])
    quality_csv = write_csv(quality_rows, out_dir / "quality_distribution.csv", ["quality_bucket", "count"])

    return {
        "report": report_path,
        "metrics": metrics_path,
        "source_distribution": source_csv,
        "filtered_reasons": filtered_csv,
        "quality_distribution": quality_csv,
    }
