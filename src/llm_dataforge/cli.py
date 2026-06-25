"""Command line interface for LLM-DataForge."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from .clean import clean_record
from .dedup import approximate_dedup_small, exact_dedup
from .ingest import generate_sample_data, load_raw_records
from .quality import add_quality_scores
from .report import generate_report
from .sampler import export_train_jsonl, sample_by_quality, sample_by_source_ratio
from .tokenize_stats import add_token_counts
from .utils import ensure_dir, load_yaml, read_jsonl, resolve_path, resolve_project_root, utc_now_string, write_jsonl


def _load_config(config_path: str | Path) -> tuple[dict[str, Any], Path]:
    config = load_yaml(config_path)
    root = resolve_project_root(config_path)
    return config, root


def _path_from_config(config: dict[str, Any], root: Path, section: str, key: str, default: str) -> Path:
    value = config.get(section, {}).get(key, default)
    return resolve_path(value, root)


def _clean_records(records: list[dict[str, Any]], config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cleaned = [clean_record(record, config) for record in records]
    reasons = Counter(record.get("filter_reason") for record in cleaned if record.get("filtered"))
    stats = {
        "input_records": len(records),
        "kept_records": sum(1 for record in cleaned if not record.get("filtered")),
        "filtered_records": sum(1 for record in cleaned if record.get("filtered")),
        "filter_reasons": {str(reason): count for reason, count in reasons.items()},
    }
    return cleaned, stats


def run_clean_command(input_path: str | Path, output_path: str | Path, config_path: str | Path) -> Path:
    """Run only ingest/parse/clean and write clean JSONL."""

    config, root = _load_config(config_path)
    input_resolved = resolve_path(input_path, root)
    output_resolved = resolve_path(output_path, root)
    raw_records = load_raw_records(input_resolved)
    cleaned_records, _ = _clean_records(raw_records, config)
    return write_jsonl(cleaned_records, output_resolved)


def run_pipeline(config_path: str | Path) -> dict[str, Path]:
    """Run the complete data quality pipeline from config."""

    config, root = _load_config(config_path)
    raw_dir = _path_from_config(config, root, "paths", "raw_dir", "data/raw")
    interim_dir = _path_from_config(config, root, "paths", "interim_dir", "data/interim")
    processed_dir = _path_from_config(config, root, "paths", "processed_dir", "outputs/processed")
    report_dir = _path_from_config(config, root, "paths", "report_dir", "outputs/reports")

    for directory in [raw_dir, interim_dir, processed_dir, report_dir]:
        ensure_dir(directory)

    raw_records = load_raw_records(raw_dir)
    if not raw_records:
        raise FileNotFoundError(f"No JSONL records found in raw_dir: {raw_dir}")

    metrics: dict[str, Any] = {
        "project": config.get("project", {"name": "LLM-DataForge"}),
        "run_time": utc_now_string(),
        "ingest": {
            "raw_records": len(raw_records),
            "records_by_source": dict(Counter(record.get("source", "unknown") for record in raw_records)),
        },
    }

    cleaned_records, cleaning_stats = _clean_records(raw_records, config)
    metrics["cleaning"] = cleaning_stats
    clean_path = write_jsonl(cleaned_records, interim_dir / "clean.jsonl")

    kept_records = [record for record in cleaned_records if not record.get("filtered")]
    dedup_cfg = config.get("dedup", {})
    dedup_input_count = len(kept_records)
    exact_stats: dict[str, Any] = {}
    approx_stats: dict[str, Any] = {}

    deduped_records = kept_records
    if dedup_cfg.get("exact", True):
        deduped_records, exact_stats = exact_dedup(deduped_records)
    if dedup_cfg.get("approximate", True):
        deduped_records, approx_stats = approximate_dedup_small(
            deduped_records,
            threshold=float(dedup_cfg.get("approximate_threshold", 0.85)),
            ngram_size=int(dedup_cfg.get("ngram_size", 5)),
        )

    metrics["dedup"] = {
        "input_records": dedup_input_count,
        "output_records": len(deduped_records),
        "duplicates_removed": dedup_input_count - len(deduped_records),
        "duplicate_rate": (dedup_input_count - len(deduped_records)) / dedup_input_count if dedup_input_count else 0,
        "exact": exact_stats,
        "approximate": approx_stats,
    }

    tokenized_records, token_stats = add_token_counts(deduped_records, config)
    metrics["tokenizer"] = token_stats

    scored_records, quality_stats = add_quality_scores(tokenized_records, config)
    metrics["quality"] = quality_stats
    processed_path = write_jsonl(scored_records, processed_dir / "processed.jsonl")

    quality_cfg = config.get("quality", {})
    sampling_cfg = config.get("sampling", {})
    quality_filtered = sample_by_quality(scored_records, min_score=float(quality_cfg.get("min_score", 0.45)))
    sampled_records = sample_by_source_ratio(
        quality_filtered,
        ratios=sampling_cfg.get("source_ratios", {}),
        max_records=int(sampling_cfg.get("max_records", 1000)),
    )
    train_path = export_train_jsonl(sampled_records, processed_dir / "train.jsonl")
    metrics["sampling"] = {
        "quality_candidates": len(quality_filtered),
        "train_records": len(sampled_records),
        "max_records": int(sampling_cfg.get("max_records", 1000)),
        "source_ratios": sampling_cfg.get("source_ratios", {}),
        "train_path": str(train_path),
    }

    report_paths = generate_report(
        sampled_records,
        output_dir=report_dir,
        metrics=metrics,
        train_path=train_path,
        all_records=cleaned_records,
        config=config,
    )

    return {
        "clean": clean_path,
        "processed": processed_path,
        "train": train_path,
        **report_paths,
    }


def run_report_command(input_path: str | Path, output_dir: str | Path) -> dict[str, Path]:
    """Generate a report from an existing train JSONL file."""

    records = read_jsonl(input_path)
    return generate_report(records, output_dir=output_dir, train_path=input_path)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""

    parser = argparse.ArgumentParser(prog="llm-dataforge", description="Lightweight LLM data quality pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sample_parser = subparsers.add_parser("generate-sample-data", help="Generate noisy sample raw JSONL files")
    sample_parser.add_argument("--output", default="data/raw", help="Output raw data directory")

    run_parser = subparsers.add_parser("run", help="Run the complete pipeline")
    run_parser.add_argument("--config", default="configs/pipeline.yaml", help="Pipeline YAML config")

    clean_parser = subparsers.add_parser("clean", help="Run only ingest/parse/clean")
    clean_parser.add_argument("--input", default="data/raw", help="Input raw JSONL file or directory")
    clean_parser.add_argument("--output", default="data/interim/clean.jsonl", help="Output clean JSONL file")
    clean_parser.add_argument("--config", default="configs/pipeline.yaml", help="Pipeline YAML config")

    report_parser = subparsers.add_parser("report", help="Generate report from train JSONL")
    report_parser.add_argument("--input", default="outputs/processed/train.jsonl", help="Input train JSONL file")
    report_parser.add_argument("--output", default="outputs/reports", help="Output report directory")

    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint."""

    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.command == "generate-sample-data":
        paths = generate_sample_data(args.output)
        for source, path in paths.items():
            print(f"wrote {source}: {path}")
    elif args.command == "run":
        paths = run_pipeline(args.config)
        for name, path in paths.items():
            print(f"{name}: {path}")
    elif args.command == "clean":
        path = run_clean_command(args.input, args.output, args.config)
        print(f"clean: {path}")
    elif args.command == "report":
        paths = run_report_command(args.input, args.output)
        for name, path in paths.items():
            print(f"{name}: {path}")
    else:  # pragma: no cover - argparse prevents this branch
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
