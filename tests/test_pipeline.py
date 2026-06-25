import yaml

from llm_dataforge.cli import run_pipeline
from llm_dataforge.ingest import generate_sample_data
from llm_dataforge.utils import read_jsonl


def test_generate_sample_data_and_run_pipeline(tmp_path):
    raw_dir = tmp_path / "data" / "raw"
    generate_sample_data(raw_dir)

    config = {
        "project": {"name": "LLM-DataForge", "version": "clean_v1"},
        "paths": {
            "raw_dir": str(raw_dir),
            "interim_dir": str(tmp_path / "data" / "interim"),
            "processed_dir": str(tmp_path / "outputs" / "processed"),
            "report_dir": str(tmp_path / "outputs" / "reports"),
        },
        "cleaning": {
            "min_chars": 50,
            "max_chars": 200000,
            "max_urls": 10,
            "repeated_char_threshold": 20,
            "enable_pii_filter": True,
        },
        "dedup": {"exact": True, "approximate": True, "approximate_threshold": 0.85, "ngram_size": 5},
        "tokenizer": {"use_hf_tokenizer": False, "model_name": "Qwen/Qwen2.5-0.5B"},
        "quality": {"min_score": 0.45},
        "sampling": {
            "max_records": 1000,
            "source_ratios": {"web": 0.35, "document": 0.25, "code": 0.20, "instruction": 0.20},
        },
    }
    config_path = tmp_path / "configs" / "pipeline.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")

    paths = run_pipeline(config_path)
    assert paths["train"].exists()
    assert paths["report"].exists()
    assert paths["metrics"].exists()
    assert (tmp_path / "outputs" / "reports" / "source_distribution.csv").exists()
    assert (tmp_path / "outputs" / "reports" / "filtered_reasons.csv").exists()
    assert (tmp_path / "outputs" / "reports" / "quality_distribution.csv").exists()
    assert len(read_jsonl(paths["train"])) > 0
