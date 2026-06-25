from llm_dataforge.quality import compute_quality_score, extract_features


def test_extract_features_basic():
    record = {
        "text": "Instruction: Explain data quality.\nOutput: Data quality affects model training.",
        "token_count": 12,
        "source": "instruction",
        "language": "en",
        "safety_flags": [],
    }
    features = extract_features(record)
    assert features["token_count"] == 12
    assert features["source"] == "instruction"
    assert features["has_instruction_structure"] is True


def test_high_quality_text_scores_high():
    features = extract_features(
        {
            "text": (
                "LLM data engineering includes cleaning, deduplication, token statistics, quality scoring, "
                "sampling, and reporting before a corpus is exported for training."
            ),
            "token_count": 28,
            "source": "document",
            "language": "en",
            "safety_flags": [],
        }
    )
    assert compute_quality_score(features, {}) >= 0.75


def test_pii_text_scores_low():
    features = extract_features(
        {
            "text": "Please keep this email user@example.com and phone 13812345678 in the training set.",
            "token_count": 20,
            "source": "web",
            "language": "en",
            "safety_flags": ["email", "phone"],
        }
    )
    assert compute_quality_score(features, {}) < 0.5
