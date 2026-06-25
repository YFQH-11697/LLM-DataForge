from llm_dataforge.dedup import exact_dedup, get_char_ngrams, jaccard_similarity, md5_text


def test_md5_text_is_stable():
    assert md5_text("abc") == md5_text("abc")
    assert md5_text("abc") != md5_text("abcd")


def test_exact_dedup_removes_duplicate_text():
    records = [
        {"id": "a", "text": "Hello   World"},
        {"id": "b", "text": "hello world"},
        {"id": "c", "text": "Different text"},
    ]
    deduped, stats = exact_dedup(records)
    assert len(deduped) == 2
    assert stats["duplicates_removed"] == 1
    assert all(record["dedup_signature"] for record in deduped)


def test_get_char_ngrams():
    assert get_char_ngrams("abcdef", n=3) == {"abc", "bcd", "cde", "def"}
    assert get_char_ngrams("ab", n=5) == {"ab"}


def test_jaccard_similarity():
    assert jaccard_similarity({"a", "b"}, {"a", "b"}) == 1.0
    assert jaccard_similarity({"a"}, {"b"}) == 0.0
    assert jaccard_similarity({"a", "b"}, {"b", "c"}) == 1 / 3
