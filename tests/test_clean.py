from llm_dataforge.clean import basic_quality_filter, detect_pii, normalize_text


def test_normalize_text_removes_control_chars_and_compacts_spaces():
    text = "  hello\x00   world\r\n\r\n\r\n中文\t文本  "
    assert normalize_text(text) == "hello world\n\n中文 文本"


def test_detect_pii_finds_email_phone_and_api_key():
    flags = detect_pii("contact user@example.com, phone 13812345678, api_key='sk-test1234567890'")
    assert "email" in flags
    assert "phone" in flags
    assert "api_key" in flags


def test_basic_quality_filter_reasons():
    config = {"min_chars": 10, "max_urls": 2, "repeated_char_threshold": 5, "enable_pii_filter": True}
    assert basic_quality_filter("", config) == (True, "empty_text")
    assert basic_quality_filter("short", config) == (True, "too_short")
    assert basic_quality_filter("valid text http://a.com http://b.com http://c.com", config) == (True, "too_many_urls")
    assert basic_quality_filter("正常文本" + "哈" * 5 + " enough content", config) == (True, "repeated_chars")
    assert basic_quality_filter("valid text with email user@example.com", config) == (True, "pii_detected")
    assert basic_quality_filter("This is a normal useful sentence for the filter.", config) == (False, "")
