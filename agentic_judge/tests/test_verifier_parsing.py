from agentic_judge.core.verifier import _extract_urls, _parse_domain, _get_priority_urls


def test_extract_urls_from_instruction():
    text = "Go to https://www.amazon.com/dp/B09YRC9Y3G and compare prices"
    assert _extract_urls(text) == ["https://www.amazon.com/dp/B09YRC9Y3G"]


def test_extract_urls_multiple():
    text = "Check https://amazon.com and https://bestbuy.com for prices"
    urls = _extract_urls(text)
    assert "https://amazon.com" in urls
    assert "https://bestbuy.com" in urls


def test_extract_urls_empty():
    assert _extract_urls("") == []
    assert _extract_urls(None) == []


def test_parse_domain_standard():
    toml_content = 'version = "1.0"\n\n[metadata]\ncategory = "Shopping"\n'
    assert _parse_domain(toml_content) == "Shopping"


def test_parse_domain_multiword():
    toml_content = 'version = "1.0"\n\n[metadata]\ncategory = "Real Estate"\n'
    assert _parse_domain(toml_content) == "Real Estate"


def test_parse_domain_with_special_chars():
    toml_content = 'version = "1.0"\n\n[metadata]\ncategory = "Data & ML Engineering"\n'
    assert _parse_domain(toml_content) == "Data & ML Engineering"


def test_parse_domain_missing():
    toml_content = 'version = "1.0"\n'  # no [metadata] section
    assert _parse_domain(toml_content) == "Unknown"


def test_get_priority_urls_instruction_first():
    instruction_urls = ["https://amazon.com/product", "https://bestbuy.com"]
    result_urls = ["https://walmart.com", "https://target.com"]
    result = _get_priority_urls(instruction_urls, result_urls, max_urls=3)
    assert result[0] == "https://amazon.com/product"
    assert result[1] == "https://bestbuy.com"
    assert result[2] == "https://walmart.com"
    assert len(result) == 3


def test_get_priority_urls_deduplication():
    instruction_urls = ["https://amazon.com"]
    result_urls = ["https://amazon.com", "https://bestbuy.com"]
    result = _get_priority_urls(instruction_urls, result_urls, max_urls=3)
    assert result.count("https://amazon.com") == 1
    assert "https://bestbuy.com" in result


def test_get_priority_urls_cap_at_max():
    instruction_urls = ["https://a.com", "https://b.com", "https://c.com", "https://d.com"]
    result = _get_priority_urls(instruction_urls, [], max_urls=3)
    assert len(result) == 3


def test_get_priority_urls_empty():
    assert _get_priority_urls([], []) == []
