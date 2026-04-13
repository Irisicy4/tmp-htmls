from agentic_judge.analysis.report import format_report, HEADLINE_TEMPLATE


SAMPLE_REPORT = {
    "total_sampled": 30,
    "verifiable_count": 22,
    "unverifiable_count": 8,
    "unverifiable_pct": 26.67,
    "unverifiable_by_reason": {
        "no_urls": {"count": 5, "pct_of_unverifiable": 62.5, "pct_of_total": 16.67},
        "navigation_error": {"count": 2, "pct_of_unverifiable": 25.0, "pct_of_total": 6.67},
        "gpt_uncertain": {"count": 1, "pct_of_unverifiable": 12.5, "pct_of_total": 3.33},
    },
    "agreement_rate": 72.73,
    "catch_rate": 33.33,
    "joint_pass_rate": 59.09,
    "agentic_verification_rate": 68.18,
    "category_breakdown": {
        "Shopping": {
            "n_verifiable": 10,
            "static_pass_rate": 80.0,
            "agentic_pass_rate": 70.0,
            "agreement_rate": 80.0,
            "catch_rate": 25.0,
        },
        "Finance & Economics": {
            "n_verifiable": 6,
            "static_pass_rate": 66.67,
            "agentic_pass_rate": 50.0,
            "agreement_rate": 66.67,
            "catch_rate": 50.0,
        },
    },
}


def test_format_report_contains_categories():
    output = format_report(SAMPLE_REPORT)
    assert "Shopping" in output
    assert "Finance & Economics" in output


def test_format_report_contains_overall_row():
    output = format_report(SAMPLE_REPORT)
    assert "Overall" in output


def test_format_report_contains_headline_numbers():
    output = format_report(SAMPLE_REPORT)
    assert "72.73%" in output
    assert "33.33%" in output
    assert "59.09%" in output


def test_format_report_contains_unverifiable_breakdown():
    output = format_report(SAMPLE_REPORT)
    assert "no_urls" in output
    assert "navigation_error" in output
    assert "gpt_uncertain" in output


def test_headline_template_interpolation():
    text = HEADLINE_TEMPLATE.format(
        agreement_rate=72.73,
        catch_rate=33.33,
        joint_pass_rate=59.09,
        unverifiable_pct=26.67,
    )
    assert "72.73%" in text
    assert "33.33%" in text
