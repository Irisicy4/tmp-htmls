import json
import os
import tempfile
from pathlib import Path

from agentic_judge.core.result_writer import write_result


def _sample_verification() -> dict:
    return {
        "task_name": "task-01-test",
        "category": "Shopping",
        "dataset": "original",
        "static_judge_passed": True,
        "static_judge_score": 3.5,
        "static_judge_dimensions": {"constraint_satisfaction": 5, "result_specificity": 3},
        "agentic_verified": True,
        "agentic_finding": "Amazon product page loaded and matches claimed backpack options.",
        "null_reason": None,
        "verification_method": "browser_navigation",
        "deliverable_url": "https://www.amazon.com/dp/B09YRC9Y3G",
        "time_seconds": 12.4,
    }


def test_write_result_creates_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = _sample_verification()
        path = write_result(result, tmpdir)
        assert Path(path).exists()
        assert path.endswith("task-01-test_agentic_verification.json")


def test_write_result_content():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = _sample_verification()
        path = write_result(result, tmpdir)
        with open(path) as f:
            data = json.load(f)
        assert data["task_name"] == "task-01-test"
        assert data["agentic_verified"] is True
        assert data["null_reason"] is None
        assert "timestamp" in data
        assert data["verification_method"] == "browser_navigation"


def test_write_result_null_verified():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = _sample_verification()
        result["agentic_verified"] = None
        result["null_reason"] = "no_urls"
        result["verification_method"] = "unverifiable"
        path = write_result(result, tmpdir)
        with open(path) as f:
            data = json.load(f)
        assert data["agentic_verified"] is None
        assert data["null_reason"] == "no_urls"


def test_write_result_creates_output_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        nested = os.path.join(tmpdir, "deep", "nested")
        result = _sample_verification()
        path = write_result(result, nested)
        assert Path(path).exists()
