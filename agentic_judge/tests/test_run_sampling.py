import json
import os
import tempfile
from pathlib import Path

from agentic_judge.run import (
    load_config,
    load_passed_results,
    stratified_sample,
    _get_task_domain,
)


def _write_result_json(base_dir: str, task_name: str, passed: bool) -> None:
    """Write result.json to base_dir/task_name/result.json (mirrors real trial structure)."""
    task_dir = os.path.join(base_dir, task_name)
    os.makedirs(task_dir, exist_ok=True)
    data = {
        "task": task_name,
        "passed": passed,
        "score": 3.0,
        "dimension_scores": {},
        "category": "Shopping",
    }
    with open(os.path.join(task_dir, "result.json"), "w") as f:
        json.dump(data, f)


def _write_task_toml(task_dir: str, task_name: str, domain: str) -> None:
    os.makedirs(os.path.join(task_dir, task_name), exist_ok=True)
    toml_path = os.path.join(task_dir, task_name, "task.toml")
    with open(toml_path, "w") as f:
        f.write(f'version = "1.0"\n\n[metadata]\ncategory = "{domain}"\n')


def test_load_config_returns_expected_keys(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("""
result_dirs:
  original: results/deprivacy-batch-1/run/run/trials
sampling:
  total_sample_size: 30
agent:
  model: gpt-4o
  timeout_seconds: 120
""")
    config = load_config(str(config_path))
    assert config["sampling"]["total_sample_size"] == 30
    assert config["agent"]["model"] == "gpt-4o"


def test_load_passed_results_filters_correctly(tmp_path):
    result_dir = str(tmp_path)
    _write_result_json(result_dir, "task-01", passed=True)
    _write_result_json(result_dir, "task-02", passed=False)
    # A stray flat JSON at root is ignored by */result.json glob
    (tmp_path / "stray.json").write_text("{}")

    results = load_passed_results(result_dir)
    assert len(results) == 1
    assert results[0]["data"]["task"] == "task-01"


def test_get_task_domain_from_toml(tmp_path):
    task_dir = str(tmp_path)
    _write_task_toml(task_dir, "task-01", "Finance & Economics")
    domain = _get_task_domain(task_dir, "task-01")
    assert domain == "Finance & Economics"


def test_get_task_domain_missing_returns_unknown(tmp_path):
    domain = _get_task_domain(str(tmp_path), "nonexistent-task")
    assert domain == "Unknown"


def test_stratified_sample_respects_size(tmp_path):
    result_dir = str(tmp_path / "results")
    task_dir = str(tmp_path / "tasks")
    os.makedirs(result_dir)

    domains = ["Shopping", "Shopping", "Finance & Economics", "Finance & Economics", "Travel & Planning", "Travel & Planning"]
    for i, domain in enumerate(domains, 1):
        name = f"task-{i:02d}"
        _write_result_json(result_dir, name, passed=True)
        _write_task_toml(task_dir, name, domain)

    results = load_passed_results(result_dir)
    sampled = stratified_sample(results, task_dir, n=3)
    assert len(sampled) == 3


def test_stratified_sample_includes_all_domains_when_possible(tmp_path):
    result_dir = str(tmp_path / "results")
    task_dir = str(tmp_path / "tasks")
    os.makedirs(result_dir)

    for i, domain in enumerate(["Shopping", "Finance & Economics", "Travel & Planning"], 1):
        name = f"task-{i:02d}"
        _write_result_json(result_dir, name, passed=True)
        _write_task_toml(task_dir, name, domain)

    results = load_passed_results(result_dir)
    sampled = stratified_sample(results, task_dir, n=3)
    domains_in_sample = {r["domain"] for r in sampled}
    assert "Shopping" in domains_in_sample
    assert "Finance & Economics" in domains_in_sample
    assert "Travel & Planning" in domains_in_sample
