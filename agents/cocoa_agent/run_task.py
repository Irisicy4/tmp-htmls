"""
Entry point for cocoabench/cocoa-agent inside a Harbor environment container.
Uses official cocoabench/cocoa-agent with overlay for skip_docker support.
"""
import sys
sys.path.insert(0, "/cocoa-agent")

import argparse
import json
import os
import sys
from pathlib import Path

# Apply Harbor overlay (skip_docker) before importing cocoa-agent
import overlay  # noqa: F401

from executor.utils import setup_logging
from agents import CocoaAgent


def wait_for_sandbox(base_url: str, timeout_sec: int = 120) -> bool:
    import requests
    for _ in range(timeout_sec // 2):
        try:
            r = requests.get(f"{base_url}/v1/ping", timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        import time
        time.sleep(2)
    return False


def _format_execution_summary(result: dict, max_chars_per_feedback: int = 2000, max_total_chars: int = 80000) -> str:
    """Format execution_trace for LLM judge (evolve_bench parity)."""
    trace = result.get("execution_trace", [])
    if not trace:
        return ""
    lines = []
    total_chars = 0
    for i, entry in enumerate(trace, 1):
        action = entry.get("action", {})
        feedback = entry.get("feedback", {})
        atype = action.get("action_type", "unknown")
        param_parts = [f"  {k}: {str(v)[:500]}{'...' if len(str(v)) > 500 else ''}"
                      for k, v in action.items() if k not in ("action_type", "tool_call_id")]
        params_block = "\n".join(param_parts) if param_parts else "  (no parameters)"
        msg = feedback.get("message", "")
        if len(msg) > max_chars_per_feedback:
            msg = msg[:max_chars_per_feedback] + f"... [truncated, {len(msg)} total chars]"
        step_text = f"Step {i}: {atype}\n{params_block}\n-> {msg}\n"
        total_chars += len(step_text)
        if total_chars > max_total_chars:
            lines.append(f"... [trace truncated at step {i}/{len(trace)}]")
            break
        lines.append(step_text)
    return "\n".join(lines)


def _strip_images(result: dict) -> dict:
    for msg in result.get("conversation", []):
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if part.get("type") == "image_url":
                    url = part.get("image_url", {}).get("url", "")
                    if url.startswith("data:"):
                        part["image_url"]["url"] = "[screenshot stripped]"
    return result


def _write_result(path: str, result: dict) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_strip_images(result), indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--task-name", default="harbor-task")
    parser.add_argument("--config", default="/cocoa-agent/configs/harbor-config.json")
    parser.add_argument("--output", default="/logs/agent/result.json")
    args = parser.parse_args()

    sandbox_url = "http://localhost:8080"
    print(f"Waiting for sandbox at {sandbox_url}...")
    if not wait_for_sandbox(sandbox_url):
        result = {"status": "error", "error": "Sandbox did not become ready", "task_name": args.task_name}
        _write_result(args.output, result)
        sys.exit(1)
    print("Sandbox ready.")

    with open(args.config) as f:
        config = json.load(f)

    config.setdefault("sandbox", {}).update({"skip_docker": True, "docker_port": 8080})

    # Apply env var overrides to config (supports UniAPI / custom endpoints + models)
    env_base_url = os.environ.get("LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    if env_base_url:
        config.setdefault("controller", {}).setdefault("args", {})["base_url"] = env_base_url
        os.environ["OPENAI_BASE_URL"] = env_base_url  # for LLM judge (test_task.py)
    env_model = os.environ.get("LLM_MODEL")
    if env_model:
        config.setdefault("controller", {}).setdefault("args", {})["model"] = env_model

    max_iter = os.environ.get("COCOA_MAX_ITERATIONS")
    if max_iter is not None and max_iter != "":
        try:
            config["sandbox"]["max_iterations"] = int(max_iter)
        except ValueError:
            pass

    setup_logging(config.get("log_level", "INFO"))
    agent = CocoaAgent(config)

    task = {
        "task_name": args.task_name,
        "instruction": args.instruction,
        "task_dir": "/tmp",
    }

    result = {}
    agent.setup_environment(task)
    try:
        result = agent.run_task(task)
    except Exception as exc:
        partial = {}
        try:
            partial["conversation"] = agent.executor.controller.get_history()
            partial["execution_trace"] = agent.executor.sandbox_client.get_history()
            partial["execution_summary"] = str(partial.get("execution_trace", []))
        except Exception:
            pass
        try:
            if hasattr(agent.executor.controller, "get_cost_stats"):
                partial["api_cost_stats"] = agent.executor.controller.get_cost_stats()
        except Exception:
            pass
        result = {**partial, "status": "error", "error": str(exc), "task_name": args.task_name}
    finally:
        agent.cleanup_environment()

    result.setdefault("task_name", args.task_name)
    if result.get("execution_trace") and not result.get("execution_summary"):
        result["execution_summary"] = _format_execution_summary(result)
    _write_result(args.output, result)
    print(f"Result written to {args.output}")


if __name__ == "__main__":
    main()
