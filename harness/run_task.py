"""
Entry point for running agents inside a Harbor/Modal container.
Supports multiple agent types via config's agent_type field.
Skill store/retrieve is agent-agnostic middleware controlled by config flags.

Used by:
  - Harbor (cocoa_harbor_agent.py calls this via environment.exec)
  - Modal  (standalone_modal_runner.py calls create_agent + run_agent_task directly)
"""
import argparse
import importlib
import json
import logging
import os
import sys
from pathlib import Path


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


# ---------------------------------------------------------------------------
# Agent types that need the browser sandbox
# ---------------------------------------------------------------------------
_SANDBOX_AGENTS = {"cocoa"}


def create_agent(config: dict):
    """Instantiate an agent based on config's agent_type."""
    agent_type = config.get("agent_type", "cocoa")

    if agent_type == "cocoa":
        from adapters.cocoa_adapter import create_agent as _create_cocoa
        return _create_cocoa(config)

    # Non-cocoa agents: lazy import from the agents package on /cocoa-agent
    # These are self-contained and only need their own deps.
    _AGENT_MAP = {
        "claude_code": ("agents", "ClaudeCodeAgent"),
        "codex": ("agents", "CodexAgent"),
        "gemini_cli": ("agents", "GeminiCLIAgent"),
        "manus": ("agents", "ManusAgent"),
        "openai_deep_research": ("agents", "OpenAIDeepResearchAgent"),
        "gemini_deep_research": ("agents", "GeminiDeepResearchAgent"),
    }

    if agent_type not in _AGENT_MAP:
        raise ValueError(f"Unknown agent_type: {agent_type}")

    module_name, class_name = _AGENT_MAP[agent_type]
    mod = importlib.import_module(module_name)
    cls = getattr(mod, class_name)
    return cls(config)


def apply_env_overrides(config: dict) -> dict:
    """Apply environment variable overrides to config."""
    config.setdefault("sandbox", {}).update({"skip_docker": True, "docker_port": 8080})

    env_base_url = (
        os.environ.get("LLM_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        # Fallback: use base_url already baked into the config (e.g. by trigger_experiment.py)
        or config.get("controller", {}).get("args", {}).get("base_url", "")
    )
    if env_base_url:
        config.setdefault("controller", {}).setdefault("args", {})["base_url"] = env_base_url
        os.environ["OPENAI_BASE_URL"] = env_base_url  # for LLM judge (test_task.py)

    # Propagate agent api_key to OPENAI_API_KEY so the judge uses the same key
    config_api_key = config.get("controller", {}).get("args", {}).get("api_key", "")
    if config_api_key:
        os.environ["OPENAI_API_KEY"] = config_api_key

    env_model = os.environ.get("LLM_MODEL")
    if env_model:
        config.setdefault("controller", {}).setdefault("args", {})["model"] = env_model

    max_iter = os.environ.get("COCOA_MAX_ITERATIONS")
    if max_iter is not None and max_iter != "":
        try:
            config["sandbox"]["max_iterations"] = int(max_iter)
        except ValueError:
            pass

    return config


# ---------------------------------------------------------------------------
# Skill middleware — agent-agnostic, controlled by config flags
# ---------------------------------------------------------------------------

def maybe_inject_skills(task: dict, config: dict) -> dict:
    """Phase 2: search stored skills and prepend relevant ones to the instruction."""
    if not config.get("use_skills"):
        return task

    from skill_store import SkillStore

    skills_dir = config.get("skills_dir", "/skills")
    store = SkillStore(skills_dir=skills_dir)
    skill_bodies = store.search_skills(
        task_description=task.get("instruction", ""),
        top_k=3,
    )
    if not skill_bodies:
        print("[skills] No relevant skills found")
        return task

    combined = "\n\n---\n\n".join(skill_bodies)
    augmented = (
        "Here are some relevant strategies from similar past tasks:\n\n"
        f"{combined}\n\n"
        "---\n\n"
        "Now, here is your actual task:\n\n"
        f"{task['instruction']}"
    )
    task = {**task, "instruction": augmented, "description": augmented}
    print(f"[skills] Injected {len(skill_bodies)} skill(s) into instruction")
    return task


def maybe_store_skill(task: dict, result: dict, eval_result: dict, config: dict) -> None:
    """Phase 1: extract and store a skill if the result is good enough."""
    if not config.get("store_skills"):
        return

    from skill_extractor import SkillExtractor
    from skill_store import SkillStore

    extractor = SkillExtractor(model=config.get("skill_model", "gpt-4o-mini"))
    decision = extractor.should_store(result, eval_result, config)
    if not decision.get("store"):
        print(f"[skills] Not storing: {decision.get('reason', 'unknown')}")
        return

    print(f"[skills] Extracting skill: {decision.get('reason', '')}")
    skill_md = extractor.extract_skill(task, result, eval_result)

    store = SkillStore(skills_dir=config.get("skills_dir", "/skills"))
    metadata = {
        "score": (eval_result.get("details", {}) or {}).get("overall_score"),
        "task_type": decision.get("task_type", ""),
        "tags": decision.get("tags", []),
    }
    task_name = task.get("task_name", "unknown")
    path = store.save_skill(skill_md, task_name, metadata)
    print(f"[skills] Saved: {path}")


# ---------------------------------------------------------------------------
# Core agent runner
# ---------------------------------------------------------------------------

def run_agent_task(agent, task: dict) -> dict:
    """Run an agent on a task, returning the result dict. Handles errors gracefully."""
    agent_type = getattr(agent, "agent_type", None) or "cocoa"

    result: dict = {}
    agent.setup_environment(task)
    try:
        result = agent.run_task(task)
    except Exception as exc:
        partial: dict = {}
        # Use cocoa adapter for cocoa-specific error recovery
        if agent_type == "cocoa":
            from adapters.cocoa_adapter import extract_error_info
            partial = extract_error_info(agent, exc)
        partial["execution_summary"] = _format_execution_summary(
            {"execution_trace": partial.get("execution_trace", [])}
        )
        result = {**partial, "status": "error", "error": str(exc), "task_name": task.get("task_name", "unknown")}
    finally:
        agent.cleanup_environment()

    result.setdefault("task_name", task.get("task_name", "unknown"))
    if result.get("execution_trace") and not result.get("execution_summary"):
        result["execution_summary"] = _format_execution_summary(result)
    return result


def _format_execution_summary(result: dict, max_chars_per_feedback: int = 2000, max_total_chars: int = 80000) -> str:
    """Format execution_trace for LLM judge."""
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


# ---------------------------------------------------------------------------
# CLI entry point (called by Harbor via cocoa_harbor_agent.py)
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--task-name", default="harbor-task")
    parser.add_argument("--config", default="/harness/configs/harbor-config.json")
    parser.add_argument("--output", default="/logs/agent/result.json")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    config = apply_env_overrides(config)

    agent_type = config.get("agent_type", "cocoa")

    # Logging: use cocoa's setup for cocoa agent, standard logging otherwise
    if agent_type == "cocoa":
        from adapters.cocoa_adapter import setup_logging
        setup_logging(config.get("log_level", "INFO"))
    else:
        logging.basicConfig(
            level=getattr(logging, config.get("log_level", "INFO").upper(), logging.INFO),
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )

    # Browser-based agents need the sandbox; pure-API agents don't
    if agent_type in _SANDBOX_AGENTS:
        sandbox_url = "http://localhost:8080"
        print(f"Waiting for sandbox at {sandbox_url}...")
        if not wait_for_sandbox(sandbox_url):
            result = {"status": "error", "error": "Sandbox did not become ready", "task_name": args.task_name}
            _write_result(args.output, result)
            sys.exit(1)
        print("Sandbox ready.")
    else:
        print(f"Agent type '{agent_type}' does not use the sandbox — skipping wait.")

    agent = create_agent(config)

    task = {
        "task_name": args.task_name,
        "instruction": args.instruction,
        "description": args.instruction,
        "task_dir": "/tmp",
        "use_encrypted": False,
    }

    # Phase 2: inject skills into instruction (agent-agnostic)
    task = maybe_inject_skills(task, config)

    result = run_agent_task(agent, task)
    _write_result(args.output, result)
    print(f"Result written to {args.output}")


if __name__ == "__main__":
    main()
