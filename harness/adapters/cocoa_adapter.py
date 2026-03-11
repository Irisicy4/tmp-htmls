"""
Cocoa-agent adapter — all cocoa-specific code lives here.

Handles:
  - sys.path setup for /cocoa-agent
  - TaskExecutor monkey-patch (skip_docker mode)
  - Agent creation
  - Error recovery (accessing executor internals)
  - Logging setup via executor.utils
"""

import sys

_prepared = False


def prepare():
    """Add /cocoa-agent to sys.path and apply the TaskExecutor monkey-patch.

    Safe to call multiple times — patches are applied only once.
    """
    global _prepared
    if _prepared:
        return
    _prepared = True

    sys.path.insert(0, "/cocoa-agent")

    # --- Monkey-patch TaskExecutor for skip_docker (Harbor) mode ---
    import executor  # noqa: F401
    from executor import TaskExecutor

    _orig_setup = TaskExecutor.setup_environment
    _orig_cleanup = TaskExecutor.cleanup_environment

    def _patched_setup(self, task: dict, wait_time: int = 30) -> None:
        skip = self.config.get("sandbox", {}).get("skip_docker", False)
        if skip:
            import requests
            import time

            base = self.sandbox_client.base_url
            for _ in range(wait_time // 2):
                try:
                    r = requests.get(f"{base}/v1/ping", timeout=2)
                    if r.status_code == 200:
                        if hasattr(self.sandbox_client, "_initialize_sdk_client"):
                            self.sandbox_client._initialize_sdk_client()
                        self.controller.clear_history()
                        return
                except Exception:
                    pass
                time.sleep(2)
            raise RuntimeError(f"Sandbox at {base} did not become ready within {wait_time}s")
        _orig_setup(self, task, wait_time)

    def _patched_cleanup(self) -> None:
        skip = self.config.get("sandbox", {}).get("skip_docker", False)
        if not skip:
            _orig_cleanup(self)
        else:
            self.controller.clear_history()

    TaskExecutor.setup_environment = _patched_setup
    TaskExecutor.cleanup_environment = _patched_cleanup


def create_agent(config: dict):
    """Import and instantiate a CocoaAgent."""
    prepare()
    _patch_agents_init()
    from agents.cocoa_agent import CocoaAgent

    return CocoaAgent(config)


_agents_init_patched = False


def _patch_agents_init():
    """Replace cocoa-agent's agents/__init__.py with a safe version.

    The upstream __init__.py eagerly imports all agent types, which fails
    when optional dependencies (decrypt_utils, etc.) are missing.
    We overwrite it with a minimal version that only exposes BaseAgent
    and CocoaAgent, with try/except for the rest.
    """
    global _agents_init_patched
    if _agents_init_patched:
        return
    _agents_init_patched = True

    import importlib

    init_path = "/cocoa-agent/agents/__init__.py"
    safe_init = '''\
"""Agents package — patched by Harbor harness for safe imports."""
from .base import BaseAgent
from .cocoa_agent import CocoaAgent

__all__ = ["BaseAgent", "CocoaAgent"]
'''
    try:
        with open(init_path, "w") as f:
            f.write(safe_init)
    except OSError:
        pass  # read-only filesystem — fall back to whatever's there

    # Force re-import if agents was already partially loaded
    if "agents" in sys.modules:
        del sys.modules["agents"]
    importlib.invalidate_caches()


def extract_error_info(agent, exc: Exception) -> dict:
    """Extract partial results from a cocoa agent after an error.

    Accesses agent.executor.controller / agent.executor.sandbox_client,
    which are cocoa-specific internals.
    """
    partial: dict = {}
    try:
        partial["conversation"] = agent.executor.controller.get_history()
        partial["execution_trace"] = agent.executor.sandbox_client.get_history()
    except Exception:
        pass
    try:
        if hasattr(agent, "executor") and hasattr(agent.executor, "controller"):
            if hasattr(agent.executor.controller, "get_cost_stats"):
                partial["api_cost_stats"] = agent.executor.controller.get_cost_stats()
            partial["iterations"] = getattr(agent.executor, "_current_iteration", 0)
    except Exception:
        pass
    return partial


def setup_logging(level: str = "INFO"):
    """Delegate to cocoa-agent's executor.utils.setup_logging."""
    prepare()
    from executor.utils import setup_logging as _setup

    _setup(level)
