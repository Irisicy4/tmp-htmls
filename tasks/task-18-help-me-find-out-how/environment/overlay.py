"""
Runtime patch for cocoabench/cocoa-agent to support Harbor (skip_docker mode).
Official cocoa-agent does not have skip_docker; we add it via monkey-patch.
Import this module BEFORE importing from executor or agents.
"""

import executor
from executor import TaskExecutor

_orig_setup = TaskExecutor.setup_environment
_orig_cleanup = TaskExecutor.cleanup_environment


def _patched_setup(self, task: dict, wait_time: int = 30) -> None:
    skip = self.config.get("sandbox", {}).get("skip_docker", False)
    if skip:
        import requests
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
            import time
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
