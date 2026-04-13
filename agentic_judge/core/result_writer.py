"""Writes per-task agentic verification results to disk."""

import json
from datetime import datetime, timezone
from pathlib import Path


def write_result(verification_dict: dict, output_dir: str) -> str:
    """Write one verification JSON file.

    Args:
        verification_dict: Result from verifier.verify_task(), must contain task_name.
        output_dir: Directory to write into (created if absent).

    Returns:
        Absolute path of the written file.
    """
    task_name = verification_dict["task_name"]
    filename = f"{task_name}_agentic_verification.json"
    path = Path(output_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)

    output = {**verification_dict, "timestamp": datetime.now(timezone.utc).isoformat()}
    with open(path, "w") as f:
        json.dump(output, f, indent=2)

    return str(path)
