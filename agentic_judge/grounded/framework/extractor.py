"""Extract the agent's summary JSON from its stdout response."""
from __future__ import annotations

import json
import re


_DELIM_RE = re.compile(
    r"===\s*JSON RESULT\s*===\s*\n(.*?)\n===\s*END JSON\s*===",
    re.S | re.I,
)
# Fallback patterns — agents sometimes drop a json code block instead of the delimiters
_FALLBACK_BLOCK_RE = re.compile(r"```json\s*\n(.*?)\n```", re.S | re.I)


def extract_summary_json(agent_text: str) -> tuple[dict | None, str]:
    """Return (summary_dict, source_used).  source_used identifies which
    parse path succeeded ('delim', 'json-fence', or 'none')."""
    if not agent_text:
        return None, "none"
    # Preferred: explicit delimiters
    m = _DELIM_RE.search(agent_text)
    if m:
        try:
            return json.loads(m.group(1).strip()), "delim"
        except json.JSONDecodeError:
            pass
    # Fallback: ```json blocks
    for m in _FALLBACK_BLOCK_RE.finditer(agent_text):
        try:
            return json.loads(m.group(1).strip()), "json-fence"
        except json.JSONDecodeError:
            continue
    # Last resort: try to find an outermost `{ ... }` block.  This is
    # brittle; the grader will flag it via a "format" deduction.
    brace_start = agent_text.find("{")
    if brace_start >= 0:
        # Naive matching balance walk
        depth = 0
        for i in range(brace_start, len(agent_text)):
            c = agent_text[i]
            if c == "{": depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    chunk = agent_text[brace_start:i+1]
                    try:
                        return json.loads(chunk), "naive-braces"
                    except json.JSONDecodeError:
                        return None, "none"
    return None, "none"
