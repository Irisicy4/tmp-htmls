"""Extract the agent's summary JSON from its stdout response."""
from __future__ import annotations

import json
import re


_DELIM_RE = re.compile(
    r"===\s*JSON RESULT\s*===\s*\r?\n(.*?)\r?\n===\s*END JSON\s*===",
    re.S | re.I,
)
# Fallback patterns — agents sometimes drop a json code block instead of the delimiters
_FALLBACK_BLOCK_RE = re.compile(r"```json\s*\r?\n(.*?)\r?\n```", re.S | re.I)

# Codex's file-dump convention prints saved files between
# `=== FILE: <name> === ... === END FILE ===` markers. When the agent saves
# the JSON summary to /output/result.txt the JSON block ends up wrapped in
# these markers, which used to confuse extraction. We strip those wrappers
# first so the inner `=== JSON RESULT ===` block parses normally.
_FILE_WRAPPER_RE = re.compile(
    r"=== FILE:\s*[^=\r\n]+===\s*\r?\n([\s\S]*?)\r?\n===\s*END FILE\s*===",
    re.I,
)


def _unwrap_file_blocks(text: str) -> str:
    """Replace `=== FILE: x === ... === END FILE ===` with just the inner
    contents, so any JSON block nested inside surfaces to the normal regex."""
    return _FILE_WRAPPER_RE.sub(lambda m: "\n" + m.group(1) + "\n", text)


def extract_summary_json_from_file(path: str) -> tuple[dict | None, str]:
    """Read the JSON written by the agent to /output/summary.json (or
    wherever the caller saved it). If the file is missing or unparseable,
    fall through to the same parsing pipeline as extract_summary_json so
    `=== JSON RESULT ===` blocks inside the file still work."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            blob = f.read()
    except OSError:
        return None, "none"
    blob = blob.strip()
    if not blob:
        return None, "none"
    # Try a direct json parse first (the simple case)
    try:
        return json.loads(blob), "summary-file"
    except json.JSONDecodeError:
        pass
    # Fall back to the same pipeline as stdout extraction
    return extract_summary_json(blob)


def extract_summary_json(agent_text: str) -> tuple[dict | None, str]:
    """Return (summary_dict, source_used).  source_used identifies which
    parse path succeeded ('delim', 'json-fence', 'naive-braces', or 'none')."""
    if not agent_text:
        return None, "none"
    # Unwrap codex's `=== FILE: ... === ... === END FILE ===` envelopes so
    # nested JSON blocks become extractable.
    unwrapped = _unwrap_file_blocks(agent_text)
    # Preferred: explicit delimiters. When several `=== JSON RESULT ===`
    # blocks are present (e.g. an agent shows an example first, or a
    # two-turn run appends the turn-2 summary after a turn-1 draft), the
    # LAST parseable block is the agent's final answer — prefer it.
    for candidate, src in ((unwrapped, "delim"), (agent_text, "delim")):
        matches = list(_DELIM_RE.finditer(candidate))
        for m in reversed(matches):
            try:
                return json.loads(m.group(1).strip()), src
            except json.JSONDecodeError:
                continue
    # Anchor for the rest of the fallbacks to the unwrapped form
    agent_text = unwrapped
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
