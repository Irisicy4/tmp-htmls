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


def _iter_candidate_dicts(agent_text: str):
    """Yield every JSON *object* embedded in agent_text, in document order.

    Sources, all treated equally:
      - `=== JSON RESULT === ... === END JSON ===` delimiter blocks
      - ```json fenced blocks
      - `=== FILE: <name> === ... === END FILE ===` blocks whose body is a
        JSON object (this is where an agent that writes its answer to
        /output/summary.json and only *points* at it inline surfaces the
        real payload).

    Order matters for tie-breaking upstream — later candidates are the
    agent's more-final work (turn-2 / end-of-run artifacts)."""
    if not agent_text:
        return
    spans: list[tuple[int, str]] = []
    for rx in (_DELIM_RE, _FALLBACK_BLOCK_RE, _FILE_WRAPPER_RE):
        for m in rx.finditer(agent_text):
            spans.append((m.start(), m.group(1)))
    for _, body in sorted(spans, key=lambda t: t[0]):
        # strict=False tolerates literal control characters (raw newlines/tabs
        # inside string values) that agent-written summary files often contain
        # and that would otherwise reject an otherwise-valid object.
        try:
            obj = json.loads(body.strip(), strict=False)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict):
            yield obj


def _schema_richness(summary, required_paths: list | None) -> int:
    """Score how completely `summary` fills the task schema.

    With an array schema (paths like `options[].course_url`): count array
    items that have every required item key present and non-empty. With a
    flat schema: count required top-level keys present and non-empty. With
    no schema: fall back to the longest embedded list, else populated-field
    count. A pointer summary (`options_count: 365` but no populated
    `options`) scores 0 and loses to the artifact that holds the rows."""
    if not isinstance(summary, dict):
        return 0
    if required_paths:
        array_field = None
        item_keys: list = []
        flat_keys: list = []
        for p in required_paths:
            if "[]." in p:
                base, _, leaf = p.partition("[].")
                array_field = base
                item_keys.append(leaf)
            else:
                flat_keys.append(p)
        if array_field is not None:
            rows = summary.get(array_field)
            if not isinstance(rows, list):
                return 0
            return sum(
                1 for r in rows
                if isinstance(r, dict)
                and all(str(r.get(k, "")).strip() for k in item_keys)
            )
        if flat_keys:
            return sum(1 for k in flat_keys if str(summary.get(k, "")).strip())
    best_list = max((len(v) for v in summary.values() if isinstance(v, list)),
                    default=0)
    if best_list:
        return best_list
    return sum(1 for v in summary.values() if v not in (None, "", [], {}))


def resolve_best_summary(agent_text: str,
                         required_paths: list | None = None
                         ) -> tuple[dict | None, str]:
    """Like extract_summary_json, but when several JSON objects are embedded
    (turn-1 draft, turn-2 answer, a `=== FILE: summary.json ===` the agent
    wrote to /output) pick the one that most completely satisfies the task
    schema, breaking ties toward the earlier extraction result.

    This fixes two real failure modes seen in two-turn runs:
      - the inline answer is a *pointer* (`{options_file, options_count}`)
        while the real rows live in the saved summary artifact; and
      - the turn-2 answer is buried behind a turn-1 draft.
    It only ever upgrades to a strictly more schema-complete object, never
    fabricates data, so it is a no-op when the primary extraction is already
    the richest candidate."""
    primary, src = extract_summary_json(agent_text)
    best, best_score, best_src = primary, _schema_richness(primary, required_paths), src
    for cand in _iter_candidate_dicts(agent_text):
        score = _schema_richness(cand, required_paths)
        if score > best_score:  # strictly richer; document order favours later blocks
            best, best_score, best_src = cand, score, "richest"
    return best, best_src


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
