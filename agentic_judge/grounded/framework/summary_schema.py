"""Over-reporting summary schema (anti-cheat by design).

A SummarySchema declares the JSON shape every agent answer must conform to.
Crucially, the schema covers MORE fields than the HARD_CONSTRAINTS check —
the agent doesn't know which subset is enforced, so the cheapest cheating
strategy degenerates into doing the task.

Each instruction.md gets an addendum like:

    Output your result as a JSON block delimited by `=== JSON RESULT ===`
    and `=== END JSON ===`.  Required fields: [...]  Optional but reward
    if present: [...]  The grader may verify any field against the live
    source URLs you cite, so be precise.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RequiredField:
    path: str          # dotted path, lists use [*]
    type_hint: str     # e.g. "string", "number", "list[object]"
    description: str
    example: Any = None


@dataclass
class OptionalField(RequiredField):
    pass


@dataclass
class SummarySchema:
    """Declares the agent's output JSON contract for a single task."""
    required: list[RequiredField] = field(default_factory=list)
    optional: list[OptionalField] = field(default_factory=list)
    examples: dict = field(default_factory=dict)

    def required_paths(self) -> list[str]:
        return [f.path for f in self.required]

    def to_instruction_addendum(self) -> str:
        """Render the prose appendix to bolt onto instruction.md."""
        lines = [
            "",
            "---",
            "",
            "## Required output format",
            "",
            "After completing the task, output your result as a JSON block "
            "delimited by `=== JSON RESULT ===` on its own line, the JSON, "
            "and `=== END JSON ===` on its own line. Include EVERY required "
            "field below; optional fields are rewarded when present.",
            "",
            "**Required fields:**",
            "",
        ]
        for f in self.required:
            lines.append(f"- `{f.path}` ({f.type_hint}) — {f.description}")
        if self.optional:
            lines += ["", "**Optional but graded if present:**", ""]
            for f in self.optional:
                lines.append(f"- `{f.path}` ({f.type_hint}) — {f.description}")
        lines += [
            "",
            "**Anti-fabrication note:** the grader fetches the URLs you cite "
            "and verifies the figures and quotes against the live page. "
            "Be precise; do not paraphrase numeric facts.",
            "",
        ]
        if self.examples:
            import json as _json
            lines += ["**Example shape (values illustrative, not literal):**",
                      "", "```json", _json.dumps(self.examples, indent=2),
                      "```", ""]
        return "\n".join(lines)
