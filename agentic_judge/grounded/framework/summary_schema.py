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
    path: str          # dotted path, lists use [] — e.g. "options[].name"
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

    def to_instruction_addendum(self, mode: str = "inline") -> str:
        """Render the prose appendix to bolt onto instruction.md.

        mode="inline"   — appended to the brief in a single-turn workflow.
                          The agent does the task AND emits the JSON tail
                          in the same response.
        mode="followup" — used as the turn-2 prompt in a two-turn workflow.
                          Turn 1 lets the agent answer the brief naturally
                          with no JSON requirement; turn 2 asks it to
                          summarise what it did, without changing the
                          substance of its previous answer.
        """
        if mode not in ("inline", "followup"):
            raise ValueError(f"unknown mode: {mode!r}")

        if mode == "followup":
            return self._render_followup()
        return self._render_inline()

    def _render_inline(self) -> str:
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
            "You may save any supporting artefacts (CSVs, screenshots, "
            "intermediate computations, etc.) under `/output/` before the "
            "JSON block — the grader can read them as evidence — but they "
            "do not replace the JSON tail. If it is easier, you may also "
            "write the same JSON to `/output/summary.json`; the grader "
            "accepts either source.",
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
            "**Infeasible escape:** if the task genuinely cannot be completed "
            "(site unreachable, data does not exist, login required, etc.) "
            "emit the same JSON block with a single key `infeasible_reason` "
            "explaining why. Producing a JSON block (with the structured "
            "answer OR `infeasible_reason`) is required; missing JSON is "
            "penalised.",
            "",
            "```",
            "=== JSON RESULT ===",
            '{ "infeasible_reason": "site requires login I do not have" }',
            "=== END JSON ===",
            "```",
            "",
        ]
        if self.examples:
            import json as _json
            lines += ["**Example shape (values illustrative, not literal):**",
                      "", "```json", _json.dumps(self.examples, indent=2),
                      "```", ""]
        return "\n".join(lines)

    def _render_followup(self) -> str:
        lines = [
            "You have already completed the task above. Now produce a JSON "
            "summary of what you did, conforming exactly to the schema "
            "below. Do not redo or amend the substance of your previous "
            "answer — this is a summary step only. Write the JSON to "
            "`/output/summary.json` AND emit it inline between "
            "`=== JSON RESULT ===` and `=== END JSON ===` delimiters.",
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
            "**URL requirements:** every `url` / `source_url` field must point "
            "to the page that DIRECTLY supports the specific values you report "
            "in its sibling fields — the exact product page, the exact filing, "
            "the exact article — not a homepage, search page, or category "
            "listing. If the supporting evidence spans several pages (e.g. the "
            "price on one page and the specification on another), provide a "
            "LIST of URLs in that field instead of a single string.",
            "",
            "If your prior answer made any factual claims you can no longer "
            "support (because the source you used isn't reachable now, "
            "etc.), summarise honestly: omit unsupported claims rather "
            "than fabricate to fill the schema. If the entire task turned "
            "out to be infeasible, emit a JSON block with a single key "
            "`infeasible_reason` explaining why.",
            "",
        ]
        if self.examples:
            import json as _json
            lines += ["**Example shape (values illustrative, not literal):**",
                      "", "```json", _json.dumps(self.examples, indent=2),
                      "```", ""]
        return "\n".join(lines)
