"""Spec for task-78-go-to-notion-weekly-planner-fill-sections (Daily Activities)."""
from ..framework.constraints import (
    JSONSchemaConforms, ContainsAllSubstrings, ListLengthInRange,
    FieldPresent, Custom,
)
from ..framework.summary_schema import SummarySchema, RequiredField, OptionalField


TASK_INSTRUCTION = (
    "Go to https://www.notion.so/templates/weekly-planner and open the Weekly "
    "Planner template. Fill in all the empty sections of the planner with "
    "appropriate placeholder data — including goals, daily tasks, notes, and "
    "any other incomplete fields — to make the page fully complete."
)

SUMMARY_SCHEMA = SummarySchema(
    required=[
        RequiredField("template_url",          "string",       "Should be the Notion weekly planner template URL."),
        RequiredField("sections_filled",       "list[object]", "Each section you filled in."),
        RequiredField("sections_filled[].name","string",       "Section label as shown on the planner."),
        RequiredField("sections_filled[].content_preview", "string", "First ~120 chars of the content you filled in."),
        RequiredField("sections_filled[].placeholder_or_real", "string", "'placeholder' (fake but realistic) or 'real'."),
        RequiredField("completion_evidence",   "string",       "URL to your filled copy OR path to a screenshot."),
    ],
    optional=[
        OptionalField("sections_total",        "integer",      "Sections found on the template, filled or not."),
        OptionalField("sections_skipped",      "list[string]", "Names of sections you couldn't access/fill."),
        OptionalField("editing_method",        "string",       "How you edited (duplicate template + edit, fork, etc.)."),
        OptionalField("notion_account_used",   "boolean",      "Did you sign into a Notion account?"),
    ],
    examples={
        "template_url": "https://www.notion.so/templates/weekly-planner",
        "sections_filled": [
            {"name": "Goals", "content_preview": "1. Ship Q3 OKR ...",
             "placeholder_or_real": "placeholder"},
            {"name": "Monday tasks", "content_preview": "Standup @ 9, ...",
             "placeholder_or_real": "placeholder"},
        ],
        "completion_evidence": "/output/notion_filled.png",
    },
)


def _has_canonical_sections(summary, ctx):
    rs = [(s or {}).get("name", "").lower() for s in (summary.get("sections_filled") or [])]
    expected_any = {"goals", "tasks", "notes"}
    hit = expected_any & {n for n in rs} or set()
    # Loose: also accept partial matches
    for n in rs:
        for e in ("goal", "task", "note", "monday", "tuesday", "wednesday",
                   "thursday", "friday", "weekly", "priority"):
            if e in n:
                hit.add(e)
    if len(hit) >= 3:
        return True, f"covered {sorted(hit)}"
    return False, f"only covered {sorted(hit)}; need ≥3 of goals/tasks/notes/days"


HARD_CONSTRAINTS = [
    JSONSchemaConforms(required_paths=SUMMARY_SCHEMA.required_paths()),
    ContainsAllSubstrings("template_url", ["notion.so"], name="template_url_on_notion"),
    ListLengthInRange("sections_filled", 3, 50, name="at_least_3_sections_filled"),
    Custom("canonical_sections_covered", _has_canonical_sections),
]


def FAITHFULNESS_CHECKS(summary: dict) -> list[dict]:
    out = []
    if (t := summary.get("template_url")):
        out.append({"url": t, "claim": "Notion"})
    return out


DIMENSIONS = ["platform_access", "section_coverage", "content_quality", "completion_confirmation"]
DIMENSION_WEIGHTS = {
    "platform_access":         0.2,
    "section_coverage":        0.35,
    "content_quality":         0.3,
    "completion_confirmation": 0.15,
}

TASK_RUBRIC = """A. Platform Access (0.20)
  5 — Notion weekly planner template reached and editable.
  4 — Reached but copy/duplicate required.
  3 — Reached but read-only.
  2 — Reached wrong template.
  1 — Did not reach Notion.

B. Section Coverage (0.35)
  5 — Every empty section filled (goals, tasks per day, notes, priorities).
  4 — Most filled; 1-2 skipped.
  3 — Half filled.
  2 — Few sections.
  1 — None.

C. Content Quality (0.30)
  5 — Sample data realistic and varied (different tasks per day, specific goals).
  4 — Realistic but repetitive.
  3 — Generic.
  2 — Lorem-ipsum.
  1 — Junk.

D. Completion Confirmation (0.15)
  5 — Screenshot or URL showing the filled planner.
  4 — Detailed prose confirmation.
  3 — Brief confirmation.
  2 — Vague.
  1 — None.
"""
