"""Spec for task-15-help-me-get-all-2025 (Data ML Engineering)."""
from ..framework.constraints import (
    JSONSchemaConforms, EqualsValue, ContainsAllSubstrings,
    ListLengthInRange, FieldPresent, AllValuesBetween,
)
from ..framework.summary_schema import SummarySchema, RequiredField, OptionalField


TASK_INSTRUCTION = (
    "Help me get all 2025 publication info from "
    "https://m.douban.com/subject_collection/ECNA7Y7GA and save as an Excel file."
)

SUMMARY_SCHEMA = SummarySchema(
    required=[
        RequiredField("source_url",                 "string",       "Exact Douban subject_collection URL used."),
        RequiredField("output_file.path",           "string",       "Absolute path to the .xlsx file you created."),
        RequiredField("output_file.format",         "string",       "'xlsx' or 'xls'."),
        RequiredField("output_file.sheet_name",     "string",       "Sheet name used."),
        RequiredField("output_file.row_count",      "integer",      "Number of data rows (excluding header)."),
        RequiredField("output_file.column_headers", "list[string]", "Column headers in order."),
        RequiredField("publications",               "list[object]", "All 2025 publications, structured."),
        RequiredField("publications[].title",       "string",       "Title."),
        RequiredField("publications[].year",        "integer",      "Publication year — must be 2025."),
    ],
    optional=[
        OptionalField("publications[].author",      "string",        "Author."),
        OptionalField("publications[].rating",      "number 0-10",   "Douban rating."),
        OptionalField("publications[].genre",       "string",        ""),
        OptionalField("publications[].url",         "string",        "Per-item URL on Douban."),
        OptionalField("filter_methodology",         "string",        "How you confirmed only 2025 items were kept."),
        OptionalField("total_items_seen",           "integer",       "Items seen on the page before year-filtering."),
    ],
    examples={
        "source_url": "https://m.douban.com/subject_collection/ECNA7Y7GA",
        "output_file": {"path": "/output/douban_2025.xlsx", "format": "xlsx",
                          "sheet_name": "publications", "row_count": 14,
                          "column_headers": ["title", "year", "author", "rating"]},
        "publications": [{"title": "...", "year": 2025, "author": "..."}],
    },
)


HARD_CONSTRAINTS = [
    JSONSchemaConforms(required_paths=SUMMARY_SCHEMA.required_paths()),
    ContainsAllSubstrings("source_url", ["ECNA7Y7GA", "douban"], name="exact_source_url"),
    EqualsValue("output_file.format", "xlsx", case_insensitive=True,
                name="format_is_xlsx"),
    AllValuesBetween("publications", "year", 2025, 2025, name="every_publication_is_2025"),
    ContainsAllSubstrings("output_file.path", [".xlsx"], name="path_ends_xlsx"),
]


def FAITHFULNESS_CHECKS(summary: dict) -> list[dict]:
    out = []
    if (src := summary.get("source_url")):
        out.append({"url": src, "claim": "ECNA7Y7GA"})
    for p in (summary.get("publications") or [])[:3]:
        if (u := p.get("url")):
            out.append({"url": u, "claim": (p.get("title") or "")[:30]})
    return out


DIMENSIONS = ["source_navigation", "data_completeness", "excel_output", "data_quality"]
DIMENSION_WEIGHTS = {
    "source_navigation": 0.20,
    "data_completeness": 0.30,
    "excel_output":      0.30,
    "data_quality":      0.20,
}

TASK_RUBRIC = """A. Source Navigation (0.20)
  5 — Used the exact ECNA7Y7GA URL; confirmed page loaded.
  4 — Reached page but indirectly (via search).
  3 — Different Douban page (general).
  2 — Other site.
  1 — No site reached.

B. Data Completeness (0.30)
  5 — All 2025 items captured; row_count matches site.
  4 — Most items captured.
  3 — Partial.
  2 — Few items.
  1 — None.

C. Excel Output (0.30)
  5 — Valid .xlsx with headers + data; openable.
  4 — .xlsx with minor formatting issue.
  3 — .xls or CSV-as-xlsx.
  2 — Wrong format (CSV/JSON).
  1 — No file.

D. Data Quality (0.20)
  5 — Title, author, year for every row; consistent format.
  4 — Mostly clean, 1 minor gap.
  3 — Inconsistencies in 30% of rows.
  2 — Many gaps.
  1 — Mostly empty.
"""
