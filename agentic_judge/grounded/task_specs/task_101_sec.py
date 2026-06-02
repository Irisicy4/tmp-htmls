"""Spec for task-101-go-to-sec-edgar-find (Finance / Legal)."""
from ..framework.constraints import (
    JSONSchemaConforms, ListLengthInRange, AllURLsMatch, FieldPresent,
    Custom,
)
from ..framework.summary_schema import SummarySchema, RequiredField, OptionalField


TASK_INSTRUCTION = (
    "Go to SEC EDGAR (https://efts.sec.gov/LATEST/search-index?q=%22climate+risk%22"
    "&dateRange=custom&startdt=2024-01-01&enddt=2025-01-01&forms=10-K) and search "
    "for the three most recent 10-K filings from S&P 500 energy companies that "
    "discuss \"climate risk\" as a legal or regulatory liability.  For each filing, "
    "extract: company name, ticker, filing date, and a verbatim quote of the key "
    "climate-risk disclosure paragraph.  Then go to CourtListener "
    "(https://www.courtlistener.com/) and search for any federal lawsuits filed "
    "against those same companies involving climate liability within the past "
    "two years; note the case name, court, and current status for any matches."
)

SUMMARY_SCHEMA = SummarySchema(
    required=[
        RequiredField("filings",                          "list[object]", "Three 10-K filings."),
        RequiredField("filings[].company",                "string",       ""),
        RequiredField("filings[].ticker",                 "string",       ""),
        RequiredField("filings[].filing_date",            "string",       "ISO date."),
        RequiredField("filings[].edgar_url",              "string",       "Canonical EDGAR filing URL."),
        RequiredField("filings[].climate_risk_quote",     "string",       "Verbatim disclosure paragraph."),
        RequiredField("related_lawsuits",                 "list[object]", "CourtListener matches OR empty list with explicit 'no_match=true'."),
        RequiredField("courtlistener_searched",           "boolean",      "True if you actually performed the search."),
    ],
    optional=[
        OptionalField("filings[].sector",                 "string",       "S&P 500 sector ('Energy' expected)."),
        OptionalField("filings[].form_type",              "string",       "'10-K'."),
        OptionalField("related_lawsuits[].case_name",     "string",       ""),
        OptionalField("related_lawsuits[].court",         "string",       ""),
        OptionalField("related_lawsuits[].status",        "string",       ""),
        OptionalField("related_lawsuits[].courtlistener_url","string",    ""),
        OptionalField("methodology_notes",                "string",       "How you narrowed S&P 500 energy."),
    ],
    examples={
        "filings": [
            {"company": "ExxonMobil", "ticker": "XOM", "filing_date": "2024-02-28",
             "edgar_url": "https://www.sec.gov/cgi-bin/browse-edgar?...",
             "climate_risk_quote": "..."}
        ],
        "related_lawsuits": [],
        "courtlistener_searched": True,
    },
)


def _sources_include_courtlistener(summary, ctx):
    if summary.get("courtlistener_searched"):
        return True, "courtlistener_searched=true"
    return False, "courtlistener was not searched"


def _quotes_present(summary, ctx):
    fs = summary.get("filings") or []
    bad = [i for i, f in enumerate(fs) if len((f.get("climate_risk_quote") or "").split()) < 8]
    if bad:
        return False, f"items {bad} have <8-word quotes"
    return True, f"all {len(fs)} quotes ≥ 8 words"


HARD_CONSTRAINTS = [
    JSONSchemaConforms(required_paths=SUMMARY_SCHEMA.required_paths()),
    ListLengthInRange("filings", 3, 3, name="exactly_3_filings"),
    AllURLsMatch("filings", "edgar_url", r"(sec\.gov|secdatabase\.com)",
                  name="all_edgar_urls_on_sec_gov"),
    Custom("courtlistener_searched", _sources_include_courtlistener),
    Custom("quotes_substantive", _quotes_present),
]


def FAITHFULNESS_CHECKS(summary: dict) -> list[dict]:
    out = []
    for f in (summary.get("filings") or [])[:3]:
        u = f.get("edgar_url")
        if not u:
            continue
        # Verify quote (first 6 words) appears on the page
        q = (f.get("climate_risk_quote") or "").strip()
        first_words = " ".join(q.split()[:6])
        out.append({"url": u, "claim": first_words or (f.get("ticker") or "")})
    for lw in (summary.get("related_lawsuits") or [])[:2]:
        if (u := (lw or {}).get("courtlistener_url")):
            out.append({"url": u, "claim": (lw.get("case_name") or "")[:30]})
    return out


DIMENSIONS = ["source_navigation", "data_accuracy", "courtlistener_search", "output_structure"]
DIMENSION_WEIGHTS = {
    "source_navigation":    0.30,
    "data_accuracy":        0.30,
    "courtlistener_search": 0.20,
    "output_structure":     0.20,
}

TASK_RUBRIC = """A. Source Navigation (0.30)
  5 — EDGAR + CourtListener both navigated with evidence.
  4 — EDGAR clean, CourtListener brief.
  3 — One source only.
  2 — Indirect (Google).
  1 — None reached.

B. Data Accuracy (0.30)
  5 — Company/ticker/date triplets all correct; quotes verifiable on EDGAR.
  4 — Minor inaccuracy.
  3 — One filing wrong.
  2 — Multiple wrong.
  1 — Fabricated.

C. CourtListener Search (0.20)
  5 — Actual search; results or explicit no-match noted.
  4 — Search done, output thin.
  3 — Claimed but not evidenced.
  2 — Skipped.
  1 — Wrong source.

D. Output Structure (0.20)
  5 — Seven-column table with all fields.
  4 — Missing 1 column.
  3 — Half columns.
  2 — Unstructured.
  1 — Nothing.
"""
