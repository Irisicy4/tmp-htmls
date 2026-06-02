"""Grounded-judge shim — delegates to
agentic_judge.grounded.framework.grounded_judge_test using the per-task spec
task_101_sec.

TASK_INSTRUCTION is a triple-quoted string copy of instruction.md (body +
agentic_judge.grounded addendum) so that scripts_batch/deprivacy_modal_runner.py
can extract it via AST and feed it to the agent unchanged.
"""
from importlib import import_module


_SPEC_MODULE = "task_101_sec"


TASK_INSTRUCTION = 'Go to SEC EDGAR (https://efts.sec.gov/LATEST/search-index?q=%22climate+risk%22&dateRange=custom&startdt=2024-01-01&enddt=2025-01-01&forms=10-K) and search for the three most recent 10-K filings from S&P 500 energy companies that discuss "climate risk" as a legal or regulatory liability. For each filing, extract: company name, ticker, filing date, and a verbatim quote of the key climate-risk disclosure paragraph. Then go to CourtListener (https://www.courtlistener.com/) and search for any federal lawsuits filed against those same companies involving climate liability within the past two years; note the case name, court, and current status for any matches you find. Compile your findings into a structured legal-exposure summary table with columns: Company, Ticker, 10-K Filing Date, Climate-Risk Disclosure Quote, Related Lawsuit (if any), Court, Case Status.\n<!-- agentic_judge.grounded addendum -->\n\n---\n\n## Required output format\n\nAfter completing the task, output your result as a JSON block delimited by `=== JSON RESULT ===` on its own line, the JSON, and `=== END JSON ===` on its own line. Include EVERY required field below; optional fields are rewarded when present.\n\n**Required fields:**\n\n- `filings` (list[object]) — Three 10-K filings.\n- `filings[].company` (string) — \n- `filings[].ticker` (string) — \n- `filings[].filing_date` (string) — ISO date.\n- `filings[].edgar_url` (string) — Canonical EDGAR filing URL.\n- `filings[].climate_risk_quote` (string) — Verbatim disclosure paragraph.\n- `related_lawsuits` (list[object]) — CourtListener matches OR empty list with explicit \'no_match=true\'.\n- `courtlistener_searched` (boolean) — True if you actually performed the search.\n\n**Optional but graded if present:**\n\n- `filings[].sector` (string) — S&P 500 sector (\'Energy\' expected).\n- `filings[].form_type` (string) — \'10-K\'.\n- `related_lawsuits[].case_name` (string) — \n- `related_lawsuits[].court` (string) — \n- `related_lawsuits[].status` (string) — \n- `related_lawsuits[].courtlistener_url` (string) — \n- `methodology_notes` (string) — How you narrowed S&P 500 energy.\n\n**Anti-fabrication note:** the grader fetches the URLs you cite and verifies the figures and quotes against the live page. Be precise; do not paraphrase numeric facts.\n\n**Example shape (values illustrative, not literal):**\n\n```json\n{\n  "filings": [\n    {\n      "company": "ExxonMobil",\n      "ticker": "XOM",\n      "filing_date": "2024-02-28",\n      "edgar_url": "https://www.sec.gov/cgi-bin/browse-edgar?...",\n      "climate_risk_quote": "..."\n    }\n  ],\n  "related_lawsuits": [],\n  "courtlistener_searched": true\n}\n```\n'


def test(result: dict) -> dict:
    from agentic_judge.grounded.framework import grounded_judge_test
    spec = import_module(f"agentic_judge.grounded.task_specs.{_SPEC_MODULE}")
    return grounded_judge_test(result, spec)
