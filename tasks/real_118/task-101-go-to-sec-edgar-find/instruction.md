Go to SEC EDGAR (https://efts.sec.gov/LATEST/search-index?q=%22climate+risk%22&dateRange=custom&startdt=2024-01-01&enddt=2025-01-01&forms=10-K) and search for the three most recent 10-K filings from S&P 500 energy companies that discuss "climate risk" as a legal or regulatory liability. For each filing, extract: company name, ticker, filing date, and a verbatim quote of the key climate-risk disclosure paragraph. Then go to CourtListener (https://www.courtlistener.com/) and search for any federal lawsuits filed against those same companies involving climate liability within the past two years; note the case name, court, and current status for any matches you find. Compile your findings into a structured legal-exposure summary table with columns: Company, Ticker, 10-K Filing Date, Climate-Risk Disclosure Quote, Related Lawsuit (if any), Court, Case Status.
<!-- agentic_judge.grounded addendum -->

---

## Required output format

After completing the task, output your result as a JSON block delimited by `=== JSON RESULT ===` on its own line, the JSON, and `=== END JSON ===` on its own line. Include EVERY required field below; optional fields are rewarded when present.

**Required fields:**

- `filings` (list[object]) — Three 10-K filings.
- `filings[].company` (string) — 
- `filings[].ticker` (string) — 
- `filings[].filing_date` (string) — ISO date.
- `filings[].edgar_url` (string) — Canonical EDGAR filing URL.
- `filings[].climate_risk_quote` (string) — Verbatim disclosure paragraph.
- `related_lawsuits` (list[object]) — CourtListener matches OR empty list with explicit 'no_match=true'.
- `courtlistener_searched` (boolean) — True if you actually performed the search.

**Optional but graded if present:**

- `filings[].sector` (string) — S&P 500 sector ('Energy' expected).
- `filings[].form_type` (string) — '10-K'.
- `related_lawsuits[].case_name` (string) — 
- `related_lawsuits[].court` (string) — 
- `related_lawsuits[].status` (string) — 
- `related_lawsuits[].courtlistener_url` (string) — 
- `methodology_notes` (string) — How you narrowed S&P 500 energy.

**Anti-fabrication note:** the grader fetches the URLs you cite and verifies the figures and quotes against the live page. Be precise; do not paraphrase numeric facts.

**Example shape (values illustrative, not literal):**

```json
{
  "filings": [
    {
      "company": "ExxonMobil",
      "ticker": "XOM",
      "filing_date": "2024-02-28",
      "edgar_url": "https://www.sec.gov/cgi-bin/browse-edgar?...",
      "climate_risk_quote": "..."
    }
  ],
  "related_lawsuits": [],
  "courtlistener_searched": true
}
```
