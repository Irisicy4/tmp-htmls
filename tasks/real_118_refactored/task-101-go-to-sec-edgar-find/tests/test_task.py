import sys
sys.path.insert(0, "/harness")
from evaluator import evaluate

TASK_INSTRUCTION = ("Go to SEC EDGAR and search for the three most recent 10-K filings from S&P 500 energy "
    "companies that discuss 'climate risk' as a legal or regulatory liability. For each filing, "
    "extract: company name, ticker, filing date, and a verbatim quote of the key climate-risk "
    "disclosure paragraph. Then go to CourtListener and search for any federal lawsuits filed "
    "against those same companies involving climate liability within the past two years; note "
    "the case name, court, and current status for any matches. Compile findings into a structured "
    "legal-exposure summary table with columns: Company, Ticker, 10-K Filing Date, Climate-Risk "
    "Disclosure Quote, Related Lawsuit (if any), Court, Case Status.")

USER_PROMPT_TEMPLATE = ("""## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Sources: Must use SEC EDGAR for 10-K filings AND CourtListener for case search (both required)
- Companies: Must be S&P 500 energy sector companies (e.g. ExxonMobil, Chevron, ConocoPhillips, EOG, Pioneer)
- EDGAR: Must retrieve actual filing details (company, ticker, date, verbatim disclosure quote)
- CourtListener: Must perform an actual search and report findings or explicitly state no matches found
- Output: A structured table with all seven specified columns

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate SEC EDGAR and find 10-K filings from energy companies? Cite evidence.
- Are the companies named actual S&P 500 energy sector companies?
- Did the agent extract verbatim climate-risk disclosure quotes from the filings?
- Did the agent search CourtListener for climate liability lawsuits?
- Is there a structured table with the required columns in the final response?

### Step 2: Dimension Scoring

#### A. Source Navigation & Data Retrieval
Did the agent navigate both SEC EDGAR and CourtListener and retrieve real data?

5 — Agent navigated EDGAR and retrieved actual 10-K filings with real data AND searched CourtListener with documented findings.
4 — Agent retrieved data from one source properly; the other source was attempted but incomplete.
3 — Agent retrieved partial data from both sources but with notable gaps (e.g., missing tickers or filing dates).
2 — Agent used only one source or relied heavily on internal knowledge without real navigation.
1 — No evidence of actual navigation; response is from prior knowledge or fabricated.

#### B. Data Accuracy & Specificity
Are the company names, tickers, filing dates, and disclosure quotes specific and plausible?

5 — All three companies are named S&P 500 energy firms with correct tickers, specific filing dates (YYYY-MM-DD), and verbatim or near-verbatim disclosure quotes.
4 — Two of three entries are fully specific; one is missing a detail (e.g., no verbatim quote).
3 — Companies are named but details are vague, approximate, or partially fabricated.
2 — Generic or clearly hallucinated company details with no specific filing evidence.
1 — No specific company data; completely generic or irrelevant response.

#### C. CourtListener Search Quality
Did the agent meaningfully search CourtListener and report findings?

5 — Agent searched CourtListener, found specific cases (or explicitly confirmed no matches with search evidence) and reported case names, courts, and status.
4 — Agent searched CourtListener and reported results but with incomplete case details.
3 — Agent mentioned CourtListener but search results are vague or not clearly tied to the EDGAR companies.
2 — Agent attempted CourtListener but could not retrieve results; no meaningful findings.
1 — CourtListener not used; lawsuit information is fabricated or omitted entirely.

#### D. Output Structure & Completeness
Is the final output a complete, well-structured table with all required columns?

5 — Complete table with all seven columns (Company, Ticker, 10-K Filing Date, Climate-Risk Disclosure Quote, Related Lawsuit, Court, Case Status) for all three companies.
4 — Table present with six of seven columns, or one company entry is incomplete.
3 — Table-like structure but missing two or more columns or one company entirely.
2 — Narrative response with no table; information present but unstructured.
1 — Missing output or completely unstructured response.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "source_navigation": <1-5>,
  "data_accuracy": <1-5>,
  "courtlistener_search": <1-5>,
  "output_structure": <1-5>,
  "dimension_reasoning": {{
    "source_navigation": "<one sentence citing specific evidence>",
    "data_accuracy": "<one sentence citing specific evidence>",
    "courtlistener_search": "<one sentence citing specific evidence>",
    "output_structure": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
""")

DIMENSION_WEIGHTS = {
    "source_navigation": 0.30,
    "data_accuracy":     0.30,
    "courtlistener_search": 0.20,
    "output_structure":  0.20,
}


def test(result):
    return evaluate(result, TASK_INSTRUCTION, USER_PROMPT_TEMPLATE, DIMENSION_WEIGHTS)
