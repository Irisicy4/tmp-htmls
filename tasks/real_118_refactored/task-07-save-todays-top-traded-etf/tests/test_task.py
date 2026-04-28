import sys
sys.path.insert(0, "/harness")
from evaluator import evaluate

TASK_INSTRUCTION = ("Save today's top traded ETF stocks from Naver Securities to Google Docs")

USER_PROMPT_TEMPLATE = ("""## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Platform: Naver Securities (finance.naver.com) specifically
- Data: today's top traded ETFs — must reflect current trading data, not historical or generic lists
- Required fields: ETF name, ticker, current price, trading volume (or value)
- Output: saved to a Google Docs document (not a local file or Google Sheets)

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate Naver Securities? Cite evidence from trace.
- Did the agent locate the top traded ETF section?
- What ETFs are listed — are they real Korean ETF tickers (e.g. KODEX, TIGER series)?
- Is the data clearly from today (date mentioned, current prices/volumes)?
- Was a Google Docs document created? Evidence (URL, creation confirmation, trace)?

### Step 2: Dimension Scoring

#### A. Platform Navigation
Did the agent navigate Naver Securities and locate today's top traded ETFs?

5 — Agent navigated finance.naver.com, located the ETF trading section, and retrieved today's top traded ETFs with tickers and data.
4 — Agent reached Naver Securities but had difficulty finding the ETF section; partial data retrieved.
3 — Agent found ETF data via a general web search referencing Naver rather than navigating directly.
2 — Agent described how to find the data without navigating the site.
1 — No navigation to Naver Securities; response is from prior knowledge or hallucinated.

#### B. Data Accuracy
Is the ETF data accurate and current for today?

5 — Real Korean ETF tickers listed (e.g. KODEX 200, TIGER ETFs); prices and volumes plausible for current trading day; date of data stated.
4 — Real ETF names listed with prices but trading volume missing or date not stated.
3 — ETF names present but prices or tickers are approximate or unverified.
2 — Generic or hallucinated ETF names; data does not appear to be from today.
1 — No ETF data or completely fabricated.

#### C. Google Docs Save
Was the data saved to a Google Docs document?

5 — Google Docs document created with ETF data; URL or creation confirmation in response or trace.
4 — Document creation attempted; trace confirms Google Docs API call but no URL provided.
3 — Agent compiled the data in the response but did not save to Google Docs.
2 — Agent mentioned creating a document without doing so.
1 — No document created.

#### D. Report Completeness
Is the report complete and well-organised?

5 — All required fields present (name, ticker, price, volume) for each ETF; clearly structured.
4 — Most fields present; one field missing for some ETFs.
3 — ETF names listed but key fields (price or volume) missing for most entries.
2 — Only ETF names listed with no trading data.
1 — No structured output.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "dimension_reasoning": {{
    "platform_navigation": "<one sentence citing specific evidence>",
    "data_accuracy": "<one sentence citing specific evidence>",
    "google_docs_save": "<one sentence citing specific evidence>",
    "report_completeness": "<one sentence citing specific evidence>"
  }},
  "platform_navigation": <1-5>,
  "data_accuracy": <1-5>,
  "google_docs_save": <1-5>,
  "report_completeness": <1-5>,
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
""")

DIMENSION_WEIGHTS = {
    "platform_navigation": 0.35,
    "data_accuracy":       0.30,
    "google_docs_save":    0.25,
    "report_completeness": 0.10,
}


def test(result):
    return evaluate(result, TASK_INSTRUCTION, USER_PROMPT_TEMPLATE, DIMENSION_WEIGHTS)
