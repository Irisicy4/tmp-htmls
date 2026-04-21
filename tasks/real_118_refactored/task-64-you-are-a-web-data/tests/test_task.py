import sys
sys.path.insert(0, "/harness")
from evaluator import evaluate

TASK_INSTRUCTION = ("""You are a web data extraction agent. Go to https://www.jalan.net/en/japan_hotels_ryokan/hokkaido/ and extract all accommodation entries listed on the page. For each entry, extract the hotel name and detail URL, deduplicate, and output as a CSV file with columns: hotel_name, hotel_url.""")

USER_PROMPT_TEMPLATE = ("""## Task Instruction
{task_instruction}

## Task-Specific Constraints
- URL: must visit the exact URL provided (jalan.net Hokkaido hotel listings)
- Fields: hotel_name and hotel_url — both required for every entry
- Deduplication: duplicate entries must be removed
- Output: valid CSV file with correct column headers

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to the correct URL?
- How many hotel entries were extracted?
- Are both hotel_name and hotel_url present for each entry?
- Was deduplication performed?
- Was a CSV file produced?

### Step 2: Dimension Scoring

#### A. Site Navigation (0.15)
Did the agent navigate to the correct URL?

5 — Agent visited the exact URL and loaded the hotel listing page.
4 — Agent reached the site but had pagination or loading issues.
3 — Agent found similar content via search rather than direct URL.
2 — Agent visited wrong section of the site.
1 — No navigation to the target URL.

#### B. Extraction Completeness (0.35)
Were all hotel entries extracted?

5 — All visible hotel entries extracted with both name and URL for each.
4 — Most entries extracted but a few missing.
3 — Partial extraction (less than 75% of entries).
2 — Only a few sample entries extracted.
1 — No extraction performed.

#### C. Data Accuracy (0.3)
Are hotel names and URLs accurate?

5 — Names match the page content; URLs are valid and point to correct detail pages.
4 — Names correct; URLs have minor issues (relative vs absolute).
3 — Names mostly correct but some garbled; URLs partially correct.
2 — Significant accuracy issues.
1 — Data is fabricated or completely wrong.

#### D. Csv Output (0.2)
Was valid CSV produced with correct columns?

5 — Valid CSV with hotel_name and hotel_url headers, one row per hotel, no duplicates.
4 — CSV produced but minor issues (e.g. extra columns, inconsistent quoting).
3 — CSV-like output but not proper format.
2 — Data in table or JSON instead of CSV.
1 — No file output.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "site_navigation": <1-5>,
  "extraction_completeness": <1-5>,
  "data_accuracy": <1-5>,
  "csv_output": <1-5>,
  "dimension_reasoning": {{
    "site_navigation": "<one sentence citing specific evidence>",
    "extraction_completeness": "<one sentence citing specific evidence>",
    "data_accuracy": "<one sentence citing specific evidence>",
    "csv_output": "<one sentence citing specific evidence>",
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
""")

DIMENSION_WEIGHTS = {
    "site_navigation": 0.15,
    "extraction_completeness": 0.35,
    "data_accuracy": 0.3,
    "csv_output": 0.2,
}


def test(result):
    return evaluate(result, TASK_INSTRUCTION, USER_PROMPT_TEMPLATE, DIMENSION_WEIGHTS)
