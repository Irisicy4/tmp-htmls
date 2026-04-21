import sys
sys.path.insert(0, "/harness")
from evaluator import evaluate

TASK_INSTRUCTION = ("Help me get all 2025 publication info from https://m.douban.com/subject_collection/ECNA7Y7GA and save as an Excel file.")

USER_PROMPT_TEMPLATE = ("""## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Source: the exact URL https://m.douban.com/subject_collection/ECNA7Y7GA (must be accessed directly, not a general Douban search)
- Filter: 2025 publications only
- Output format: Excel file (.xlsx or .xls)
- Data expected: publication titles, authors, and any available metadata (rating, date, genre); at minimum title and year

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate the specific Douban URL? Cite evidence.
- How many publications did the agent retrieve? Are they filtered to 2025?
- What fields were captured (title, author, date, rating, etc.)?
- Was an Excel file created? Is there evidence of file creation (path, confirmation)?
- Did the agent handle pagination to get all entries, or only the first page?

### Step 2: Dimension Scoring

#### A. Source Navigation
Did the agent access the specific Douban URL as instructed?

5 — Agent navigated the exact URL and retrieved content from that specific collection page.
4 — Agent accessed the URL but may have only retrieved the first page without paginating.
3 — Agent searched Douban generally without accessing the specific collection URL.
2 — Agent used a different source or described what to do without navigating.
1 — No navigation performed; response is fabricated or refused.

#### B. Data Completeness
Did the agent retrieve all 2025 publications with adequate fields?

5 — All 2025 entries retrieved with at minimum title and publication date; additional fields (author, rating) are a bonus.
4 — Most 2025 entries retrieved; pagination may have missed some (e.g. only first 2 pages of 5).
3 — Some 2025 entries retrieved but coverage is clearly incomplete (e.g. only 10-20 entries when more exist).
2 — Data retrieved but not filtered to 2025, or very few entries captured.
1 — No data retrieved or data is fabricated.

#### C. Excel Output
Was the data saved as an Excel file?

5 — Excel file created and confirmed (.xlsx/.xls); file path or download link provided.
4 — Excel creation attempted and likely succeeded per trace; no explicit confirmation in response.
3 — Agent produced data in a tabular format (e.g. CSV or JSON) but not Excel specifically.
2 — Agent displayed the data in the response without saving to any file format.
1 — No file output; agent only described the data or refused.

#### D. Data Quality
Is the extracted data clean, structured, and accurate?

5 — Data is well-structured in Excel with clear column headers; entries are accurate and consistently formatted.
4 — Data is structured but minor formatting issues (e.g. inconsistent date formats, missing some fields for a few entries).
3 — Data is present but poorly structured or requires significant cleaning.
2 — Data is raw/unprocessed text dumped into Excel without proper column structure.
1 — No structured data output.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "source_navigation": <1-5>,
  "data_completeness": <1-5>,
  "excel_output": <1-5>,
  "data_quality": <1-5>,
  "dimension_reasoning": {{
    "source_navigation": "<one sentence citing specific evidence>",
    "data_completeness": "<one sentence citing specific evidence>",
    "excel_output": "<one sentence citing specific evidence>",
    "data_quality": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
""")

DIMENSION_WEIGHTS = {
    "source_navigation": 0.20,
    "data_completeness": 0.30,
    "excel_output":      0.30,
    "data_quality":      0.20,
}


def test(result):
    return evaluate(result, TASK_INSTRUCTION, USER_PROMPT_TEMPLATE, DIMENSION_WEIGHTS)
