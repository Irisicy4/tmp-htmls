import sys
sys.path.insert(0, "/harness")
from evaluator import evaluate

TASK_INSTRUCTION = ("Go to the USPTO Patent Full-Text and Image Database or Google Patents and search for patents "
    "filed in the last three years by Apple, Samsung, and Google related to 'on-device large language "
    "model inference.' For each company, identify the two most relevant patents and record: patent "
    "number, title, filing date, key independent claim (claim 1), and cited prior art. Then "
    "cross-reference each patent on Google Patents to check forward citations and any patent family "
    "members. Produce a competitive IP landscape table with columns: Company, Patent Number, Title, "
    "Filing Date, Key Claim Summary, Forward Citations, Family Size.")

USER_PROMPT_TEMPLATE = ("""## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Sources: Must use USPTO patent database AND Google Patents (both required)
- Companies: Apple, Samsung, and Google specifically — all three required
- Per company: Two patents minimum
- Patent details required: patent number (US format), title, filing date, key claim 1 summary, prior art
- Cross-reference: forward citations and patent family size from Google Patents
- Output: Structured table with all seven specified columns for all six patents (2 per company)

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate USPTO and/or Google Patents and retrieve real patent data?
- Are patents found for all three companies (Apple, Samsung, Google)?
- Are patent numbers in plausible US patent format (e.g., US11234567B2)?
- Did the agent retrieve forward citation counts and family sizes?
- Is there a structured table with the required columns?

### Step 2: Dimension Scoring

#### A. Patent Database Navigation
Did the agent navigate USPTO and Google Patents and retrieve real patent records?

5 — Agent navigated both USPTO and Google Patents, retrieved actual patent records with real patent numbers and details.
4 — Agent navigated one platform thoroughly and the other partially.
3 — Agent retrieved patents from at least one platform but records appear incomplete or partially fabricated.
2 — Agent searched but could not retrieve specific patents; response relies heavily on prior knowledge.
1 — No evidence of actual database navigation; patents are invented or generic.

#### B. Patent Data Completeness
Are all required fields present for all six patents (2 per company × 3 companies)?

5 — All six patents present with patent number, title, filing date, claim 1 summary, and prior art for all.
4 — Five of six patents fully detailed; one is missing a minor field.
3 — At least four patents with most required fields; notable gaps in one or two entries.
2 — Fewer than four patents with complete data, or companies are missing.
1 — Fewer than three patents found or missing one company entirely.

#### C. Forward Citations & Family Data
Did the agent cross-reference Google Patents for forward citations and family size?

5 — Forward citation count and patent family size retrieved from Google Patents for all or most patents.
4 — Cross-reference done for at least four patents with specific numbers reported.
3 — Cross-reference mentioned but data is vague (e.g., "several citations" without numbers) or covers fewer than four patents.
2 — Cross-reference attempted but no specific citation/family data retrieved.
1 — No cross-referencing done; forward citation and family data absent.

#### D. Output Structure & Analytical Value
Is the final output a complete, well-structured IP landscape table with analytical value?

5 — Complete table with all seven columns for all six patents, well-organized by company, with claim summaries that clearly describe the patented invention.
4 — Table present with minor structural gaps (one missing column or one incomplete entry).
3 — Table-like output with two or more missing columns or entries, or claim summaries are too vague.
2 — Narrative response with no table; information is unstructured.
1 — No meaningful structured output.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "patent_db_navigation": <1-5>,
  "patent_data_completeness": <1-5>,
  "forward_citation_data": <1-5>,
  "output_structure": <1-5>,
  "dimension_reasoning": {{
    "patent_db_navigation": "<one sentence citing specific evidence>",
    "patent_data_completeness": "<one sentence citing specific evidence>",
    "forward_citation_data": "<one sentence citing specific evidence>",
    "output_structure": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
""")

DIMENSION_WEIGHTS = {
    "patent_db_navigation":     0.25,
    "patent_data_completeness": 0.30,
    "forward_citation_data":    0.25,
    "output_structure":         0.20,
}


def test(result):
    return evaluate(result, TASK_INSTRUCTION, USER_PROMPT_TEMPLATE, DIMENSION_WEIGHTS)
