"""LLM-as-judge evaluator for task-44-go-to-betman-co-kr-click.

Category: Data & ML Engineering
Task: Go to https://www.betman.co.kr/, click on game purchase hit results, set the period from January 1, 2025 to the current date, set the game type to soccer toto win/draw/loss, search, and save the resul
"""

import sys
sys.path.insert(0, "/harness")
from evaluator import run_judge

TASK_INSTRUCTION = 'Go to https://www.betman.co.kr/, click on game purchase hit results, set the period from January 1, 2025 to the current date, set the game type to soccer toto win/draw/loss, search, and save the results to Google Sheets.'

SYSTEM_PROMPT_EXTRA = 'Your job is to assess whether an AI agent successfully completed a web data extraction and spreadsheet saving task.'

CONSTRAINTS = """- Platform: betman.co.kr specifically — the agent must navigate this site
- Section: "game purchase hit results" (게임구매 적중결과) — correct section must be selected
- Date filter: January 1, 2025 to the current date — both date bounds required
- Game type filter: soccer toto win/draw/loss (축구토토 승무패) — correct game type required
- Output: results must be saved to Google Sheets (not just displayed or saved locally)"""

EVIDENCE_QUESTIONS = """- Did the agent navigate to betman.co.kr? Cite evidence.
- Did the agent locate and click the correct "game purchase hit results" section?
- Were both date filters set correctly (Jan 1, 2025 to current date)?
- Was the game type filter set to soccer toto win/draw/loss?
- Was data successfully extracted and saved to Google Sheets?"""

DIMENSION_RUBRICS = """#### A. Site Navigation & Section Selection
Did the agent navigate betman.co.kr and reach the correct section?

5 — Agent navigated betman.co.kr and reached the game purchase hit results section.
4 — Agent reached betman.co.kr but had difficulty finding the correct section.
3 — Agent reached the site but landed on a different section.
2 — Agent attempted to navigate but failed to load the site or find the section.
1 — No navigation to betman.co.kr; response is from prior knowledge or hallucinated.

#### B. Filter Configuration
Did the agent correctly apply all required filters?

5 — Both date range (Jan 1 2025 to current) and game type (soccer toto win/draw/loss) correctly set.
4 — One filter correctly set; the other is approximate or partially correct.
3 — Filters applied but one is clearly wrong (e.g. wrong date range or wrong game type).
2 — Agent attempted to set filters but could not interact with the form correctly.
1 — No filter configuration attempted.

#### C. Data Extraction Quality
Was the search executed and data successfully retrieved?

5 — Search executed, results retrieved, and data is clearly structured with match results.
4 — Search executed and results retrieved but data is incomplete or poorly structured.
3 — Search attempted but only partial results retrieved or agent encountered errors.
2 — Agent described the data without actually extracting it.
1 — No data extracted.

#### D. Google Sheets Save
Was the extracted data saved to Google Sheets?

5 — Data saved to a new or existing Google Sheet with clear confirmation (sheet name, URL, or screenshot evidence).
4 — Data pasted into Google Sheets but without clear confirmation or with minor issues.
3 — Agent attempted to save to Google Sheets but encountered errors; partial save.
2 — Data saved locally (CSV/file) instead of Google Sheets.
1 — No saving attempted."""

DIMENSION_WEIGHTS = {
    'site_navigation': 0.2,
    'filter_configuration': 0.3,
    'data_extraction_quality': 0.25,
    'google_sheets_save': 0.25,
}


def test(result):
    return run_judge(
        result,
        task_instruction=TASK_INSTRUCTION,
        system_prompt_extra=SYSTEM_PROMPT_EXTRA,
        constraints=CONSTRAINTS,
        evidence_questions=EVIDENCE_QUESTIONS,
        dimension_rubrics=DIMENSION_RUBRICS,
        dimension_weights=DIMENSION_WEIGHTS,
    )
