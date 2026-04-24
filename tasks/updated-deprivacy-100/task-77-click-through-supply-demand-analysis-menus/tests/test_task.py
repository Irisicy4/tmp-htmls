"""LLM-as-judge evaluator for task-77-click-through-supply-demand-analysis-menus.

Category: Finance & Economics
Task: Click through all the supply-demand analysis menus, analyze by institution and by foreign investor, and recommend stocks.
"""

import sys
sys.path.insert(0, "/harness")
from evaluator import run_judge

TASK_INSTRUCTION = 'Click through all the supply-demand analysis menus, analyze by institution and by foreign investor, and recommend stocks.'

SYSTEM_PROMPT_EXTRA = "Your job is to assess whether an AI agent successfully completed a browser-based task.\n\nThe task involves navigating a Korean stock market platform's supply-demand analysis features, reviewing institutional and foreign investor activity, and making stock recommendations based on the data."

CONSTRAINTS = """- Platform: Korean stock market tool (likely HTS like Kiwoom, Shinhan, or a web platform)
- Coverage: must navigate both 'by institution' AND 'by foreign investor' menus
- Output: stock recommendations grounded in the supply-demand data
- Data: must be current (recent trading data)"""

EVIDENCE_QUESTIONS = """- Did the agent navigate supply-demand analysis menus?
- Were both institutional and foreign investor tabs/sections covered?
- What specific stocks showed significant institutional or foreign buying?
- Were stock recommendations made with data backing?"""

DIMENSION_RUBRICS = """#### A. Menu Navigation (0.25)
Did the agent navigate the supply-demand analysis menus?

5 — Agent navigated to supply-demand analysis and accessed both institutional and foreign investor sections.
4 — Accessed one section but not the other.
3 — Found supply-demand analysis but navigation was incomplete.
2 — Described the menu structure without navigating.
1 — No navigation.

#### B. Data Extraction (0.3)
Was relevant trading data extracted?

5 — Specific stocks with institutional/foreign net buying amounts and dates extracted.
4 — Stock data found but less precise.
3 — General trends identified without specific stock data.
2 — Data described without being extracted.
1 — No data extracted.

#### C. Analysis Quality (0.3)
Was the data properly analyzed?

5 — Clear analysis: which stocks show consistent institutional accumulation vs distribution; foreign investor trends.
4 — Good analysis but one dimension (inst or foreign) weaker.
3 — Basic pattern identification without depth.
2 — Data presented without analysis.
1 — No analysis.

#### D. Recommendation Quality (0.15)
Were stock recommendations grounded in the data?

5 — Specific stock recommendations with ticker, reasoning from supply-demand data, and risk note.
4 — Recommendations made with some data backing.
3 — Recommendations made but thinly supported.
2 — Generic recommendations not tied to supply-demand data.
1 — No recommendations."""

DIMENSION_WEIGHTS = {
    'menu_navigation': 0.25,
    'data_extraction': 0.3,
    'analysis_quality': 0.3,
    'recommendation_quality': 0.15,
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
