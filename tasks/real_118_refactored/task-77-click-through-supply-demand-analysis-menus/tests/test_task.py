import sys
sys.path.insert(0, "/harness")
from evaluator import evaluate

TASK_INSTRUCTION = ("""Click through all the supply-demand analysis menus, analyze by institution and by foreign investor, and recommend stocks.""")

USER_PROMPT_TEMPLATE = ("""## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Platform: Korean stock market tool (likely HTS like Kiwoom, Shinhan, or a web platform)
- Coverage: must navigate both 'by institution' AND 'by foreign investor' menus
- Output: stock recommendations grounded in the supply-demand data
- Data: must be current (recent trading data)

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate supply-demand analysis menus?
- Were both institutional and foreign investor tabs/sections covered?
- What specific stocks showed significant institutional or foreign buying?
- Were stock recommendations made with data backing?

### Step 2: Dimension Scoring

#### A. Menu Navigation (0.25)
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
1 — No recommendations.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "menu_navigation": <1-5>,
  "data_extraction": <1-5>,
  "analysis_quality": <1-5>,
  "recommendation_quality": <1-5>,
  "dimension_reasoning": {{
    "menu_navigation": "<one sentence citing specific evidence>",
    "data_extraction": "<one sentence citing specific evidence>",
    "analysis_quality": "<one sentence citing specific evidence>",
    "recommendation_quality": "<one sentence citing specific evidence>",
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
""")

DIMENSION_WEIGHTS = {
    "menu_navigation": 0.25,
    "data_extraction": 0.3,
    "analysis_quality": 0.3,
    "recommendation_quality": 0.15,
}


def test(result):
    return evaluate(result, TASK_INSTRUCTION, USER_PROMPT_TEMPLATE, DIMENSION_WEIGHTS)
