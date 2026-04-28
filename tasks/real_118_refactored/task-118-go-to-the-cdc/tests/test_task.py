import sys
sys.path.insert(0, "/harness")
from evaluator import evaluate

TASK_INSTRUCTION = ("Go to CDC WONDER and use the Underlying Cause of Death database to query age-adjusted "
    "cardiovascular disease mortality rates (ICD-10 codes I00-I99) for all 50 US states for the "
    "most recent available year. Identify the five states with the highest and the five with the "
    "lowest age-adjusted cardiovascular mortality rates; record each state's rate per 100,000 "
    "population. Then go to the Commonwealth Fund's State Health System Performance scorecard "
    "and find each of those ten states' overall health system performance rank and their "
    "cardiovascular care rank specifically. Compile a cardiovascular mortality analysis table "
    "with all five specified columns including a 'disparity note' for states where CV mortality "
    "and health system rank diverge significantly.")

USER_PROMPT_TEMPLATE = ("""## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Sources: CDC WONDER (wonder.cdc.gov) AND Commonwealth Fund scorecard (commonwealthfund.org) — both required
- CDC WONDER: Underlying Cause of Death database; ICD-10 I00-I99; age-adjusted rates; all 50 states; most recent year
- States selected: Top 5 HIGHEST and bottom 5 LOWEST by age-adjusted CVD mortality (10 states total)
- Commonwealth Fund: Overall health system rank AND cardiovascular care rank for each of the 10 states
- Disparity note: Flag states where mortality rank and health system rank diverge significantly
- Output: Five-column table for all 10 states plus disparity notes

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate CDC WONDER and query CVD mortality (ICD-10 I00-I99) by state?
- Are age-adjusted rates per 100,000 population reported for 10 states (5 highest, 5 lowest)?
- Did the agent access the Commonwealth Fund scorecard for state health system ranks?
- Are overall rank and CV care rank reported for all 10 states?
- Is there a five-column table with disparity notes?

### Step 2: Dimension Scoring

#### A. CDC WONDER Navigation & Query
Did the agent navigate CDC WONDER and query cardiovascular mortality data by state?

5 — Navigated wonder.cdc.gov, used the Underlying Cause of Death database, specified ICD-10 I00-I99, grouped by state, and retrieved age-adjusted rates per 100,000 for the most recent available year.
4 — Navigated CDC WONDER and retrieved state-level CVD mortality data but ICD-10 specification or age-adjustment was not confirmed.
3 — Navigated CDC WONDER and retrieved mortality data but the query parameters (ICD codes or age-adjustment) are unclear or approximated.
2 — CDC WONDER mentioned; data appears to be from prior knowledge or secondary sources rather than a direct query.
1 — CDC WONDER not navigated; mortality rates are entirely from prior knowledge or fabricated.

#### B. State Selection Accuracy
Are the 5 highest and 5 lowest CVD mortality states correctly identified with specific rates?

5 — Five highest and five lowest states identified with specific age-adjusted rates per 100,000 (e.g., Mississippi ~380, Hawaii ~150); states are plausible given known regional health disparities.
4 — Nine of ten states identified correctly; one state appears misranked or the rate for one state is missing.
3 — States identified but rates are approximate or the top/bottom split is not clearly based on a ranked query.
2 — Ten states listed but the selection criteria (highest/lowest by CVD mortality) is not clearly applied.
1 — States are random or do not reflect CVD mortality extremes; rates absent or implausible.

#### C. Commonwealth Fund Scorecard Data
Did the agent access the Commonwealth Fund scorecard and retrieve ranks for all 10 states?

5 — Commonwealth Fund scorecard accessed; overall health system rank AND cardiovascular care rank found for all 10 states with specific rank numbers.
4 — Commonwealth Fund data found for eight or nine of ten states; one or two states are missing one rank.
3 — Commonwealth Fund data found for at least seven states; cardiovascular care rank specifically is less often available than overall rank.
2 — Commonwealth Fund referenced for fewer than seven states; most ranks estimated.
1 — Commonwealth Fund not used; all rank data is from prior knowledge or fabricated.

#### D. Output Table & Disparity Notes
Is the five-column table complete with specific disparity notes for divergent states?

5 — Complete five-column table (State, Age-Adjusted CVD Mortality Rate, Data Year, Commonwealth Overall Rank, CV Care Rank) for all 10 states; disparity notes present for at least two states with specific divergence described (e.g., "Oklahoma: high mortality rank 45 but overall health system rank 35 — moderate divergence").
4 — Table with all five columns; disparity notes present but one state that clearly diverges lacks a note.
3 — Table with four columns; or disparity notes are generic for all states.
2 — Table with three or fewer columns; no disparity notes.
1 — No structured table.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "dimension_reasoning": {{
    "cdc_wonder_navigation": "<one sentence citing specific evidence>",
    "state_selection_accuracy": "<one sentence citing specific evidence>",
    "commonwealth_fund_data": "<one sentence citing specific evidence>",
    "output_table": "<one sentence citing specific evidence>"
  }},
  "cdc_wonder_navigation": <1-5>,
  "state_selection_accuracy": <1-5>,
  "commonwealth_fund_data": <1-5>,
  "output_table": <1-5>,
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
""")

DIMENSION_WEIGHTS = {
    "cdc_wonder_navigation": 0.35,
    "state_selection_accuracy": 0.23,
    "commonwealth_fund_data": 0.23,
    "output_table": 0.19,
}


def test(result):
    return evaluate(result, TASK_INSTRUCTION, USER_PROMPT_TEMPLATE, DIMENSION_WEIGHTS)
