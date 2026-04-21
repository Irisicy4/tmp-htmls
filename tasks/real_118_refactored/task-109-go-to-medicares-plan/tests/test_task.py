import sys
sys.path.insert(0, "/harness")
from evaluator import evaluate

TASK_INSTRUCTION = ("Go to Medicare's Plan Finder and search for Part D prescription drug plans available in "
    "ZIP code 33101 (Miami, FL) for a beneficiary who takes atorvastatin 40mg, metformin 500mg, "
    "and lisinopril 10mg. Identify the three plans with the lowest estimated annual drug cost. "
    "For each plan, record: plan name, insurer, monthly premium, annual deductible, and estimated "
    "annual drug cost. Then visit the formulary lookup page for the top-ranked plan on the insurer's "
    "website and confirm that all three drugs are covered and at what tier. Produce a Medicare Part D "
    "plan comparison table with all specified columns plus a 'total cost' row.")

USER_PROMPT_TEMPLATE = ("""## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Primary source: medicare.gov/plan-compare for Part D plan search
- ZIP code: 33101 (Miami, FL) specifically
- Medications: atorvastatin 40mg, metformin 500mg, lisinopril 10mg — all three must be entered
- Selection: Three plans with LOWEST estimated annual drug cost
- Required fields: plan name, insurer, monthly premium, annual deductible, estimated annual drug cost
- Formulary check: Top-ranked plan's formulary on the insurer's own website; drug tier for each medication
- Output: Comparison table with all six columns plus a total cost row (premiums × 12 + drug costs)

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate Medicare Plan Finder and enter ZIP 33101 and the three medications?
- Are three specific Part D plans retrieved with insurer names and cost details?
- Did the agent access the top plan's formulary on the insurer's website?
- Are drug tier assignments for all three medications reported?
- Is there a comparison table with a total cost row?

### Step 2: Dimension Scoring

#### A. Medicare Plan Finder Navigation
Did the agent use medicare.gov/plan-compare with correct inputs?

5 — Navigated Medicare Plan Finder, entered ZIP 33101, and added all three medications (atorvastatin, metformin, lisinopril) before retrieving plan results.
4 — Navigated Plan Finder with correct ZIP but only two of three medications entered.
3 — Navigated Plan Finder with correct ZIP but medications not entered; plans retrieved without drug-cost personalization.
2 — Mentioned Medicare Plan Finder but plan data appears generic or not from actual navigation with the specified inputs.
1 — Medicare Plan Finder not used; plans are from prior knowledge or fabricated.

#### B. Plan Cost Data Completeness
Are monthly premium, annual deductible, and estimated annual drug cost present for all three plans?

5 — All three plans have plan name, insurer, monthly premium, annual deductible, and estimated annual drug cost.
4 — Two of three plans are fully detailed; one is missing one cost field.
3 — All three plans named but cost data incomplete for at least two plans.
2 — Only one plan has complete cost data.
1 — No specific plan cost data.

#### C. Formulary Verification
Did the agent check the top plan's formulary on the insurer's own website for drug tier assignments?

5 — Agent navigated to the top-ranked insurer's formulary website and confirmed tier (1-5 or equivalent) for all three drugs.
4 — Agent checked the formulary and confirmed tiers for two of three drugs.
3 — Agent checked the formulary but reported only covered/not covered (yes/no) without tier numbers.
2 — Agent mentioned checking formulary but no specific tier information retrieved.
1 — Formulary not checked on insurer website; tier information absent or fabricated.

#### D. Output Table & Total Cost Row
Is the comparison table complete with all required columns and a total cost row?

5 — Complete table with all six columns for all three plans plus a correctly calculated total cost row (monthly premium × 12 + estimated annual drug cost).
4 — Table with five of six columns; total cost row present but calculation not shown.
3 — Table present with four or fewer columns; or total cost row missing.
2 — Narrative comparison without structured table.
1 — No table.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "plan_finder_navigation": <1-5>,
  "plan_cost_data": <1-5>,
  "formulary_verification": <1-5>,
  "output_table": <1-5>,
  "dimension_reasoning": {{
    "plan_finder_navigation": "<one sentence citing specific evidence>",
    "plan_cost_data": "<one sentence citing specific evidence>",
    "formulary_verification": "<one sentence citing specific evidence>",
    "output_table": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
""")

DIMENSION_WEIGHTS = {
    "plan_finder_navigation": 0.30,
    "plan_cost_data":         0.25,
    "formulary_verification": 0.25,
    "output_table":           0.20,
}


def test(result):
    return evaluate(result, TASK_INSTRUCTION, USER_PROMPT_TEMPLATE, DIMENSION_WEIGHTS)
