import sys
sys.path.insert(0, "/harness")
from evaluator import evaluate

TASK_INSTRUCTION = ("Go to the Bureau of Labor Statistics Occupational Outlook Handbook and look up the job "
    "outlook, median annual wage, and projected 10-year growth rate for three roles: Machine "
    "Learning Engineer, Data Scientist, and AI Research Scientist. Then go to LinkedIn Jobs and "
    "note the number of open positions in the United States for each role. Finally, visit "
    "Levels.fyi and find the median total compensation (base + bonus + equity) at the L4/E4 "
    "equivalent level at Google, Meta, and Amazon for each of the three roles. Produce a talent "
    "market intelligence report table with all specified columns and a 'market heat' rating "
    "(Hot / Warm / Cooling) for each role.")

USER_PROMPT_TEMPLATE = ("""## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Sources: BLS OOH (bls.gov/ooh), LinkedIn Jobs, AND Levels.fyi — all three required
- Roles: Machine Learning Engineer, Data Scientist, AND AI Research Scientist — all three
- BLS fields: median annual wage, projected 10-year growth %, job outlook description
- LinkedIn: US open position count (approximate is acceptable)
- Levels.fyi: median TC at L4/E4 equivalent at Google, Meta, AND Amazon for each role
- Output: Seven-column table with market heat rating (Hot/Warm/Cooling) per role

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate BLS OOH and find wage and growth data for the three roles?
- Did the agent search LinkedIn Jobs for US open position counts?
- Did the agent visit Levels.fyi and retrieve TC data for all three companies × three roles?
- Is there a complete seven-column table with market heat ratings?

### Step 2: Dimension Scoring

#### A. BLS OOH Data Retrieval
Did the agent navigate BLS OOH and retrieve median wage and growth data for all three roles?

5 — Navigated bls.gov/ooh, found median annual wage and projected 10-year growth rate for all three roles with specific figures.
4 — BLS data found for two of three roles with specific figures; third role approximated or matched to nearest BLS occupation.
3 — BLS data referenced for all three roles but one or two figures are approximate (e.g., "around $100k") or matched loosely.
2 — BLS OOH mentioned but data appears to be from prior knowledge rather than current navigation.
1 — No BLS navigation; wages and growth rates are fabricated or entirely from prior knowledge.

#### B. LinkedIn Open Position Count
Did the agent search LinkedIn Jobs and report US open position counts for each role?

5 — LinkedIn Jobs searched for all three roles in the United States; specific position counts (e.g., "12,453 results") reported for each.
4 — LinkedIn counts found for two of three roles; third role count is estimated.
3 — LinkedIn referenced for all three roles but counts are approximate or clearly dated (e.g., not from a current search).
2 — LinkedIn counts cited for one role or all counts are generic round numbers without search evidence.
1 — LinkedIn Jobs not searched; position counts absent or fabricated.

#### C. Levels.fyi Compensation Data
Did the agent retrieve TC data from Levels.fyi for all three companies and all three roles?

5 — Levels.fyi consulted; median TC (base + bonus + equity) at L4/E4 equivalent found at Google, Meta, AND Amazon for all three roles (nine data points).
4 — Levels.fyi data found for seven or eight of nine combinations; one or two company-role pairs are missing.
3 — Levels.fyi data found for at least six combinations; three missing.
2 — Levels.fyi data found for fewer than six combinations or data appears fabricated.
1 — Levels.fyi not used; all compensation figures are from prior knowledge.

#### D. Output Table & Market Heat Rating
Is the talent market intelligence table complete and are market heat ratings logical?

5 — Complete seven-column table (Role, BLS Median Wage, Growth Rate, LinkedIn Positions, Google TC, Meta TC, Amazon TC) plus market heat rating (Hot/Warm/Cooling) for all three roles, with ratings justified by the data.
4 — Table with six columns; market heat ratings present for all three roles.
3 — Table with five columns; or one role is missing from the table.
2 — Partial table with fewer than five columns; no market heat rating.
1 — No structured table.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "dimension_reasoning": {{
    "bls_data_retrieval": "<one sentence citing specific evidence>",
    "linkedin_position_count": "<one sentence citing specific evidence>",
    "levelsfyi_compensation": "<one sentence citing specific evidence>",
    "output_table": "<one sentence citing specific evidence>"
  }},
  "bls_data_retrieval": <1-5>,
  "linkedin_position_count": <1-5>,
  "levelsfyi_compensation": <1-5>,
  "output_table": <1-5>,
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
""")

DIMENSION_WEIGHTS = {
    "bls_data_retrieval": 0.35,
    "linkedin_position_count": 0.17,
    "levelsfyi_compensation": 0.26,
    "output_table": 0.22,
}


def test(result):
    return evaluate(result, TASK_INSTRUCTION, USER_PROMPT_TEMPLATE, DIMENSION_WEIGHTS)
