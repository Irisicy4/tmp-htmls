import sys
sys.path.insert(0, "/harness")
from evaluator import evaluate

TASK_INSTRUCTION = ("""Analyze the following student profile data and generate a structured report covering political affiliation, English proficiency exam results, other certification exam results, graduate school recommendations, scholarships and honors, arts and sports, academic competitions, research output, student leadership roles, volunteer hours, and social practice activities.""")

USER_PROMPT_TEMPLATE = ("""## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Input: student profile data must be provided (structured data with multiple fields)
- Coverage: all 11 categories must be addressed
- Output: structured report (not just raw data)
- Analysis: each category should have an analytical summary, not just listed facts

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent analyze actual student profile data?
- How many of the 11 categories are covered?
- Is the output structured with clear sections?
- Does each section provide analysis beyond raw data?
- Is the report professionally formatted?

### Step 2: Dimension Scoring

#### A. Data Processing (0.2)
Did the agent process actual student profile data?

5 — Agent clearly read and processed the student data; specific data points referenced.
4 — Data processed but some fields missed.
3 — Data partially processed.
2 — Generic template filled without specific student data.
1 — No data processing.

#### B. Category Coverage (0.35)
How many of the 11 categories are covered?

5 — All 11 categories addressed.
4 — 9-10 categories.
3 — 7-8 categories.
2 — 4-6 categories.
1 — Fewer than 4 categories.

#### C. Analytical Depth (0.3)
Is each category analyzed rather than just listed?

5 — Each category has interpretation: strengths, gaps, comparison to standards, recommendations.
4 — Most categories analyzed; some just listed.
3 — About half analyzed; rest listed.
2 — Mostly raw data without analysis.
1 — No analysis.

#### D. Report Structure (0.15)
Is the report well-structured and professional?

5 — Clear headers for each category, executive summary, consistent formatting.
4 — Good structure but missing summary or inconsistent.
3 — Sections present but formatting rough.
2 — Wall of text.
1 — No structure.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "dimension_reasoning": {{
    "data_processing": "<one sentence citing specific evidence>",
    "category_coverage": "<one sentence citing specific evidence>",
    "analytical_depth": "<one sentence citing specific evidence>",
    "report_structure": "<one sentence citing specific evidence>",
  }},
  "data_processing": <1-5>,
  "category_coverage": <1-5>,
  "analytical_depth": <1-5>,
  "report_structure": <1-5>,
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
""")

DIMENSION_WEIGHTS = {
    "data_processing": 0.2,
    "category_coverage": 0.35,
    "analytical_depth": 0.3,
    "report_structure": 0.15,
}


def test(result):
    return evaluate(result, TASK_INSTRUCTION, USER_PROMPT_TEMPLATE, DIMENSION_WEIGHTS)
