import sys
sys.path.insert(0, "/harness")
from evaluator import evaluate

TASK_INSTRUCTION = ("""This GitHub project claims to have reached 5,000 stars in 48 hours. I'm certain the data is fake. Please compile a record of star inflation and also analyze the forks from the same angle.""")

USER_PROMPT_TEMPLATE = ("""## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Target: a specific GitHub repository (must be identified from context or searched)
- Analysis: star growth timeline — look for unnatural spikes, bot-like patterns
- Forks: same analysis applied to fork count
- Evidence: must use data (star history charts, API data, timestamps) not just assertion

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent access GitHub star history data for the repository?
- What growth patterns were found (spikes, timing, geographic concentration)?
- Was fork analysis also performed?
- What specific evidence of inflation was identified?
- Were external tools (star-history.com, GitHub API) used?

### Step 2: Dimension Scoring

#### A. Star History Analysis (0.35)
Did the agent analyze star growth timeline?

5 — Star growth timeline retrieved (via star-history.com or GitHub API) showing specific dates, volumes, and suspicious spikes.
4 — Star growth analyzed but with less precise data.
3 — Growth pattern described without timeline data.
2 — Agent asserted inflation without data.
1 — No star history analysis.

#### B. Inflation Evidence (0.3)
Was specific evidence of star inflation identified?

5 — Specific red flags cited: unnatural spike timing, accounts with no other activity, geographic clustering, similar account creation dates.
4 — 2-3 specific indicators identified.
3 — 1 specific indicator identified.
2 — Vague inflation claims without evidence.
1 — No inflation evidence.

#### C. Fork Analysis (0.25)
Was fork analysis also performed?

5 — Fork count and growth pattern analyzed; empty/inactive forks identified as additional evidence.
4 — Forks analyzed but less thoroughly than stars.
3 — Forks mentioned but not analyzed.
2 — Fork analysis described without data.
1 — No fork analysis.

#### D. Report Quality (0.1)
Is the report well-organized and evidence-based?

5 — Structured report with timeline, evidence, and conclusion; data cited.
4 — Good report but organization could be clearer.
3 — Information present but loosely organized.
2 — Disorganized or hard to follow.
1 — No report structure.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "star_history_analysis": <1-5>,
  "inflation_evidence": <1-5>,
  "fork_analysis": <1-5>,
  "report_quality": <1-5>,
  "dimension_reasoning": {{
    "star_history_analysis": "<one sentence citing specific evidence>",
    "inflation_evidence": "<one sentence citing specific evidence>",
    "fork_analysis": "<one sentence citing specific evidence>",
    "report_quality": "<one sentence citing specific evidence>",
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
""")

DIMENSION_WEIGHTS = {
    "star_history_analysis": 0.35,
    "inflation_evidence": 0.3,
    "fork_analysis": 0.25,
    "report_quality": 0.1,
}


def test(result):
    return evaluate(result, TASK_INSTRUCTION, USER_PROMPT_TEMPLATE, DIMENSION_WEIGHTS)
