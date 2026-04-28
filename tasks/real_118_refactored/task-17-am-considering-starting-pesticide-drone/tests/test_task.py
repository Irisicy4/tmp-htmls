import sys
sys.path.insert(0, "/harness")
from evaluator import evaluate

TASK_INSTRUCTION = ("""I am considering starting a pesticide drone business.
Please research and tell me in detail:
- Future positioning and how to prepare for starting the business
- How much money is needed
- Essential products to purchase (e.g. vehicles, generators, drones), and which drone brand is good (e.g. DJI)
- Whether it can be run as a standalone business
- How much it would cost to operate (e.g. price per unit area)
- Which month is the peak season, and how much can be earned in a year
- How long it takes to recoup the investment
I am 35 years old (born 1991); please consider what is feasible at my age.
I like using computers and have some photography skills – please see if this helps with the work.

Compile everything and create a Google Docs document.""")

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Seven specific research topics required (see task instruction above)
- Personal context must be considered: user is 35 years old, computer-literate, has photography skills
- Drone brand recommendation explicitly required (e.g. DJI)
- Output must be compiled into a Google Docs document, not just returned as inline text

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
Carefully read the agent response and trace, then answer each question:

- Did the agent address all 7 required topics (future positioning, capital needed, equipment to buy, standalone viability, operating cost, peak season/annual revenue, payback period)?
- Were specific drone brands recommended (e.g. DJI)?
- Were concrete cost and revenue figures provided?
- Was the user's age (35) and background (computers, photography) referenced?
- Did the agent create a Google Docs document?
- Was a document URL or creation confirmation provided?

### Step 2: Dimension Scoring
Score each dimension 1–5 using the rubrics below.

#### A. Topic Coverage (0.40)
Did the agent research all required topics and produce a comprehensive response?

5 — All 7 topics addressed with specific information; personal context (age, skills) considered.
4 — 6 of 7 topics addressed, or personal context only lightly addressed.
3 — 4–5 topics addressed; notable gaps.
2 — Fewer than 4 topics addressed.
1 — Task abandoned or no research performed.

#### B. Information Specificity (0.30)
Is the information specific, accurate, and credible?

5 — Concrete figures (startup costs, revenue ranges, payback period), named drone brands, credible sourcing or reasoning.
4 — Mostly specific; minor gaps in figures or sourcing.
3 — General information without specific figures; vague recommendations.
2 — Vague or potentially inaccurate; no concrete figures or sources.
1 — Hallucinated or clearly wrong information.

#### C. Document Saved (0.20)
Was the compiled output saved to a Google Docs document?

5 — Google Doc created and URL or clear confirmation of creation provided.
4 — Google Doc creation attempted with partial success (e.g. doc created but content incomplete).
3 — Agent indicated intent to save to Google Docs but outcome is unclear.
2 — Output returned as inline text only; no document created.
1 — No attempt to save to a document.

#### D. Scope Adherence (0.10)
Were all aspects of the task addressed including personal context and drone brand recommendation?

5 — All 7 topics, personal context, drone brand recommendation, and document all present.
4 — Minor gap in one area.
3 — Moderate gaps across multiple areas.
2 — Significant portions of the task unaddressed.
1 — Task largely incomplete.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising your Step 1 findings>",
  "dimension_reasoning": {{
    "topic_coverage": "<one sentence citing specific evidence>",
    "information_specificity": "<one sentence citing specific evidence>",
    "document_saved": "<one sentence citing specific evidence>",
    "scope_adherence": "<one sentence citing specific evidence>"
  }},
  "topic_coverage": <1-5>,
  "information_specificity": <1-5>,
  "document_saved": <1-5>,
  "scope_adherence": <1-5>,
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "topic_coverage":       0.40,
    "information_specificity": 0.30,
    "document_saved":       0.20,
    "scope_adherence":      0.10,
}


def test(result):
    return evaluate(result, TASK_INSTRUCTION, USER_PROMPT_TEMPLATE, DIMENSION_WEIGHTS)
