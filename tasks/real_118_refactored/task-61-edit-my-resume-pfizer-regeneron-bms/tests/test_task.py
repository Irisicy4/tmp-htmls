import sys
sys.path.insert(0, "/harness")
from evaluator import evaluate

TASK_INSTRUCTION = ("""Can you edit my resume? I don't get any interviews. I usually apply to Pfizer, Regeneron, BMS, and Eli Lilly.""")

USER_PROMPT_TEMPLATE = ("""## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Target companies: Pfizer, Regeneron, Bristol Myers Squibb, Eli Lilly — all pharma/biotech
- The agent must work with an actual resume (either uploaded or obtained from the user)
- Edits must be substantive — not just surface formatting changes
- Agent should align resume language with pharma industry keywords and job requirements

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent access the user's resume?
- What specific edits were made (or proposed)?
- Were pharma-specific keywords and requirements addressed?
- Was the resume tailored to Pfizer/Regeneron/BMS/Eli Lilly specifically?
- Did the agent provide actionable, substantive changes?

### Step 2: Dimension Scoring

#### A. Resume Access (0.2)
Did the agent access and work with the actual resume?

5 — Agent accessed and read the resume content and made specific edits.
4 — Agent referenced resume content but edits were partial.
3 — Agent provided generic advice without accessing resume content.
2 — Agent asked for the resume without proceeding.
1 — No resume engagement.

#### B. Edit Quality (0.35)
Are the edits substantive and high-quality?

5 — Specific rewrites of bullet points, summary, and skills sections with strong action verbs and quantified achievements.
4 — Good edits but not all sections addressed.
3 — Surface-level edits (formatting, grammar) without substantive content improvement.
2 — Generic advice rather than actual edits.
1 — No edits made.

#### C. Pharma Alignment (0.3)
Is the resume tailored to pharma/biotech companies?

5 — Incorporates pharma-specific keywords (GMP, clinical trials, regulatory, drug development) and company culture alignment.
4 — Some industry-specific language but not fully targeted.
3 — Generic corporate resume improvements without pharma focus.
2 — Minimal industry awareness.
1 — No pharma/biotech tailoring.

#### D. Interview Strategy (0.15)
Does the agent explain why changes help get interviews?

5 — Clear explanation of how each major change addresses ATS keywords or recruiter priorities at target companies.
4 — Some explanation provided.
3 — Changes made without explanation.
2 — Vague rationale.
1 — No rationale.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "dimension_reasoning": {{
    "resume_access": "<one sentence citing specific evidence>",
    "edit_quality": "<one sentence citing specific evidence>",
    "pharma_alignment": "<one sentence citing specific evidence>",
    "interview_strategy": "<one sentence citing specific evidence>",
  }},
  "resume_access": <1-5>,
  "edit_quality": <1-5>,
  "pharma_alignment": <1-5>,
  "interview_strategy": <1-5>,
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
""")

DIMENSION_WEIGHTS = {
    "resume_access": 0.2,
    "edit_quality": 0.35,
    "pharma_alignment": 0.3,
    "interview_strategy": 0.15,
}


def test(result):
    return evaluate(result, TASK_INSTRUCTION, USER_PROMPT_TEMPLATE, DIMENSION_WEIGHTS)
