import sys
sys.path.insert(0, "/harness")
from evaluator import evaluate

TASK_INSTRUCTION = ("""Please design a comprehensive learning plan for Web 3.0. Do not take any action yet—first list the steps you will use to help me create the plan.""")

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- The agent must NOT execute any actions yet — the response must be a list of planned steps only
- Topic domain: Web 3.0 (blockchain, decentralised apps, DeFi, smart contracts, NFTs, etc.)
- Output: a structured, numbered list of the steps the agent will follow to build the learning plan
- The learning plan itself must NOT be created yet — only the meta-steps for creating it

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
Carefully read the agent response and trace, then answer each question:

- Did the agent list steps rather than execute them?
- How many steps were listed?
- Are the steps specific to designing a Web 3.0 learning plan (not generic)?
- Are the steps logically ordered (e.g. define scope → research → structure → curate resources → assess)?
- Did the agent take any premature actions (if so, this violates the constraint)?
- Was any document created or content saved beyond the step list itself?

### Step 2: Dimension Scoring
Score each dimension 1–5 using the rubrics below.

#### A. Task Completion (0.40)
Did the agent list steps for creating a Web 3.0 learning plan without taking premature action?

5 — Clear, numbered list of steps presented; no premature actions taken; steps are specific to Web 3.0.
4 — Steps listed but one minor action was taken, or steps are slightly generic.
3 — Steps partially listed or mixed with actual plan content; constraint partially violated.
2 — Agent began executing the plan instead of listing steps, or only vague steps provided.
1 — No steps listed, or task misunderstood entirely.

#### B. Information Quality (0.30)
Are the planned steps relevant, accurate, and well-conceived for a Web 3.0 learning plan?

5 — Steps reflect genuine Web 3.0 curriculum design, covering key topic areas (blockchain, DeFi, smart contracts, NFTs, dApps) with a logical pedagogical structure.
4 — Mostly relevant steps; minor gaps in Web 3.0 topic coverage.
3 — Steps are generic learning-plan steps not specifically tailored to Web 3.0.
2 — Steps show limited understanding of Web 3.0 or learning plan design.
1 — Steps are irrelevant or inaccurate.

#### C. Document Saved (0.20)
Were the steps presented in a clear, structured, and retrievable format?

5 — Steps clearly written in a structured format (numbered list, outline, or document).
4 — Steps present but formatting is inconsistent or hard to follow.
3 — Steps mentioned but not clearly laid out.
2 — Steps implied but not explicitly listed.
1 — No discernible step list presented.

#### D. Completeness (0.10)
Did the agent address the full scope of what was asked?

5 — Comprehensive set of steps covering the full lifecycle of creating a Web 3.0 learning plan (scoping, research, structuring, resource curation, assessment design).
4 — Most phases covered; one step or phase missing.
3 — Steps cover only part of the process.
2 — Very few steps; significant scope missing.
1 — Task not meaningfully addressed.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising your Step 1 findings>",
  "task_completion": <1-5>,
  "information_quality": <1-5>,
  "document_saved": <1-5>,
  "completeness": <1-5>,
  "dimension_reasoning": {{
    "task_completion": "<one sentence citing specific evidence>",
    "information_quality": "<one sentence citing specific evidence>",
    "document_saved": "<one sentence citing specific evidence>",
    "completeness": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "task_completion":    0.40,
    "information_quality": 0.30,
    "document_saved":     0.20,
    "completeness":       0.10,
}


def test(result):
    return evaluate(result, TASK_INSTRUCTION, USER_PROMPT_TEMPLATE, DIMENSION_WEIGHTS)
