import sys
sys.path.insert(0, "/harness")
from evaluator import evaluate

TASK_INSTRUCTION = ("Please design a comprehensive learning plan for Web 3.0. Do not take any action yet—first list the steps you will use to help me create the plan.")

USER_PROMPT_TEMPLATE = ("""## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Critical: agent must list steps/approach ONLY — not produce the actual learning plan
- Producing the full plan when asked to only list steps is a violation of the task constraint
- Steps should be specific to Web 3.0 (not generic learning plan steps)
- Steps should be logical, ordered, and comprehensive enough to lead to a good learning plan

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis
- Did the agent list steps/approach rather than producing the full plan?
- How many steps were listed? Are they specific to Web 3.0?
- Did the agent accidentally produce a full learning plan instead of just steps?
- Are the steps logically ordered and comprehensive?

### Step 2: Dimension Scoring

#### A. Constraint Adherence
Did the agent correctly list steps WITHOUT producing the full plan?

5 — Agent listed steps only; explicitly stated it will not act yet; no full plan produced.
4 — Agent listed steps; briefly elaborated on one or two but did not produce a full plan.
3 — Agent listed steps but also began producing partial plan content (borderline violation).
2 — Agent produced a partial or full learning plan instead of listing steps.
1 — Agent ignored the constraint entirely and produced a full learning plan.

#### B. Step Specificity
Are the steps specific to Web 3.0 rather than generic learning plan steps?

5 — Steps reference Web 3.0-specific topics (blockchain, smart contracts, DeFi, NFTs, DAOs, wallets, Solidity, etc.) and how they would be covered.
4 — Most steps are Web 3.0-specific; 1–2 are generic (e.g. "assess current knowledge").
3 — Steps are relevant to technology learning broadly but Web 3.0-specific content is mentioned vaguely.
2 — Steps are completely generic (could apply to learning any technology).
1 — No steps listed or steps are irrelevant.

#### C. Step Completeness & Logic
Are the steps comprehensive and logically ordered?

5 — 5+ distinct steps covering: assessment, topic sequencing, resource identification, practice, and review; logical progression.
4 — 4–5 steps with clear logic; one phase (e.g. practice or review) missing.
3 — 3–4 steps; logical but missing key phases.
2 — 1–2 steps; far too sparse to constitute a plan for creating a plan.
1 — No structured steps.

#### D. Response Clarity
Is the step list clear, well-formatted, and easy to follow?

5 — Numbered or bulleted list; each step has a clear title and brief description.
4 — List format with step titles; descriptions are minimal but understandable.
3 — Steps present but loosely formatted (e.g. paragraph form).
2 — Steps are buried in prose; hard to extract.
1 — No structured output.

### Step 3: Output
<Answer>
{{
  "evidence_summary": "<2-3 sentences>",
  "constraint_adherence": <1-5>,
  "step_specificity": <1-5>,
  "step_completeness": <1-5>,
  "response_clarity": <1-5>,
  "dimension_reasoning": {{"constraint_adherence": "<one sentence>", "step_specificity": "<one sentence>", "step_completeness": "<one sentence>", "response_clarity": "<one sentence>"}},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false>
}}
</Answer>
""")

DIMENSION_WEIGHTS = {"constraint_adherence": 0.40, "step_specificity": 0.25, "step_completeness": 0.20, "response_clarity": 0.15}


def test(result):
    return evaluate(result, TASK_INSTRUCTION, USER_PROMPT_TEMPLATE, DIMENSION_WEIGHTS)
