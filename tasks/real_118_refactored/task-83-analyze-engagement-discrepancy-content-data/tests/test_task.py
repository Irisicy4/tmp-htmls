import sys
sys.path.insert(0, "/harness")
from evaluator import evaluate

TASK_INSTRUCTION = ("""Analyze the engagement discrepancy in this content performance data. Diagnose breakouts, failures, and baselines, then extract replicable formulas for future newsletters.""")

USER_PROMPT_TEMPLATE = ("""## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Input: content performance data (provided by user — open rates, click rates, engagement metrics)
- Analysis: must distinguish breakouts (top performers), failures (underperformers), and baseline
- Output: replicable formulas — specific patterns that can be replicated
- Depth: must go beyond averages to identify specific causal factors

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent analyze actual performance data?
- Were breakouts, failures, and baselines clearly identified?
- What specific patterns were found for each category?
- Are replicable formulas extracted with clear rationale?
- Is the output actionable for future newsletter creation?

### Step 2: Dimension Scoring

#### A. Data Analysis (0.3)
Did the agent analyze the actual performance data?

5 — Specific metrics analyzed: open rates, CTR, engagement; compared across content pieces with statistical framing.
4 — Good analysis but less precise numerically.
3 — General trends identified without piece-by-piece analysis.
2 — Described what to look for without analyzing actual data.
1 — No data analysis.

#### B. Pattern Identification (0.35)
Were breakouts, failures, and baselines clearly identified?

5 — Specific pieces categorized as breakout/failure/baseline with metrics to support each classification.
4 — Two of three categories clearly identified.
3 — One category clearly identified.
2 — Vague categorization without metric support.
1 — No categorization.

#### C. Formula Extraction (0.25)
Were replicable formulas extracted?

5 — Specific formulas stated: e.g. 'subject lines with numbers + urgency words outperform by X%'; actionable and testable.
4 — Good formulas but less specific or testable.
3 — Patterns identified but not stated as formulas.
2 — Generic best practices not grounded in the data.
1 — No formulas extracted.

#### D. Actionability (0.1)
Is the output directly usable for future newsletters?

5 — Formulas are ready to apply with clear next steps.
4 — Mostly actionable with minor gaps.
3 — Insights present but require interpretation to apply.
2 — Too abstract to directly apply.
1 — Not actionable.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "data_analysis": <1-5>,
  "pattern_identification": <1-5>,
  "formula_extraction": <1-5>,
  "actionability": <1-5>,
  "dimension_reasoning": {{
    "data_analysis": "<one sentence citing specific evidence>",
    "pattern_identification": "<one sentence citing specific evidence>",
    "formula_extraction": "<one sentence citing specific evidence>",
    "actionability": "<one sentence citing specific evidence>",
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
""")

DIMENSION_WEIGHTS = {
    "data_analysis": 0.3,
    "pattern_identification": 0.35,
    "formula_extraction": 0.25,
    "actionability": 0.1,
}


def test(result):
    return evaluate(result, TASK_INSTRUCTION, USER_PROMPT_TEMPLATE, DIMENSION_WEIGHTS)
