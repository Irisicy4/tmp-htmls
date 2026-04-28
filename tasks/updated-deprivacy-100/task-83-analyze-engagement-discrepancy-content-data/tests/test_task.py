"""LLM-as-judge evaluator for task-83-analyze-engagement-discrepancy-content-data.

Category: Marketing & Analytics
Task: Analyze the engagement discrepancy in this content performance data. Diagnose breakouts, failures, and baselines, then extract replicable formulas for future newsletters.
"""

import sys
sys.path.insert(0, "/harness")
from evaluator import run_judge

TASK_INSTRUCTION = 'Analyze the engagement discrepancy in this content performance data. Diagnose breakouts, failures, and baselines, then extract replicable formulas for future newsletters.'

SYSTEM_PROMPT_EXTRA = 'Your job is to assess whether an AI agent successfully completed a browser-based task.\n\nThe task involves analyzing newsletter or content performance data to identify what drove high-performing vs low-performing pieces, and extracting actionable patterns.'

CONSTRAINTS = """- Input: content performance data (provided by user — open rates, click rates, engagement metrics)
- Analysis: must distinguish breakouts (top performers), failures (underperformers), and baseline
- Output: replicable formulas — specific patterns that can be replicated
- Depth: must go beyond averages to identify specific causal factors"""

EVIDENCE_QUESTIONS = """- Did the agent analyze actual performance data?
- Were breakouts, failures, and baselines clearly identified?
- What specific patterns were found for each category?
- Are replicable formulas extracted with clear rationale?
- Is the output actionable for future newsletter creation?"""

DIMENSION_RUBRICS = """#### A. Data Analysis (0.3)
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
1 — Not actionable."""

DIMENSION_WEIGHTS = {
    'data_analysis': 0.3,
    'pattern_identification': 0.35,
    'formula_extraction': 0.25,
    'actionability': 0.1,
}


def test(result):
    return run_judge(
        result,
        task_instruction=TASK_INSTRUCTION,
        system_prompt_extra=SYSTEM_PROMPT_EXTRA,
        constraints=CONSTRAINTS,
        evidence_questions=EVIDENCE_QUESTIONS,
        dimension_rubrics=DIMENSION_RUBRICS,
        dimension_weights=DIMENSION_WEIGHTS,
    )
