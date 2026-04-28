"""LLM-as-judge evaluator for task-50-pork-bone-broth-price-changed-usage.

Category: Finance & Economics
Task: The price we receive from the manufacturer for the following product has changed. The usage amount has also changed, so I need a persuasive explanation of whether the actual unit cost for our store ha
"""

import sys
sys.path.insert(0, "/harness")
from evaluator import run_judge

TASK_INSTRUCTION = 'The price we receive from the manufacturer for the following product has changed. The usage amount has also changed, so I need a persuasive explanation of whether the actual unit cost for our store has gone up or down.\n  \n  Product name: Pork bone broth (domestic)\n  Weight: 3kg\n  Original price (VAT excluded): 3,700 won/3kg, mixed with water at 1:1 ratio (2x dilution)\n  New price (VAT excluded): 5,600 won/3kg, mixed with water at 3:1 ratio (3x dilution)'

SYSTEM_PROMPT_EXTRA = 'Assess whether an AI agent correctly computed the actual unit cost impact of a price change combined with a dilution ratio change, and presented a persuasive argument.'

CONSTRAINTS = """- Math must be correct: original cost per unit = 3700/2 = 1850 won/kg-equivalent; new cost per unit = 5600/3 ≈ 1867 won/kg-equivalent — the actual unit cost increased slightly
- The agent must compute and present the per-unit cost, not just compare raw prices
- The argument must be persuasive and logically structured
- The conclusion must be mathematically accurate"""

EVIDENCE_QUESTIONS = """- Did the agent compute the actual per-unit cost for both old and new pricing?
- Is the math correct (1850 vs 1867 won per unit)?
- Did the agent present a persuasive argument?
- Is the conclusion accurate (slight cost increase)?"""

DIMENSION_RUBRICS = """#### A. Mathematical Accuracy (0.4)
Is the per-unit cost calculation correct?

5 — Correctly computes: old = 3700÷2 = 1850 won/unit, new = 5600÷3 ≈ 1867 won/unit; concludes cost increased.
4 — Correct approach but minor rounding error or slightly different framing.
3 — Partially correct — identifies the dilution ratio matters but makes a computational error.
2 — Compares raw prices (3700 vs 5600) without accounting for dilution ratio.
1 — No calculation or clearly wrong answer.

#### B. Persuasive Logic (0.3)
Is the argument well-structured and convincing?

5 — Clear logical flow: states the problem, shows calculation, explains implication, draws conclusion.
4 — Mostly persuasive but one step is weak or unclear.
3 — Argument present but logic gaps make it less convincing.
2 — Descriptive rather than persuasive.
1 — No argument structure.

#### C. Practical Framing (0.2)
Is the output framed in a way useful for a store owner?

5 — Uses business language, explains impact per serving/use, gives actionable insight.
4 — Good framing but slightly too technical or academic.
3 — Correct info but not framed for a store owner audience.
2 — Very generic framing.
1 — No practical framing.

#### D. Output Clarity (0.1)
Is the response clear and easy to understand?

5 — Well-organized, concise, easy to read.
4 — Clear but slightly verbose.
3 — Understandable but requires effort.
2 — Confusing or poorly organized.
1 — Incomprehensible."""

DIMENSION_WEIGHTS = {
    'mathematical_accuracy': 0.4,
    'persuasive_logic': 0.3,
    'practical_framing': 0.2,
    'output_clarity': 0.1,
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
