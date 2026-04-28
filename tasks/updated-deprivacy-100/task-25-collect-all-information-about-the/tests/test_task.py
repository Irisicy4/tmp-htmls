"""LLM-as-judge evaluator for task-25-collect-all-information-about-the.

Category: Finance & Economics
Task: Collect all information about the US stock market from 2025-12-15 to 2025-12-16. Explain why the market was largely down (mostly in the red). Using all collected information, think deeply and give you
"""

import sys
sys.path.insert(0, "/harness")
from evaluator import run_judge

TASK_INSTRUCTION = 'Collect all information about the US stock market from 2025-12-15 to 2025-12-16. Explain why the market was largely down (mostly in the red). Using all collected information, think deeply and give your own conclusion and reasoning for your prediction of the market trend for 2025-12-16 (5 hours before market open).'

SYSTEM_PROMPT_EXTRA = 'You are assessing a financial research and prediction task covering specific dates (Dec 15-16, 2025).\n\nImportant: The task asks for a prediction "5 hours before market open on Dec 16" — this means the agent was operating in a pre-market context. Evaluate the quality of the reasoning and data collection, not whether the prediction turned out to be correct.'

CONSTRAINTS = """- Date range: December 15–16, 2025 specifically
- Three required components: (1) data collection, (2) explanation of why market was down, (3) prediction for Dec 16
- Prediction context: 5 hours before Dec 16 market open (pre-market reasoning)
- Sources: must use credible financial sources (not just prior knowledge)
- Note: task assumes market was "largely down" on these dates — agent should find evidence for this"""

EVIDENCE_QUESTIONS = """- Did the agent search for Dec 15-16 2025 market data? What sources were used?
- What specific market data was collected (indices, % changes, volumes)?
- Did the agent identify reasons for the market being down?
- Did the agent make a prediction for Dec 16 with reasoning?
- Is the reasoning analytical and grounded in the collected data?"""

DIMENSION_RUBRICS = """#### A. Data Collection
Did the agent collect specific, dated market data for Dec 15-16 2025?

5 — Specific data collected: index levels or % changes for Dec 15 (S&P 500, NASDAQ, or Dow); sources cited; data is from Dec 15-16 2025 specifically.
4 — Data collected but less specific (e.g. general market direction without exact figures) or one date missing.
3 — Market data mentioned but vague or not clearly from the specified dates.
2 — Generic market commentary without specific Dec 15-16 data.
1 — No market data collected; response from prior knowledge or refused.

#### B. Explanation Quality
Does the agent provide a credible explanation for why the market was down?

5 — 2+ specific causal factors identified (e.g. Fed decision, economic data release, geopolitical event) with evidence linking them to the Dec 15 selloff.
4 — 1–2 factors identified with some evidence; reasoning is plausible.
3 — Explanation is directionally plausible but vague (e.g. "interest rate concerns" without specifics).
2 — Explanation is generic (e.g. "markets go up and down") without specific factors.
1 — No explanation provided or explanation is clearly wrong.

#### C. Prediction & Reasoning
Does the agent make a specific Dec 16 prediction with analytical reasoning?

5 — Clear directional prediction (up/down/flat) with 2+ reasoned factors drawn from collected data; reasoning is coherent and data-driven.
4 — Prediction made with 1–2 supporting reasons; reasoning present but not deeply analytical.
3 — Prediction made but reasoning is thin or generic (e.g. "likely to continue downward trend" without justification).
2 — Prediction mentioned without any reasoning.
1 — No prediction made.

#### D. Analytical Depth
Does the response demonstrate genuine analytical thinking beyond surface-level summary?

5 — Agent synthesises multiple data points, identifies market sentiment, considers catalysts, and provides nuanced pre-market analysis.
4 — Good analysis with some synthesis; one dimension (e.g. sentiment or catalysts) underexplored.
3 — Adequate summary of facts with basic analysis; limited synthesis.
2 — Mostly a data dump without analytical commentary.
1 — No analysis; pure narrative or refusal."""

DIMENSION_WEIGHTS = {
    'data_collection': 0.25,
    'explanation_quality': 0.3,
    'prediction_reasoning': 0.3,
    'analytical_depth': 0.15,
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
