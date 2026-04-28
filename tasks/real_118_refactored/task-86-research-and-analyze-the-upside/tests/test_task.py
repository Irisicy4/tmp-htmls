import sys
sys.path.insert(0, "/harness")
from evaluator import evaluate

TASK_INSTRUCTION = ("""Research and analyze the upside potential for TSMC and Samsung Electronics stock prices over the next 30 days based on publicly available analyst reports, recent earnings data, and news. Infer a peak price for each.""")

USER_PROMPT_TEMPLATE = ("""## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Stocks: TSMC and Samsung Electronics — both required
- Horizon: over the next 30 days (forward-looking)
- Analysis: must be based on publicly available analyst reports, earnings data, and news
- Output: peak price inference with reasoning for each stock

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent research publicly available materials (analyst reports, earnings, news)?
- Were both stocks analyzed individually?
- What price targets or peak prices were inferred?
- Is the analysis grounded in fundamentals, technicals, or both?
- Is the 30-day forward-looking window addressed?

### Step 2: Dimension Scoring

#### A. Material Analysis (0.25)
Did the agent analyze publicly available financial materials?

5 — Agent researched and referenced specific data from analyst reports, earnings releases, or financial news for the global semiconductor market.
4 — Materials referenced but analysis is shallow.
3 — General financial analysis without clear reference to sources.
2 — Agent described the task without analyzing materials.
1 — No material analysis.

#### B. Stock Coverage (0.2)
Were both stocks analyzed?

5 — Both TSMC and Samsung Electronics analyzed individually with separate assessments.
4 — Both covered but one more superficially.
3 — Only one stock analyzed in depth.
2 — Both mentioned but neither analyzed.
1 — Only one stock or neither.

#### C. Price Inference (0.35)
Were peak price levels inferred with reasoning?

5 — Specific peak price range stated for each stock over the next 30 days, with technical (resistance levels, momentum) or fundamental (valuation, catalyst) rationale.
4 — Price range inferred with partial rationale.
3 — Upside direction identified without specific price target.
2 — Vague 'could go up' without quantification.
1 — No price inference.

#### D. Analytical Rigor (0.2)
Is the analysis methodologically sound?

5 — Clear methodology: technical analysis, fundamental valuation, catalyst identification, risk factors.
4 — Good methodology but one element weak.
3 — Basic analysis with limited rigor.
2 — Opinion-based without analytical framework.
1 — No analytical rigor.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "dimension_reasoning": {{
    "material_analysis": "<one sentence citing specific evidence>",
    "stock_coverage": "<one sentence citing specific evidence>",
    "price_inference": "<one sentence citing specific evidence>",
    "analytical_rigor": "<one sentence citing specific evidence>",
  }},
  "material_analysis": <1-5>,
  "stock_coverage": <1-5>,
  "price_inference": <1-5>,
  "analytical_rigor": <1-5>,
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
""")

DIMENSION_WEIGHTS = {
    "material_analysis": 0.25,
    "stock_coverage": 0.2,
    "price_inference": 0.35,
    "analytical_rigor": 0.2,
}


def test(result):
    return evaluate(result, TASK_INSTRUCTION, USER_PROMPT_TEMPLATE, DIMENSION_WEIGHTS)
