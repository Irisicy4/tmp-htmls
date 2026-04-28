import sys
sys.path.insert(0, "/harness")
from evaluator import evaluate

TASK_INSTRUCTION = ("""Research the cheapest shops where you can buy the iPhone 15 Plus and iPhone 16 Plus under a 2-year return program.""")

USER_PROMPT_TEMPLATE = ("""## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Products: iPhone 15 Plus AND iPhone 16 Plus — both required
- Program: 2-year return/残価設定型 program (不要になったら返却できるプログラム)
- Market: Japan (based on language context)
- Comparison: multiple shops/carriers compared for each device

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent search for both iPhone models under 2-year return programs?
- Which shops/carriers were checked (docomo, au, SoftBank, Apple, etc.)?
- What are the monthly or total prices for each model?
- Which shop was identified as cheapest for each model?

### Step 2: Dimension Scoring

#### A. Product Coverage (0.2)
Were both iPhone 15 Plus and 16 Plus covered?

5 — Both models researched with prices under 2-year return program.
4 — Both models covered but one less thoroughly.
3 — Only one model researched.
2 — Prices found but not specifically for 2-year return program.
1 — No product research.

#### B. Shop Comparison (0.3)
Were multiple shops compared?

5 — 4+ shops/carriers compared for each model (docomo, au, SoftBank, Apple, etc.).
4 — 3 shops compared.
3 — 2 shops compared.
2 — Only 1 shop.
1 — No comparison.

#### C. Price Accuracy (0.35)
Are prices accurate and program-specific?

5 — Monthly installment and total program cost clearly stated; 2-year return program confirmed at each carrier.
4 — Prices present but program type not always confirmed.
3 — Approximate prices without program-specific detail.
2 — General iPhone prices without return program context.
1 — No reliable prices.

#### D. Best Deal (0.15)
Was the cheapest option identified?

5 — Clear winner for each model with total cost breakdown.
4 — Best deal identified but without full cost breakdown.
3 — Best deal suggested without clear comparison basis.
2 — Prices listed without identifying best deal.
1 — No recommendation.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "dimension_reasoning": {{
    "product_coverage": "<one sentence citing specific evidence>",
    "shop_comparison": "<one sentence citing specific evidence>",
    "price_accuracy": "<one sentence citing specific evidence>",
    "best_deal": "<one sentence citing specific evidence>",
  }},
  "product_coverage": <1-5>,
  "shop_comparison": <1-5>,
  "price_accuracy": <1-5>,
  "best_deal": <1-5>,
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
""")

DIMENSION_WEIGHTS = {
    "product_coverage": 0.2,
    "shop_comparison": 0.3,
    "price_accuracy": 0.35,
    "best_deal": 0.15,
}


def test(result):
    return evaluate(result, TASK_INSTRUCTION, USER_PROMPT_TEMPLATE, DIMENSION_WEIGHTS)
