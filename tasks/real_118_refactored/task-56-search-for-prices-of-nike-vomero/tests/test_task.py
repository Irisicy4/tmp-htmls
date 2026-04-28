import sys
sys.path.insert(0, "/harness")
from evaluator import evaluate

TASK_INSTRUCTION = ("""Search for prices of the Nike Vomero Plus sneaker in Brazil and compare prices across stores to find the best deal.""")

USER_PROMPT_TEMPLATE = ("""## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Product: Nike Vomero Plus specifically (not Air Vomero or other models)
- Region: Brazil — prices should be in BRL
- Multi-store comparison: at least 3 stores should be compared
- Best deal: agent must identify which store has the lowest price

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent search for Nike Vomero Plus in Brazil?
- How many stores were compared?
- What prices were found and in what currency?
- Which store was identified as having the best deal?
- Were prices from Google Shopping or direct retailer pages?

### Step 2: Dimension Scoring

#### A. Search Execution (0.35)
Did the agent search for Nike Vomero Plus in Brazil?

5 — Agent used Google Shopping Brazil or visited Brazilian retailer sites directly.
4 — Agent searched but used general web search rather than shopping platforms.
3 — Agent found some prices but search was not Brazil-specific.
2 — Agent described what to search without actually searching.
1 — No search performed.

#### B. Store Coverage (0.2)
How many stores were compared?

5 — 4 or more Brazilian stores compared with specific prices.
4 — 3 stores compared.
3 — 2 stores compared.
2 — Only 1 store found.
1 — No stores compared.

#### C. Price Accuracy (0.28)
Are prices accurate and in BRL?

5 — Prices in BRL from credible Brazilian retailers (Netshoes, Nike.com.br, Centauro, etc.) with specific amounts.
4 — Prices in BRL but from less authoritative sources.
3 — Prices found but currency or accuracy uncertain.
2 — Prices found but in USD or without clear source.
1 — No actual prices found.

#### D. Recommendation Quality (0.17)
Did the agent identify the best deal clearly?

5 — Clear winner identified with price, store name, and any relevant conditions (shipping, installments).
4 — Best deal identified but missing one detail.
3 — Best deal mentioned vaguely.
2 — Prices listed without identifying best deal.
1 — No recommendation.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "dimension_reasoning": {{
    "search_execution": "<one sentence citing specific evidence>",
    "store_coverage": "<one sentence citing specific evidence>",
    "price_accuracy": "<one sentence citing specific evidence>",
    "recommendation_quality": "<one sentence citing specific evidence>",
  }},
  "search_execution": <1-5>,
  "store_coverage": <1-5>,
  "price_accuracy": <1-5>,
  "recommendation_quality": <1-5>,
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
""")

DIMENSION_WEIGHTS = {
    "search_execution": 0.35,
    "store_coverage": 0.2,
    "price_accuracy": 0.28,
    "recommendation_quality": 0.17,
}


def test(result):
    return evaluate(result, TASK_INSTRUCTION, USER_PROMPT_TEMPLATE, DIMENSION_WEIGHTS)
