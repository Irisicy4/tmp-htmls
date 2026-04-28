import sys
sys.path.insert(0, "/harness")
from evaluator import evaluate

TASK_INSTRUCTION = ("""Where in the world are Salomon prices the lowest and where is it most likely to find clearance stock?""")

USER_PROMPT_TEMPLATE = ("""## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Brand: Salomon specifically
- Coverage: global comparison — multiple countries/regions
- Clearance: must identify where clearance stock is most likely (seasonal, outlet stores, end-of-line)
- Output: ranked list of regions with price reasoning

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent research Salomon prices across multiple countries?
- What specific regions/countries were identified as cheapest?
- What explains the price differences (currency, VAT, market positioning)?
- Where is clearance stock most likely to be found and why?

### Step 2: Dimension Scoring

#### A. Geographic Coverage (0.25)
Were multiple global regions compared?

5 — 5+ regions compared (e.g. Japan, US, EU, China, Outlet regions) with specific price data.
4 — 3-4 regions compared.
3 — 2 regions compared.
2 — Only one region analyzed.
1 — No geographic comparison.

#### B. Price Analysis (0.3)
Are price differences explained with evidence?

5 — Specific prices in local currency with reasoning (VAT, import duty, regional positioning, currency strength).
4 — Prices found with partial explanation.
3 — Regions identified as cheaper but without clear price data.
2 — Vague 'Japan is cheaper' without evidence.
1 — No price analysis.

#### C. Clearance Insights (0.3)
Were clearance opportunities identified?

5 — Specific clearance channels identified: outlet stores, end-of-season markets, specific retailers known for Salomon clearance.
4 — Good clearance insights but less specific.
3 — General 'check outlets' advice without specifics.
2 — Clearance mentioned without insight.
1 — No clearance analysis.

#### D. Actionability (0.15)
Is the output actionable for a buyer?

5 — Clear ranked list with specific stores, websites, or regions to target for best prices.
4 — Mostly actionable.
3 — Informative but requires further research to act.
2 — Too vague to act on.
1 — Not actionable.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "dimension_reasoning": {{
    "geographic_coverage": "<one sentence citing specific evidence>",
    "price_analysis": "<one sentence citing specific evidence>",
    "clearance_insights": "<one sentence citing specific evidence>",
    "actionability": "<one sentence citing specific evidence>",
  }},
  "geographic_coverage": <1-5>,
  "price_analysis": <1-5>,
  "clearance_insights": <1-5>,
  "actionability": <1-5>,
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
""")

DIMENSION_WEIGHTS = {
    "geographic_coverage": 0.25,
    "price_analysis": 0.3,
    "clearance_insights": 0.3,
    "actionability": 0.15,
}


def test(result):
    return evaluate(result, TASK_INSTRUCTION, USER_PROMPT_TEMPLATE, DIMENSION_WEIGHTS)
