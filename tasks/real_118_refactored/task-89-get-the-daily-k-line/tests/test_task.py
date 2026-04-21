import sys
sys.path.insert(0, "/harness")
from evaluator import evaluate

TASK_INSTRUCTION = ("""Get the daily K-line chart for CATL from East Money (eastmoney.com).""")

USER_PROMPT_TEMPLATE = ("""## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Platform: eastmoney.com specifically
- Stock: CATL (宁德时代 (CATL))
- Chart type: daily K-line (candlestick) chart
- Output: chart retrieved and key data points reported (current price, recent highs/lows)

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to eastmoney.com?
- Was CATL found by name or stock code?
- Was the daily K-line chart accessed?
- What chart data was reported (price, date range, highs/lows)?
- Was a screenshot or description of the chart provided?

### Step 2: Dimension Scoring

#### A. Platform Navigation (0.25)
Did the agent navigate to eastmoney.com?

5 — Agent navigated to eastmoney.com and accessed the stock section.
4 — Agent reached eastmoney.com but had navigation difficulty.
3 — Agent found the stock via a different Chinese financial platform.
2 — Agent described eastmoney.com without navigating.
1 — No navigation.

#### B. Stock Identification (0.2)
Was the correct stock found?

5 — CATL (宁德时代 (CATL)) found with correct stock code and exchange.
4 — Stock found but code or exchange detail unclear.
3 — Similar company found but not the exact stock.
2 — Wrong stock accessed.
1 — Stock not found.

#### C. Chart Retrieval (0.35)
Was the daily K-line chart accessed?

5 — Daily K-line chart accessed; OHLC data for recent days reported.
4 — Chart accessed but only closing price reported.
3 — Stock page accessed but chart not specifically retrieved.
2 — General stock info without chart data.
1 — No chart retrieved.

#### D. Data Reporting (0.2)
Was chart data clearly reported?

5 — Recent OHLC prices, date range, 52-week high/low, and trend description.
4 — Price and trend reported; some data missing.
3 — Only current price reported.
2 — Vague description without specific numbers.
1 — No data reported.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "platform_navigation": <1-5>,
  "stock_identification": <1-5>,
  "chart_retrieval": <1-5>,
  "data_reporting": <1-5>,
  "dimension_reasoning": {{
    "platform_navigation": "<one sentence citing specific evidence>",
    "stock_identification": "<one sentence citing specific evidence>",
    "chart_retrieval": "<one sentence citing specific evidence>",
    "data_reporting": "<one sentence citing specific evidence>",
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
""")

DIMENSION_WEIGHTS = {
    "platform_navigation": 0.25,
    "stock_identification": 0.2,
    "chart_retrieval": 0.35,
    "data_reporting": 0.2,
}


def test(result):
    return evaluate(result, TASK_INSTRUCTION, USER_PROMPT_TEMPLATE, DIMENSION_WEIGHTS)
