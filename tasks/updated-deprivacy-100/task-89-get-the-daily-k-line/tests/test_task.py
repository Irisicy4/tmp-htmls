"""LLM-as-judge evaluator for task-89-get-the-daily-k-line.

Category: Finance & Economics
Task: Get the daily K-line chart for CATL from East Money (eastmoney.com).
"""

import sys
sys.path.insert(0, "/harness")
from evaluator import run_judge

TASK_INSTRUCTION = 'Get the daily K-line chart for CATL from East Money (eastmoney.com).'

SYSTEM_PROMPT_EXTRA = 'Your job is to assess whether an AI agent successfully completed a browser-based task.\n\nThe task involves navigating eastmoney.com (a major Chinese financial platform) to find the daily candlestick (K-line) chart for CATL (宁德时代 (CATL), stock code 300750).'

CONSTRAINTS = """- Platform: eastmoney.com specifically
- Stock: CATL (宁德时代 (CATL))
- Chart type: daily K-line (candlestick) chart
- Output: chart retrieved and key data points reported (current price, recent highs/lows)"""

EVIDENCE_QUESTIONS = """- Did the agent navigate to eastmoney.com?
- Was CATL found by name or stock code?
- Was the daily K-line chart accessed?
- What chart data was reported (price, date range, highs/lows)?
- Was a screenshot or description of the chart provided?"""

DIMENSION_RUBRICS = """#### A. Platform Navigation (0.25)
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
1 — No data reported."""

DIMENSION_WEIGHTS = {
    'platform_navigation': 0.25,
    'stock_identification': 0.2,
    'chart_retrieval': 0.35,
    'data_reporting': 0.2,
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
