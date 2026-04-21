import sys
sys.path.insert(0, "/harness")
from evaluator import evaluate

TASK_INSTRUCTION = ("Extract data and plot the UV (Unique Visitors) data for the past 5 years.")

USER_PROMPT_TEMPLATE = ("""## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Data: UV (Unique Visitors) or equivalent web traffic metric — must span approximately 5 years
- Source: data must come from a credible source (analytics platform, public report, or authoritative website)
- Output: a chart/plot must be produced (HTML/JS chart, image file, or equivalent visualization)
- Coverage: data points should cover at least 4 of the 5 most recent years

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent search for and find UV/traffic data? From what source?
- What time range does the data cover? Is it approximately 5 years?
- Was a chart or visualization produced? What type?
- Are the data points specific (actual numbers) or vague estimates?
- Was a file saved or an interactive chart generated?

### Step 2: Dimension Scoring

#### A. Data Source Quality
Did the agent find credible UV data?

5 — Data sourced from a credible analytics platform, official report, or authoritative site with specific yearly figures.
4 — Data sourced from a credible source but some values are estimated or interpolated.
3 — Data found from a secondary source (blog, article citing stats); numbers present but sourcing is weak.
2 — Agent described what UV data looks like without finding actual numbers.
1 — No data found or data is clearly fabricated.

#### B. Temporal Coverage
Does the data span approximately 5 years?

5 — Data covers 5 years with data points for each year (or equivalent granularity).
4 — Data covers 4 years or has one gap year.
3 — Data covers 3 years.
2 — Data covers fewer than 3 years or only shows a trend without specific years.
1 — No temporal data provided.

#### C. Visualization Quality
Was a chart produced and is it clear?

5 — Clear chart produced (HTML/JS interactive chart or image) with labeled axes, title, and data points.
4 — Chart produced but missing labels or title.
3 — Chart attempted but poorly formatted or hard to read.
2 — Data presented as a table or list instead of a chart.
1 — No visualization produced.

#### D. Report Completeness
Did the agent deliver a complete, useful output?

5 — Full report with data source, extracted numbers, and chart — saved as file or clearly presented.
4 — Report present but one element missing (e.g. no source citation or no saved file).
3 — Partial output — either data or chart present but not both.
2 — Only a description of the approach without actual data or chart.
1 — No useful output.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "data_source_quality": <1-5>,
  "temporal_coverage": <1-5>,
  "visualization_quality": <1-5>,
  "report_completeness": <1-5>,
  "dimension_reasoning": {{
    "data_source_quality": "<one sentence citing specific evidence>",
    "temporal_coverage": "<one sentence citing specific evidence>",
    "visualization_quality": "<one sentence citing specific evidence>",
    "report_completeness": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
""")

DIMENSION_WEIGHTS = {
    "data_source_quality":   0.25,
    "temporal_coverage":     0.25,
    "visualization_quality": 0.30,
    "report_completeness":   0.20,
}


def test(result):
    return evaluate(result, TASK_INSTRUCTION, USER_PROMPT_TEMPLATE, DIMENSION_WEIGHTS)
