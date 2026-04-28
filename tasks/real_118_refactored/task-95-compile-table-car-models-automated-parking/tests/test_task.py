import sys
sys.path.insert(0, "/harness")
from evaluator import evaluate

TASK_INSTRUCTION = ("""Please help me compile a table of car models that currently support automated parking into mechanical or multi-level parking spaces via parking assist. Include any height restrictions for the parking spaces they support, as well as the vehicle's own height dimensions. Cite your sources.""")

USER_PROMPT_TEMPLATE = ("""## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Feature: automated parking assist compatible with mechanical/multi-level parking structures
- Data: both vehicle height AND parking space height restriction required
- Coverage: multiple car brands/models
- Output: table format with sources cited

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent search for cars with mechanical parking space compatibility?
- How many car models were found?
- Are vehicle heights and parking space height limits both included?
- Are sources cited?
- Is output in table format?

### Step 2: Dimension Scoring

#### A. Research Quality (0.25)
Did the agent find relevant car models?

5 — Multiple car models found with specific parking assist feature names (e.g. Toyota Advanced Park, BMW Parking Assistant Plus).
4 — Several models found but feature details thin.
3 — Some models identified without feature specifics.
2 — Very limited findings.
1 — No research.

#### B. Height Data (0.35)
Are both vehicle and parking height data included?

5 — Vehicle height AND supported parking space max height both present for each model.
4 — Vehicle height present; parking space limit for most models.
3 — One height dimension present for most models.
2 — Heights mentioned without specific values.
1 — No height data.

#### C. Table Format (0.25)
Is output in table format?

5 — Clear table: car model, brand, vehicle height, max parking height supported, source.
4 — Table present but missing one column.
3 — List format instead of table.
2 — Narrative without table.
1 — No structured output.

#### D. Source Citation (0.15)
Are sources cited for each entry?

5 — Each model's data attributed to a specific source (manufacturer spec page, review, news).
4 — Most entries sourced.
3 — Some sources cited.
2 — General sources without per-model attribution.
1 — No sources.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "dimension_reasoning": {{
    "research_quality": "<one sentence citing specific evidence>",
    "height_data": "<one sentence citing specific evidence>",
    "table_format": "<one sentence citing specific evidence>",
    "source_citation": "<one sentence citing specific evidence>",
  }},
  "research_quality": <1-5>,
  "height_data": <1-5>,
  "table_format": <1-5>,
  "source_citation": <1-5>,
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
""")

DIMENSION_WEIGHTS = {
    "research_quality": 0.25,
    "height_data": 0.35,
    "table_format": 0.25,
    "source_citation": 0.15,
}


def test(result):
    return evaluate(result, TASK_INSTRUCTION, USER_PROMPT_TEMPLATE, DIMENSION_WEIGHTS)
