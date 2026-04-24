"""LLM-as-judge evaluator for task-95-compile-table-car-models-automated-parking.

Category: Daily Activities
Task: Please help me compile a table of car models that currently support automated parking into mechanical or multi-level parking spaces via parking assist. Include any height restrictions for the parking 
"""

import sys
sys.path.insert(0, "/harness")
from evaluator import run_judge

TASK_INSTRUCTION = "Please help me compile a table of car models that currently support automated parking into mechanical or multi-level parking spaces via parking assist. Include any height restrictions for the parking spaces they support, as well as the vehicle's own height dimensions. Cite your sources."

SYSTEM_PROMPT_EXTRA = 'Your job is to assess whether an AI agent successfully completed a browser-based task.\n\nThe task involves researching which car models have automated/remote parking assist that works with mechanical or multi-level (立体駐車場) parking structures, including height constraint compatibility.'

CONSTRAINTS = """- Feature: automated parking assist compatible with mechanical/multi-level parking structures
- Data: both vehicle height AND parking space height restriction required
- Coverage: multiple car brands/models
- Output: table format with sources cited"""

EVIDENCE_QUESTIONS = """- Did the agent search for cars with mechanical parking space compatibility?
- How many car models were found?
- Are vehicle heights and parking space height limits both included?
- Are sources cited?
- Is output in table format?"""

DIMENSION_RUBRICS = """#### A. Research Quality (0.25)
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
1 — No sources."""

DIMENSION_WEIGHTS = {
    'research_quality': 0.25,
    'height_data': 0.35,
    'table_format': 0.25,
    'source_citation': 0.15,
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
