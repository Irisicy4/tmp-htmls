"""LLM-as-judge evaluator for task-46-extract-data-and-plot-uv.

Category: Data Analysis and Visualization
Task: Extract data and plot the UV data for the past 5 years.
"""

import sys
sys.path.insert(0, "/harness")
from evaluator import run_judge

TASK_INSTRUCTION = 'Extract data and plot the UV data for the past 5 years.'

SYSTEM_PROMPT_EXTRA = 'Your job is to assess whether an AI agent successfully extracted accurate UV data for the past 5 years and produced a clear and informative plot based on the processed data. The evaluation emphasizes the correctness of data extraction, the appropriateness of processing methods, and the clarity of the visualization.'

CONSTRAINTS = """- The data must cover exactly the past 5 years from the current date.
- The UV data must be sourced from a reliable and verifiable source.
- The plot must clearly represent the trends or patterns in the UV data over the 5-year period.
- The axes, labels, and legends in the plot must be appropriately labeled and easy to understand.
- The agent must handle any missing or incomplete data in a reasonable and documented manner."""

EVIDENCE_QUESTIONS = """- Did the agent extract UV data covering the past 5 years from a reliable source?
- Did the agent process the data correctly, including handling missing or incomplete data?
- Does the plot clearly and accurately represent the trends or patterns in the UV data?
- Are the axes, labels, and legends in the plot clear and appropriate?
- Did the agent document the data source and processing steps adequately?"""

DIMENSION_RUBRICS = """#### A. Data Extraction Accuracy
Measures the correctness and completeness of the extracted UV data.

5 — The data covers exactly the past 5 years and is sourced from a reliable, verifiable source.
4 — The data covers the past 5 years with minor gaps or ambiguities in the source.
3 — The data partially covers the past 5 years or the source is not fully reliable.
2 — The data is incomplete or the source is questionable.
1 — The data is missing or entirely unreliable.

#### B. Data Processing Quality
Evaluates how well the agent handled and processed the data.

5 — The data was processed correctly, with clear handling of missing or incomplete data.
4 — The data was processed correctly with minor issues in handling missing or incomplete data.
3 — The data processing had noticeable issues or lacked clarity in handling missing data.
2 — The data processing was flawed or poorly documented.
1 — The data processing was incorrect or not performed.

#### C. Visualization Clarity
Assesses the clarity and informativeness of the plot.

5 — The plot is clear, well-labeled, and effectively communicates trends or patterns.
4 — The plot is mostly clear and informative with minor labeling or clarity issues.
3 — The plot is somewhat clear but has noticeable issues in labeling or communication of trends.
2 — The plot is unclear or poorly labeled, making it hard to interpret.
1 — The plot is missing or entirely incomprehensible.

#### D. Documentation Completeness
Measures how well the agent documented the data source and processing steps.

5 — The data source and processing steps are thoroughly and clearly documented.
4 — The documentation is mostly complete with minor omissions or ambiguities.
3 — The documentation is partially complete or lacks clarity.
2 — The documentation is incomplete or unclear.
1 — The documentation is missing or entirely inadequate."""

DIMENSION_WEIGHTS = {
    'data_extraction_accuracy': 0.3,
    'data_processing_quality': 0.25,
    'visualization_clarity': 0.25,
    'documentation_completeness': 0.2,
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
