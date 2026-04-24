"""LLM-as-judge evaluator for task-07-save-todays-top-traded-etf.

Category: Data Collection and Documentation
Task: Save today’s top traded ETF stocks from Naver Securities to Google Docs
"""

import sys
sys.path.insert(0, "/harness")
from evaluator import run_judge

TASK_INSTRUCTION = 'Save today’s top traded ETF stocks from Naver Securities to Google Docs'

SYSTEM_PROMPT_EXTRA = "Your job is to assess whether an AI agent successfully retrieved today's top traded ETF stocks from Naver Securities and documented them in Google Docs. The evaluation emphasizes accuracy, completeness, proper formatting, and adherence to task requirements."

CONSTRAINTS = """- Retrieve data specifically from Naver Securities.
- Focus only on ETF stocks and identify the top traded ones.
- Save the data in a Google Docs document with clear formatting.
- Ensure the document is accessible and properly titled.
- Avoid including irrelevant or outdated information."""

EVIDENCE_QUESTIONS = """- Did the agent retrieve the data from Naver Securities as instructed?
- Does the document include only ETF stocks and identify the top traded ones?
- Is the Google Docs document properly formatted and titled?
- Is the information accurate and complete?
- Is the document accessible and free of irrelevant content?"""

DIMENSION_RUBRICS = """#### A. Data Accuracy
Measures whether the retrieved ETF stock data is accurate and matches Naver Securities.

5 — All ETF stock data is accurate and matches Naver Securities perfectly.
4 — Most ETF stock data is accurate, with minor discrepancies.
3 — Some ETF stock data is accurate, but there are noticeable errors.
2 — Significant inaccuracies in the ETF stock data retrieved.
1 — ETF stock data is entirely inaccurate or missing.

#### B. Data Completeness
Evaluates whether all top traded ETF stocks are included in the document.

5 — All top traded ETF stocks are included without omissions.
4 — Most top traded ETF stocks are included, with minor omissions.
3 — Some top traded ETF stocks are included, but key entries are missing.
2 — Few top traded ETF stocks are included, with major omissions.
1 — No top traded ETF stocks are included or the list is irrelevant.

#### C. Document Formatting
Assesses the clarity and organization of the Google Docs document.

5 — The document is well-organized, clearly formatted, and easy to read.
4 — The document is mostly well-organized, with minor formatting issues.
3 — The document has noticeable formatting issues but is still readable.
2 — The document is poorly formatted and difficult to read.
1 — The document is completely disorganized and unreadable.

#### D. Task Adherence
Checks whether the agent followed all task constraints and instructions.

5 — All task constraints and instructions were followed perfectly.
4 — Most task constraints and instructions were followed, with minor deviations.
3 — Some task constraints and instructions were followed, but there are noticeable deviations.
2 — Few task constraints and instructions were followed, with major deviations.
1 — None of the task constraints or instructions were followed."""

DIMENSION_WEIGHTS = {
    'data_accuracy': 0.35,
    'data_completeness': 0.25,
    'document_formatting': 0.2,
    'task_adherence': 0.2,
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
