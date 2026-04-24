"""LLM-as-judge evaluator for task-68-cut-out-two-books-right-side-image.

Category: (Self) Media
Task: This is my product main image. Please cut out the 2 books on the right side of the image separately.
"""

import sys
sys.path.insert(0, "/harness")
from evaluator import run_judge

TASK_INSTRUCTION = 'This is my product main image. Please cut out the 2 books on the right side of the image separately.'

SYSTEM_PROMPT_EXTRA = 'Your job is to assess whether an AI agent successfully completed a browser-based task.\n\nThe task involves using an image editing tool (likely Photopea) to isolate and extract two book objects from a product image as separate files.'

CONSTRAINTS = """- Tool: Photopea or equivalent browser-based image editor
- Action: extract/cut out 2 specific books from the right side of the image
- Output: 2 separate image files (one per book) with transparent or clean backgrounds
- Precision: cutouts should be clean without significant artifacts"""

EVIDENCE_QUESTIONS = """- Did the agent open an image editing tool?
- Were the two books on the right side identified and selected?
- Were separate cutout files produced for each book?
- Is the background clean (transparent or white) around the cutouts?"""

DIMENSION_RUBRICS = """#### A. Tool Usage (0.25)
Did the agent use an appropriate image editing tool?

5 — Agent opened Photopea or equivalent and loaded the product image.
4 — Agent used a tool but with setup difficulties.
3 — Agent described using a tool without clear evidence of execution.
2 — Agent suggested a tool without using it.
1 — No tool used.

#### B. Object Identification (0.2)
Were the correct objects (2 books on right side) identified?

5 — Both books on the right side correctly identified and selected.
4 — Books identified but selection slightly off.
3 — One book correctly identified.
2 — Wrong objects selected.
1 — No object identification.

#### C. Extraction Quality (0.4)
Were clean cutouts produced?

5 — Both books extracted as separate files with clean transparent/white background and minimal artifacts.
4 — Both extracted but with minor artifacts.
3 — One book extracted cleanly; other has issues.
2 — Both extracted but with significant artifacts.
1 — No cutouts produced.

#### D. File Output (0.15)
Were separate files saved for each book?

5 — Two separate image files saved (e.g. book1.png, book2.png).
4 — Files saved but both in one file or wrong format.
3 — One file saved.
2 — Files described but not saved.
1 — No output files."""

DIMENSION_WEIGHTS = {
    'tool_usage': 0.25,
    'object_identification': 0.2,
    'extraction_quality': 0.4,
    'file_output': 0.15,
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
