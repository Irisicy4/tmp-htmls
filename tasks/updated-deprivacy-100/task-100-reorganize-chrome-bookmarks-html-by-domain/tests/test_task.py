"""LLM-as-judge evaluator for task-100-reorganize-chrome-bookmarks-html-by-domain.

Category: Daily Activities
Task: The file /Users/user/Downloads/bookmarks.html is my exported Chrome bookmarks. Please reorganize them, automatically categorizing by domain, and generate a new file that I can import back into Chrome.
"""

import sys
sys.path.insert(0, "/harness")
from evaluator import run_judge

TASK_INSTRUCTION = 'The file /Users/user/Downloads/bookmarks.html is my exported Chrome bookmarks. Please reorganize them, automatically categorizing by domain, and generate a new file that I can import back into Chrome.'

SYSTEM_PROMPT_EXTRA = 'Your job is to assess whether an AI agent successfully completed a browser-based task.\n\nThe task involves parsing an exported Chrome bookmarks HTML file, reorganizing bookmarks by domain category, and generating a new valid Chrome-importable bookmarks HTML file.'

CONSTRAINTS = """- Input: Chrome bookmarks HTML file (Netscape bookmark format)
- Processing: categorize by domain automatically
- Output: new HTML file in Chrome-importable format (Netscape Bookmark File Format)
- Validity: output must be importable into Chrome without errors"""

EVIDENCE_QUESTIONS = """- Did the agent read/parse the bookmarks file?
- Were bookmarks categorized by domain?
- Was a new HTML file generated in correct format?
- Would the output file be importable into Chrome?
- How many bookmarks were processed?"""

DIMENSION_RUBRICS = """#### A. File Parsing (0.2)
Did the agent parse the bookmarks file?

5 — Bookmarks HTML correctly parsed; bookmark titles, URLs, and existing folders extracted.
4 — Parsing mostly correct with minor misses.
3 — Partial parsing.
2 — Agent read the file but parsing is incomplete.
1 — No parsing.

#### B. Domain Categorization (0.3)
Were bookmarks categorized by domain?

5 — All bookmarks grouped into domain-based folders (e.g. GitHub, Google, YouTube); ungrouped items in 'Other'.
4 — Most bookmarks categorized; some missed.
3 — Categorization present but inconsistent.
2 — Minimal categorization.
1 — No categorization.

#### C. Output Validity (0.35)
Is the output a valid Chrome-importable bookmarks file?

5 — Valid Netscape Bookmark File Format with correct DOCTYPE, DL/DT structure, and ADD_DATE attributes.
4 — Mostly valid but minor format issues.
3 — HTML produced but format deviates from Netscape standard.
2 — HTML produced but not importable.
1 — No output file.

#### D. Completeness (0.15)
Were all bookmarks preserved in the output?

5 — All input bookmarks present in output with no loss.
4 — Most bookmarks preserved; a few dropped.
3 — Significant bookmark loss.
2 — Only sample bookmarks in output.
1 — No bookmarks in output."""

DIMENSION_WEIGHTS = {
    'file_parsing': 0.2,
    'domain_categorization': 0.3,
    'output_validity': 0.35,
    'completeness': 0.15,
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
