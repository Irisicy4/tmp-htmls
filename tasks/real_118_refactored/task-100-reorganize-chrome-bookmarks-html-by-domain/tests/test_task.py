import sys
sys.path.insert(0, "/harness")
from evaluator import evaluate

TASK_INSTRUCTION = ("""The file /Users/user/Downloads/bookmarks.html is my exported Chrome bookmarks. Please reorganize them, automatically categorizing by domain, and generate a new file that I can import back into Chrome.""")

USER_PROMPT_TEMPLATE = ("""## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Input: Chrome bookmarks HTML file (Netscape bookmark format)
- Processing: categorize by domain automatically
- Output: new HTML file in Chrome-importable format (Netscape Bookmark File Format)
- Validity: output must be importable into Chrome without errors

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent read/parse the bookmarks file?
- Were bookmarks categorized by domain?
- Was a new HTML file generated in correct format?
- Would the output file be importable into Chrome?
- How many bookmarks were processed?

### Step 2: Dimension Scoring

#### A. File Parsing (0.2)
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

#### D. Bookmark Preservation (0.15)
Were all bookmarks preserved in the output?

5 — All input bookmarks present in output with no loss.
4 — Most bookmarks preserved; a few dropped.
3 — Significant bookmark loss.
2 — Only sample bookmarks in output.
1 — No bookmarks in output.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "dimension_reasoning": {{
    "file_parsing": "<one sentence citing specific evidence>",
    "domain_categorization": "<one sentence citing specific evidence>",
    "output_validity": "<one sentence citing specific evidence>",
    "bookmark_preservation": "<one sentence citing specific evidence>"
  }},
  "file_parsing": <1-5>,
  "domain_categorization": <1-5>,
  "output_validity": <1-5>,
  "bookmark_preservation": <1-5>,
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
""")

DIMENSION_WEIGHTS = {
    "file_parsing":         0.20,
    "domain_categorization": 0.30,
    "output_validity":      0.35,
    "bookmark_preservation": 0.15,
}


def test(result):
    return evaluate(result, TASK_INSTRUCTION, USER_PROMPT_TEMPLATE, DIMENSION_WEIGHTS)
