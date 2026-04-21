import sys
sys.path.insert(0, "/harness")
from evaluator import evaluate

TASK_INSTRUCTION = ("Please rebuild the reservation program.")

USER_PROMPT_TEMPLATE = ("""## Task Instruction
{task_instruction}

## Task-Specific Constraints
- The task is intentionally vague ("rebuild the reservation program") — the agent must interpret this reasonably and build a functional reservation/booking web app
- A working UI is required — not just a description or plan
- Core reservation features expected: ability to make/view/cancel a reservation with date/time selection
- Code must be runnable (HTML file, React app, or similar)

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent produce actual code or just a description?
- What type of application was built (HTML, React, etc.)?
- What reservation features are present (booking form, date picker, list view, cancellation)?
- Was the app tested or previewed in a browser?
- Was a file saved?

### Step 2: Dimension Scoring

#### A. Application Completeness
Does the app cover core reservation functionality?

5 — App has booking form with date/time, reservation list view, and cancellation functionality.
4 — App has booking form and list view but no cancellation, or one feature is incomplete.
3 — App has only a booking form or only a list view — core loop is incomplete.
2 — App skeleton exists but no functional reservation logic.
1 — No working application produced.

#### B. Code Quality
Is the code well-structured and runnable?

5 — Clean, well-structured code that runs without errors; uses appropriate framework (React, HTML/JS).
4 — Code runs but has minor issues or is poorly organized.
3 — Code has bugs or incomplete sections but the intent is clear.
2 — Code exists but has major errors that prevent it from running.
1 — No code produced or code is clearly non-functional.

#### C. UI/UX Quality
Is the interface usable and reasonably well-designed?

5 — Clean, intuitive UI with clear labels, responsive layout, and good user experience.
4 — Functional UI but aesthetically basic or with minor UX issues.
3 — UI works but is hard to use or poorly laid out.
2 — UI exists but is broken or barely usable.
1 — No UI produced.

#### D. Delivery
Was the application properly saved and deliverable?

5 — File saved locally and/or dev server launched; agent confirmed it works.
4 — File saved but not tested or confirmed.
3 — Code provided in response but not saved as a file.
2 — Only partial code provided.
1 — Nothing delivered.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "application_completeness": <1-5>,
  "code_quality": <1-5>,
  "ui_ux_quality": <1-5>,
  "delivery": <1-5>,
  "dimension_reasoning": {{
    "application_completeness": "<one sentence citing specific evidence>",
    "code_quality": "<one sentence citing specific evidence>",
    "ui_ux_quality": "<one sentence citing specific evidence>",
    "delivery": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
""")

DIMENSION_WEIGHTS = {
    "application_completeness": 0.35,
    "code_quality":             0.25,
    "ui_ux_quality":            0.25,
    "delivery":                 0.15,
}


def test(result):
    return evaluate(result, TASK_INSTRUCTION, USER_PROMPT_TEMPLATE, DIMENSION_WEIGHTS)
