import sys
sys.path.insert(0, "/harness")
from evaluator import evaluate

TASK_INSTRUCTION = ("""Please create an ear training (ear copy) tool as a web application. Build it as a single HTML file using Web Audio API that plays musical intervals or chords and lets the user identify them.""")

USER_PROMPT_TEMPLATE = ("""## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Format: single HTML file with embedded CSS and JavaScript
- API: Web Audio API for sound generation (no external audio files)
- Features: plays musical intervals or chords; user can identify them; feedback provided
- Quality: must be functional and usable — not just a mockup

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent produce a single HTML file?
- Does it use Web Audio API for sound generation?
- What musical content is included (intervals, chords, both)?
- Is there an identification UI (buttons, input) with feedback?
- Does the app function correctly?

### Step 2: Dimension Scoring

#### A. Web Audio Implementation (0.3)
Is Web Audio API used correctly?

5 — AudioContext, OscillatorNode, frequency calculations for musical intervals all correctly implemented.
4 — Web Audio API used but with minor errors.
3 — Web Audio API attempted but fundamental issues.
2 — External audio files or HTML audio elements used instead.
1 — No audio implementation.

#### B. Musical Content (0.25)
Is the musical content accurate and varied?

5 — Correct frequencies for intervals (unison through octave) and/or chords; covers major and minor at minimum.
4 — Mostly correct but missing some intervals/chords.
3 — Basic intervals only (major/minor/perfect).
2 — Inaccurate frequencies or very limited content.
1 — No musical content.

#### C. Ui Completeness (0.25)
Is the UI functional for ear training?

5 — Play button, answer selection (buttons/dropdown), feedback (correct/wrong), score tracking.
4 — Play + answer selection + feedback but no score.
3 — Play + answer but no feedback.
2 — Play only without identification UI.
1 — No functional UI.

#### D. Code Quality (0.2)
Is the single-file HTML well-written?

5 — Clean, readable HTML/CSS/JS in one file; no external dependencies; works on open.
4 — Functional but slightly messy code.
3 — Works but has notable quality issues.
2 — Significant bugs; partially functional.
1 — Does not work.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "web_audio_implementation": <1-5>,
  "musical_content": <1-5>,
  "ui_completeness": <1-5>,
  "code_quality": <1-5>,
  "dimension_reasoning": {{
    "web_audio_implementation": "<one sentence citing specific evidence>",
    "musical_content": "<one sentence citing specific evidence>",
    "ui_completeness": "<one sentence citing specific evidence>",
    "code_quality": "<one sentence citing specific evidence>",
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
""")

DIMENSION_WEIGHTS = {
    "web_audio_implementation": 0.3,
    "musical_content": 0.25,
    "ui_completeness": 0.25,
    "code_quality": 0.2,
}


def test(result):
    return evaluate(result, TASK_INSTRUCTION, USER_PROMPT_TEMPLATE, DIMENSION_WEIGHTS)
