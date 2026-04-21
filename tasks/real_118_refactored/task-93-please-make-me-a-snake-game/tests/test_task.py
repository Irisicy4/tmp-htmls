import sys
sys.path.insert(0, "/harness")
from evaluator import evaluate

TASK_INSTRUCTION = ("""Please make me a Snake game.""")

USER_PROMPT_TEMPLATE = ("""## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Format: single HTML file (self-contained)
- Gameplay: standard Snake mechanics — snake grows on eating food, dies on wall/self collision
- Controls: arrow keys or WASD
- Features: score display, game over state, restart option

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent produce a single HTML file?
- Does the Snake game have correct mechanics (growth, collision detection)?
- Are controls implemented?
- Is score displayed?
- Is game over and restart handled?

### Step 2: Dimension Scoring

#### A. Game Mechanics (0.4)
Are core Snake mechanics correctly implemented?

5 — Snake moves continuously, grows on food, dies on wall and self-collision — all correct.
4 — Core mechanics work but one edge case wrong (e.g. self-collision detection off).
3 — Snake moves and grows but collision detection is buggy.
2 — Snake moves but growth or collision missing.
1 — No functional game mechanics.

#### B. Controls (0.2)
Are controls properly implemented?

5 — Arrow keys or WASD work correctly; can't reverse direction instantly.
4 — Controls work but reverse direction is allowed.
3 — Controls work partially.
2 — Controls implemented but unresponsive.
1 — No controls.

#### C. Ui Completeness (0.25)
Are UI elements present?

5 — Score display, game over screen, and restart option all present.
4 — Score and game over but no restart.
3 — Score only.
2 — No UI elements.
1 — Blank canvas only.

#### D. Code Quality (0.15)
Is the single-file HTML clean and functional?

5 — Clean code, no external dependencies, works immediately on open.
4 — Functional with minor code quality issues.
3 — Works but notable quality issues.
2 — Partially functional.
1 — Does not work.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "game_mechanics": <1-5>,
  "controls": <1-5>,
  "ui_completeness": <1-5>,
  "code_quality": <1-5>,
  "dimension_reasoning": {{
    "game_mechanics": "<one sentence citing specific evidence>",
    "controls": "<one sentence citing specific evidence>",
    "ui_completeness": "<one sentence citing specific evidence>",
    "code_quality": "<one sentence citing specific evidence>",
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
""")

DIMENSION_WEIGHTS = {
    "game_mechanics": 0.4,
    "controls": 0.2,
    "ui_completeness": 0.25,
    "code_quality": 0.15,
}


def test(result):
    return evaluate(result, TASK_INSTRUCTION, USER_PROMPT_TEMPLATE, DIMENSION_WEIGHTS)
