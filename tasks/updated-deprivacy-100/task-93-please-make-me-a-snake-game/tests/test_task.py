"""LLM-as-judge evaluator for task-93-please-make-me-a-snake-game.

Category: Software Engineering
Task: Please make me a Snake game.
"""

import sys
sys.path.insert(0, "/harness")
from evaluator import run_judge

TASK_INSTRUCTION = 'Please make me a Snake game.'

SYSTEM_PROMPT_EXTRA = 'Your job is to assess whether an AI agent successfully completed a browser-based task.\n\nThe task involves building a functional Snake game as a single HTML file with JavaScript.'

CONSTRAINTS = """- Format: single HTML file (self-contained)
- Gameplay: standard Snake mechanics — snake grows on eating food, dies on wall/self collision
- Controls: arrow keys or WASD
- Features: score display, game over state, restart option"""

EVIDENCE_QUESTIONS = """- Did the agent produce a single HTML file?
- Does the Snake game have correct mechanics (growth, collision detection)?
- Are controls implemented?
- Is score displayed?
- Is game over and restart handled?"""

DIMENSION_RUBRICS = """#### A. Game Mechanics (0.4)
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
1 — Does not work."""

DIMENSION_WEIGHTS = {
    'game_mechanics': 0.4,
    'controls': 0.2,
    'ui_completeness': 0.25,
    'code_quality': 0.15,
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
