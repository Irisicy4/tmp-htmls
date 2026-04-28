"""LLM-as-judge evaluator for task-87-create-ear-training-tool-single-html-file.

Category: Web Development
Task: Create a web-based ear training tool as a single HTML file. The tool should: (1) play a musical note or interval using the Web Audio API, (2) let the user guess what note or interval was played from m
"""

import sys
sys.path.insert(0, "/harness")
from evaluator import run_judge

TASK_INSTRUCTION = "Create a web-based ear training tool as a single HTML file. The tool should: (1) play a musical note or interval using the Web Audio API, (2) let the user guess what note or interval was played from multiple choice options, (3) show whether the answer was correct, and (4) track the user's score. Save the file locally."

SYSTEM_PROMPT_EXTRA = "Your job is to assess whether an AI agent successfully created a functional and user-friendly web-based ear training tool as a single HTML file. The tool must use the Web Audio API to play musical notes or intervals, offer multiple-choice options for user guesses, display feedback on correctness, and track the user's score. Evaluation emphasizes technical accuracy, usability, and compliance with the specified requirements."

CONSTRAINTS = """- The tool must be implemented as a single HTML file.
- The Web Audio API must be used to generate and play musical notes or intervals.
- The interface must allow users to select their answers from multiple-choice options.
- The tool must display feedback indicating whether the user's answer was correct.
- The tool must track and display the user's score."""

EVIDENCE_QUESTIONS = """- Does the HTML file include functional code to play musical notes or intervals using the Web Audio API?
- Does the tool provide multiple-choice options for users to guess the note or interval played?
- Does the tool display feedback indicating whether the user's answer was correct?
- Does the tool track and display the user's score accurately?
- Is the tool implemented as a single HTML file without external dependencies?"""

DIMENSION_RUBRICS = """#### A. Web Audio Api Functionality
Evaluates whether the Web Audio API is correctly implemented to play notes or intervals.

5 — The tool reliably plays musical notes or intervals using the Web Audio API without errors.
4 — The tool plays musical notes or intervals using the Web Audio API, but with minor inconsistencies or occasional errors.
3 — The tool plays musical notes or intervals, but the implementation has noticeable flaws or frequent errors.
2 — The tool attempts to use the Web Audio API but fails to play notes or intervals correctly.
1 — The tool does not use the Web Audio API to play musical notes or intervals.

#### B. User Interface Design
Evaluates the usability and clarity of the tool's interface for guessing notes or intervals.

5 — The interface is intuitive, visually clear, and easy to use for guessing notes or intervals.
4 — The interface is functional and mostly clear, with minor usability issues.
3 — The interface is somewhat confusing or lacks clarity, impacting usability.
2 — The interface is poorly designed and difficult to use for guessing notes or intervals.
1 — The interface is non-functional or missing entirely.

#### C. Feedback And Score Tracking
Evaluates the tool's ability to provide feedback and track the user's score accurately.

5 — The tool provides immediate and accurate feedback on answers and reliably tracks the user's score.
4 — The tool provides feedback and tracks the score, but with minor inaccuracies or delays.
3 — The tool provides feedback and tracks the score, but with noticeable issues or inconsistencies.
2 — The tool attempts to provide feedback and track the score but fails to do so reliably.
1 — The tool does not provide feedback or track the user's score.

#### D. Task Constraints Adherence
Evaluates whether the tool adheres to the constraints outlined in the task instruction.

5 — The tool fully adheres to all task constraints, including being a single HTML file and meeting all functional requirements.
4 — The tool adheres to most task constraints, with minor deviations.
3 — The tool adheres to some task constraints but has noticeable deviations.
2 — The tool barely adheres to the task constraints, with significant omissions.
1 — The tool does not adhere to the task constraints."""

DIMENSION_WEIGHTS = {
    'web_audio_api_functionality': 0.3,
    'user_interface_design': 0.25,
    'feedback_and_score_tracking': 0.25,
    'task_constraints_adherence': 0.2,
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
