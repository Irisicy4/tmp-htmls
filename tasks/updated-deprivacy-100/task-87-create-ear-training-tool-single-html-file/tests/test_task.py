"""
LLM-as-judge evaluator for EvolveBench task-87-create-ear-training-tool-single-html-file.

Category: Web Development
Task: Create a web-based ear training tool as a single HTML file. The tool should: (1) play a musical note or interval using the Web Audio API, (2) let the user guess what note or interval was played from m
"""

import os, json, re

TASK_INSTRUCTION = """Create a web-based ear training tool as a single HTML file. The tool should: (1) play a musical note or interval using the Web Audio API, (2) let the user guess what note or interval was played from multiple choice options, (3) show whether the answer was correct, and (4) track the user's score. Save the file locally."""
PASS_THRESHOLD = 3.0

SYSTEM_PROMPT = """The judge is evaluating the agent's ability to create a functional and user-friendly web-based ear training tool as a single HTML file. The tool should utilize the Web Audio API to play musical notes or intervals, provide multiple-choice options for user guesses, display feedback on correctness, and track the user's score effectively. The evaluation focuses on technical implementation, usability, and adherence to the task requirements.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- The tool must be implemented as a single HTML file.
- The Web Audio API must be used to generate and play musical notes or intervals.
- The interface must allow users to select their answers from multiple-choice options.
- The tool must display feedback indicating whether the user's answer was correct.
- The tool must track and display the user's score.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Does the HTML file include functional code to play musical notes or intervals using the Web Audio API?
- Does the tool provide multiple-choice options for users to guess the note or interval played?
- Does the tool display feedback indicating whether the user's answer was correct?
- Does the tool track and display the user's score accurately?
- Is the tool implemented as a single HTML file without external dependencies?

### Step 2: Dimension Scoring

#### A. Web Audio Api Functionality
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
1 — The tool does not adhere to the task constraints.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "web_audio_api_functionality": <1-5>,
  "user_interface_design": <1-5>,
  "feedback_and_score_tracking": <1-5>,
  "task_constraints_adherence": <1-5>,
  "dimension_reasoning": {{
    "web_audio_api_functionality": "<one sentence citing specific evidence>",
    "user_interface_design": "<one sentence citing specific evidence>",
    "feedback_and_score_tracking": "<one sentence citing specific evidence>",
    "task_constraints_adherence": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "web_audio_api_functionality": 0.3,
    "user_interface_design": 0.25,
    "feedback_and_score_tracking": 0.25,
    "task_constraints_adherence": 0.2,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())

