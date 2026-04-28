import sys
sys.path.insert(0, "/harness")
from evaluator import evaluate

TASK_INSTRUCTION = ("""Please create a program to manage orders and customer inquiries using the Naver Smart Store API.""")

USER_PROMPT_TEMPLATE = ("""## Task Instruction
{task_instruction}

## Task-Specific Constraints
- API: Naver Smart Store API specifically
- Features: order management AND customer inquiry handling — both required
- Implementation: working code, not just pseudocode or description
- Language: Python preferred; should include API authentication setup

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent produce actual working code?
- Does the code integrate with Naver Smart Store API?
- Are both order management and customer inquiry features implemented?
- Is API authentication handled?
- Was the code saved as a file?

### Step 2: Dimension Scoring

#### A. Api Integration (0.3)
Does the code properly integrate with Naver Smart Store API?

5 — Correct API endpoints used; OAuth/token authentication implemented; API calls are syntactically correct.
4 — API integration present but authentication is incomplete.
3 — API references correct but implementation has significant gaps.
2 — Generic HTTP code without Naver-specific API knowledge.
1 — No API integration.

#### B. Feature Completeness (0.35)
Are both required features implemented?

5 — Order management (list, detail, status update) AND customer inquiry (list, respond) both implemented.
4 — One feature fully implemented; other partially done.
3 — One feature implemented; other missing.
2 — Only basic order listing with no inquiry handling.
1 — Neither feature implemented.

#### C. Code Quality (0.25)
Is the code well-structured and runnable?

5 — Clean, modular code with error handling, docstrings, and configuration management.
4 — Functional code with minor quality issues.
3 — Code runs but is poorly structured.
2 — Code has major bugs or won't run.
1 — No runnable code.

#### D. Delivery (0.1)
Was the code saved as a deliverable file?

5 — Python file saved with clear setup instructions.
4 — Code saved but minimal instructions.
3 — Code provided in response but not saved as file.
2 — Only partial code snippets.
1 — Nothing delivered.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "dimension_reasoning": {{
    "api_integration": "<one sentence citing specific evidence>",
    "feature_completeness": "<one sentence citing specific evidence>",
    "code_quality": "<one sentence citing specific evidence>",
    "delivery": "<one sentence citing specific evidence>",
  }},
  "api_integration": <1-5>,
  "feature_completeness": <1-5>,
  "code_quality": <1-5>,
  "delivery": <1-5>,
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
""")

DIMENSION_WEIGHTS = {
    "api_integration": 0.3,
    "feature_completeness": 0.35,
    "code_quality": 0.25,
    "delivery": 0.1,
}


def test(result):
    return evaluate(result, TASK_INSTRUCTION, USER_PROMPT_TEMPLATE, DIMENSION_WEIGHTS)
