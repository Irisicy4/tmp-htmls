import sys
sys.path.insert(0, "/harness")
from evaluator import evaluate

TASK_INSTRUCTION = ("""Please update my N8N installation on Hostinger.""")

USER_PROMPT_TEMPLATE = ("""## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Platform: Hostinger (VPS or shared hosting)
- Action: update/upgrade N8N to latest version
- Method: agent may use SSH terminal, Hostinger admin panel, or Docker depending on installation type
- Confirmation: agent should verify the updated version number

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent access Hostinger or an SSH terminal?
- What N8N update method was used (npm, Docker, hPanel)?
- Was the update command executed successfully?
- Was the new N8N version confirmed?

### Step 2: Dimension Scoring

#### A. Platform Access (0.25)
Did the agent access Hostinger or a terminal?

5 — Agent accessed Hostinger hPanel or SSH terminal and authenticated successfully.
4 — Agent accessed the platform but with some difficulty.
3 — Agent described update steps without accessing the platform.
2 — Agent gave generic N8N update instructions without Hostinger context.
1 — No platform access attempted.

#### B. Update Execution (0.4)
Was the N8N update command executed?

5 — Update command run (e.g. npm update -g n8n, docker pull n8nio/n8n, or equivalent) with output shown.
4 — Update initiated but completion unclear.
3 — Correct command identified but not executed.
2 — Wrong update method attempted.
1 — No update attempt.

#### C. Version Verification (0.25)
Was the updated version verified?

5 — New N8N version confirmed (e.g. n8n --version output shown after update).
4 — Update appeared successful but version not explicitly verified.
3 — Agent mentioned version check but result unclear.
2 — No version verification.
1 — Update failed or not attempted.

#### D. Error Handling (0.1)
Did the agent handle any errors gracefully?

5 — Any errors encountered were addressed and resolved.
4 — Errors acknowledged with reasonable workaround.
3 — Errors noted but not resolved.
2 — Errors ignored.
1 — Task failed due to unhandled errors.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "platform_access": <1-5>,
  "update_execution": <1-5>,
  "version_verification": <1-5>,
  "error_handling": <1-5>,
  "dimension_reasoning": {{
    "platform_access": "<one sentence citing specific evidence>",
    "update_execution": "<one sentence citing specific evidence>",
    "version_verification": "<one sentence citing specific evidence>",
    "error_handling": "<one sentence citing specific evidence>",
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
""")

DIMENSION_WEIGHTS = {
    "platform_access": 0.25,
    "update_execution": 0.4,
    "version_verification": 0.25,
    "error_handling": 0.1,
}


def test(result):
    return evaluate(result, TASK_INSTRUCTION, USER_PROMPT_TEMPLATE, DIMENSION_WEIGHTS)
