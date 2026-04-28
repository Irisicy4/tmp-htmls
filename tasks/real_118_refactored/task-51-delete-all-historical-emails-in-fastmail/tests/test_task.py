import sys
sys.path.insert(0, "/harness")
from evaluator import evaluate

TASK_INSTRUCTION = ("""Delete all historical emails in Fastmail that are related to verification codes.""")

USER_PROMPT_TEMPLATE = ("""## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Platform: Fastmail specifically
- Scope: all historical verification code emails — the agent should be thorough
- Action: actual deletion required, not just identification
- Search strategy: agent should use iterative search refinement to catch all variants

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Fastmail?
- What search queries were used to find verification code emails?
- How many emails were found and deleted?
- Did the agent use multiple search iterations to ensure completeness?
- Was deletion confirmed?

### Step 2: Dimension Scoring

#### A. Platform Access (0.35)
Did the agent access Fastmail?

5 — Agent navigated to Fastmail and accessed the inbox.
4 — Agent accessed Fastmail but with some difficulty.
3 — Agent attempted Fastmail but had login or access issues.
2 — Agent described Fastmail without navigating to it.
1 — No Fastmail navigation.

#### B. Search Strategy (0.24)
Did the agent use effective search to find verification code emails?

5 — Used multiple search queries covering various keywords (verification, 验证码, code, OTP, etc.) with iterative refinement.
4 — Used 2-3 good search queries.
3 — Used only one broad search query.
2 — Browsed inbox manually without searching.
1 — No search strategy.

#### C. Deletion Execution (0.28)
Were emails actually deleted?

5 — Agent selected and deleted emails with confirmation of deletion count or empty results after.
4 — Agent deleted emails but without clear confirmation of completeness.
3 — Agent deleted some emails but process was incomplete.
2 — Agent identified emails without deleting them.
1 — No deletion performed.

#### D. Deletion Thoroughness (0.13)
Did the agent ensure thorough coverage and verify no verification emails remain?

5 — Agent explicitly checked for remaining emails after deletion and confirmed inbox clear of verification emails.
4 — Agent did multiple passes but did not verify completeness.
3 — Agent did one pass only.
2 — Agent acknowledged potential incompleteness without addressing it.
1 — No completeness consideration.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "dimension_reasoning": {{
    "platform_access": "<one sentence citing specific evidence>",
    "search_strategy": "<one sentence citing specific evidence>",
    "deletion_execution": "<one sentence citing specific evidence>",
    "deletion_thoroughness": "<one sentence citing specific evidence>"
  }},
  "platform_access": <1-5>,
  "search_strategy": <1-5>,
  "deletion_execution": <1-5>,
  "deletion_thoroughness": <1-5>,
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
""")

DIMENSION_WEIGHTS = {
    "platform_access": 0.35,
    "search_strategy": 0.24,
    "deletion_execution": 0.28,
    "deletion_thoroughness": 0.13,
}


def test(result):
    return evaluate(result, TASK_INSTRUCTION, USER_PROMPT_TEMPLATE, DIMENSION_WEIGHTS)
