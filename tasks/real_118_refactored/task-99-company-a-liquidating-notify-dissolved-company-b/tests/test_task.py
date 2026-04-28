import sys
sys.path.insert(0, "/harness")
from evaluator import evaluate

TASK_INSTRUCTION = ("""Company A is preparing for liquidation and dissolution. It has discovered that it still has accounts payable to Company B on its books. However, Company B has already been dissolved. After Company A begins its liquidation process, does it still have an obligation to notify Company B? Please answer based on Chinese law.""")

USER_PROMPT_TEMPLATE = ("""## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Jurisdiction: Chinese law specifically
- Topic: liquidation notification obligations when creditor is dissolved
- Required: cite specific Chinese legal provisions (Company Law, Liquidation regulations)
- Answer: must address the notification obligation and what Company A should do instead

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent research Chinese law on this specific scenario?
- What specific legal provisions were cited (Company Law articles)?
- Is the notification obligation clearly addressed?
- What practical guidance is provided for Company A?

### Step 2: Dimension Scoring

#### A. Legal Research (0.25)
Did the agent research the relevant Chinese legal framework?

5 — Researched Chinese Company Law, liquidation regulations, and relevant judicial interpretations.
4 — Found relevant law but search was less comprehensive.
3 — General Chinese corporate law found without specific liquidation provisions.
2 — Generic legal answer without Chinese law research.
1 — No legal research.

#### B. Legal Accuracy (0.35)
Is the legal analysis accurate under Chinese law?

5 — Correct analysis: under Chinese Company Law, notification obligation exists but must be directed to known creditors; when creditor is dissolved, obligation typically transforms to public announcement; specific articles cited.
4 — Mostly correct with minor gaps.
3 — Generally correct direction but missing key provisions.
2 — Partially correct but significant errors.
1 — Incorrect or no legal analysis.

#### C. Citation Quality (0.25)
Are specific legal provisions cited?

5 — Specific articles from Chinese Company Law, liquidation procedures, or relevant regulations cited.
4 — Some articles cited.
3 — Legal framework referenced without specific articles.
2 — Vague reference to Chinese law.
1 — No citations.

#### D. Practical Guidance (0.15)
Is practical guidance provided for Company A?

5 — Clear next steps: what Company A should do (public announcement, hold funds, court guidance).
4 — Some guidance provided.
3 — Legal conclusion reached without practical next steps.
2 — Very vague guidance.
1 — No guidance.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "dimension_reasoning": {{
    "legal_research": "<one sentence citing specific evidence>",
    "legal_accuracy": "<one sentence citing specific evidence>",
    "citation_quality": "<one sentence citing specific evidence>",
    "practical_guidance": "<one sentence citing specific evidence>",
  }},
  "legal_research": <1-5>,
  "legal_accuracy": <1-5>,
  "citation_quality": <1-5>,
  "practical_guidance": <1-5>,
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
""")

DIMENSION_WEIGHTS = {
    "legal_research": 0.25,
    "legal_accuracy": 0.35,
    "citation_quality": 0.25,
    "practical_guidance": 0.15,
}


def test(result):
    return evaluate(result, TASK_INSTRUCTION, USER_PROMPT_TEMPLATE, DIMENSION_WEIGHTS)
