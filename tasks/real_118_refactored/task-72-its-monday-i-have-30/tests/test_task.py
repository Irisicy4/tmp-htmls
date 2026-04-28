import sys
sys.path.insert(0, "/harness")
from evaluator import evaluate

TASK_INSTRUCTION = ("""It's Monday. I have 30 minutes before the all-hands meeting. Summarize the major tech shifts from last week and simulate 3 tough questions investors might ask us today based on those trends.""")

USER_PROMPT_TEMPLATE = ("""## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Must identify major tech developments from the past week (recent, not evergreen content)
- Must produce exactly 3 simulated investor questions
- Questions must directly reference the identified tech shifts (not generic questions)
- Output must be concise and structured for a 30-minute pre-meeting scan
- Questions should be genuinely challenging (probing assumptions, risks, or strategic implications)

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent search for recent tech news? What sources or topics were identified?
- Are the tech developments named specifically (company names, product launches, regulatory actions, etc.)?
- How many investor questions were generated, and are they directly tied to the identified trends?
- Is the output structured and brief enough for a 30-minute pre-meeting read?

### Step 2: Dimension Scoring

#### A. Tech Shift Identification (0.35)
Did the agent identify specific, recent major tech developments from the past week?

5 — 3 or more distinct, named tech developments with specific companies, dates, or events (e.g. "OpenAI released GPT-X on [date]", "EU passed AI regulation", "Apple acquired...").
4 — 2-3 tech developments identified with reasonable specificity.
3 — Tech topics mentioned but vague, or clearly not from the past week (evergreen topics).
2 — Only one development identified, or developments are generic without specifics.
1 — No tech shifts identified or researched.

#### B. Investor Question Quality (0.30)
Are exactly 3 tough, specific investor questions provided that are grounded in the identified trends?

5 — Exactly 3 questions, each directly referencing a named tech development; questions probe risks, competitive dynamics, or strategic implications in a way that challenges the presenter.
4 — 3 questions provided, mostly tied to the trends but some are slightly generic.
3 — 3 questions provided but generic, or not clearly tied to the specific tech shifts identified.
2 — Fewer than 3 questions, or questions are low-quality/softball.
1 — No investor questions generated.

#### C. Recency and Specificity (0.20)
Is the content clearly from the past week with concrete details?

5 — Content references specific dates, events, or announcements from the past week; shows evidence of actual web research.
4 — Recent content with implicit timeframe (e.g. references to recent announcements without specific dates).
3 — Some recent content mixed with older or evergreen material.
2 — Content appears outdated or could apply to any time period.
1 — No evidence of recent research; entirely generic.

#### D. Briefing Readability (0.15)
Is the output well-organized and scannable within 30 minutes?

5 — Clear sections (tech summary + investor questions), concise bullet points or headers, actionable takeaways; digestible in under 5 minutes.
4 — Organized output with some structure, but slightly verbose.
3 — Information present but dense or disorganized.
2 — Poorly structured wall of text.
1 — No structure; output unusable as a briefing.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "dimension_reasoning": {{
    "tech_shift_identification": "<one sentence citing specific evidence>",
    "investor_question_quality": "<one sentence citing specific evidence>",
    "recency_and_specificity": "<one sentence citing specific evidence>",
    "briefing_readability": "<one sentence citing specific evidence>"
  }},
  "tech_shift_identification": <1-5>,
  "investor_question_quality": <1-5>,
  "recency_and_specificity": <1-5>,
  "briefing_readability": <1-5>,
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
""")

DIMENSION_WEIGHTS = {
    "tech_shift_identification": 0.35,
    "investor_question_quality": 0.30,
    "recency_and_specificity": 0.20,
    "briefing_readability": 0.15,
}


def test(result):
    return evaluate(result, TASK_INSTRUCTION, USER_PROMPT_TEMPLATE, DIMENSION_WEIGHTS)
