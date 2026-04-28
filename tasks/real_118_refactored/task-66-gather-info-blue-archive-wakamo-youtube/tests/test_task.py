import sys
sys.path.insert(0, "/harness")
from evaluator import evaluate

TASK_INSTRUCTION = ("""For the smartphone game Blue Archive's Total Assault mode featuring Wakamo Hovercraft on Torment difficulty, gather and organize information about effective team compositions, focusing primarily on YouTube videos.""")

USER_PROMPT_TEMPLATE = ("""## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Game: Blue Archive specifically
- Mode: Total Assault (总力战)
- Boss: Wakamo Hovercraft on Torment difficulty
- Primary source: YouTube videos — not just text guides
- Output: organized compilation of team compositions with source references

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent search YouTube for Wakamo Hovercraft Total Assault guides?
- What specific team compositions were found?
- Are YouTube video sources cited?
- Is information organized by team composition type or ranking?
- Is difficulty (Torment) confirmed in the sources?

### Step 2: Dimension Scoring

#### A. Youtube Research (0.3)
Did the agent find relevant YouTube content?

5 — Multiple YouTube videos found specifically for Wakamo Hovercraft Torment with video titles/channels cited.
4 — YouTube content found but not all specifically Torment difficulty.
3 — YouTube searched but limited results; supplemented with other sources.
2 — General web search instead of YouTube focus.
1 — No YouTube research.

#### B. Team Comp Detail (0.35)
Are specific team compositions reported?

5 — Multiple team compositions with character names, equipment, and roles (DPS/support/healer).
4 — Team compositions listed with character names but limited detail.
3 — Character names mentioned without team context.
2 — Vague descriptions (e.g. 'use strong characters') without specifics.
1 — No team compositions found.

#### C. Source Organization (0.2)
Is content organized and sources cited?

5 — Content organized by composition type with YouTube source links or channel names.
4 — Organized but sources only partially cited.
3 — Information present but not well-organized.
2 — Dump of information without structure.
1 — No organization or sourcing.

#### D. Difficulty Accuracy (0.15)
Is the content specifically for Torment difficulty?

5 — All content confirmed for Torment difficulty explicitly.
4 — Mostly Torment but some content from lower difficulties.
3 — Difficulty level unclear for most content.
2 — Content from wrong difficulty.
1 — Difficulty not addressed.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "dimension_reasoning": {{
    "youtube_research": "<one sentence citing specific evidence>",
    "team_comp_detail": "<one sentence citing specific evidence>",
    "source_organization": "<one sentence citing specific evidence>",
    "difficulty_accuracy": "<one sentence citing specific evidence>",
  }},
  "youtube_research": <1-5>,
  "team_comp_detail": <1-5>,
  "source_organization": <1-5>,
  "difficulty_accuracy": <1-5>,
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
""")

DIMENSION_WEIGHTS = {
    "youtube_research": 0.3,
    "team_comp_detail": 0.35,
    "source_organization": 0.2,
    "difficulty_accuracy": 0.15,
}


def test(result):
    return evaluate(result, TASK_INSTRUCTION, USER_PROMPT_TEMPLATE, DIMENSION_WEIGHTS)
