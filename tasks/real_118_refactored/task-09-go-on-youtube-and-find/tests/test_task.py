import sys
sys.path.insert(0, "/harness")
from evaluator import evaluate

TASK_INSTRUCTION = ("""go on youtube and find videos of silent hill 3 that explains the story""")

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Platform: YouTube specifically (not a general web search)
- Topic: Silent Hill 3 story/narrative explanation (lore videos, not pure gameplay or boss guides)
- Required output: specific video titles, channels, or links — not just a description of what to search for

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
Carefully read the agent response and trace, then answer each question with direct observations:

- Did the agent navigate to YouTube or perform a YouTube search?
- Were specific video titles or channel names identified?
- Are the videos clearly about Silent Hill 3 story/narrative (not unrelated content)?
- Did the agent provide enough detail for the user to find and watch the videos?
- Were any links, video IDs, or other verifiable identifiers included?

### Step 2: Dimension Scoring
Score each dimension 1–5 using the rubrics below.

#### A. Task Completion (0.36)
Did the agent find and present YouTube videos of Silent Hill 3 story explanations?

5 — Agent navigated to YouTube, found multiple relevant story-explanation videos, and presented them with titles/channels/links.
4 — Agent found relevant videos but presentation is incomplete (e.g. missing links, or one video is off-topic).
3 — Agent partially completed the task (e.g. described search results without identifying specific videos, or found only one).
2 — Agent attempted but failed to find relevant videos, or only described how to search rather than doing it.
1 — Agent did not attempt the task or abandoned it.

#### B. Information Quality (0.27)
Are the videos real, relevant, and accurately described?

5 — Videos are clearly relevant to Silent Hill 3 story/lore; titles and channels are plausible and descriptions match the task.
4 — Mostly relevant; minor inaccuracies or one video is questionable.
3 — Some relevant content but vague, unverifiable, or mixed with off-topic results.
2 — Information appears hallucinated or inaccurate (non-existent videos, wrong game).
1 — No real video information provided.

#### C. Response Quality (0.18)
Is the response well-organised and useful to the user?

5 — Clear, structured list of videos with enough detail (title, channel, description) for the user to act immediately.
4 — Adequate structure; some useful details missing.
3 — Response is present but disorganised or requires extra effort to interpret.
2 — Sparse or hard to use.
1 — Not useful.

#### D. Completeness (0.18)
Did the agent address all aspects of the request?

5 — Found multiple videos (2+), all relevant to the story, with actionable details.
4 — Found videos but quantity or coverage is slightly short.
3 — Task partially addressed — e.g. only one video, or story focus is unclear.
2 — Significant gaps; most of the request unaddressed.
1 — Request not addressed.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising your Step 1 findings>",
  "task_completion": <1-5>,
  "information_quality": <1-5>,
  "response_quality": <1-5>,
  "completeness": <1-5>,
  "dimension_reasoning": {{
    "task_completion": "<one sentence citing specific evidence>",
    "information_quality": "<one sentence citing specific evidence>",
    "response_quality": "<one sentence citing specific evidence>",
    "completeness": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "task_completion":    0.36,
    "information_quality": 0.27,
    "response_quality":   0.18,
    "completeness":       0.18,
}


def test(result):
    return evaluate(result, TASK_INSTRUCTION, USER_PROMPT_TEMPLATE, DIMENSION_WEIGHTS)
