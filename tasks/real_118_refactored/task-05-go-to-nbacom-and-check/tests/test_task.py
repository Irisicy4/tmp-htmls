import sys
sys.path.insert(0, "/harness")
from evaluator import evaluate

TASK_INSTRUCTION = ("go to nba.com and check for Jayson Tatum's current 3-point status")

USER_PROMPT_TEMPLATE = ("""## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Platform: nba.com specifically — not ESPN, Basketball-Reference, or general web search
- Player: Jayson Tatum specifically
- Stat: 3-point shooting status — this could mean current season 3PT stats (attempts, makes, percentage), all-time 3PT record progress, or both; the agent should surface whatever is most current and relevant
- Recency: the data must be current (from this season or the most recent game), not career averages alone

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to nba.com? Cite evidence from the trace or response.
- What 3-point statistics did the agent report for Jayson Tatum? List them explicitly (e.g. 3PM, 3PA, 3P%).
- Are the stats current (this season or recent game), or are they career/historical only?
- Did the agent interpret "3-point status" meaningfully (e.g. ranking, record chase, season performance)?
- Did the agent fabricate or estimate stats, or retrieve them from the live site?

### Step 2: Dimension Scoring

#### A. Platform Execution
Did the agent actually navigate nba.com as instructed?

5 — Clear evidence agent visited nba.com and retrieved live stats from the page.
4 — Agent used nba.com but retrieved limited data (e.g. only landed on player profile without drilling into stats).
3 — Agent retrieved Tatum stats but from a different source (ESPN, Basketball-Reference, etc.).
2 — Agent searched the web generally without visiting a specific stats platform.
1 — No navigation performed; stats appear fabricated or from prior knowledge.

#### B. Stat Accuracy & Recency
Are the reported 3-point statistics accurate and current?

5 — Stats are clearly sourced from nba.com, include current season figures, and are specific (3PM, 3PA, 3P% with game/season context).
4 — Stats are current but one figure is missing or slightly vague (e.g. percentage given but not attempts).
3 — Stats are present but only career averages or outdated; no current season breakdown.
2 — Stats are mentioned but vague (e.g. "he's a great 3-point shooter") without specific numbers for Tatum.
1 — No stats provided, or stats are clearly incorrect.

#### C. Interpretive Completeness
Did the agent interpret "3-point status" in a meaningful, useful way?

5 — Agent reports current season 3PT stats AND contextualises them (e.g. league ranking, record chase, recent game performance).
4 — Agent reports current stats with minimal context (numbers only, no ranking or significance).
3 — Agent reports only one aspect (e.g. all-time record progress but no season stats, or vice versa).
2 — Agent reports generic information about Tatum being a good 3-point shooter without specific current status.
1 — Response does not address 3-point status at all.

#### D. Response Clarity
Is the output concise and directly useful?

5 — Stats are clearly presented, labelled, and easy to read at a glance.
4 — Stats are present but formatting is slightly cluttered or requires parsing.
3 — Stats are buried in a long paragraph; requires effort to extract the key numbers.
2 — Response is mostly narrative with stats hard to find.
1 — No useful stats present.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "platform_execution": <1-5>,
  "stat_accuracy_recency": <1-5>,
  "interpretive_completeness": <1-5>,
  "response_clarity": <1-5>,
  "dimension_reasoning": {{
    "platform_execution": "<one sentence citing specific evidence>",
    "stat_accuracy_recency": "<one sentence citing specific evidence>",
    "interpretive_completeness": "<one sentence citing specific evidence>",
    "response_clarity": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
""")

DIMENSION_WEIGHTS = {
    "platform_execution":        0.30,
    "stat_accuracy_recency":     0.35,
    "interpretive_completeness": 0.20,
    "response_clarity":          0.15,
}


def test(result):
    return evaluate(result, TASK_INSTRUCTION, USER_PROMPT_TEMPLATE, DIMENSION_WEIGHTS)
