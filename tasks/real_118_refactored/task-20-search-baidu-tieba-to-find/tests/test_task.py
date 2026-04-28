import sys
sys.path.insert(0, "/harness")
from evaluator import evaluate

TASK_INSTRUCTION = ("Search Baidu Tieba to find and compile a cultivation/system progression guide for the mobile game 境界刀鸣 (Jingjie Daoming).")

USER_PROMPT_TEMPLATE = ("""## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Platform: Baidu Tieba (百度贴吧) specifically — not Baidu search, Bilibili, or other platforms
- Game: 境界刀鸣 (Jingjie Daoming) specifically — a Chinese mobile RPG
- Content type: cultivation/system progression guide — covering how the game's character advancement, skill systems, or resource progression works
- Output: a compiled guide (not just a list of links), synthesised from Tieba posts

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate Baidu Tieba? Cite evidence from trace or response.
- What game-specific content did the agent find? Describe briefly.
- Is the content about 境界刀鸣 specifically?
- Does the content cover cultivation/progression systems (not just general gameplay tips)?
- Did the agent compile content from multiple posts, or just summarise one?
- Is the output a usable guide or just a description of what was found?

### Step 2: Dimension Scoring

#### A. Platform Execution
Did the agent actually navigate Baidu Tieba as instructed?

5 — Clear evidence agent searched and browsed Baidu Tieba for 境界刀鸣 content; multiple posts accessed.
4 — Agent used Baidu Tieba but accessed only 1–2 posts, or trace confirms platform but response doesn't cite it.
3 — Ambiguous: response mentions Baidu Tieba but trace shows no navigation, or agent used Tieba alongside other platforms.
2 — Agent used Baidu search or a different platform instead of Tieba.
1 — No search performed; content appears fabricated or from prior knowledge.

#### B. Game Specificity
Is the content specifically about 境界刀鸣?

5 — All content is clearly about 境界刀鸣 (game name appears in sources; content references game-specific mechanics).
4 — Most content is game-specific; 1 section is generic mobile RPG advice that could apply to any game.
3 — Content is broadly relevant to Chinese mobile RPG cultivation systems but lacks 境界刀鸣-specific detail.
2 — Content is generic mobile game advice with only superficial mention of the game name.
1 — Content is off-topic or clearly about a different game.

#### C. Guide Depth
Does the compiled content constitute a useful cultivation/progression guide?

5 — Guide covers 3+ distinct progression systems (e.g. character levels, skill trees, equipment upgrade, resource farming) with specific mechanics and tips.
4 — Guide covers 2–3 systems with useful detail; one area is shallow.
3 — Guide covers 1–2 systems; useful but limited in scope.
2 — Content mentions cultivation/progression but provides only surface-level overview without actionable guidance.
1 — No guide content; output is a description of the search process or a list of post titles only.

#### D. Compilation Quality
Is the output a well-organised, readable guide compiled from multiple sources?

5 — Content is synthesised from 2+ Tieba posts into a coherent, structured guide with clear sections.
4 — Content is organised but lightly edited from sources; structure is present but synthesis is minimal.
3 — Content is assembled but reads as disconnected fragments from different posts.
2 — Output is a single post's content without compilation or a list of URLs without content.
1 — No compiled output; agent only described what it searched for.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "dimension_reasoning": {{
    "platform_execution": "<one sentence citing specific evidence>",
    "game_specificity": "<one sentence citing specific evidence>",
    "guide_depth": "<one sentence citing specific evidence>",
    "compilation_quality": "<one sentence citing specific evidence>"
  }},
  "platform_execution": <1-5>,
  "game_specificity": <1-5>,
  "guide_depth": <1-5>,
  "compilation_quality": <1-5>,
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
""")

DIMENSION_WEIGHTS = {
    "platform_execution": 0.35,
    "game_specificity": 0.22,
    "guide_depth": 0.26,
    "compilation_quality": 0.17,
}


def test(result):
    return evaluate(result, TASK_INSTRUCTION, USER_PROMPT_TEMPLATE, DIMENSION_WEIGHTS)
