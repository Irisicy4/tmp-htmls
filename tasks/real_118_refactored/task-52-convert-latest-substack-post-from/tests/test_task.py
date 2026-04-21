import sys
sys.path.insert(0, "/harness")
from evaluator import evaluate

TASK_INSTRUCTION = ("""Convert the latest Substack post from The Biblical Man (biblicalman.substack.com) into: (1) an Instagram Carousel with 10 slides, caption, and hashtags, (2) a TikTok Carousel with 7 slides, caption, and hashtags, and (3) 5 Instagram Story frames. Follow the hook-build-payoff-CTA arc and use a confrontational, declarative voice.""")

USER_PROMPT_TEMPLATE = ("""## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Source: must fetch actual content from biblicalman.substack.com — not invented content
- Instagram Carousel: exactly 10 slides + caption + hashtags
- TikTok Carousel: exactly 7 slides + caption + hashtags
- Instagram Stories: 5 frames
- Arc: Hook → Build → Build → Build → Bridge → Payoff → CTA
- Voice: confrontational, declarative, no therapeutic language

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent fetch the Substack post from biblicalman.substack.com?
- Were all three output formats produced (IG carousel, TikTok carousel, IG Stories)?
- Do slide counts match (10, 7, 5)?
- Does the content follow the hook-build-payoff-CTA arc?
- Is the voice confrontational and declarative?

### Step 2: Dimension Scoring

#### A. Content Sourcing (0.25)
Did the agent fetch actual content from biblicalman.substack.com?

5 — Agent navigated to biblicalman.substack.com, fetched the latest post, and based content on it.
4 — Agent fetched content but from a cached or slightly outdated version.
3 — Agent referenced the Substack but content seems partially invented.
2 — Agent created generic Biblical Man-style content without fetching actual post.
1 — No sourcing from biblicalman.substack.com.

#### B. Format Completeness (0.3)
Were all three output formats produced with correct slide counts?

5 — All three formats present: 10-slide IG carousel, 7-slide TikTok carousel, 5 IG Story frames.
4 — Two of three formats present, or slide counts slightly off.
3 — One format present and complete.
2 — Formats present but significantly incomplete (e.g. 3 slides instead of 10).
1 — No carousel content produced.

#### C. Narrative Arc (0.25)
Does the content follow the hook-build-payoff-CTA arc?

5 — Clear arc: scroll-stopping hook → tension-building slides → bridge → payoff → CTA.
4 — Arc mostly followed but one element is weak or missing.
3 — Some arc structure but not consistently applied.
2 — Content is good but arc is not intentionally structured.
1 — No arc — slides are random or disjointed.

#### D. Voice Quality (0.2)
Is the voice confrontational, declarative, and on-brand?

5 — Short punchy sentences, declarative statements, physical/visceral language, no emojis in slides, no therapeutic language.
4 — Mostly on-brand but a few slides drift soft.
3 — Mix of confrontational and soft voice.
2 — Mostly motivational-poster tone rather than confrontational.
1 — Wrong voice entirely — too soft, too casual, or too corporate.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "content_sourcing": <1-5>,
  "format_completeness": <1-5>,
  "narrative_arc": <1-5>,
  "voice_quality": <1-5>,
  "dimension_reasoning": {{
    "content_sourcing": "<one sentence citing specific evidence>",
    "format_completeness": "<one sentence citing specific evidence>",
    "narrative_arc": "<one sentence citing specific evidence>",
    "voice_quality": "<one sentence citing specific evidence>",
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
""")

DIMENSION_WEIGHTS = {
    "content_sourcing": 0.25,
    "format_completeness": 0.3,
    "narrative_arc": 0.25,
    "voice_quality": 0.2,
}


def test(result):
    return evaluate(result, TASK_INSTRUCTION, USER_PROMPT_TEMPLATE, DIMENSION_WEIGHTS)
