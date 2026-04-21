import sys
sys.path.insert(0, "/harness")
from evaluator import evaluate

TASK_INSTRUCTION = ("""Compile a comprehensive intelligence dossier on the target online persona, covering identity mapping, content forensics, platform and network audit, psychological profiling, and asset valuation.""")

USER_PROMPT_TEMPLATE = ("""## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Scope: public information only — social media profiles, public posts, published content
- Coverage: identity mapping, content analysis, platform presence, network connections, estimated reach/value
- Structure: organized report with all five sections
- Ethics: public information only; no private data, no hacking

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent research the target persona across multiple platforms?
- Were all five sections covered (identity, content, platform, network, valuation)?
- Is the information clearly sourced from public data?
- Is the report well-organized with each section distinct?
- Is asset/reach valuation grounded in methodology?

### Step 2: Dimension Scoring

#### A. Research Breadth (0.25)
Did the agent research across multiple platforms?

5 — Searched 4+ platforms (Twitter/X, LinkedIn, Instagram, YouTube, etc.) with specific findings from each.
4 — 2-3 platforms researched.
3 — 1-2 platforms with limited findings.
2 — Single source research.
1 — No research.

#### B. Section Completeness (0.35)
Were all five dossier sections covered?

5 — All five: identity mapping, content forensics, platform audit, network analysis, asset valuation — all present with substance.
4 — Four of five sections.
3 — Three of five sections.
2 — One or two sections.
1 — No structured sections.

#### C. Insight Depth (0.25)
Is the analysis insightful beyond surface facts?

5 — Deep analysis: content patterns, audience psychology, influence network structure, monetization signals.
4 — Good depth in most sections.
3 — Surface-level facts without analytical insight.
2 — Mostly a list of profile details.
1 — No analytical depth.

#### D. Report Structure (0.15)
Is the dossier well-organized and professional?

5 — Clear sections with headers, findings, and methodology notes; professional tone.
4 — Well-organized but missing some formatting polish.
3 — Content present but loosely organized.
2 — Wall of text.
1 — No structure.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "research_breadth": <1-5>,
  "section_completeness": <1-5>,
  "insight_depth": <1-5>,
  "report_structure": <1-5>,
  "dimension_reasoning": {{
    "research_breadth": "<one sentence citing specific evidence>",
    "section_completeness": "<one sentence citing specific evidence>",
    "insight_depth": "<one sentence citing specific evidence>",
    "report_structure": "<one sentence citing specific evidence>",
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
""")

DIMENSION_WEIGHTS = {
    "research_breadth": 0.25,
    "section_completeness": 0.35,
    "insight_depth": 0.25,
    "report_structure": 0.15,
}


def test(result):
    return evaluate(result, TASK_INSTRUCTION, USER_PROMPT_TEMPLATE, DIMENSION_WEIGHTS)
