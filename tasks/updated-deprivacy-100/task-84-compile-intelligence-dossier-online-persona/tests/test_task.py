"""LLM-as-judge evaluator for task-84-compile-intelligence-dossier-online-persona.

Category: Marketing & Analytics
Task: Compile a comprehensive intelligence dossier on the target online persona, covering identity mapping, content forensics, platform and network audit, psychological profiling, and asset valuation.
"""

import sys
sys.path.insert(0, "/harness")
from evaluator import run_judge

TASK_INSTRUCTION = 'Compile a comprehensive intelligence dossier on the target online persona, covering identity mapping, content forensics, platform and network audit, psychological profiling, and asset valuation.'

SYSTEM_PROMPT_EXTRA = 'Your job is to assess whether an AI agent successfully completed a browser-based task.\n\nThe task involves researching a specific online persona across platforms and compiling a structured intelligence report. This is a legitimate competitive intelligence or OSINT research task for marketing/analytics purposes.'

CONSTRAINTS = """- Scope: public information only — social media profiles, public posts, published content
- Coverage: identity mapping, content analysis, platform presence, network connections, estimated reach/value
- Structure: organized report with all five sections
- Ethics: public information only; no private data, no hacking"""

EVIDENCE_QUESTIONS = """- Did the agent research the target persona across multiple platforms?
- Were all five sections covered (identity, content, platform, network, valuation)?
- Is the information clearly sourced from public data?
- Is the report well-organized with each section distinct?
- Is asset/reach valuation grounded in methodology?"""

DIMENSION_RUBRICS = """#### A. Research Breadth (0.25)
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
1 — No structure."""

DIMENSION_WEIGHTS = {
    'research_breadth': 0.25,
    'section_completeness': 0.35,
    'insight_depth': 0.25,
    'report_structure': 0.15,
}


def test(result):
    return run_judge(
        result,
        task_instruction=TASK_INSTRUCTION,
        system_prompt_extra=SYSTEM_PROMPT_EXTRA,
        constraints=CONSTRAINTS,
        evidence_questions=EVIDENCE_QUESTIONS,
        dimension_rubrics=DIMENSION_RUBRICS,
        dimension_weights=DIMENSION_WEIGHTS,
    )
