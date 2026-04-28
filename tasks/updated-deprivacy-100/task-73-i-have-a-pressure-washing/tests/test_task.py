"""LLM-as-judge evaluator for task-73-i-have-a-pressure-washing.

Category: Marketing & Analytics
Task: I have a pressure washing business that operates in jacksonvile florida. i want to start a local newsletter to own my distristribution, and market my business essentially for free. Please find best pr
"""

import sys
sys.path.insert(0, "/harness")
from evaluator import run_judge

TASK_INSTRUCTION = 'I have a pressure washing business that operates in jacksonvile florida. i want to start a local newsletter to own my distristribution, and market my business essentially for free. Please find best practices to start a local newsletter here, what content ideas people would like best, and outline real growth strategies to start the newsletter. I want to follow the naptown scoop format. find what is currently working in the local newsletter idustry and outline strategies'

SYSTEM_PROMPT_EXTRA = 'Your job is to assess whether an AI agent successfully completed a browser-based task.\n\nThe task asks the agent to research and compile a practical guide for starting a local newsletter for a pressure washing business in Jacksonville, Florida. The agent must find best practices, content ideas suited to the local market, and growth strategies — specifically referencing the Naptown Scoop as a format model.'

CONSTRAINTS = """- Business context: pressure washing company in Jacksonville, Florida
- Goal: start a local newsletter for free distribution/marketing
- Must cover: best practices for launching a local newsletter
- Must cover: specific content ideas (ideally relevant to Jacksonville community or home services)
- Must cover: concrete growth strategies for building the subscriber base
- Format reference: Naptown Scoop model (local-focused, community-driven newsletter)
- Research requirement: findings from what is currently working in the local newsletter industry"""

EVIDENCE_QUESTIONS = """- Did the agent research local newsletter best practices? What sources or examples were found?
- Are content ideas provided and are they relevant to Jacksonville or home services?
- Are growth strategies specific and actionable (e.g. platforms, referral tactics, cross-promotions)?
- Did the agent reference or analyze the Naptown Scoop or a comparable local newsletter model?"""

DIMENSION_RUBRICS = """#### A. Best Practices Coverage (0.30)
Did the agent find specific, actionable best practices for launching a local newsletter?

5 — 5 or more concrete best practices with specific tools, platforms (e.g. Beehiiv, Substack, Mailchimp), or examples drawn from the local newsletter industry.
4 — 3-4 solid best practices with some platform-level or operational specifics.
3 — General best practices mentioned with limited actionable detail.
2 — Very generic newsletter advice not tailored to local or small-business context.
1 — No best practices provided.

#### B. Content Strategy (0.25)
Are content ideas specific and relevant to Jacksonville, FL and/or the pressure washing business?

5 — 5 or more content ideas, at least some tied to Jacksonville community (local events, neighborhoods, home maintenance tips, seasonal demand) or the pressure washing trade.
4 — 3-4 relevant content ideas with some local or business-specific angle.
3 — Generic content ideas applicable to any local newsletter without Jacksonville or trade specificity.
2 — Very few (1-2) or entirely irrelevant content ideas.
1 — No content strategy provided.

#### C. Growth Strategies (0.25)
Are growth strategies concrete and actionable?

5 — 3 or more specific growth tactics with implementation detail (e.g. referral programs with incentives, partnerships with local real estate agents or HOAs, cross-promotion with complementary businesses, newsletter aggregator listings).
4 — 2-3 growth strategies with reasonable implementation detail.
3 — General growth tactics (e.g. "post on social media") without specifics.
2 — Vague growth mentions without strategies.
1 — No growth strategies provided.

#### D. Naptown Scoop Format Reference (0.20)
Did the agent reference and apply the Naptown Scoop format model?

5 — Specifically describes key Naptown Scoop format elements (e.g. local event roundups, sponsor model, community-first tone, consistent cadence) and directly adapts them to the Jacksonville pressure washing context.
4 — Mentions Naptown Scoop by name and draws at least one relevant structural or strategic lesson.
3 — Naptown Scoop referenced but analysis is superficial (name-drop only without explanation).
2 — No mention of Naptown Scoop but analyzes another successful local newsletter as a model.
1 — No reference to Naptown Scoop or any comparable local newsletter example."""

DIMENSION_WEIGHTS = {
    'best_practices_coverage': 0.3,
    'content_strategy': 0.25,
    'growth_strategies': 0.25,
    'naptown_scoop_reference': 0.2,
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
