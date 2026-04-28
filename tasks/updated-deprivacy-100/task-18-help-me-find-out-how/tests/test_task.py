"""LLM-as-judge evaluator for task-18-help-me-find-out-how.

Category: Finance & Economics
Task: Help me find out how many subsidiaries HeidelbergCement has that are currently registered and in existence. Please list them all.
"""

import sys
sys.path.insert(0, "/harness")
from evaluator import run_judge

TASK_INSTRUCTION = 'Help me find out how many subsidiaries HeidelbergCement has that are currently registered and in existence. Please list them all.'

SYSTEM_PROMPT_EXTRA = "Your job is to assess whether an AI agent successfully completed a research task about a company's corporate subsidiaries."

CONSTRAINTS = """- Company: HeidelbergCement specifically — not a parent company, competitor, or subsidiary itself
- Status filter: subsidiaries that are currently registered and in existence — not dissolved, liquidated, or historical entities
- Output format: a list of all subsidiaries (not a narrative description, not a count alone)
- Research: agent must actively look up current subsidiary information rather than relying solely on prior knowledge"""

EVIDENCE_QUESTIONS = """Carefully read the agent response and trace. Then answer each of the following questions with a direct observation — do not infer or assume:

- Did the agent actively search for HeidelbergCement subsidiary information? Cite evidence from the trace or response.
- How many subsidiaries did the agent list? Provide the count.
- Are the listed entities confirmed as currently registered and in existence, or does the agent include dissolved/historical entities?
- Did the agent acknowledge any limitations (e.g. partial list, data source limitations)?
- Is the output structured as a list, or presented as prose/narrative?"""

DIMENSION_RUBRICS = """Score each dimension from 1 to 5 using the rubrics below. Each score must be justified by specific evidence from Step 1.

#### A. Research Execution
Did the agent actively search for HeidelbergCement subsidiary information?

5 — Clear evidence in trace and/or response that the agent searched for and retrieved HeidelbergCement subsidiary data from a credible source (e.g. company filings, corporate registry, official website).
4 — Agent searched for subsidiary information but used a less authoritative source (e.g. general web search, Wikipedia) without verifying against official records.
3 — Ambiguous: response mentions subsidiary information but trace shows no clear research activity, or agent combined prior knowledge with light research.
2 — Agent described what subsidiaries HeidelbergCement might have without performing an actual search.
1 — Agent did not perform any research; response is hallucinated or generated from prior knowledge only.

#### B. Completeness
How many subsidiaries were found and listed?

5 — Agent listed a substantial number of subsidiaries (10+) with clear sourcing; acknowledged if the list may be partial.
4 — Agent listed several subsidiaries (5–9) with sourcing; minor gaps acknowledged.
3 — Agent listed a small number of subsidiaries (2–4); list is clearly incomplete but some real entities are present.
2 — Agent listed only 1 subsidiary or provided only a count without a list.
1 — No subsidiaries listed; agent only described the company or refused the task.

#### C. Registration Status Accuracy
Are the listed subsidiaries confirmed as currently registered and in existence?

5 — All listed subsidiaries are explicitly confirmed as currently active/registered; dissolved or inactive entities are excluded or clearly flagged.
4 — Most listed subsidiaries appear to be active; 1–2 may be ambiguous but none are clearly dissolved.
3 — Agent listed subsidiaries without verifying registration status; mix of active and potentially inactive entities possible.
2 — Agent included entities that are known to be dissolved or did not address the registration status requirement.
1 — Registration status requirement was ignored entirely; list contains historical or dissolved entities without distinction.

#### D. Presentation Quality
Is the output a clear, usable list of subsidiaries?

5 — Output is a well-organized list with subsidiary names clearly presented; each entry is distinct and easy to reference.
4 — Output is a list with minor formatting issues or slight redundancy between items.
3 — Output contains subsidiary information but is presented as prose or a loosely structured summary rather than a clean list.
2 — Output mentions subsidiaries but does not present them as an actual list; or list has fewer than 3 items with no explanation.
1 — Output is a refusal, an error message, or contains no subsidiary information."""

DIMENSION_WEIGHTS = {
    'research_execution': 0.25,
    'completeness': 0.35,
    'registration_status_accuracy': 0.25,
    'presentation_quality': 0.15,
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
