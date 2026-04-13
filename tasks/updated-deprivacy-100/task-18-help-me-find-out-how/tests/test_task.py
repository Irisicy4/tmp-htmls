"""
LLM-as-judge evaluator for EvolveBench task-18.

Category: Finance & Economics
Approach: Hybrid — category-level rubric axes (Finance & Economics) + task-specific constraint injection.
Design grounded in:
  - AutoMetrics: axes-from-examples pipeline, observable 5-point rubric anchors
  - SWE-ABS: forced analysis before verdict, orthogonal sub-dimensions, structured output tag

Task:
  "Help me find out how many subsidiaries HeidelbergCement has that are currently
   registered and in existence. Please list them all."
"""

import os
import json
import re

# ---------------------------------------------------------------------------
# Task metadata
# ---------------------------------------------------------------------------

TASK_INSTRUCTION = (
    "Help me find out how many subsidiaries HeidelbergCement has that are currently "
    "registered and in existence. Please list them all."
)

PASS_THRESHOLD = 3.0

# ---------------------------------------------------------------------------
# Judge prompt
#
# Structure (from SWE-ABS pattern):
#   1. Forced analysis section — prevents jumping to verdict
#   2. Per-dimension scoring — orthogonal axes, each with observable anchors
#   3. Structured output tag — enables reliable parsing + majority voting
#
# Axes (Finance & Economics category):
#   A. Research Execution        — did the agent actively search for subsidiary information?
#   B. Completeness              — how many subsidiaries were found and listed?
#   C. Registration Status       — are the listed subsidiaries confirmed as currently registered and in existence?
#   D. Presentation Quality      — is the output a clear, usable list of subsidiaries?
#
# Task-specific constraints injected:
#   - Company: HeidelbergCement specifically
#   - Status filter: currently registered and in existence (not dissolved, not historical)
#   - Output format: a list of all subsidiaries
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a research task about a company's corporate subsidiaries.

You will be given:
1. The task instruction the agent was given
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

You must evaluate the agent's performance across four dimensions. Follow the evaluation steps exactly in order."""


USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Company: HeidelbergCement specifically — not a parent company, competitor, or subsidiary itself
- Status filter: subsidiaries that are currently registered and in existence — not dissolved, liquidated, or historical entities
- Output format: a list of all subsidiaries (not a narrative description, not a count alone)
- Research: agent must actively look up current subsidiary information rather than relying solely on prior knowledge

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
Carefully read the agent response and trace. Then answer each of the following questions with a direct observation — do not infer or assume:

- Did the agent actively search for HeidelbergCement subsidiary information? Cite evidence from the trace or response.
- How many subsidiaries did the agent list? Provide the count.
- Are the listed entities confirmed as currently registered and in existence, or does the agent include dissolved/historical entities?
- Did the agent acknowledge any limitations (e.g. partial list, data source limitations)?
- Is the output structured as a list, or presented as prose/narrative?

### Step 2: Dimension Scoring
Score each dimension from 1 to 5 using the rubrics below. Each score must be justified by specific evidence from Step 1.

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
1 — Output is a refusal, an error message, or contains no subsidiary information.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags, exactly as shown:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising your Step 1 findings>",
  "research_execution": <1-5>,
  "completeness": <1-5>,
  "registration_status_accuracy": <1-5>,
  "presentation_quality": <1-5>,
  "dimension_reasoning": {{
    "research_execution": "<one sentence citing specific evidence>",
    "completeness": "<one sentence citing specific evidence>",
    "registration_status_accuracy": "<one sentence citing specific evidence>",
    "presentation_quality": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

# Dimension weights — Completeness is highest because the task explicitly asks
# to list all subsidiaries. Registration Status Accuracy is second because
# the task specifically requires currently registered and in existence entities.
DIMENSION_WEIGHTS = {
    "research_execution":          0.25,
    "completeness":                0.35,
    "registration_status_accuracy": 0.25,
    "presentation_quality":        0.15,
}

DIMENSIONS = list(DIMENSION_WEIGHTS.keys())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_response(result: dict) -> str:
    task_result = result.get("task_result") or ""
    if isinstance(task_result, str) and task_result.strip():
        return task_result
    for message in reversed(result.get("conversation") or []):
        if not isinstance(message, dict):
            continue
        if message.get("role") == "assistant":
            content = message.get("content") or ""
            if isinstance(content, str) and len(content) > 20:
                return content
    return ""


def _parse_answer_tag(text):
    m = re.search(r"<Answer>(.*?)</Answer>", text, re.DOTALL | re.IGNORECASE)
    if m:
        try: return json.loads(m.group(1).strip())
        except Exception: pass
    try: return json.loads(text.strip())
    except Exception: pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try: return json.loads(m.group())
        except Exception: pass
    return None

def _call_judge_once(agent_response: str, execution_summary: str) -> dict | None:
    """Single judge call. Returns parsed dict or None on failure."""
    try:
        import openai
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

        user_content = USER_PROMPT_TEMPLATE.format(
            task_instruction=TASK_INSTRUCTION,
            agent_response=agent_response,
            execution_summary=execution_summary or "Not available.",
        )

        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_content},
            ],
            max_tokens=1024,
        )
        raw = completion.choices[0].message.content
        return _parse_answer_tag(raw)
    except Exception as e:
        return {"error": str(e)}


def _majority_vote(votes: list[dict]) -> dict:
    """
    Aggregate up to 3 judge calls via majority vote on each dimension.
    Overall score is recomputed from voted dimension scores using weights.
    Used for borderline cases (overall_score within 0.5 of threshold).
    """
    valid = [v for v in votes if v and "error" not in v and all(d in v for d in DIMENSIONS)]
    if not valid:
        return votes[0] if votes else {"error": "All judge calls failed"}

    aggregated = {}
    for dim in DIMENSIONS:
        scores = [v[dim] for v in valid]
        aggregated[dim] = sorted(scores)[len(scores) // 2]

    overall = sum(aggregated[d] * DIMENSION_WEIGHTS[d] for d in DIMENSIONS)
    aggregated["overall_score"] = round(overall, 2)
    aggregated["passed"] = overall >= PASS_THRESHOLD

    median_call = sorted(valid, key=lambda v: abs(v.get("overall_score", 0) - overall))[0]
    aggregated["evidence_summary"] = median_call.get("evidence_summary", "")
    aggregated["dimension_reasoning"] = median_call.get("dimension_reasoning", {})
    aggregated["_votes_used"] = len(valid)

    return aggregated


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def test(result: dict) -> dict:
    agent_response = _extract_response(result)
    execution_summary = result.get("execution_summary", "")

    if not agent_response.strip():
        return {
            "passed": False,
            "feedback": "No response found from agent.",
            "details": {"task_completed": result.get("status") == "success"},
        }

    first_call = _call_judge_once(agent_response, execution_summary)

    if first_call and "error" not in first_call:
        overall = first_call.get("overall_score", 0)
        borderline = abs(float(overall) - PASS_THRESHOLD) <= 0.5

        if borderline:
            call2 = _call_judge_once(agent_response, execution_summary)
            call3 = _call_judge_once(agent_response, execution_summary)
            scores = _majority_vote([first_call, call2, call3])
        else:
            scores = first_call
            scores["_votes_used"] = 1
    else:
        scores = first_call or {"error": "Judge call failed", "overall_score": 0}

    overall = scores.get("overall_score", 0)
    passed = scores.get("passed", float(overall) >= PASS_THRESHOLD)

    feedback_lines = [f"Overall score: {overall}/5  (threshold: {PASS_THRESHOLD})"]
    for dim in DIMENSIONS:
        if dim in scores:
            feedback_lines.append(f"  {dim}: {scores[dim]}/5")
    if scores.get("evidence_summary"):
        feedback_lines.append(f"\nEvidence summary: {scores['evidence_summary']}")
    reasoning = scores.get("dimension_reasoning", {})
    if reasoning:
        feedback_lines.append("\nDimension reasoning:")
        for dim, reason in reasoning.items():
            feedback_lines.append(f"  {dim}: {reason}")
    if scores.get("_votes_used", 1) > 1:
        feedback_lines.append(f"\n(Borderline case: {scores['_votes_used']} judge calls used, majority vote applied)")

    return {
        "passed": bool(passed),
        "feedback": "\n".join(feedback_lines),
        "details": {
            "task_completed": result.get("status") == "success",
            "overall_score": overall,
            "dimension_scores": {d: scores.get(d) for d in DIMENSIONS},
            "evidence_summary": scores.get("evidence_summary"),
            "dimension_reasoning": scores.get("dimension_reasoning"),
            "pass_threshold": PASS_THRESHOLD,
            "votes_used": scores.get("_votes_used", 1),
        },
    }