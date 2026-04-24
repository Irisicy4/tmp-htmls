"""
LLM-as-judge evaluator for EvolveBench task-01.

Category: Shopping
Approach: Hybrid — category-level rubric axes (Shopping) + task-specific constraint injection.
Design grounded in:
  - AutoMetrics: axes-from-examples pipeline, observable 5-point rubric anchors
  - SWE-ABS: forced analysis before verdict, orthogonal sub-dimensions, structured output tag

Task:
  "im looking for a backpack under $75 that has all the features of this one:
   https://www.amazon.com/dp/B09YRC9Y3G please do some research and find 3-5 options
   and summarize their key features and prices, comparing them to the original."
"""

import os
import json
import re

# ---------------------------------------------------------------------------
# Task metadata
# ---------------------------------------------------------------------------

TASK_INSTRUCTION = """im looking for a backpack under $75 that has all the features of this one: https://www.amazon.com/dp/B09YRC9Y3G please do some research and find 3-5 optionsand summarize their key features and prices, comparing them to the original."""

PASS_THRESHOLD = 3.0

# ---------------------------------------------------------------------------
# Judge prompt
#
# Structure (from SWE-ABS pattern):
#   1. Forced analysis section — prevents jumping to verdict
#   2. Per-dimension scoring — orthogonal axes, each with observable anchors
#   3. Structured output tag — enables reliable parsing + majority voting
#
# Axes (from AutoMetrics Shopping category):
#   A. Constraint Satisfaction   — did the agent respect all hard constraints?
#   B. Result Specificity        — are actual products found, or vague search pages?
#   C. Comparison Quality        — does the agent meaningfully compare to the original?
#   D. Task Completeness         — did the agent address every part of the request?
#
# Task-specific constraints injected:
#   - Budget: all options must be <= $75
#   - Quantity: 3-5 options required
#   - Reference product: https://www.amazon.com/dp/B09YRC9Y3G
#   - Required output: key features, prices, comparison to original
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based shopping research task.

You will be given:
1. The task instruction the agent was given
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

You must evaluate the agent's performance across four dimensions. Follow the evaluation steps exactly in order."""


USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Budget ceiling: $75 per item (hard constraint — any option above this is a failure)
- Required quantity: 3 to 5 backpack options (not fewer, not more)
- Reference product: https://www.amazon.com/dp/B09YRC9Y3G (agent must have researched this product's features)
- Required output components: (a) key features per option, (b) price per option, (c) comparison to the original product

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
Carefully read the agent response and trace. Then answer each of the following questions with a direct observation from the response — do not infer or assume:

- How many backpack options did the agent present? List them briefly.
- What prices were mentioned? List all dollar amounts found.
- Did any price exceed $75? If so, which?
- Did the agent reference or research the original Amazon product (B09YRC9Y3G)?
- Did the agent compare candidates to the original? Quote or describe the comparison language used.
- Were key features (e.g. capacity, material, straps, pockets) described for each option?

### Step 2: Dimension Scoring
Score each dimension from 1 to 5 using the rubrics below. Each score must be justified by specific evidence from Step 1.

#### A. Constraint Satisfaction
Did the agent respect all hard constraints (budget and quantity)?

5 — All 3–5 options are explicitly priced at or below $75; quantity is exactly 3–5.
4 — All prices are at or below $75; quantity is 3–5 but one option is missing a price.
3 — Quantity is 3–5 but 1 option has an ambiguous or missing price; no option clearly exceeds $75.
2 — Quantity is outside 3–5 (e.g. only 2 options, or 6+), OR 1 option exceeds $75.
1 — Multiple options exceed $75, OR fewer than 2 options presented, OR task was abandoned.

#### B. Result Specificity
Are the options real, specific products (with names, links, or identifiable details) rather than vague suggestions?

5 — All 3–5 options have: product name, specific price, and at least one verifiable detail (ASIN, URL, brand+model).
4 — Most options (3+) have name and price; 1–2 are missing a verifiable identifier.
3 — Options are named but lack prices or identifiers for 2+ of them; still clearly distinct products.
2 — Options are described generically (e.g. "a hiking backpack from Amazon") without specific names or prices.
1 — No specific products identified; agent only described a search or listed categories.

#### C. Comparison Quality
Did the agent meaningfully compare candidates to the original product?

5 — Each option is explicitly compared to the original on 2+ specific features (e.g. capacity, material, pockets); comparison is structured and actionable.
4 — Most options (3+) are compared to the original on at least 1 specific feature; comparison is present but uneven.
3 — A general comparison summary exists (e.g. "similar features to the original") but feature-by-feature breakdown is missing or shallow.
2 — Comparison language is present (e.g. "vs", "similar") but no actual feature comparison is made.
1 — No comparison to the original product at all.

#### D. Task Completeness
Did the agent address every component of the request?

5 — All four required components present for all options: key features, price, comparison, and the original product was researched.
4 — All components present for most options; 1 minor gap (e.g. one option missing a feature description).
3 — 2–3 components present consistently; one component (e.g. original product research, or comparison) is missing or superficial.
2 — Only 1–2 components addressed; significant parts of the task are skipped.
1 — Agent failed to complete the task or provided an off-topic response.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags, exactly as shown:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising your Step 1 findings>",
  "constraint_satisfaction": <1-5>,
  "result_specificity": <1-5>,
  "comparison_quality": <1-5>,
  "task_completeness": <1-5>,
  "dimension_reasoning": {{
    "constraint_satisfaction": "<one sentence citing specific evidence>",
    "result_specificity": "<one sentence citing specific evidence>",
    "comparison_quality": "<one sentence citing specific evidence>",
    "task_completeness": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

# Dimension weights — Constraint Satisfaction is highest because budget/quantity
# are the only hard, verifiable constraints in this task.
DIMENSION_WEIGHTS = {
    "constraint_satisfaction": 0.35,
    "result_specificity":      0.25,
    "comparison_quality":      0.25,
    "task_completeness":       0.15,
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
        # Majority: median of up to 3 integer scores
        aggregated[dim] = sorted(scores)[len(scores) // 2]

    overall = sum(aggregated[d] * DIMENSION_WEIGHTS[d] for d in DIMENSIONS)
    aggregated["overall_score"] = round(overall, 2)
    aggregated["passed"] = overall >= PASS_THRESHOLD

    # Use reasoning from the call closest to the median overall score
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

    # First judge call
    first_call = _call_judge_once(agent_response, execution_summary)

    if first_call and "error" not in first_call:
        overall = first_call.get("overall_score", 0)
        borderline = abs(float(overall) - PASS_THRESHOLD) <= 0.5

        if borderline:
            # Majority vote: run 2 more calls for borderline cases
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

    # Build feedback string
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