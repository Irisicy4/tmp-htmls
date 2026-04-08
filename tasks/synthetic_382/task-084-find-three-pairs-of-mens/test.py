"""
LLM-as-judge evaluator for EvolveBench task.

Category: Shopping
Task: Find three pairs of men's running shoes in size US 10 that are rated 4 stars or higher on Amazon, and extract details like brand, price, and customer rating.
"""

import os, json, re
PASS_THRESHOLD = 3.0

def _extract_response(result):
    task_result = result.get("task_result") or ""
    if isinstance(task_result, str) and task_result.strip(): return task_result
    for message in reversed(result.get("conversation") or []):
        if not isinstance(message, dict): continue
        if message.get("role") == "assistant":
            content = message.get("content") or ""
            if isinstance(content, str) and len(content) > 20: return content
    return ""

def _parse(text):
    match = re.search(r"<Answer>(.*?)</Answer>", text, re.DOTALL | re.IGNORECASE)
    if not match: return None
    try: return json.loads(match.group(1).strip())
    except json.JSONDecodeError: return None

def _call(agent_response, execution_summary):
    try:
        import openai
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT_TEMPLATE.format(
                    task_instruction=TASK_INSTRUCTION,
                    agent_response=agent_response,
                    execution_summary=execution_summary or "Not available.",
                )}
            ],
            max_tokens=1024,
        )
        return _parse(completion.choices[0].message.content)
    except Exception as e: return {"error": str(e)}

def _vote(votes):
    valid = [v for v in votes if v and "error" not in v and all(d in v for d in DIMENSIONS)]
    if not valid: return votes[0] if votes else {"error": "All judge calls failed"}
    aggregated = {dim: sorted([v[dim] for v in valid])[len(valid) // 2] for dim in DIMENSIONS}
    overall = sum(aggregated[d] * DIMENSION_WEIGHTS[d] for d in DIMENSIONS)
    aggregated["overall_score"] = round(overall, 2); aggregated["passed"] = overall >= PASS_THRESHOLD
    median_call = sorted(valid, key=lambda v: abs(v.get("overall_score", 0) - overall))[0]
    aggregated["evidence_summary"] = median_call.get("evidence_summary", "")
    aggregated["dimension_reasoning"] = median_call.get("dimension_reasoning", {})
    aggregated["_votes_used"] = len(valid)
    return aggregated

def test(result):
    agent_response = _extract_response(result)
    execution_summary = result.get("execution_summary", "")
    if not agent_response.strip():
        return {"passed": False, "feedback": "No response found from agent.",
                "details": {"task_completed": result.get("status") == "success"}}
    first = _call(agent_response, execution_summary)
    if first and "error" not in first:
        overall = first.get("overall_score", 0)
        if abs(float(overall) - PASS_THRESHOLD) <= 0.5:
            scores = _vote([first, _call(agent_response, execution_summary), _call(agent_response, execution_summary)])
        else:
            scores = first; scores["_votes_used"] = 1
    else:
        scores = first or {"error": "Judge call failed", "overall_score": 0}
    overall = scores.get("overall_score", 0)
    passed = scores.get("passed", float(overall) >= PASS_THRESHOLD)
    lines = [f"Overall score: {overall}/5  (threshold: {PASS_THRESHOLD})"]
    for dim in DIMENSIONS:
        if dim in scores: lines.append(f"  {dim}: {scores[dim]}/5")
    if scores.get("evidence_summary"): lines.append(f"\nEvidence summary: {scores['evidence_summary']}")
    reasoning = scores.get("dimension_reasoning", {})
    if reasoning:
        lines.append("\nDimension reasoning:")
        for dim, reason in reasoning.items(): lines.append(f"  {dim}: {reason}")
    if scores.get("_votes_used", 1) > 1:
        lines.append(f"\n(Borderline case: {scores['_votes_used']} judge calls used, majority vote applied)")
    return {
        "passed": bool(passed), "feedback": "\n".join(lines),
        "details": {"task_completed": result.get("status") == "success", "overall_score": overall,
                    "dimension_scores": {d: scores.get(d) for d in DIMENSIONS},
                    "evidence_summary": scores.get("evidence_summary"),
                    "dimension_reasoning": scores.get("dimension_reasoning"),
                    "pass_threshold": PASS_THRESHOLD, "votes_used": scores.get("_votes_used", 1)},
    }


TASK_INSTRUCTION = """Find three pairs of men's running shoes in size US 10 that are rated 4 stars or higher on Amazon. Use the filters to narrow down the options and extract details such as brand, price, and customer rating for each pair. Ensure the shoes are in stock and eligible for free shipping."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves finding three pairs of men's running shoes in size US 10 that are rated 4 stars or higher on Amazon. The agent must use filters to narrow down the options and extract details such as brand, price, and customer rating for each pair. The shoes must be in stock and eligible for free shipping.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Find three pairs of men's running shoes in size US 10 that are rated 4 stars or higher on Amazon. Use the filters to narrow down the options and extract details such as brand, price, and customer rating for each pair. Ensure the shoes are in stock and eligible for free shipping.

## Task-Specific Constraints
- Must use Amazon.com as the platform for shopping.
- Must apply filters for size (US 10) and customer rating (4 stars or higher).
- Must verify that the shoes are in stock and eligible for free shipping.
- Must extract and include brand, price, and customer rating for each pair.
- Output must list exactly three pairs of shoes in a structured format (e.g., table or JSON).

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Amazon.com and use the required filters?
- Are three pairs of shoes listed in the response?
- Does the response include brand, price, and customer rating for each pair?
- Are the shoes verified to be in stock and eligible for free shipping?
- Is the output organized in a structured format (e.g., table or JSON)?

### Step 2: Dimension Scoring

#### A. Deliverable Accuracy (0.35)
Measures whether the agent correctly identified three pairs of men's running shoes that meet all specified criteria.

5 — All three pairs meet the criteria (size US 10, 4+ stars, in stock, free shipping) with accurate details.
4 — Two pairs meet the criteria fully; the third has minor issues.
3 — At least one pair meets the criteria fully; others have significant issues.
2 — None of the pairs meet the criteria fully, but partial attempts are present.
1 — No valid pairs identified.

#### B. Coverage of Requirements (0.30)
Measures whether the agent followed all task constraints and used the required filters.

5 — All constraints (size, rating, stock, free shipping) and filters are applied correctly.
4 — Most constraints are applied, but one minor filter or requirement is missing.
3 — Some constraints are applied, but major filters or requirements are missing.
2 — Few constraints are applied correctly.
1 — No constraints are applied.

#### C. Detail Specificity (0.20)
Measures whether the agent provided detailed information for each pair of shoes.

5 — Includes brand, price, and customer rating for all three pairs with no errors.
4 — Includes details for all pairs, but minor inaccuracies are present.
3 — Includes details for at least one pair; others are incomplete or inaccurate.
2 — Details are mostly missing or incorrect.
1 — No details provided.

#### D. Output Structure and Credibility (0.15)
Measures whether the output is well-organized and credible.

5 — Output is structured (e.g., table or JSON) and all data appears credible.
4 — Output is structured but contains minor formatting or credibility issues.
3 — Output is partially structured or has significant credibility issues.
2 — Output is unstructured and lacks credibility.
1 — Output is absent or completely unstructured.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "deliverable_accuracy": <1-5>,
  "coverage_of_requirements": <1-5>,
  "detail_specificity": <1-5>,
  "output_structure_and_credibility": <1-5>,
  "dimension_reasoning": {{
    "deliverable_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_requirements": "<one sentence citing specific evidence>",
    "detail_specificity": "<one sentence citing specific evidence>",
    "output_structure_and_credibility": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "deliverable_accuracy": 0.35,
    "coverage_of_requirements": 0.30,
    "detail_specificity": 0.20,
    "output_structure_and_credibility": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())