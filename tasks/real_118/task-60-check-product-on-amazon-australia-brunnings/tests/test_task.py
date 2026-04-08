"""
LLM-as-judge evaluator for EvolveBench task-60.

Category: Shopping
Task: Please check this product on Amazon Australia: https://www.amazon.com.au/Brunnings-Ant-Killer-Powder...
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


TASK_INSTRUCTION = """Please check this product on Amazon Australia: https://www.amazon.com.au/Brunnings-Ant-Killer-Powder-500/dp/B0DR94VX18/ and tell me if it is effective for killing ants, cockroaches, and bull ants."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves checking a specific Amazon Australia product listing and evaluating its effectiveness against three pest types based on product details, reviews, and active ingredients.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- URL: must visit the exact Amazon Australia product page provided
- Pests: must specifically address ants, cockroaches, AND bull ants (three separate assessments)
- Evidence: effectiveness claims must be grounded in product description, active ingredients, or customer reviews
- Not just star rating — must assess actual pest-control efficacy

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent visit the exact Amazon Australia URL?
- What is the product name and active ingredient?
- Does the product claim to kill ants, cockroaches, and bull ants?
- What evidence (ingredients, reviews, product description) supports the effectiveness assessment?
- Was each pest type addressed individually?

### Step 2: Dimension Scoring

#### A. Product Page Access (0.2)
Did the agent visit the correct product page?

5 — Agent navigated to the exact Amazon.com.au URL and retrieved product details.
4 — Agent reached Amazon Australia but had trouble loading the full page.
3 — Agent found the product via search rather than the direct URL.
2 — Agent used Amazon.com (US) instead of Amazon.com.au.
1 — No product page accessed.

#### B. Pest Coverage (0.3)
Were all three pest types addressed?

5 — Ants, cockroaches, and bull ants all assessed individually with specific evidence.
4 — Two pest types assessed with evidence; one is missing or vague.
3 — All three mentioned but without individual assessment.
2 — Only one or two pest types mentioned.
1 — Pest effectiveness not addressed.

#### C. Evidence Quality (0.35)
Is the effectiveness assessment grounded in product evidence?

5 — Assessment based on active ingredients (e.g. permethrin, deltamethrin) AND product claims AND reviews.
4 — Based on two of three evidence types.
3 — Based on product claims alone or reviews alone.
2 — Based only on star rating or vague impressions.
1 — No evidence cited.

#### D. Recommendation Clarity (0.15)
Is the final assessment clear and actionable?

5 — Clear yes/no/partially effective for each pest with reasoning.
4 — Clear overall recommendation but individual pest breakdown is weak.
3 — Recommendation present but hedged without clear conclusion.
2 — Inconclusive or contradictory.
1 — No recommendation.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "product_page_access": <1-5>,
  "pest_coverage": <1-5>,
  "evidence_quality": <1-5>,
  "recommendation_clarity": <1-5>,
  "dimension_reasoning": {{
    "product_page_access": "<one sentence citing specific evidence>",
    "pest_coverage": "<one sentence citing specific evidence>",
    "evidence_quality": "<one sentence citing specific evidence>",
    "recommendation_clarity": "<one sentence citing specific evidence>",
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "product_page_access": 0.2,
    "pest_coverage": 0.3,
    "evidence_quality": 0.35,
    "recommendation_clarity": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())