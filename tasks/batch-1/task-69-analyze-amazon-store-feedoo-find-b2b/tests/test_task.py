"""
LLM-as-judge evaluator for EvolveBench task-69.

Category: Marketing & Analytics
Task: Please analyze the Amazon store at https://www.amazon.com/stores/Feedoo/page/87628091-2876-4623-81CF...
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


TASK_INSTRUCTION = """Please analyze the Amazon store at https://www.amazon.com/stores/Feedoo/page/87628091-2876-4623-81CF-899D72B5CBEF and help me find potential B2B customers to sell our products to."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves analyzing an Amazon brand store to identify the types of products sold, then researching potential B2B customers who might buy those products wholesale or in bulk.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- URL: must visit the exact Amazon store URL
- Store: Feedoo brand store
- Goal: identify B2B customer segments for the products in the store
- Output: list of potential B2B customer types with reasoning

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent visit the Feedoo Amazon store?
- What products/categories were identified in the store?
- What B2B customer segments were identified?
- Is the B2B identification grounded in the store's actual products?

### Step 2: Dimension Scoring

#### A. Store Analysis (0.3)
Did the agent analyze the Amazon store?

5 — Agent visited the store URL and identified product categories, price points, and brand positioning.
4 — Store visited but analysis is shallow.
3 — Visited store but only noted a few products.
2 — Described what an Amazon store looks like without visiting.
1 — No store visit.

#### B. B2B Identification (0.35)
Were relevant B2B customer segments identified?

5 — 4+ specific B2B segments identified that logically need these products (e.g. pet stores, veterinary clinics, groomers if it's a pet brand).
4 — 2-3 specific B2B segments.
3 — 1-2 segments identified.
2 — Vague 'businesses that buy products' without specificity.
1 — No B2B segments identified.

#### C. Reasoning Quality (0.25)
Is B2B identification grounded in the store's products?

5 — Each customer segment directly linked to specific products in the store with clear rationale.
4 — Good reasoning but not all segments linked to specific products.
3 — General reasoning without product-specific links.
2 — Generic B2B advice not tied to the store.
1 — No reasoning.

#### D. Actionability (0.1)
Is the output actionable for sales outreach?

5 — Includes how to reach each B2B segment (industry associations, LinkedIn, trade shows, etc.).
4 — Outreach channels mentioned for some segments.
3 — Segments identified without outreach guidance.
2 — Very generic.
1 — Not actionable.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "store_analysis": <1-5>,
  "b2b_identification": <1-5>,
  "reasoning_quality": <1-5>,
  "actionability": <1-5>,
  "dimension_reasoning": {{
    "store_analysis": "<one sentence citing specific evidence>",
    "b2b_identification": "<one sentence citing specific evidence>",
    "reasoning_quality": "<one sentence citing specific evidence>",
    "actionability": "<one sentence citing specific evidence>",
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "store_analysis": 0.3,
    "b2b_identification": 0.35,
    "reasoning_quality": 0.25,
    "actionability": 0.1,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())