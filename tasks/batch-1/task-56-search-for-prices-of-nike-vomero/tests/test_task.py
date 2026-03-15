"""
LLM-as-judge evaluator for EvolveBench task-56.

Category: Shopping
Task: Search Nike Vomero Plus prices in Brazil and compare across stores.
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

def _call(agent_response, execution_summary, system_prompt, user_prompt_template, task_instruction):
    try:
        import openai
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt_template.format(
                    task_instruction=task_instruction,
                    agent_response=agent_response,
                    execution_summary=execution_summary or "Not available.",
                )}
            ],
            max_tokens=1024,
        )
        return _parse(completion.choices[0].message.content)
    except Exception as e: return {"error": str(e)}

def _vote(votes, dimensions, weights, pass_threshold):
    valid = [v for v in votes if v and "error" not in v and all(d in v for d in dimensions)]
    if not valid: return votes[0] if votes else {"error": "All judge calls failed"}
    aggregated = {dim: sorted([v[dim] for v in valid])[len(valid) // 2] for dim in dimensions}
    overall = sum(aggregated[d] * weights[d] for d in dimensions)
    aggregated["overall_score"] = round(overall, 2); aggregated["passed"] = overall >= pass_threshold
    median_call = sorted(valid, key=lambda v: abs(v.get("overall_score", 0) - overall))[0]
    aggregated["evidence_summary"] = median_call.get("evidence_summary", "")
    aggregated["dimension_reasoning"] = median_call.get("dimension_reasoning", {})
    aggregated["_votes_used"] = len(valid)
    return aggregated


TASK_INSTRUCTION = """Search for prices of the Nike Vomero Plus sneaker in Brazil and compare prices across stores to find the best deal."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Assess whether an AI agent successfully found and compared Nike Vomero Plus prices across Brazilian retailers."""

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Product: Nike Vomero Plus specifically (not Air Vomero or other models)
- Region: Brazil — prices should be in BRL
- Multi-store comparison: at least 3 stores should be compared
- Best deal: agent must identify which store has the lowest price

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent search for Nike Vomero Plus in Brazil?
- How many stores were compared?
- What prices were found and in what currency?
- Which store was identified as having the best deal?
- Were prices from Google Shopping or direct retailer pages?

### Step 2: Dimension Scoring

#### A. Search Execution (0.2)
Did the agent search for Nike Vomero Plus in Brazil?

5 — Agent used Google Shopping Brazil or visited Brazilian retailer sites directly.
4 — Agent searched but used general web search rather than shopping platforms.
3 — Agent found some prices but search was not Brazil-specific.
2 — Agent described what to search without actually searching.
1 — No search performed.

#### B. Store Coverage (0.25)
How many stores were compared?

5 — 4 or more Brazilian stores compared with specific prices.
4 — 3 stores compared.
3 — 2 stores compared.
2 — Only 1 store found.
1 — No stores compared.

#### C. Price Accuracy (0.35)
Are prices accurate and in BRL?

5 — Prices in BRL from credible Brazilian retailers (Netshoes, Nike.com.br, Centauro, etc.) with specific amounts.
4 — Prices in BRL but from less authoritative sources.
3 — Prices found but currency or accuracy uncertain.
2 — Prices found but in USD or without clear source.
1 — No actual prices found.

#### D. Recommendation Quality (0.2)
Did the agent identify the best deal clearly?

5 — Clear winner identified with price, store name, and any relevant conditions (shipping, installments).
4 — Best deal identified but missing one detail.
3 — Best deal mentioned vaguely.
2 — Prices listed without identifying best deal.
1 — No recommendation.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "search_execution": <1-5>,
  "store_coverage": <1-5>,
  "price_accuracy": <1-5>,
  "recommendation_quality": <1-5>,
  "dimension_reasoning": {{
    "search_execution": "<one sentence citing specific evidence>",
    "store_coverage": "<one sentence citing specific evidence>",
    "price_accuracy": "<one sentence citing specific evidence>",
    "recommendation_quality": "<one sentence citing specific evidence>",
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "search_execution": 0.2,
    "store_coverage": 0.25,
    "price_accuracy": 0.35,
    "recommendation_quality": 0.2,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())

def test(result):
    agent_response = _extract_response(result)
    execution_summary = result.get("execution_summary", "")
    if not agent_response.strip():
        return {"passed": False, "feedback": "No response found from agent.",
                "details": {"task_completed": result.get("status") == "success"}}
    first = _call(agent_response, execution_summary, SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, TASK_INSTRUCTION)
    if first and "error" not in first:
        overall = first.get("overall_score", 0)
        if abs(float(overall) - PASS_THRESHOLD) <= 0.5:
            scores = _vote([first,
                            _call(agent_response, execution_summary, SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, TASK_INSTRUCTION),
                            _call(agent_response, execution_summary, SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, TASK_INSTRUCTION)],
                           DIMENSIONS, DIMENSION_WEIGHTS, PASS_THRESHOLD)
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