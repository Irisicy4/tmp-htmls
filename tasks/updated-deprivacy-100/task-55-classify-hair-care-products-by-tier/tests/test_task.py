"""
LLM-as-judge evaluator for EvolveBench task-55.

Category: Shopping
Task: Classify hair care products by ingredient tier, then analyze a specific shampoo.
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


TASK_INSTRUCTION = """Classify hair care products on the market by their ingredient lists and rank them by tier. Then analyze the tier of the shampoo shown in the image based on its ingredients."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Assess whether an AI agent produced a credible ingredient-based tier classification of hair care products and correctly analyzed a specific shampoo."""

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Tier classification must be based on actual ingredient analysis (not brand prestige)
- Must produce at least 3 tiers (e.g. Basic, Mid-range, Premium)
- Must analyze the specific shampoo from the image
- Analysis must reference specific ingredients and explain their tier implications

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent research ingredient-based tier classification systems?
- How many tiers were defined and what criteria distinguish them?
- Did the agent analyze the specific shampoo from the image?
- What ingredients were identified and what tier was assigned?

### Step 2: Dimension Scoring

#### A. Tier Framework (0.3)
Did the agent produce a credible ingredient-based tier classification?

5 — Clear 3+ tier framework with specific ingredient criteria for each tier (e.g. sulfate-free, silicone types, active ingredients).
4 — Good framework but one tier is poorly defined.
3 — Basic tiers present but criteria are vague or brand-based rather than ingredient-based.
2 — Only 2 tiers or very simplistic classification.
1 — No tier framework.

#### B. Ingredient Knowledge (0.25)
Does the agent demonstrate knowledge of hair care ingredients?

5 — References specific ingredients by name (surfactants, conditioning agents, actives) and explains their quality indicators.
4 — Good ingredient knowledge but incomplete coverage.
3 — Basic ingredient knowledge — knows good vs bad but lacks specifics.
2 — Very generic ingredient discussion.
1 — No ingredient knowledge demonstrated.

#### C. Product Analysis (0.35)
Did the agent analyze the specific shampoo and assign it a tier?

5 — Specific shampoo analyzed with ingredient-by-ingredient breakdown and clear tier assignment with reasoning.
4 — Tier assigned with some ingredient analysis but not comprehensive.
3 — Tier assigned but reasoning is thin.
2 — Tier mentioned without specific ingredient analysis.
1 — No product analysis.

#### D. Output Structure (0.1)
Is the output well-organized?

5 — Clear structure: tier framework first, then product analysis, with consistent formatting.
4 — Good structure but minor organizational issues.
3 — Content present but hard to navigate.
2 — Wall of text.
1 — No structure.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "tier_framework": <1-5>,
  "ingredient_knowledge": <1-5>,
  "product_analysis": <1-5>,
  "output_structure": <1-5>,
  "dimension_reasoning": {{
    "tier_framework": "<one sentence citing specific evidence>",
    "ingredient_knowledge": "<one sentence citing specific evidence>",
    "product_analysis": "<one sentence citing specific evidence>",
    "output_structure": "<one sentence citing specific evidence>",
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "tier_framework": 0.3,
    "ingredient_knowledge": 0.25,
    "product_analysis": 0.35,
    "output_structure": 0.1,
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