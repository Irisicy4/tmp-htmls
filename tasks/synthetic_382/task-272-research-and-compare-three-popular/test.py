"""
LLM-as-judge evaluator for EvolveBench task.

Category: Travel & Planning
Task: Research and compare three popular overland options for traveling across South America.
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


TASK_INSTRUCTION = """Research and compare three popular overland options for traveling across South America (e.g., bus tours, self-driving rentals, or train routes). Include details on cost, availability, recommended routes, and environmental impact. Use information from Lonely Planet, Rome2Rio, and Viator."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to research and compare three popular overland travel options across South America, including bus tours, self-driving rentals, and train routes. The deliverable must include details on cost, availability, recommended routes, and environmental impact, sourced from Lonely Planet, Rome2Rio, and Viator.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Research and compare three popular overland options for traveling across South America (e.g., bus tours, self-driving rentals, or train routes). Include details on cost, availability, recommended routes, and environmental impact. Use information from Lonely Planet, Rome2Rio, and Viator.

## Task-Specific Constraints
- Must visit Lonely Planet, Rome2Rio, and Viator platforms.
- Must include price data for all three options compared.
- Must provide availability details for each option.
- Must address environmental impact for each option.
- Output must be organized as a structured list or table.
- Must include recommended routes for each option.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Lonely Planet, Rome2Rio, and Viator platforms? Which ones were actually visited?
- Are cost, availability, environmental impact, and recommended routes present for all three options?
- Is the output organized as a structured list or table?
- Are the claims about cost and environmental impact accurate and sourced?
- Are there any missing or incomplete details in the response?

### Step 2: Dimension Scoring

#### A. Deliverable Accuracy (0.35)
Measures whether the agent's output includes correct and complete details for cost, availability, recommended routes, and environmental impact.

5 — All required details are correct and complete for all three options.
4 — Minor inaccuracies or omissions in one area, but overall usable.
3 — Partial completion; some major details missing or incorrect.
2 — Poor; most details missing or incorrect.
1 — Nothing; no usable information provided.

#### B. Coverage of Platforms and Sources (0.30)
Measures whether the agent used all required platforms (Lonely Planet, Rome2Rio, Viator) and included sourced information.

5 — All three platforms visited and sourced information included.
4 — Two platforms visited and sourced information included.
3 — At least one platform visited and some sourced information included.
2 — Minimal platform usage or no sourced information.
1 — No platform usage or sourced information.

#### C. Depth and Specificity (0.20)
Measures whether the agent provided detailed and specific comparisons, including numerical data and qualitative insights.

5 — Highly detailed comparisons with specific numerical data and qualitative insights for all three options.
4 — Good detail but missing minor specifics in one or two areas.
3 — Basic comparisons with limited numerical data or qualitative insights.
2 — Poor detail; mostly generic or vague comparisons.
1 — No detail; generic or absent comparisons.

#### D. Output Structure and Credibility (0.15)
Measures whether the output is well-organized and sourced from credible platforms.

5 — Output is well-organized, structured as a table or list, and sources are credible.
4 — Minor formatting issues but credible sources used.
3 — Basic structure; some formatting issues or unclear sourcing.
2 — Poor structure; unclear or missing sources.
1 — No structure or credible sources.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "deliverable_accuracy": <1-5>,
  "coverage_of_platforms_and_sources": <1-5>,
  "depth_and_specificity": <1-5>,
  "output_structure_and_credibility": <1-5>,
  "dimension_reasoning": {{
    "deliverable_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_platforms_and_sources": "<one sentence citing specific evidence>",
    "depth_and_specificity": "<one sentence citing specific evidence>",
    "output_structure_and_credibility": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "deliverable_accuracy": 0.35,
    "coverage_of_platforms_and_sources": 0.30,
    "depth_and_specificity": 0.20,
    "output_structure_and_credibility": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())