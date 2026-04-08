"""
LLM-as-judge evaluator for EvolveBench task.

Category: (Self) Media
Task: Research and compare three podcast hosting platforms (Anchor, Podbean, Buzzsprout) on monetization, analytics, and pricing.
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


TASK_INSTRUCTION = """Research and compare three popular podcast hosting platforms: Anchor, Podbean, and Buzzsprout. Specifically, evaluate their monetization options, audience analytics features, and pricing plans. Create a comparison table summarizing the differences."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to research and compare three podcast hosting platforms (Anchor, Podbean, Buzzsprout) on monetization options, audience analytics features, and pricing plans. The deliverable is a structured comparison table summarizing these differences.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Research and compare three popular podcast hosting platforms: Anchor, Podbean, and Buzzsprout. Specifically, evaluate their monetization options, audience analytics features, and pricing plans. Create a comparison table summarizing the differences.

## Task-Specific Constraints
- Must visit at least 3 of the specified platforms (Anchor, Podbean, Buzzsprout).
- Must include monetization options, audience analytics features, and pricing plans for each platform.
- Output must be organized as a comparison table.
- Must provide specific details (e.g., pricing numbers, analytics features) for each platform.
- Must ensure factual accuracy of the claims made.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to the required platforms? Which ones were actually visited?
- Are monetization options, audience analytics features, and pricing plans present for all three platforms?
- Is the output organized as a comparison table?
- Are specific details (e.g., pricing numbers, analytics features) included and accurate?
- Are the claims made factually accurate and sourced?

### Step 2: Dimension Scoring

#### A. Deliverable Accuracy (0.35)
Measures whether the comparison table is correct and complete.

5 — Table includes all required categories (monetization, analytics, pricing) for all three platforms, with accurate and detailed data.
4 — Table includes all categories for all platforms but lacks some details or has minor inaccuracies.
3 — Table includes most categories and platforms but is incomplete or contains notable inaccuracies.
2 — Table is missing multiple categories or platforms, or contains major inaccuracies.
1 — No meaningful comparison table provided.

#### B. Coverage of Platforms (0.30)
Measures whether all three specified platforms were researched and included.

5 — All three platforms (Anchor, Podbean, Buzzsprout) are fully covered.
4 — All three platforms are included but coverage is incomplete for one platform.
3 — At least two platforms are covered with partial data.
2 — Only one platform is covered, or data is very incomplete.
1 — No platforms are covered.

#### C. Depth of Details (0.20)
Measures the specificity and depth of the data provided.

5 — Includes detailed pricing numbers, specific analytics features, and monetization options for all platforms.
4 — Includes detailed data for most platforms but lacks depth in one or two areas.
3 — Includes some specific details but lacks depth across multiple areas.
2 — Includes very few specific details or mostly vague information.
1 — No specific details provided.

#### D. Output Structure and Credibility (0.15)
Measures the organization and credibility of the output.

5 — Output is well-organized as a table, with credible and sourced data.
4 — Output is organized as a table but lacks sourcing or has minor structural issues.
3 — Output is somewhat organized but lacks clarity or credibility.
2 — Output is poorly organized or mostly unclear.
1 — Output is completely disorganized or not credible.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "deliverable_accuracy": <1-5>,
  "coverage_of_platforms": <1-5>,
  "depth_of_details": <1-5>,
  "output_structure_and_credibility": <1-5>,
  "dimension_reasoning": {{
    "deliverable_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_platforms": "<one sentence citing specific evidence>",
    "depth_of_details": "<one sentence citing specific evidence>",
    "output_structure_and_credibility": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "deliverable_accuracy": 0.35,
    "coverage_of_platforms": 0.30,
    "depth_of_details": 0.20,
    "output_structure_and_credibility": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())