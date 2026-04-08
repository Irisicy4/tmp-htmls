"""
LLM-as-judge evaluator for EvolveBench task.

Category: Daily Activities
Task: Compare three local gyms in Manhattan, NY based on membership fees, amenities, and reviews, and recommend the best option for strength training and group fitness classes.
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


TASK_INSTRUCTION = """Find and compare three local gyms in Manhattan, NY. Research their monthly membership fees, amenities (e.g., pool, yoga classes, free weights), and customer reviews from their official websites and aggregator sites like Yelp. Create a table summarizing the key differences and recommend the best option for someone interested in strength training and group fitness classes."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to compare three gyms in Manhattan, NY based on membership fees, amenities, and customer reviews. The agent must create a table summarizing the differences and recommend the best gym for strength training and group fitness classes.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Find and compare three local gyms in Manhattan, NY. Research their monthly membership fees, amenities (e.g., pool, yoga classes, free weights), and customer reviews from their official websites and aggregator sites like Yelp. Create a table summarizing the key differences and recommend the best option for someone interested in strength training and group fitness classes.

## Task-Specific Constraints
- Must visit at least three platforms (yelp.com, planetfitness.com, equinox.com).
- Must include monthly membership fees for all gyms compared.
- Must list amenities for each gym, including pools, yoga classes, and free weights.
- Must summarize customer reviews from Yelp or similar sources.
- Output must be organized as a table with clear comparisons.
- Recommendation must address strength training and group fitness explicitly.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to the required platforms? Which ones were actually visited?
- Are monthly membership fees for all three gyms present in the response?
- Are amenities like pools, yoga classes, and free weights listed for each gym?
- Is the output organized as a table with clear comparisons?
- Does the recommendation explicitly address strength training and group fitness?

### Step 2: Dimension Scoring

#### A. Deliverable Accuracy (0.35)
Measures whether the main output (table and recommendation) is correct and complete.

5 — Table includes fees, amenities, and reviews for all three gyms; recommendation is clear and justified.
4 — Table includes most required data; recommendation is clear but lacks justification.
3 — Table includes partial data; recommendation is vague or incomplete.
2 — Table is missing key data; recommendation is unclear or unjustified.
1 — No usable table or recommendation provided.

#### B. Coverage of Sources (0.30)
Measures whether the agent used all required platforms and sources.

5 — Agent visited all three specified platforms and used customer reviews from Yelp.
4 — Agent visited two platforms and used customer reviews.
3 — Agent visited one platform and included minimal reviews.
2 — Agent visited one platform but did not include reviews.
1 — No evidence of platform usage or reviews.

#### C. Depth of Comparison (0.25)
Measures the specificity and detail in the comparison.

5 — Includes detailed comparisons of fees, amenities, and reviews for all gyms.
4 — Includes comparisons but lacks depth in one area (e.g., reviews or amenities).
3 — Includes basic comparisons but lacks depth in multiple areas.
2 — Comparisons are vague or incomplete.
1 — No meaningful comparisons provided.

#### D. Output Structure and Credibility (0.10)
Measures the organization and credibility of the response.

5 — Output is well-organized as a table; sources are credible and cited.
4 — Output is mostly organized; sources are credible but not cited.
3 — Output is partially organized; sources are unclear or missing.
2 — Output is disorganized; sources are unclear or missing.
1 — Output is unusable or lacks credibility.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "deliverable_accuracy": <1-5>,
  "coverage_of_sources": <1-5>,
  "depth_of_comparison": <1-5>,
  "output_structure_and_credibility": <1-5>,
  "dimension_reasoning": {{
    "deliverable_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_sources": "<one sentence citing specific evidence>",
    "depth_of_comparison": "<one sentence citing specific evidence>",
    "output_structure_and_credibility": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "deliverable_accuracy": 0.35,
    "coverage_of_sources": 0.30,
    "depth_of_comparison": 0.25,
    "output_structure_and_credibility": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())