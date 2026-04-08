"""
LLM-as-judge evaluator for EvolveBench task.

Category: Daily Activities
Task: Research and compare three meal kit delivery services in New York City based on price per meal, dietary options, and customer reviews.
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


TASK_INSTRUCTION = """Research and compare three meal kit delivery services available in New York City, focusing on price per meal, dietary options (vegetarian, gluten-free, etc.), and customer reviews. Visit the official websites of Blue Apron, HelloFresh, and Home Chef, along with third-party review platforms like Trustpilot or Yelp, to gather the necessary information."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to research and compare three meal kit delivery services in New York City based on price per meal, dietary options, and customer reviews. The agent must visit both official websites and third-party review platforms to gather accurate and comprehensive information.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Research and compare three meal kit delivery services available in New York City, focusing on price per meal, dietary options (vegetarian, gluten-free, etc.), and customer reviews. Visit the official websites of Blue Apron, HelloFresh, and Home Chef, along with third-party review platforms like Trustpilot or Yelp, to gather the necessary information.

## Task-Specific Constraints
- Must visit at least 3 of the specified platforms (Blue Apron, HelloFresh, Home Chef, Trustpilot/Yelp).
- Must include price data for all three services compared.
- Must include dietary options (e.g., vegetarian, gluten-free) for all services.
- Must include customer review summaries for all services.
- Output must be organized as a structured table or list.
- Must provide specific sources for all claims made.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to the required platforms? Which ones were actually visited?
- Are price data, dietary options, and customer reviews present for all three services?
- Is the output organized as a structured table or list?
- Are specific sources cited for all claims made?
- Are the claims accurate and consistent with the sources?

### Step 2: Dimension Scoring

#### A. Deliverable Accuracy (0.35)
Measures whether the agent's output is correct, complete, and fulfills the task requirements.

5 — All required data (price, dietary options, reviews) is accurate and complete for all three services.
4 — Minor omissions or inaccuracies in one area (e.g., missing one dietary option).
3 — Partial completion with significant omissions or inaccuracies in multiple areas.
2 — Mostly incomplete or inaccurate data.
1 — No meaningful data provided.

#### B. Coverage of Sources (0.30)
Measures whether the agent visited and utilized all required platforms.

5 — Agent visited and utilized all specified platforms (Blue Apron, HelloFresh, Home Chef, Trustpilot/Yelp).
4 — Agent missed one platform but utilized the others effectively.
3 — Agent missed two platforms or only partially utilized them.
2 — Agent visited only one platform or failed to utilize them effectively.
1 — No platforms visited.

#### C. Depth and Specificity (0.25)
Measures whether the agent provided detailed comparisons and specific data points.

5 — Detailed comparisons with specific data points (e.g., price per meal, dietary options, review summaries).
4 — Comparisons provided but lacking detail in one area.
3 — Basic comparisons with limited detail or missing specific data points.
2 — Very superficial comparisons with minimal detail.
1 — No meaningful comparisons provided.

#### D. Output Structure and Source Credibility (0.10)
Measures whether the output is well-organized and sources are credible.

5 — Output is structured as a clear table or list, with credible sources cited for all claims.
4 — Output is mostly well-organized but lacks clarity or misses some citations.
3 — Output is usable but poorly organized or missing citations for some claims.
2 — Output is disorganized or lacks credibility.
1 — Output is completely unstructured or sources are absent.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "deliverable_accuracy": <1-5>,
  "coverage_of_sources": <1-5>,
  "depth_and_specificity": <1-5>,
  "output_structure_and_source_credibility": <1-5>,
  "dimension_reasoning": {{
    "deliverable_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_sources": "<one sentence citing specific evidence>",
    "depth_and_specificity": "<one sentence citing specific evidence>",
    "output_structure_and_source_credibility": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "deliverable_accuracy": 0.35,
    "coverage_of_sources": 0.30,
    "depth_and_specificity": 0.25,
    "output_structure_and_source_credibility": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())