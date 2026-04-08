"""
LLM-as-judge evaluator for EvolveBench task.

Category: Design
Task: Verify whether Material Design currently recommends using floating action buttons (FABs) in mobile UI designs.
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


TASK_INSTRUCTION = """Verify whether Material Design currently recommends using floating action buttons (FABs) in mobile UI designs. Check the Material Design guidelines and recent articles that discuss UI/UX trends and report whether FAB is still recommended."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to verify whether Material Design guidelines currently recommend using floating action buttons (FABs) in mobile UI designs. The agent must consult the official Material Design guidelines and recent articles from UX/Design platforms to determine the current recommendation status. Successful completion involves a clear, accurate summary of the findings, citing sources and trends.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Verify whether Material Design currently recommends using floating action buttons (FABs) in mobile UI designs. Check the Material Design guidelines and recent articles that discuss UI/UX trends and report whether FAB is still recommended.

## Task-Specific Constraints
- Must visit at least 3 of the specified platforms: material.io, uxdesign.cc, smashingmagazine.com.
- Must include direct quotes or summaries from the Material Design guidelines.
- Must reference at least one recent article discussing UI/UX trends.
- Output must include a clear recommendation (e.g., "FABs are recommended" or "FABs are not recommended").
- Response must be organized in a structured format (e.g., bullet points or paragraphs).

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to the required platforms? Which ones were actually visited?
- Does the response include direct quotes or summaries from the Material Design guidelines?
- Does the response reference at least one recent article discussing UI/UX trends?
- Is the output organized in a structured format (e.g., bullet points or paragraphs)?
- Does the response provide a clear recommendation about FABs?

### Step 2: Dimension Scoring

#### A. Recommendation Accuracy (0.35)
Measures whether the agent provided a clear, correct recommendation based on evidence.

5 — Recommendation is clear, correct, and fully supported by evidence from guidelines and articles.
4 — Recommendation is correct but lacks full supporting evidence or clarity.
3 — Recommendation is partially correct but incomplete or unclear.
2 — Recommendation is mostly incorrect or unsupported.
1 — Recommendation is absent or entirely incorrect.

#### B. Source Coverage (0.30)
Measures whether the agent consulted all required platforms and referenced them appropriately.

5 — Consulted all 3 platforms and referenced each appropriately.
4 — Consulted 2 platforms and referenced them appropriately.
3 — Consulted 1 platform or referenced sources minimally.
2 — Consulted no platforms or referenced sources poorly.
1 — No sources consulted or referenced.

#### C. Depth of Evidence (0.20)
Measures the specificity and depth of evidence provided in the response.

5 — Includes detailed quotes, summaries, and trends from multiple sources.
4 — Includes quotes or summaries but lacks depth or breadth.
3 — Includes minimal evidence or lacks specificity.
2 — Evidence is mostly absent or vague.
1 — No evidence provided.

#### D. Output Structure and Credibility (0.15)
Measures the organization and credibility of the response.

5 — Response is well-organized, structured, and cites credible sources.
4 — Response is organized but lacks full credibility or polish.
3 — Response is minimally organized or credible.
2 — Response is poorly organized or lacks credibility.
1 — Response is unstructured and not credible.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "recommendation_accuracy": <1-5>,
  "source_coverage": <1-5>,
  "depth_of_evidence": <1-5>,
  "output_structure_and_credibility": <1-5>,
  "dimension_reasoning": {{
    "recommendation_accuracy": "<one sentence citing specific evidence>",
    "source_coverage": "<one sentence citing specific evidence>",
    "depth_of_evidence": "<one sentence citing specific evidence>",
    "output_structure_and_credibility": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "recommendation_accuracy": 0.35,
    "source_coverage": 0.30,
    "depth_of_evidence": 0.20,
    "output_structure_and_credibility": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())