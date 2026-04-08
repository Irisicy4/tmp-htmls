"""
LLM-as-judge evaluator for EvolveBench task.

Category: Design
Task: Produce a brand mood board for a tech-centric startup targeting Gen Z using design tools.
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


TASK_INSTRUCTION = """Produce a brand mood board for a new tech-centric startup targeting Gen Z. Use publicly available design tools to combine a color palette inspired by modern neon trends, typography examples, and abstract shapes. Include at least 6 distinct visual elements."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to produce a brand mood board for a tech-centric startup targeting Gen Z. The deliverable must include a color palette inspired by modern neon trends, typography examples, and abstract shapes. At least 6 distinct visual elements must be included.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Produce a brand mood board for a new tech-centric startup targeting Gen Z. Use publicly available design tools to combine a color palette inspired by modern neon trends, typography examples, and abstract shapes. Include at least 6 distinct visual elements.

## Task-Specific Constraints
- Must visit at least 3 of the specified platforms: canva.com, unsplash.com, coolors.co.
- Must include a color palette inspired by modern neon trends.
- Must include typography examples relevant to Gen Z preferences.
- Must include abstract shapes as part of the mood board.
- Must include at least 6 distinct visual elements in total.
- Final output must be organized and visually cohesive.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to the required platforms? Which ones were actually visited?
- Does the response include a color palette inspired by modern neon trends?
- Are typography examples relevant to Gen Z preferences included?
- Are abstract shapes present in the mood board?
- Are there at least 6 distinct visual elements in the final output?
- Is the final output organized and visually cohesive?

### Step 2: Dimension Scoring

#### A. Deliverable Accuracy (0.35)
Measures whether the mood board meets the task requirements and includes all specified elements.

5 — Includes all required elements (color palette, typography, abstract shapes, 6+ visual elements) with clear relevance to Gen Z.
4 — Includes most required elements; minor omissions or less relevance to Gen Z.
3 — Includes some required elements but lacks completeness or relevance.
2 — Includes few required elements; significant omissions or irrelevance.
1 — Includes none of the required elements.

#### B. Coverage of Platforms (0.30)
Measures whether the agent used the specified platforms effectively.

5 — Successfully used all 3 platforms (canva.com, unsplash.com, coolors.co) and incorporated their outputs.
4 — Used 2 platforms effectively; minor omissions in platform usage.
3 — Used 1 platform effectively; significant omissions in platform usage.
2 — Attempted platform usage but failed to incorporate outputs meaningfully.
1 — Did not use any of the specified platforms.

#### C. Depth and Specificity (0.20)
Measures the level of detail and specificity in the mood board.

5 — Provides detailed and specific elements (e.g., named colors, font names, shape descriptions).
4 — Provides moderately detailed elements; minor omissions in specificity.
3 — Provides basic elements with limited detail or specificity.
2 — Provides vague or generic elements; lacks meaningful detail.
1 — Provides no detail or specificity.

#### D. Organization and Cohesion (0.15)
Measures the visual organization and cohesiveness of the mood board.

5 — Mood board is visually cohesive, well-organized, and aesthetically appealing.
4 — Mood board is mostly cohesive and organized; minor aesthetic issues.
3 — Mood board is somewhat cohesive but lacks organization or aesthetic appeal.
2 — Mood board is poorly organized and lacks visual cohesion.
1 — Mood board is completely disorganized and incoherent.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "deliverable_accuracy": <1-5>,
  "coverage_of_platforms": <1-5>,
  "depth_and_specificity": <1-5>,
  "organization_and_cohesion": <1-5>,
  "dimension_reasoning": {{
    "deliverable_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_platforms": "<one sentence citing specific evidence>",
    "depth_and_specificity": "<one sentence citing specific evidence>",
    "organization_and_cohesion": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "deliverable_accuracy": 0.35,
    "coverage_of_platforms": 0.30,
    "depth_and_specificity": 0.20,
    "organization_and_cohesion": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())