"""
LLM-as-judge evaluator for EvolveBench task.

Category: Travel & Planning
Task: Create a packing checklist template for a summer beach vacation using Canva, incorporating recommendations from TripAdvisor and Good Housekeeping.
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


TASK_INSTRUCTION = """Using Canva (no login required), create a packing checklist template for a summer beach vacation. Include sections for clothing, toiletries, electronics, and miscellaneous items. Use recommendations from TripAdvisor and Good Housekeeping to design the checklist."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to create a packing checklist template for a summer beach vacation using Canva. The checklist must include sections for clothing, toiletries, electronics, and miscellaneous items. The agent must incorporate recommendations from TripAdvisor and Good Housekeeping to ensure the checklist is comprehensive and relevant.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Using Canva (no login required), create a packing checklist template for a summer beach vacation. Include sections for clothing, toiletries, electronics, and miscellaneous items. Use recommendations from TripAdvisor and Good Housekeeping to design the checklist.

## Task-Specific Constraints
- Must visit both TripAdvisor and Good Housekeeping to gather recommendations.
- Checklist must include at least 4 sections: clothing, toiletries, electronics, and miscellaneous items.
- Each section must contain at least 3 specific, relevant items.
- Output must be structured as a visually appealing template created in Canva.
- Recommendations must be clearly sourced from TripAdvisor and Good Housekeeping.
- The final response must describe the checklist and its design.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to TripAdvisor and Good Housekeeping? Were recommendations sourced from both?
- Does the checklist include all required sections (clothing, toiletries, electronics, miscellaneous)?
- Are there at least 3 specific items in each section?
- Is the output structured as a visually appealing Canva template?
- Are the recommendations clearly attributed to TripAdvisor and Good Housekeeping?

### Step 2: Dimension Scoring

#### A. Checklist Completeness (0.35)
Measures whether the checklist includes all required sections and items.

5 — Includes all 4 sections with at least 3 specific items per section.
4 — Includes all 4 sections but 1 section has fewer than 3 items.
3 — Includes at least 3 sections with 3 items each.
2 — Includes fewer than 3 sections or fewer than 3 items per section.
1 — Checklist is missing or incomplete.

#### B. Source Utilization (0.30)
Measures whether recommendations were sourced from TripAdvisor and Good Housekeeping.

5 — Recommendations are clearly sourced from both platforms with specific attribution.
4 — Recommendations are sourced from both platforms but attribution is vague.
3 — Recommendations are sourced from only one platform.
2 — Recommendations are mostly missing or unsourced.
1 — No recommendations were sourced.

#### C. Design Quality (0.25)
Measures the visual appeal and organization of the Canva template.

5 — Template is visually appealing, well-organized, and professional.
4 — Template is visually appealing but slightly disorganized.
3 — Template is usable but lacks visual polish.
2 — Template is poorly designed or disorganized.
1 — No template was created.

#### D. Specificity of Items (0.10)
Measures the specificity and relevance of items listed in the checklist.

5 — Items are highly specific and relevant to a summer beach vacation.
4 — Items are mostly specific and relevant.
3 — Items are generic but usable.
2 — Items are vague or irrelevant.
1 — Items are missing or nonsensical.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "checklist_completeness": <1-5>,
  "source_utilization": <1-5>,
  "design_quality": <1-5>,
  "specificity_of_items": <1-5>,
  "dimension_reasoning": {{
    "checklist_completeness": "<one sentence citing specific evidence>",
    "source_utilization": "<one sentence citing specific evidence>",
    "design_quality": "<one sentence citing specific evidence>",
    "specificity_of_items": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "checklist_completeness": 0.35,
    "source_utilization": 0.30,
    "design_quality": 0.25,
    "specificity_of_items": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())