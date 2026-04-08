"""
LLM-as-judge evaluator for EvolveBench task.

Category: Daily Activities
Task: Verify whether The Cheesecake Factory in Los Angeles (Beverly Center location) is currently offering their seasonal pumpkin cheesecake by checking their official menu page and Yelp reviews.
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


TASK_INSTRUCTION = """Verify whether The Cheesecake Factory in Los Angeles (Beverly Center location) is currently offering their seasonal pumpkin cheesecake. Check their official menu page along with recent customer reviews on Yelp to confirm availability."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves determining whether The Cheesecake Factory in Los Angeles (Beverly Center location) is offering their seasonal pumpkin cheesecake. The agent must check both the restaurant's official menu page and recent Yelp reviews for confirmation. A successful completion requires accurate and sourced information about the availability of the item.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Verify whether The Cheesecake Factory in Los Angeles (Beverly Center location) is currently offering their seasonal pumpkin cheesecake. Check their official menu page along with recent customer reviews on Yelp to confirm availability.

## Task-Specific Constraints
- Must visit both the official Cheesecake Factory menu page and Yelp reviews for the Beverly Center location.
- Must explicitly confirm whether the seasonal pumpkin cheesecake is available.
- Must provide evidence or citations from both sources.
- Output must clearly state the conclusion (available or not available).
- Must include timestamps or dates for any cited reviews or menu updates.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to both the official Cheesecake Factory menu page and Yelp reviews?
- Does the response explicitly confirm the availability of the seasonal pumpkin cheesecake?
- Are citations or evidence from both sources included?
- Are timestamps or dates provided for the cited reviews or menu updates?
- Is the output clear and conclusive?

### Step 2: Dimension Scoring

#### A. Primary Deliverable Accuracy (0.35)
Measures whether the agent correctly determined the availability of the seasonal pumpkin cheesecake.

5 — Accurately confirms availability with evidence from both sources.
4 — Confirms availability but evidence is incomplete or partially missing.
3 — Provides partial confirmation with minimal evidence.
2 — Incorrect or unclear confirmation with little evidence.
1 — No confirmation or completely incorrect.

#### B. Source Coverage (0.30)
Measures whether the agent used both required sources (menu page and Yelp reviews).

5 — Uses both sources and cites specific details from each.
4 — Uses both sources but lacks specific details from one.
3 — Uses one source with minimal details.
2 — Attempts to use sources but fails to extract meaningful information.
1 — Does not use the required sources.

#### C. Depth of Evidence (0.20)
Measures the specificity and detail of the evidence provided.

5 — Provides detailed evidence, including timestamps or dates for all sources.
4 — Provides evidence with some details missing (e.g., timestamps).
3 — Provides minimal evidence with no timestamps or dates.
2 — Evidence is vague or unclear.
1 — No evidence provided.

#### D. Output Clarity and Structure (0.15)
Measures whether the response is well-organized and easy to understand.

5 — Output is clear, well-structured, and directly answers the task.
4 — Output is mostly clear but could be better organized.
3 — Output is understandable but lacks structure or clarity.
2 — Output is poorly organized or difficult to follow.
1 — Output is incoherent or irrelevant.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "primary_deliverable_accuracy": <1-5>,
  "source_coverage": <1-5>,
  "depth_of_evidence": <1-5>,
  "output_clarity_and_structure": <1-5>,
  "dimension_reasoning": {{
    "primary_deliverable_accuracy": "<one sentence citing specific evidence>",
    "source_coverage": "<one sentence citing specific evidence>",
    "depth_of_evidence": "<one sentence citing specific evidence>",
    "output_clarity_and_structure": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "primary_deliverable_accuracy": 0.35,
    "source_coverage": 0.30,
    "depth_of_evidence": 0.20,
    "output_clarity_and_structure": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())