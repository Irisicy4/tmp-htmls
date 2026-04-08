"""
LLM-as-judge evaluator for EvolveBench task.

Category: Software Engineering
Task: Verify whether the FastAPI framework is actively maintained by checking its GitHub repository.
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


TASK_INSTRUCTION = """Verify whether the FastAPI framework is actively maintained by checking its GitHub repository. Look for the last commit date, current number of open issues, and latest release version and date."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves verifying the maintenance status of the FastAPI framework by analyzing its GitHub repository. A successful completion requires the agent to extract the last commit date, the current number of open issues, and the latest release version and date.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Verify whether the FastAPI framework is actively maintained by checking its GitHub repository. Look for the last commit date, current number of open issues, and latest release version and date.

## Task-Specific Constraints
- Must navigate to the FastAPI GitHub repository.
- Must extract the last commit date and verify it is recent (within the last year).
- Must extract the current number of open issues.
- Must extract the latest release version and its release date.
- Must present the extracted information in a structured format (e.g., table or JSON).
- Must ensure all data is accurate and sourced directly from the repository.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to the FastAPI GitHub repository?
- Did the agent extract the last commit date? Is it recent (within the last year)?
- Did the agent extract the current number of open issues?
- Did the agent extract the latest release version and its release date?
- Is the output presented in a structured format (e.g., table or JSON)?

### Step 2: Dimension Scoring

#### A. Accuracy of Extracted Data (0.35)
Measures whether the agent correctly extracted the required data from the GitHub repository.

5 — All required data (last commit date, open issues, latest release version and date) is accurate and complete.
4 — Most required data is accurate, but one minor detail is incomplete or slightly incorrect.
3 — Some required data is accurate, but one major detail is missing or incorrect.
2 — Most required data is missing or incorrect.
1 — No required data is accurate or present.

#### B. Coverage of Required Items (0.30)
Measures whether the agent included all specified items in its response.

5 — Includes all required items (last commit date, open issues, latest release version and date).
4 — Includes most required items, but one is missing.
3 — Includes at least two required items, but others are missing.
2 — Includes only one required item.
1 — Includes none of the required items.

#### C. Specificity of Details (0.20)
Measures whether the agent provided detailed and precise information.

5 — Provides highly specific details (e.g., exact dates, issue counts, version numbers).
4 — Provides specific details, but one is slightly vague or incomplete.
3 — Provides some details, but lacks precision in multiple areas.
2 — Provides very few details or vague information.
1 — Provides no specific details.

#### D. Output Structure and Credibility (0.15)
Measures whether the response is well-organized and sourced from credible evidence.

5 — Output is well-structured (e.g., table or JSON) and clearly sourced from the GitHub repository.
4 — Output is structured but lacks clarity or sourcing in one area.
3 — Output is partially structured but disorganized or unclear in multiple areas.
2 — Output is poorly structured and lacks sourcing.
1 — Output is unstructured and lacks credibility.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "accuracy_of_extracted_data": <1-5>,
  "coverage_of_required_items": <1-5>,
  "specificity_of_details": <1-5>,
  "output_structure_and_credibility": <1-5>,
  "dimension_reasoning": {{
    "accuracy_of_extracted_data": "<one sentence citing specific evidence>",
    "coverage_of_required_items": "<one sentence citing specific evidence>",
    "specificity_of_details": "<one sentence citing specific evidence>",
    "output_structure_and_credibility": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "accuracy_of_extracted_data": 0.35,
    "coverage_of_required_items": 0.30,
    "specificity_of_details": 0.20,
    "output_structure_and_credibility": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())