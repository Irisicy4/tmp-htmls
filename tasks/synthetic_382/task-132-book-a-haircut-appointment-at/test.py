"""
LLM-as-judge evaluator for EvolveBench task.

Category: Daily Activities
Task: Book a haircut appointment at a salon in San Francisco using Booksy.com.
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


TASK_INSTRUCTION = """Book a haircut appointment at a salon in San Francisco using Booksy.com. Search for salons specializing in men's haircuts, select one with at least four stars, and proceed through the multi-step booking workflow, choosing an available slot for next Saturday. Report the salon name, appointment time, and price shown at the final step."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to book a haircut appointment at a salon in San Francisco using Booksy.com. The agent must search for salons specializing in men's haircuts, select one with at least four stars, and complete the multi-step booking workflow. A successful completion includes reporting the salon name, appointment time, and price shown at the final step.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Book a haircut appointment at a salon in San Francisco using Booksy.com. Search for salons specializing in men's haircuts, select one with at least four stars, and proceed through the multi-step booking workflow, choosing an available slot for next Saturday. Report the salon name, appointment time, and price shown at the final step.

## Task-Specific Constraints
- Must navigate to Booksy.com and perform a search for men's haircut salons in San Francisco.
- Must filter results to include only salons with at least four stars.
- Must select an available appointment slot for next Saturday.
- Must report the salon name, appointment time, and price shown at the final step.
- Output must be structured as a JSON object containing the required details.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Booksy.com and perform the required search?
- Did the agent filter results to include only salons with at least four stars?
- Did the agent select an appointment slot for next Saturday?
- Are the salon name, appointment time, and price present in the response?
- Is the output structured as a JSON object containing the required details?

### Step 2: Dimension Scoring

#### A. Primary Deliverable Accuracy (0.35)
Measures whether the agent correctly reported the salon name, appointment time, and price.

5 — All three details (salon name, appointment time, price) are correct and complete.
4 — Two details are correct; one is missing or slightly incorrect.
3 — At least one detail is correct; others are missing or incorrect.
2 — Details are mostly incorrect or missing.
1 — No details are provided or completely wrong.

#### B. Coverage of Required Steps (0.30)
Measures whether the agent performed all required steps in the booking workflow.

5 — All required steps (search, filter, select slot, report) are completed correctly.
4 — Most steps are completed; minor omissions.
3 — Partial completion; at least one major step is missing.
2 — Minimal completion; most steps are missing or incorrect.
1 — No steps are completed correctly.

#### C. Depth of Information (0.20)
Measures the specificity and completeness of the reported details.

5 — Includes all required details with specific values (e.g., exact price, time).
4 — Includes most details with minor omissions or generalizations.
3 — Includes some details but lacks specificity or completeness.
2 — Includes minimal details; mostly vague or incomplete.
1 — No details provided.

#### D. Output Structure and Credibility (0.15)
Measures whether the output is well-organized and sourced from credible actions.

5 — Output is structured as a JSON object and clearly sourced from Booksy.com.
4 — Output is structured but contains minor formatting issues.
3 — Output is partially structured; sourcing is unclear.
2 — Output is poorly structured or lacks credibility.
1 — Output is unstructured and lacks credibility.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "primary_deliverable_accuracy": <1-5>,
  "coverage_of_required_steps": <1-5>,
  "depth_of_information": <1-5>,
  "output_structure_and_credibility": <1-5>,
  "dimension_reasoning": {{
    "primary_deliverable_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_required_steps": "<one sentence citing specific evidence>",
    "depth_of_information": "<one sentence citing specific evidence>",
    "output_structure_and_credibility": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "primary_deliverable_accuracy": 0.35,
    "coverage_of_required_steps": 0.30,
    "depth_of_information": 0.20,
    "output_structure_and_credibility": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())