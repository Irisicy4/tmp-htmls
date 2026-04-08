"""
LLM-as-judge evaluator for EvolveBench task.

Category: Shopping
Task: Check the availability, model, price, and store location of the latest Samsung Galaxy S series phone in ZIP code 90210 on Best Buy's website.
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


TASK_INSTRUCTION = """Navigate to Best Buy's website and complete the workflow for checking the availability of the latest Samsung Galaxy S series phone in ZIP code 90210. Report the model, price, and store availability."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to navigate Best Buy's website to check the availability of the latest Samsung Galaxy S series phone in ZIP code 90210. The agent must provide the phone's model, price, and store availability as the deliverable. This task is in the shopping domain and requires accurate and complete information retrieval.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Navigate to Best Buy's website and complete the workflow for checking the availability of the latest Samsung Galaxy S series phone in ZIP code 90210. Report the model, price, and store availability.

## Task-Specific Constraints
- Must navigate to bestbuy.com to retrieve the required information.
- Must specify the exact model name of the latest Samsung Galaxy S series phone.
- Must include the price of the phone as listed on the website.
- Must provide store availability for ZIP code 90210.
- Output must be structured as a clear list or table.
- Information must be accurate and sourced from Best Buy's website.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to bestbuy.com to retrieve the required information?
- Is the model name of the latest Samsung Galaxy S series phone clearly specified?
- Is the price of the phone included and accurate?
- Is store availability for ZIP code 90210 provided?
- Is the output structured as a clear list or table?

### Step 2: Dimension Scoring

#### A. Primary Deliverable Accuracy (0.35)
Measures whether the agent correctly identified the model, price, and store availability.

5 — All three elements (model, price, store availability) are correct and complete.
4 — Two elements are correct and complete; one is partially correct or missing.
3 — At least one element is correct and complete; others are missing or incorrect.
2 — Minimal correctness; mostly incorrect or missing.
1 — Completely incorrect or missing.

#### B. Coverage of Required Information (0.30)
Measures whether the agent included all required details from the task instruction.

5 — Includes all required details (model, price, store availability) with no omissions.
4 — Includes most required details; minor omissions.
3 — Includes some required details; significant omissions.
2 — Includes minimal required details; major omissions.
1 — Includes none of the required details.

#### C. Depth and Specificity (0.20)
Measures the level of detail and specificity in the agent's response.

5 — Provides highly detailed and specific information (e.g., exact model name, price breakdown).
4 — Provides moderately detailed information; some minor lack of specificity.
3 — Provides basic information; lacks depth or specificity.
2 — Provides minimal information; vague or incomplete.
1 — Provides no meaningful information.

#### D. Output Structure and Credibility (0.15)
Measures whether the output is well-organized and sourced from credible evidence.

5 — Output is well-structured (e.g., list or table) and clearly sourced from bestbuy.com.
4 — Output is mostly well-structured; minor formatting issues or unclear sourcing.
3 — Output is usable but poorly structured or lacks clear sourcing.
2 — Output is disorganized or credibility is questionable.
1 — Output is completely disorganized or lacks credibility.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "primary_deliverable_accuracy": <1-5>,
  "coverage_of_required_information": <1-5>,
  "depth_and_specificity": <1-5>,
  "output_structure_and_credibility": <1-5>,
  "dimension_reasoning": {{
    "primary_deliverable_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_required_information": "<one sentence citing specific evidence>",
    "depth_and_specificity": "<one sentence citing specific evidence>",
    "output_structure_and_credibility": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "primary_deliverable_accuracy": 0.35,
    "coverage_of_required_information": 0.30,
    "depth_and_specificity": 0.20,
    "output_structure_and_credibility": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())