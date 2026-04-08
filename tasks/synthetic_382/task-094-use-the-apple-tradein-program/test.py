"""
LLM-as-judge evaluator for EvolveBench task.

Category: Shopping
Task: Estimate the trade-in value of a used iPhone 12 (128GB, good condition) using apple.com.
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


TASK_INSTRUCTION = """Use the Apple Trade-In program to estimate the trade-in value of a used iPhone 12 (128GB, good condition) on apple.com. Complete the trade-in value workflow and summarize the value quoted at the final step."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to use the Apple Trade-In program on apple.com to estimate the trade-in value of a used iPhone 12 (128GB, good condition). The agent must complete the trade-in workflow and provide the quoted value at the final step. A successful completion includes accurate navigation of the platform, correct estimation of the trade-in value, and a clear summary of the quoted value.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Use the Apple Trade-In program to estimate the trade-in value of a used iPhone 12 (128GB, good condition) on apple.com. Complete the trade-in value workflow and summarize the value quoted at the final step.

## Task-Specific Constraints
- Must navigate to apple.com and access the Trade-In program.
- Must specify the device as an iPhone 12 with 128GB storage.
- Must indicate the condition of the device as "good."
- Must complete the trade-in workflow to obtain a quoted value.
- Must summarize the quoted value clearly in the response.
- Output must be concise and free of extraneous information.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to apple.com and access the Trade-In program?
- Did the agent specify the correct device model, storage capacity, and condition?
- Did the agent complete the trade-in workflow to obtain a quoted value?
- Is the quoted value summarized clearly in the response?
- Is the response concise and free of extraneous information?

### Step 2: Dimension Scoring

#### A. Trade-In Value Accuracy (0.35)
Measures whether the quoted trade-in value is accurate and matches the Apple Trade-In program's output.

5 — Quoted value is completely accurate and matches the Apple Trade-In program's output.
4 — Quoted value is mostly accurate but slightly off.
3 — Quoted value is partially accurate but missing key details.
2 — Quoted value is mostly incorrect or incomplete.
1 — Quoted value is absent or completely wrong.

#### B. Workflow Completion (0.30)
Measures whether the agent completed all required steps in the trade-in workflow.

5 — All steps completed correctly, including device selection and condition specification.
4 — Most steps completed correctly, with minor omissions.
3 — Partial completion of steps, with significant omissions.
2 — Minimal steps completed, with major omissions.
1 — No steps completed or completely incorrect workflow.

#### C. Detail Specificity (0.20)
Measures the specificity and clarity of the agent's response.

5 — Response includes all required details (device model, storage, condition, quoted value).
4 — Response includes most required details, with minor omissions.
3 — Response includes some required details but lacks clarity.
2 — Response includes minimal details and lacks clarity.
1 — Response is vague or missing required details.

#### D. Output Structure (0.15)
Measures the organization and conciseness of the response.

5 — Response is well-organized, concise, and free of extraneous information.
4 — Response is mostly organized and concise, with minor issues.
3 — Response is partially organized but includes extraneous information.
2 — Response is poorly organized and includes significant extraneous information.
1 — Response is disorganized and unclear.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "trade_in_value_accuracy": <1-5>,
  "workflow_completion": <1-5>,
  "detail_specificity": <1-5>,
  "output_structure": <1-5>,
  "dimension_reasoning": {{
    "trade_in_value_accuracy": "<one sentence citing specific evidence>",
    "workflow_completion": "<one sentence citing specific evidence>",
    "detail_specificity": "<one sentence citing specific evidence>",
    "output_structure": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "trade_in_value_accuracy": 0.35,
    "workflow_completion": 0.30,
    "detail_specificity": 0.20,
    "output_structure": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())