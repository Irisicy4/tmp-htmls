"""
LLM-as-judge evaluator for EvolveBench task.

Category: Data & ML Engineering
Task: Verify benchmark performance claims for ResNet-50 on ImageNet between MLPerf and Papers with Code.
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


TASK_INSTRUCTION = """Verify whether the benchmark performance claims for the ResNet-50 model on ImageNet dataset match between MLPerf's latest leaderboard and the benchmark details published on Papers with Code. Check accuracy percentages and training times reported in both sources."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to verify benchmark performance claims for the ResNet-50 model on the ImageNet dataset by comparing data from MLPerf's latest leaderboard and Papers with Code. The agent must ensure the accuracy percentages and training times match between the two sources.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Verify whether the benchmark performance claims for the ResNet-50 model on ImageNet dataset match between MLPerf's latest leaderboard and the benchmark details published on Papers with Code. Check accuracy percentages and training times reported in both sources.

## Task-Specific Constraints
- Must visit both MLPerf.org and Paperswithcode.com to gather data.
- Must include accuracy percentages and training times for ResNet-50 from both sources.
- Output must clearly compare the data from both platforms side-by-side.
- Must identify and explain any discrepancies between the sources.
- Output must be structured as a table or well-organized list.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to both MLPerf.org and Paperswithcode.com to collect data?
- Are accuracy percentages and training times for ResNet-50 present in the response?
- Is the output organized as a table or structured list?
- Are discrepancies between the sources identified and explained?
- Is the data sourced correctly and accurately?

### Step 2: Dimension Scoring

#### A. Data Accuracy (0.35)
Measures whether the accuracy percentages and training times reported for ResNet-50 are correct and match the sources.

5 — All data is correct and matches both sources without discrepancies.
4 — Minor errors or omissions in the data, but most is correct.
3 — Data is partially correct but contains significant errors or omissions.
2 — Data is mostly incorrect or missing.
1 — No data provided or completely incorrect.

#### B. Platform Coverage (0.30)
Measures whether the agent visited and used both MLPerf.org and Paperswithcode.com as required.

5 — Both platforms were visited, and data from both was used.
4 — Both platforms were visited, but data from one is incomplete.
3 — Only one platform was visited, but some relevant data was used.
2 — Only one platform was visited, and data is incomplete.
1 — Neither platform was visited.

#### C. Discrepancy Analysis (0.25)
Measures whether the agent identified and explained discrepancies between the two sources.

5 — Discrepancies are clearly identified and explained with evidence.
4 — Discrepancies are partially identified or explained.
3 — Discrepancies are mentioned but not explained clearly.
2 — Discrepancies are mostly ignored or poorly explained.
1 — No attempt to identify discrepancies.

#### D. Output Structure (0.10)
Measures whether the output is well-organized and easy to interpret.

5 — Output is structured as a clear table or list with excellent formatting.
4 — Output is structured but formatting is slightly unclear.
3 — Output is usable but lacks clear structure or formatting.
2 — Output is poorly organized and hard to interpret.
1 — Output is completely unstructured or missing.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "data_accuracy": <1-5>,
  "platform_coverage": <1-5>,
  "discrepancy_analysis": <1-5>,
  "output_structure": <1-5>,
  "dimension_reasoning": {{
    "data_accuracy": "<one sentence citing specific evidence>",
    "platform_coverage": "<one sentence citing specific evidence>",
    "discrepancy_analysis": "<one sentence citing specific evidence>",
    "output_structure": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "data_accuracy": 0.35,
    "platform_coverage": 0.30,
    "discrepancy_analysis": 0.25,
    "output_structure": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())