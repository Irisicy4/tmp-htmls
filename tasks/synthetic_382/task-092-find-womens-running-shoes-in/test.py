"""
LLM-as-judge evaluator for EvolveBench task.

Category: Shopping
Task: Find women's running shoes in size 8 with arch support priced under $120 on Zappos and extract the top 5 options with brand names, prices, ratings, and colors.
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


TASK_INSTRUCTION = """Find women's running shoes in size 8 with arch support priced under $120 on Zappos. Use the filtering options to narrow down results, and extract the top 5 options with brand names, prices, ratings, and colors available."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves finding women's running shoes in size 8 with arch support priced under $120 on Zappos. The agent must use filtering options to narrow down results and extract the top 5 options with brand names, prices, ratings, and colors available. A successful completion requires accurate filtering and structured output.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Find women's running shoes in size 8 with arch support priced under $120 on Zappos. Use the filtering options to narrow down results, and extract the top 5 options with brand names, prices, ratings, and colors available.

## Task-Specific Constraints
- Must use Zappos.com as the platform for searching.
- Must apply filters for size 8, arch support, and price under $120.
- Must extract exactly 5 options, no more or less.
- Output must include brand names, prices, ratings, and available colors for each option.
- Output must be organized as a structured list or table.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Zappos.com and apply the required filters (size, arch support, price)?
- Are exactly 5 options present in the response?
- Does the output include brand names, prices, ratings, and colors for each option?
- Is the output organized as a structured list or table?
- Are the extracted details accurate and consistent with the task constraints?

### Step 2: Dimension Scoring

#### A. Filtering Accuracy (0.35)
Measures whether the agent correctly applied the required filters on Zappos.com.

5 — All filters (size, arch support, price) applied correctly and results match constraints.
4 — Most filters applied correctly; minor inaccuracies in results.
3 — Some filters applied; noticeable inaccuracies in results.
2 — Few filters applied; results mostly incorrect.
1 — No filters applied or completely incorrect results.

#### B. Completeness of Output (0.30)
Measures whether the agent provided all required details for the top 5 options.

5 — All 5 options include brand names, prices, ratings, and colors.
4 — 4-5 options include most required details; minor omissions.
3 — At least 3 options include partial details; noticeable omissions.
2 — 1-2 options include minimal details; major omissions.
1 — No options or completely missing details.

#### C. Detail Specificity (0.25)
Measures the depth and specificity of the extracted information.

5 — All details (prices, ratings, colors) are specific and accurate.
4 — Most details specific; minor inaccuracies or vagueness.
3 — Some details specific; noticeable vagueness or inaccuracies.
2 — Few details specific; mostly vague or incorrect.
1 — No specific details provided.

#### D. Output Structure (0.10)
Measures whether the response is well-organized and easy to interpret.

5 — Output is structured as a clear table or list; easy to interpret.
4 — Output mostly structured; minor formatting issues.
3 — Output partially structured; noticeable formatting issues.
2 — Output poorly structured; difficult to interpret.
1 — Output completely unstructured or incoherent.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "filtering_accuracy": <1-5>,
  "completeness_of_output": <1-5>,
  "detail_specificity": <1-5>,
  "output_structure": <1-5>,
  "dimension_reasoning": {{
    "filtering_accuracy": "<one sentence citing specific evidence>",
    "completeness_of_output": "<one sentence citing specific evidence>",
    "detail_specificity": "<one sentence citing specific evidence>",
    "output_structure": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "filtering_accuracy": 0.35,
    "completeness_of_output": 0.30,
    "detail_specificity": 0.25,
    "output_structure": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())