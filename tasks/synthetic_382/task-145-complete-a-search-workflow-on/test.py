"""
LLM-as-judge evaluator for EvolveBench task.

Category: Software Engineering
Task: Evaluate whether the agent successfully searched npmjs.com for image compression packages, applied filters for active maintenance and 1,000+ weekly downloads, and extracted the top three results with package name, description, and download count.
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


TASK_INSTRUCTION = """Complete a search workflow on npmjs.com to find packages for image compression. Apply filters for actively maintained packages with 1,000+ weekly downloads and extract the top three results with package name, description, and download count."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to search npmjs.com for image compression packages, apply filters for actively maintained packages with at least 1,000 weekly downloads, and extract the top three results. The output must include the package name, description, and download count for each result. A successful completion requires accurate filtering, correct extraction of data, and a structured output.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Complete a search workflow on npmjs.com to find packages for image compression. Apply filters for actively maintained packages with 1,000+ weekly downloads and extract the top three results with package name, description, and download count.

## Task-Specific Constraints
- Must navigate to npmjs.com and perform a search for image compression packages.
- Must apply filters for actively maintained packages.
- Must apply filters for packages with at least 1,000 weekly downloads.
- Must extract the top three results based on relevance or popularity.
- Output must include package name, description, and download count for each result.
- Output must be structured as a clear list or table.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to npmjs.com and perform a search for image compression packages?
- Did the agent apply the required filters (actively maintained and 1,000+ weekly downloads)?
- Are the top three results included in the response with package name, description, and download count?
- Is the output structured as a clear list or table?
- Are the extracted details accurate and match the task requirements?

### Step 2: Dimension Scoring

#### A. Filtering Accuracy (0.35)
Measures whether the agent correctly applied the required filters on npmjs.com.

5 — All required filters (actively maintained, 1,000+ weekly downloads) were applied correctly.
4 — One filter was partially applied or slightly incorrect.
3 — At least one filter was applied, but others were missing.
2 — Filters were mostly incorrect or missing.
1 — No filters were applied.

#### B. Data Extraction Accuracy (0.30)
Measures whether the agent correctly extracted the top three results with all required details.

5 — All three results are correct and include package name, description, and download count.
4 — Two results are correct and complete; one is partially correct or missing details.
3 — At least one result is correct and complete.
2 — Results are mostly incorrect or incomplete.
1 — No correct results were extracted.

#### C. Output Structure and Clarity (0.20)
Measures whether the output is well-organized and clearly presented.

5 — Output is structured as a clear table or list with all required details.
4 — Output is mostly clear but has minor formatting issues.
3 — Output is understandable but poorly structured.
2 — Output is disorganized or difficult to interpret.
1 — Output is completely unclear or missing.

#### D. Platform Navigation and Workflow (0.15)
Measures whether the agent successfully navigated npmjs.com and followed the workflow.

5 — Agent navigated npmjs.com and completed the search workflow correctly.
4 — Agent navigated npmjs.com but had minor issues in the workflow.
3 — Agent partially navigated npmjs.com but missed key steps.
2 — Agent attempted navigation but failed to complete the workflow.
1 — Agent did not navigate npmjs.com.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "filtering_accuracy": <1-5>,
  "data_extraction_accuracy": <1-5>,
  "output_structure_and_clarity": <1-5>,
  "platform_navigation_and_workflow": <1-5>,
  "dimension_reasoning": {{
    "filtering_accuracy": "<one sentence citing specific evidence>",
    "data_extraction_accuracy": "<one sentence citing specific evidence>",
    "output_structure_and_clarity": "<one sentence citing specific evidence>",
    "platform_navigation_and_workflow": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "filtering_accuracy": 0.35,
    "data_extraction_accuracy": 0.30,
    "output_structure_and_clarity": 0.20,
    "platform_navigation_and_workflow": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())