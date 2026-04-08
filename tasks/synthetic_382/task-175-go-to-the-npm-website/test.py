"""
LLM-as-judge evaluator for EvolveBench task.

Category: Software Engineering
Task: Extract the top 10 'eslint plugin' packages from npm, sorted by weekly download count, with their name, description, and last updated date.
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


TASK_INSTRUCTION = """Go to the npm website and search for packages tagged with 'eslint plugin'. Extract the top 10 packages sorted by weekly download count with their name, description, and last updated date."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to navigate to npmjs.com, search for packages tagged with 'eslint plugin', and extract the top 10 packages sorted by weekly download count. The output must include the package name, description, and last updated date. A successful completion involves accurate data extraction and proper formatting of the output.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Go to the npm website and search for packages tagged with 'eslint plugin'. Extract the top 10 packages sorted by weekly download count with their name, description, and last updated date.

## Task-Specific Constraints
- Must navigate to npmjs.com and perform a search for 'eslint plugin'.
- Must extract the top 10 packages sorted by weekly download count.
- Output must include package name, description, and last updated date.
- Output must be organized as a structured list or table.
- Must ensure data accuracy and proper formatting.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to npmjs.com and perform the required search?
- Are the top 10 packages sorted by weekly download count included in the response?
- Does the output include package name, description, and last updated date for each package?
- Is the output organized as a structured list or table?
- Is the extracted data accurate and properly formatted?

### Step 2: Dimension Scoring

#### A. Data Accuracy (0.35)
Measures whether the extracted data is correct and matches the actual data on npmjs.com.

5 — All 10 packages are correct, with accurate names, descriptions, and last updated dates.
4 — 8-9 packages are correct, with minor inaccuracies in descriptions or dates.
3 — 6-7 packages are correct, with noticeable inaccuracies in descriptions or dates.
2 — 3-5 packages are correct, with significant inaccuracies or missing data.
1 — Fewer than 3 packages are correct, or data is entirely wrong.

#### B. Coverage (0.30)
Measures whether the agent included all required items and followed the task constraints.

5 — Includes all 10 packages sorted by weekly download count, with all required fields.
4 — Includes 8-9 packages, with minor omissions or sorting errors.
3 — Includes 6-7 packages, with noticeable omissions or sorting errors.
2 — Includes 3-5 packages, with major omissions or sorting errors.
1 — Includes fewer than 3 packages, or sorting is entirely wrong.

#### C. Depth of Information (0.25)
Measures the level of detail and specificity in the extracted data.

5 — Provides detailed and complete descriptions and dates for all packages.
4 — Provides detailed descriptions and dates for 8-9 packages.
3 — Provides partial descriptions and dates for 6-7 packages.
2 — Provides minimal descriptions or dates for 3-5 packages.
1 — Provides no meaningful descriptions or dates.

#### D. Output Structure (0.10)
Measures whether the output is well-organized and formatted correctly.

5 — Output is structured as a clear table or list, easy to read and understand.
4 — Output is mostly structured well, with minor formatting issues.
3 — Output is partially structured, with noticeable formatting issues.
2 — Output is poorly structured, difficult to read or understand.
1 — Output is unstructured or completely disorganized.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "data_accuracy": <1-5>,
  "coverage": <1-5>,
  "depth_of_information": <1-5>,
  "output_structure": <1-5>,
  "dimension_reasoning": {{
    "data_accuracy": "<one sentence citing specific evidence>",
    "coverage": "<one sentence citing specific evidence>",
    "depth_of_information": "<one sentence citing specific evidence>",
    "output_structure": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "data_accuracy": 0.35,
    "coverage": 0.30,
    "depth_of_information": 0.25,
    "output_structure": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())