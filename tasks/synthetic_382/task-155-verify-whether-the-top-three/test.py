"""
LLM-as-judge evaluator for EvolveBench task.

Category: Software Engineering
Task: Verify whether the top three most-used Python web frameworks (Flask, Django, and FastAPI) are actively maintained by checking their GitHub repositories for last commit date, number of open issues, and recent releases.
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


TASK_INSTRUCTION = """Verify whether the top three most-used Python web frameworks (Flask, Django, and FastAPI) are actively maintained by checking their GitHub repositories for last commit date, number of open issues, and recent releases."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to verify the active maintenance of the top three Python web frameworks (Flask, Django, and FastAPI) by checking their GitHub repositories for last commit date, number of open issues, and recent releases. This task falls under the domain of software engineering.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Verify whether the top three most-used Python web frameworks (Flask, Django, and FastAPI) are actively maintained by checking their GitHub repositories for last commit date, number of open issues, and recent releases.

## Task-Specific Constraints
- Must visit the GitHub repositories for Flask, Django, and FastAPI.
- Must extract and report the last commit date for each framework.
- Must extract and report the number of open issues for each framework.
- Must extract and report the most recent release date for each framework.
- Output must be presented in a structured format (e.g., table or JSON).
- Must include evidence or citations for the extracted data.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to the GitHub repositories for Flask, Django, and FastAPI?
- Did the agent extract the last commit date, number of open issues, and recent release date for each framework?
- Is the output structured in a clear and organized format (e.g., table or JSON)?
- Are the extracted data points accurate and sourced from the GitHub repositories?
- Did the agent include evidence or citations for the extracted data?

### Step 2: Dimension Scoring

#### A. Data Accuracy (0.35)
Measures whether the extracted data (last commit date, open issues, recent releases) is correct and matches the GitHub repositories.

5 — All data points are accurate for all three frameworks.
4 — One minor inaccuracy or missing data point.
3 — Some inaccuracies or missing data, but most information is correct.
2 — Significant inaccuracies or missing data for multiple frameworks.
1 — Data is entirely incorrect or missing.

#### B. Coverage of Frameworks (0.30)
Measures whether the agent covered all three frameworks (Flask, Django, FastAPI) as required.

5 — All three frameworks are fully covered with all required data points.
4 — All three frameworks are covered, but one is missing a minor data point.
3 — At least two frameworks are covered with most data points.
2 — Only one framework is covered or most data points are missing.
1 — No frameworks are covered.

#### C. Depth of Evidence (0.20)
Measures whether the agent provided sufficient evidence or citations for the extracted data.

5 — Evidence or citations are provided for all data points.
4 — Evidence is provided for most data points, with minor omissions.
3 — Some evidence is provided, but it is incomplete or inconsistent.
2 — Little evidence is provided for the extracted data.
1 — No evidence or citations are provided.

#### D. Output Structure (0.15)
Measures whether the output is well-organized and presented in the required structured format.

5 — Output is fully structured and easy to read (e.g., table or JSON).
4 — Output is mostly structured, with minor formatting issues.
3 — Output is partially structured but lacks clarity or completeness.
2 — Output is poorly structured and difficult to interpret.
1 — Output is unstructured or entirely missing.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "data_accuracy": <1-5>,
  "coverage_of_frameworks": <1-5>,
  "depth_of_evidence": <1-5>,
  "output_structure": <1-5>,
  "dimension_reasoning": {{
    "data_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_frameworks": "<one sentence citing specific evidence>",
    "depth_of_evidence": "<one sentence citing specific evidence>",
    "output_structure": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "data_accuracy": 0.35,
    "coverage_of_frameworks": 0.30,
    "depth_of_evidence": 0.20,
    "output_structure": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())