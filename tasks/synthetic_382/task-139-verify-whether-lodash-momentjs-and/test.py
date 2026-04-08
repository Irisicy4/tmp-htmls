"""
LLM-as-judge evaluator for EvolveBench task.

Category: Software Engineering
Task: Verify whether Lodash, Moment.js, and D3.js are actively maintained by analyzing their GitHub repositories.
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


TASK_INSTRUCTION = """Verify whether Lodash, Moment.js, and D3.js are still actively maintained. Check their GitHub repositories for last commit dates, number of open issues, and latest release dates. Produce a report indicating whether they are actively maintained, partially maintained, or abandoned."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves verifying the maintenance status of three JavaScript libraries: Lodash, Moment.js, and D3.js. The agent must analyze their GitHub repositories for last commit dates, number of open issues, and latest release dates. A successful completion requires a structured report categorizing each library as actively maintained, partially maintained, or abandoned based on the evidence.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Verify whether Lodash, Moment.js, and D3.js are still actively maintained. Check their GitHub repositories for last commit dates, number of open issues, and latest release dates. Produce a report indicating whether they are actively maintained, partially maintained, or abandoned.

## Task-Specific Constraints
- Must visit the GitHub repositories for Lodash, Moment.js, and D3.js.
- Must extract last commit dates for all three libraries.
- Must extract the number of open issues for all three libraries.
- Must extract the latest release dates for all three libraries.
- Output must categorize each library as actively maintained, partially maintained, or abandoned.
- Output must be structured as a table or structured list.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to the GitHub repositories for Lodash, Moment.js, and D3.js?
- Are the last commit dates for all three libraries present in the response?
- Are the number of open issues for all three libraries present in the response?
- Are the latest release dates for all three libraries present in the response?
- Is the output structured as a table or structured list?

### Step 2: Dimension Scoring

#### A. Maintenance Status Accuracy (0.35)
Measures whether the agent correctly categorized each library's maintenance status based on evidence.

5 — Correctly categorizes all three libraries with accurate evidence.
4 — Correctly categorizes two libraries; minor errors in the third.
3 — Correctly categorizes one library; significant errors in the others.
2 — Incorrect categorization for most libraries; evidence misinterpreted.
1 — No categorization or completely incorrect.

#### B. Evidence Coverage (0.30)
Measures whether the agent included all required evidence (commit dates, open issues, release dates).

5 — Includes all required evidence for all three libraries.
4 — Includes most evidence; minor omissions for one library.
3 — Includes partial evidence; significant omissions for one or more libraries.
2 — Includes minimal evidence; major omissions across libraries.
1 — No evidence included.

#### C. Specificity and Detail (0.25)
Measures the depth and specificity of the evidence presented.

5 — Provides detailed evidence with specific dates, issue counts, and release versions.
4 — Provides mostly detailed evidence; minor lack of specificity for one library.
3 — Provides partial evidence; lacks specificity for multiple libraries.
2 — Provides vague or minimal evidence; lacks key details.
1 — No specific evidence provided.

#### D. Output Structure and Clarity (0.10)
Measures whether the output is well-organized and easy to interpret.

5 — Output is structured as a clear table or list; easy to interpret.
4 — Output is mostly clear; minor formatting issues.
3 — Output is partially clear; significant formatting issues.
2 — Output is poorly structured; difficult to interpret.
1 — Output is unstructured or incomprehensible.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "maintenance_status_accuracy": <1-5>,
  "evidence_coverage": <1-5>,
  "specificity_and_detail": <1-5>,
  "output_structure_and_clarity": <1-5>,
  "dimension_reasoning": {{
    "maintenance_status_accuracy": "<one sentence citing specific evidence>",
    "evidence_coverage": "<one sentence citing specific evidence>",
    "specificity_and_detail": "<one sentence citing specific evidence>",
    "output_structure_and_clarity": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "maintenance_status_accuracy": 0.35,
    "evidence_coverage": 0.30,
    "specificity_and_detail": 0.25,
    "output_structure_and_clarity": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())