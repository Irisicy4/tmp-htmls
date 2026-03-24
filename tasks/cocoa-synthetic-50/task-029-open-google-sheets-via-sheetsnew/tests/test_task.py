"""
LLM-as-judge evaluator for EvolveBench task.

Category: HR & Recruiting
Task: Build a candidate sourcing tracker for a software engineering role using data from multiple platforms.
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


TASK_INSTRUCTION = """Open Google Sheets (via sheets.new) and build a candidate sourcing tracker for a software engineering role. Use job posting data from Stack Overflow Jobs, LinkedIn, and AngelList to fill the sheet. Create columns for Candidate Name, Role, Contact Info, Source Platform, Resume Link, and Status (e.g., Applied, Interviewing, Hired). Populate the sheet with information for 10 candidates you identify through these platforms."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves creating a candidate sourcing tracker for a software engineering role. The agent must gather data from Stack Overflow Jobs, LinkedIn, and AngelList, and organize it into a Google Sheet with specific columns. A successful completion requires data for 10 candidates, with all required fields populated.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Open Google Sheets (via sheets.new) and build a candidate sourcing tracker for a software engineering role. Use job posting data from Stack Overflow Jobs, LinkedIn, and AngelList to fill the sheet. Create columns for Candidate Name, Role, Contact Info, Source Platform, Resume Link, and Status (e.g., Applied, Interviewing, Hired). Populate the sheet with information for 10 candidates you identify through these platforms.

## Task-Specific Constraints
- Must visit Stack Overflow Jobs, LinkedIn, and AngelList to source candidates.
- Must create a Google Sheet with the specified columns.
- Must populate data for exactly 10 candidates.
- Each candidate must have all columns filled (no missing data).
- The "Source Platform" column must indicate the platform where the candidate was found.
- The "Status" column must include one of the following: Applied, Interviewing, Hired.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Stack Overflow Jobs, LinkedIn, and AngelList? Which platforms were actually visited?
- Did the agent create a Google Sheet with the specified columns?
- Did the agent populate data for exactly 10 candidates?
- Are all required fields (Candidate Name, Role, Contact Info, Source Platform, Resume Link, Status) filled for each candidate?
- Is the "Source Platform" column accurate and consistent with the platforms visited?

### Step 2: Dimension Scoring

#### A. Deliverable Completeness (0.35)
Measures whether the Google Sheet contains all required data for 10 candidates.

5 — All 10 candidates are included with every required field filled.
4 — 8-9 candidates are included, or some fields are incomplete.
3 — 6-7 candidates are included, or many fields are incomplete.
2 — Fewer than 6 candidates are included, or most fields are missing.
1 — No candidates or data were included.

#### B. Platform Coverage (0.30)
Measures whether the agent used all three specified platforms.

5 — All three platforms (Stack Overflow Jobs, LinkedIn, AngelList) were used.
4 — Two platforms were used.
3 — Only one platform was used.
2 — No platforms were used, but some attempt was made to gather data.
1 — No attempt to use any platform.

#### C. Data Accuracy and Specificity (0.20)
Measures whether the data provided is accurate and specific.

5 — All data is accurate, specific, and matches the task requirements.
4 — Most data is accurate, but some minor errors or omissions exist.
3 — Data contains noticeable inaccuracies or lacks specificity.
2 — Data is mostly incorrect or vague.
1 — Data is entirely incorrect or irrelevant.

#### D. Output Organization (0.15)
Measures whether the Google Sheet is well-structured and easy to interpret.

5 — The sheet is well-organized, with clear formatting and no errors.
4 — The sheet is mostly well-organized, with minor formatting issues.
3 — The sheet is somewhat disorganized but usable.
2 — The sheet is poorly organized and difficult to interpret.
1 — The sheet is entirely disorganized or unusable.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "deliverable_completeness": <1-5>,
  "platform_coverage": <1-5>,
  "data_accuracy_and_specificity": <1-5>,
  "output_organization": <1-5>,
  "dimension_reasoning": {{
    "deliverable_completeness": "<one sentence citing specific evidence>",
    "platform_coverage": "<one sentence citing specific evidence>",
    "data_accuracy_and_specificity": "<one sentence citing specific evidence>",
    "output_organization": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "deliverable_completeness": 0.35,
    "platform_coverage": 0.30,
    "data_accuracy_and_specificity": 0.20,
    "output_organization": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())