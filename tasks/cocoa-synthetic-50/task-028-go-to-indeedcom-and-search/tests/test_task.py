"""
LLM-as-judge evaluator for EvolveBench task.

Category: HR & Recruiting
Task: Extract and record details of Technical Recruiter job postings in NYC from Indeed.com based on specific filters.
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


TASK_INSTRUCTION = """Go to Indeed.com and search for 'Technical Recruiter' roles in New York City posted in the last 14 days. Apply filters for salaries above $80,000 and job types marked as full-time. Extract the first 10 matching job postings and record the following details for each: company name, job title, salary range, work mode (remote/hybrid/on-site), and posting date."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves extracting job postings for 'Technical Recruiter' roles in New York City from Indeed.com, applying specific filters (posted in the last 14 days, salary above $80,000, full-time), and recording details for the first 10 matches. The deliverable must include company name, job title, salary range, work mode, and posting date for each job.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Go to Indeed.com and search for 'Technical Recruiter' roles in New York City posted in the last 14 days. Apply filters for salaries above $80,000 and job types marked as full-time. Extract the first 10 matching job postings and record the following details for each: company name, job title, salary range, work mode (remote/hybrid/on-site), and posting date.

## Task-Specific Constraints
- Must use Indeed.com as the primary platform for job search.
- Must apply filters for salary above $80,000 and job type as full-time.
- Must include only jobs posted in the last 14 days.
- Must extract and record details for exactly 10 job postings.
- Output must include company name, job title, salary range, work mode, and posting date for each job.
- Output must be organized as a structured list or table.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Indeed.com and apply the required filters?
- Are the extracted job postings limited to the last 14 days and full-time roles with salaries above $80,000?
- Does the output include exactly 10 job postings with all required details (company name, job title, salary range, work mode, posting date)?
- Is the output organized as a structured list or table?
- Are there any missing or incorrect details in the extracted data?

### Step 2: Dimension Scoring

#### A. Primary Deliverable Accuracy (0.35)
Measures whether the agent correctly extracted and recorded the required job details.

5 — All 10 job postings are correct, complete, and match the filters.
4 — 8-9 job postings are correct and complete, with minor errors.
3 — 6-7 job postings are correct, but some details are missing or incorrect.
2 — Fewer than 6 job postings are correct, with significant errors.
1 — No correct job postings or completely wrong output.

#### B. Coverage of Filters and Platforms (0.30)
Measures whether the agent used Indeed.com and applied all required filters.

5 — Used Indeed.com and applied all filters (salary, job type, date).
4 — Used Indeed.com and applied most filters, with minor omissions.
3 — Used Indeed.com but missed one or more key filters.
2 — Used Indeed.com but applied filters incorrectly or inconsistently.
1 — Did not use Indeed.com or failed to apply filters.

#### C. Detail and Specificity (0.20)
Measures whether the extracted data includes all required details for each job posting.

5 — All postings include company name, job title, salary range, work mode, and posting date.
4 — Most postings include all details, with minor omissions.
3 — Some postings include all details, but others are incomplete.
2 — Few postings include all details, with significant omissions.
1 — No postings include the required details.

#### D. Output Structure and Organization (0.15)
Measures whether the output is well-organized and easy to interpret.

5 — Output is a structured list or table, clear and well-formatted.
4 — Output is mostly structured, with minor formatting issues.
3 — Output is partially structured but somewhat disorganized.
2 — Output is poorly structured and difficult to interpret.
1 — Output is unstructured or unreadable.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "primary_deliverable_accuracy": <1-5>,
  "coverage_of_filters_and_platforms": <1-5>,
  "detail_and_specificity": <1-5>,
  "output_structure_and_organization": <1-5>,
  "dimension_reasoning": {{
    "primary_deliverable_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_filters_and_platforms": "<one sentence citing specific evidence>",
    "detail_and_specificity": "<one sentence citing specific evidence>",
    "output_structure_and_organization": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "primary_deliverable_accuracy": 0.35,
    "coverage_of_filters_and_platforms": 0.30,
    "detail_and_specificity": 0.20,
    "output_structure_and_organization": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())