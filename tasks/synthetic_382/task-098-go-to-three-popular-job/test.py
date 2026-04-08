"""
LLM-as-judge evaluator for EvolveBench task.

Category: Daily Activities
Task: Extract top 5 'Software Engineer' job listings in San Francisco with salaries over $100,000 from three job posting platforms.
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


TASK_INSTRUCTION = """Go to three popular job posting sites, filter for 'Software Engineer' jobs in San Francisco with salaries over $100,000, and extract the top 5 listings from each. Include job title, company name, and salary range."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to extract job listings for 'Software Engineer' roles in San Francisco with salaries over $100,000 from three specified job posting platforms. A successful completion requires the agent to provide the top 5 listings from each platform, including job title, company name, and salary range.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Go to three popular job posting sites, filter for 'Software Engineer' jobs in San Francisco with salaries over $100,000, and extract the top 5 listings from each. Include job title, company name, and salary range.

## Task-Specific Constraints
- Must visit indeed.com, linkedin.com/jobs, and glassdoor.com.
- Must filter for 'Software Engineer' jobs in San Francisco.
- Must filter for salary ranges above $100,000.
- Must extract exactly 5 listings per platform.
- Output must include job title, company name, and salary range.
- Output must be organized as a structured list or table.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to indeed.com, linkedin.com/jobs, and glassdoor.com? Which platforms were actually visited?
- Did the agent filter for 'Software Engineer' jobs in San Francisco with salaries over $100,000?
- Did the agent extract exactly 5 listings per platform?
- Are job title, company name, and salary range present for each listing?
- Is the output organized as a structured list or table?

### Step 2: Dimension Scoring

#### A. Deliverable Accuracy (0.35)
Measures whether the extracted job listings are correct and complete.

5 — All 15 listings (5 per platform) are correct, with accurate job titles, company names, and salary ranges.
4 — At least 12 listings are correct, with minor inaccuracies in details.
3 — At least 9 listings are correct, but some details are missing or incorrect.
2 — Fewer than 9 listings are correct, with significant missing or incorrect details.
1 — No correct listings provided.

#### B. Platform Coverage (0.30)
Measures whether the agent visited all required platforms and extracted listings.

5 — All three platforms were visited, and listings were extracted from each.
4 — Two platforms were visited, and listings were extracted from both.
3 — At least one platform was visited, with listings extracted.
2 — Platforms were visited, but no listings were extracted.
1 — No platforms were visited.

#### C. Filtering Specificity (0.25)
Measures whether the agent applied the correct filters (location, role, salary).

5 — Correct filters applied for all platforms (location, role, salary).
4 — Correct filters applied for at least two platforms.
3 — Correct filters applied for at least one platform.
2 — Filters applied incorrectly or inconsistently.
1 — No filters applied.

#### D. Output Structure (0.10)
Measures the organization and readability of the output.

5 — Output is organized as a clear table or structured list, easy to read.
4 — Output is mostly organized, with minor formatting issues.
3 — Output is partially organized, but difficult to interpret.
2 — Output is disorganized or poorly formatted.
1 — Output is absent or unreadable.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "deliverable_accuracy": <1-5>,
  "platform_coverage": <1-5>,
  "filtering_specificity": <1-5>,
  "output_structure": <1-5>,
  "dimension_reasoning": {{
    "deliverable_accuracy": "<one sentence citing specific evidence>",
    "platform_coverage": "<one sentence citing specific evidence>",
    "filtering_specificity": "<one sentence citing specific evidence>",
    "output_structure": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "deliverable_accuracy": 0.35,
    "platform_coverage": 0.30,
    "filtering_specificity": 0.25,
    "output_structure": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())