"""
LLM-as-judge evaluator for EvolveBench task-94.

Category: Data & ML Engineering
Task: Go to https://www.certosa.com/Services/FastTrac/Index.aspx, search for ID 17161, and give me a summary of the results.
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


TASK_INSTRUCTION = """Go to https://www.certosa.com/Services/FastTrac/Index.aspx, search for ID 17161, and give me a summary of the results."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves navigating a specific web form, entering a search ID, and summarizing the results returned.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- URL: must navigate to the exact URL provided
- Search: ID 17161 specifically
- Output: summary of whatever results the search returns
- If the site is unavailable or the ID returns no results, this should be noted

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to the certosa.com FastTrac URL?
- Was the search form found and ID 17161 entered?
- What results were returned?
- Is the summary accurate to the returned data?

### Step 2: Dimension Scoring

#### A. Site Navigation (0.25)
Did the agent navigate to the correct URL?

5 — Agent navigated to certosa.com/Services/FastTrac/Index.aspx successfully.
4 — Agent reached certosa.com but had difficulty finding FastTrac.
3 — Agent found a related certosa.com page but not FastTrac.
2 — Agent described the URL without navigating.
1 — No navigation.

#### B. Search Execution (0.3)
Was the search for ID 17161 executed?

5 — ID 17161 entered in the search form and search submitted.
4 — Search attempted but form interaction had issues.
3 — Agent found the form but did not submit the search.
2 — Agent described the search without executing it.
1 — No search.

#### C. Result Accuracy (0.35)
Are the reported results accurate?

5 — Results accurately summarized from the actual page response.
4 — Results mostly accurate with minor gaps.
3 — Partial results reported.
2 — Results described without accuracy confidence.
1 — No results reported or site unavailable without noting it.

#### D. Summary Quality (0.1)
Is the summary well-written?

5 — Clear, concise summary covering key result fields.
4 — Good summary with minor gaps.
3 — Adequate but verbose.
2 — Incomplete or confusing.
1 — No summary.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "site_navigation": <1-5>,
  "search_execution": <1-5>,
  "result_accuracy": <1-5>,
  "summary_quality": <1-5>,
  "dimension_reasoning": {{
    "site_navigation": "<one sentence citing specific evidence>",
    "search_execution": "<one sentence citing specific evidence>",
    "result_accuracy": "<one sentence citing specific evidence>",
    "summary_quality": "<one sentence citing specific evidence>",
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "site_navigation": 0.25,
    "search_execution": 0.3,
    "result_accuracy": 0.35,
    "summary_quality": 0.1,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())