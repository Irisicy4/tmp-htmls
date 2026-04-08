"""
LLM-as-judge evaluator for EvolveBench task.

Category: Design
Task: Search for modern office layout illustrations on Shutterstock with specific filters and report top three results.
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


TASK_INSTRUCTION = """Visit the Shutterstock website and search for illustrations of modern office layouts using filters for 'vector' format and 'blue' as the dominant color. Complete the workflow to narrow results and report the final top three illustrations by title and URL."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to search for modern office layout illustrations on Shutterstock using specific filters: 'vector' format and 'blue' as the dominant color. The agent must report the top three results by title and URL. A successful completion includes correctly applying the filters and providing accurate, structured output.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Visit the Shutterstock website and search for illustrations of modern office layouts using filters for 'vector' format and 'blue' as the dominant color. Complete the workflow to narrow results and report the final top three illustrations by title and URL.

## Task-Specific Constraints
- Must apply the 'vector' format filter correctly.
- Must apply the 'blue' dominant color filter correctly.
- Must report exactly three illustrations, including their titles and URLs.
- Output must be structured as a list or table.
- Titles and URLs must match the actual search results from Shutterstock.
- Must demonstrate evidence of navigating Shutterstock and applying filters.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Shutterstock and apply the required filters ('vector' format and 'blue' dominant color)?
- Are exactly three illustrations reported, including titles and URLs?
- Does the output match the required structured format (list or table)?
- Are the reported titles and URLs accurate and sourced from Shutterstock?
- Is there evidence in the tool-call trace that the agent completed the workflow correctly?

### Step 2: Dimension Scoring

#### A. Filter Application Accuracy (0.35)
Measures whether the agent correctly applied the 'vector' format and 'blue' dominant color filters.

5 — Both filters applied correctly, with evidence in the tool-call trace.
4 — Filters applied correctly, but evidence is incomplete or unclear.
3 — At least one filter applied correctly, but the other is missing or incorrect.
2 — Filters mostly missing or incorrectly applied.
1 — No filters applied.

#### B. Result Completeness (0.30)
Measures whether the agent reported exactly three illustrations with titles and URLs.

5 — Reports exactly three illustrations with accurate titles and URLs.
4 — Reports three illustrations, but one or more titles/URLs are slightly inaccurate.
3 — Reports fewer than three illustrations or has significant inaccuracies in titles/URLs.
2 — Reports one illustration or mostly inaccurate data.
1 — No illustrations reported.

#### C. Output Structure (0.20)
Measures whether the output is organized as a structured list or table.

5 — Output is fully structured as a clear list or table.
4 — Output is mostly structured but has minor formatting issues.
3 — Output is partially structured but lacks clarity.
2 — Output is poorly structured or disorganized.
1 — Output is completely unstructured.

#### D. Evidence Credibility (0.15)
Measures whether the agent's response and tool-call trace demonstrate credible evidence of task completion.

5 — Evidence is fully credible and matches the task requirements.
4 — Evidence is mostly credible but has minor gaps.
3 — Evidence is partially credible but lacks clarity or completeness.
2 — Evidence is mostly missing or unclear.
1 — No credible evidence provided.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "filter_application_accuracy": <1-5>,
  "result_completeness": <1-5>,
  "output_structure": <1-5>,
  "evidence_credibility": <1-5>,
  "dimension_reasoning": {{
    "filter_application_accuracy": "<one sentence citing specific evidence>",
    "result_completeness": "<one sentence citing specific evidence>",
    "output_structure": "<one sentence citing specific evidence>",
    "evidence_credibility": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "filter_application_accuracy": 0.35,
    "result_completeness": 0.30,
    "output_structure": 0.20,
    "evidence_credibility": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())