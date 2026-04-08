"""
LLM-as-judge evaluator for EvolveBench task.

Category: Software Engineering
Task: Extract and summarize major breaking changes from React 18 and Angular 16 release notes.
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


TASK_INSTRUCTION = """Go to the GitHub release pages for React and Angular, and extract all major breaking changes listed for versions React 18 and Angular 16. Include URLs of the release notes and summarize the context of each breaking change."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to navigate to the GitHub release pages for React and Angular, identify major breaking changes for React 18 and Angular 16, and summarize them. The deliverable must include URLs to the release notes and clear context for each breaking change.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Go to the GitHub release pages for React and Angular, and extract all major breaking changes listed for versions React 18 and Angular 16. Include URLs of the release notes and summarize the context of each breaking change.

## Task-Specific Constraints
- Must navigate to the GitHub release pages for both React and Angular.
- Must identify breaking changes specifically for React 18 and Angular 16.
- Must include URLs to the release notes for both versions.
- Must summarize the context of each breaking change in clear language.
- Output must be structured as a list or table for readability.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to the GitHub release pages for React and Angular?
- Are the breaking changes for React 18 and Angular 16 clearly identified?
- Are URLs to the release notes included in the response?
- Is the context for each breaking change summarized clearly?
- Is the output structured as a readable list or table?

### Step 2: Dimension Scoring

#### A. Deliverable Accuracy (0.35)
Measures whether the breaking changes for React 18 and Angular 16 are correctly identified and summarized.

5 — All breaking changes are correctly identified and summarized with accurate context.
4 — Most breaking changes are correctly identified and summarized, with minor omissions or inaccuracies.
3 — Some breaking changes are identified and summarized, but significant omissions or inaccuracies exist.
2 — Few breaking changes are identified, with major omissions or inaccuracies.
1 — No breaking changes are identified or summarized.

#### B. Coverage of Sources (0.30)
Measures whether the agent navigated to the required GitHub release pages and included URLs for both React and Angular.

5 — URLs for both React 18 and Angular 16 release notes are included, and both pages were clearly visited.
4 — URLs for both release notes are included, but evidence of visiting one page is unclear.
3 — URLs for at least one release note are included, with partial evidence of navigation.
2 — URLs are missing or navigation evidence is mostly absent.
1 — No URLs or navigation evidence provided.

#### C. Context Depth (0.20)
Measures the depth and clarity of the context provided for each breaking change.

5 — Context for all breaking changes is detailed, clear, and specific.
4 — Context for most breaking changes is clear, with minor gaps in detail.
3 — Context for some breaking changes is provided, but lacks depth or clarity.
2 — Context is mostly unclear or missing for breaking changes.
1 — No context is provided.

#### D. Output Structure (0.15)
Measures whether the response is well-organized and easy to read.

5 — Output is structured as a clear list or table, with excellent readability.
4 — Output is structured, but readability could be improved slightly.
3 — Output is partially structured, but lacks clarity or consistency.
2 — Output is poorly structured and hard to read.
1 — Output is unstructured or absent.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "deliverable_accuracy": <1-5>,
  "coverage_of_sources": <1-5>,
  "context_depth": <1-5>,
  "output_structure": <1-5>,
  "dimension_reasoning": {{
    "deliverable_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_sources": "<one sentence citing specific evidence>",
    "context_depth": "<one sentence citing specific evidence>",
    "output_structure": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "deliverable_accuracy": 0.35,
    "coverage_of_sources": 0.30,
    "context_depth": 0.20,
    "output_structure": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())