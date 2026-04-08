"""
LLM-as-judge evaluator for EvolveBench task.

Category: (Self) Media
Task: Research and compare the monetization terms, audience reach, and feature sets of Patreon, Substack, and Ko-fi, and summarize findings in a comparison table.
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


TASK_INSTRUCTION = """Research and compare the monetization terms, audience reach, and feature sets of Patreon, Substack, and Ko-fi as platforms for creators looking to monetize their content. Summarize your findings in a comparison table."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to research and compare the monetization terms, audience reach, and feature sets of Patreon, Substack, and Ko-fi. The deliverable must be a structured comparison table summarizing the findings. A successful completion includes accurate data from all three platforms, clear comparisons, and adherence to the required format.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Research and compare the monetization terms, audience reach, and feature sets of Patreon, Substack, and Ko-fi as platforms for creators looking to monetize their content. Summarize your findings in a comparison table.

## Task-Specific Constraints
- Must visit all three platforms: Patreon, Substack, and Ko-fi.
- Must include monetization terms, audience reach, and feature sets for all platforms.
- Output must be organized as a comparison table.
- Must include specific data points (e.g., fees, audience size, key features).
- Must provide accurate and sourced information.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to all three required platforms (Patreon, Substack, Ko-fi)?
- Does the response include monetization terms, audience reach, and feature sets for all platforms?
- Is the output organized as a comparison table?
- Are specific data points (e.g., fees, audience size, key features) present and accurate?
- Is the information sourced or verifiable?

### Step 2: Dimension Scoring

#### A. Deliverable Accuracy (0.35)
Measures whether the main output (comparison table) is correct and complete.

5 — Includes accurate and complete data for all three platforms, with no errors.
4 — Includes mostly accurate data, with minor omissions or errors.
3 — Includes partially accurate data but lacks significant details or has notable errors.
2 — Includes mostly incorrect or incomplete data.
1 — No meaningful data provided.

#### B. Coverage of Required Platforms (0.30)
Measures whether the agent covered all three platforms and included required elements.

5 — Covers all three platforms and includes monetization terms, audience reach, and feature sets for each.
4 — Covers all three platforms but misses minor details in one or more categories.
3 — Covers at least two platforms but lacks significant details or categories.
2 — Covers only one platform or provides very limited information.
1 — Does not cover any platform meaningfully.

#### C. Depth of Comparison (0.20)
Measures the level of detail and specificity in the comparisons.

5 — Provides detailed comparisons with specific data points (e.g., fees, audience size, features).
4 — Provides comparisons with some specific data points but lacks depth in some areas.
3 — Provides basic comparisons but lacks specific data points or depth.
2 — Provides minimal comparisons with very little detail.
1 — No meaningful comparisons provided.

#### D. Output Structure and Clarity (0.15)
Measures the organization and clarity of the output.

5 — Output is well-organized as a clear comparison table, easy to read and understand.
4 — Output is mostly well-organized but has minor formatting or clarity issues.
3 — Output is somewhat organized but difficult to follow or incomplete.
2 — Output is poorly organized and hard to understand.
1 — Output is not organized in a meaningful way.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "deliverable_accuracy": <1-5>,
  "coverage_of_required_platforms": <1-5>,
  "depth_of_comparison": <1-5>,
  "output_structure_and_clarity": <1-5>,
  "dimension_reasoning": {{
    "deliverable_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_required_platforms": "<one sentence citing specific evidence>",
    "depth_of_comparison": "<one sentence citing specific evidence>",
    "output_structure_and_clarity": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "deliverable_accuracy": 0.35,
    "coverage_of_required_platforms": 0.30,
    "depth_of_comparison": 0.20,
    "output_structure_and_clarity": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())