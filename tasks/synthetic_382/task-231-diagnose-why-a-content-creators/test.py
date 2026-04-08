"""
LLM-as-judge evaluator for EvolveBench task.

Category: (Self) Media
Task: Diagnose why a content creator's Instagram reels have experienced a sudden drop in reach over the past two weeks.
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


TASK_INSTRUCTION = """Diagnose why a content creator's Instagram reels have experienced a sudden drop in reach over the past two weeks. Check Instagram's public documentation, recent patch notes, and creator feedback on community forums for potential algorithm changes or feature updates."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves diagnosing why a content creator's Instagram reels have experienced a sudden drop in reach over the past two weeks. The agent must investigate algorithm changes or feature updates by consulting Instagram's public documentation, recent patch notes, and creator feedback on community forums.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Diagnose why a content creator's Instagram reels have experienced a sudden drop in reach over the past two weeks. Check Instagram's public documentation, recent patch notes, and creator feedback on community forums for potential algorithm changes or feature updates.

## Task-Specific Constraints
- Must visit at least 3 of the specified platforms: instagram.com, reddit.com, medium.com.
- Must identify specific algorithm changes or feature updates affecting reach.
- Must include evidence or citations from credible sources (e.g., patch notes, official documentation, or community feedback).
- Output must be structured as a clear list or table summarizing findings.
- Must address whether the drop is due to external factors (e.g., platform-wide changes) or creator-specific issues.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to the required platforms? Which ones were actually visited?
- Did the agent identify specific algorithm changes or feature updates affecting reach?
- Are citations or evidence from credible sources included in the response?
- Is the output organized as a clear list or table summarizing findings?
- Does the response address whether the drop is due to external factors or creator-specific issues?

### Step 2: Dimension Scoring

#### A. Primary Deliverable Accuracy (0.35)
Measures whether the agent correctly identified algorithm changes or feature updates affecting reach.

5 — Identifies 3 or more specific changes/updates with credible evidence.
4 — Identifies 2 specific changes/updates with credible evidence.
3 — Identifies 1 specific change/update with credible evidence.
2 — Identifies vague or unsupported changes/updates.
1 — No relevant changes/updates identified.

#### B. Coverage of Required Platforms (0.30)
Measures whether the agent visited all required platforms and used them effectively.

5 — Visited all 3 platforms and extracted relevant information from each.
4 — Visited 2 platforms and extracted relevant information.
3 — Visited 1 platform and extracted relevant information.
2 — Visited platforms but extracted little/no relevant information.
1 — Did not visit any required platforms.

#### C. Depth and Specificity (0.25)
Measures the level of detail and specificity in the agent's findings.

5 — Provides detailed findings with specific examples and comparisons.
4 — Provides moderately detailed findings with some examples.
3 — Provides basic findings with limited detail or examples.
2 — Provides vague findings with minimal detail.
1 — Provides no meaningful findings.

#### D. Source Credibility and Output Structure (0.10)
Measures the credibility of sources and organization of the output.

5 — All sources are credible, and the output is well-organized as a table or structured list.
4 — Most sources are credible, and the output is moderately organized.
3 — Some sources are credible, and the output is minimally organized.
2 — Few sources are credible, and the output is poorly organized.
1 — No credible sources, and the output is disorganized.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "primary_deliverable_accuracy": <1-5>,
  "coverage_of_required_platforms": <1-5>,
  "depth_and_specificity": <1-5>,
  "source_credibility_and_output_structure": <1-5>,
  "dimension_reasoning": {{
    "primary_deliverable_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_required_platforms": "<one sentence citing specific evidence>",
    "depth_and_specificity": "<one sentence citing specific evidence>",
    "source_credibility_and_output_structure": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "primary_deliverable_accuracy": 0.35,
    "coverage_of_required_platforms": 0.30,
    "depth_and_specificity": 0.25,
    "source_credibility_and_output_structure": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())