"""
LLM-as-judge evaluator for EvolveBench task.

Category: Design
Task: Research and compare three popular icon libraries for UI design and summarize findings in a comparison table.
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


TASK_INSTRUCTION = """Research and compare three popular icon libraries for UI design: Material Design Icons, Font Awesome, and Feather Icons. Focus on the following aspects: total icon count, licensing terms, and availability of outlined vs. filled styles. Summarize your findings in a comparison table."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to research and compare three popular icon libraries for UI design: Material Design Icons, Font Awesome, and Feather Icons. The agent must provide information on total icon count, licensing terms, and availability of outlined vs. filled styles. A successful completion requires summarizing findings in a structured comparison table.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Research and compare three popular icon libraries for UI design: Material Design Icons, Font Awesome, and Feather Icons. Focus on the following aspects: total icon count, licensing terms, and availability of outlined vs. filled styles. Summarize your findings in a comparison table.

## Task-Specific Constraints
- Must visit material.io, fontawesome.com, and feathericons.com.
- Must include total icon count for each library.
- Must describe licensing terms for each library.
- Must specify availability of outlined vs. filled styles for each library.
- Output must be organized as a comparison table.
- Must provide accurate and sourced information.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to material.io, fontawesome.com, and feathericons.com? Which ones were actually visited?
- Does the response include total icon count for all three libraries?
- Are licensing terms described for all three libraries?
- Does the response specify availability of outlined vs. filled styles for all three libraries?
- Is the output organized as a comparison table?

### Step 2: Dimension Scoring

#### A. Deliverable Accuracy (0.35)
Measures whether the comparison table is complete and accurate.

5 — Table includes all required data (icon count, licensing terms, outlined vs. filled styles) for all three libraries.
4 — Table includes most required data but has minor omissions or inaccuracies.
3 — Table includes partial data but is missing key elements or contains notable inaccuracies.
2 — Table is mostly incomplete or inaccurate.
1 — Table is absent or completely wrong.

#### B. Coverage of Sources (0.30)
Measures whether the agent visited all required platforms and used them effectively.

5 — Agent visited all three platforms and sourced data from each.
4 — Agent visited at least two platforms and sourced most required data.
3 — Agent visited at least one platform and sourced partial data.
2 — Agent visited platforms but sourced little to no usable data.
1 — Agent did not visit any required platforms.

#### C. Depth of Information (0.25)
Measures the level of detail and specificity in the response.

5 — Response includes detailed numbers, licensing terms, and style availability for all libraries.
4 — Response includes most details but lacks minor specifics.
3 — Response includes partial details but omits significant specifics.
2 — Response includes very few details or is vague.
1 — Response lacks meaningful details entirely.

#### D. Output Structure and Credibility (0.10)
Measures whether the output is well-organized and sources are credible.

5 — Output is organized as a clear comparison table with credible sourcing.
4 — Output is mostly organized but has minor structural or credibility issues.
3 — Output is partially organized but lacks clarity or credible sourcing.
2 — Output is poorly organized or lacks credible sourcing.
1 — Output is disorganized and lacks credibility entirely.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "deliverable_accuracy": <1-5>,
  "coverage_of_sources": <1-5>,
  "depth_of_information": <1-5>,
  "output_structure_and_credibility": <1-5>,
  "dimension_reasoning": {{
    "deliverable_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_sources": "<one sentence citing specific evidence>",
    "depth_of_information": "<one sentence citing specific evidence>",
    "output_structure_and_credibility": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "deliverable_accuracy": 0.35,
    "coverage_of_sources": 0.30,
    "depth_of_information": 0.25,
    "output_structure_and_credibility": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())