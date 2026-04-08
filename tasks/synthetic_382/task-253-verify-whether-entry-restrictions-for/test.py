"""
LLM-as-judge evaluator for EvolveBench task.

Category: Travel & Planning
Task: Verify whether entry restrictions for US citizens traveling to Thailand have changed in the past month by checking official government advisories and reporting current requirements and updates.
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


TASK_INSTRUCTION = """Verify whether entry restrictions for US citizens traveling to Thailand have changed in the past month. Check official government advisories on Travel.State.Gov, the Thai Embassy website, and IATA's Travel Restrictions Map. Report the current requirements and any updates found."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to verify entry restrictions for US citizens traveling to Thailand by consulting three specific platforms: Travel.State.Gov, the Thai Embassy website, and IATA's Travel Restrictions Map. The deliverable is a report summarizing the current requirements and highlighting any changes in the past month.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Verify whether entry restrictions for US citizens traveling to Thailand have changed in the past month. Check official government advisories on Travel.State.Gov, the Thai Embassy website, and IATA's Travel Restrictions Map. Report the current requirements and any updates found.

## Task-Specific Constraints
- Must visit all three specified platforms: Travel.State.Gov, Thai Embassy website, and IATA Travel Restrictions Map.
- Must explicitly mention whether entry restrictions have changed in the past month.
- Must provide the current entry requirements for US citizens traveling to Thailand.
- Must cite sources for all claims made in the response.
- Output must be organized as a structured list or table.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to all three required platforms? Which ones were actually visited?
- Does the response explicitly state whether entry restrictions have changed in the past month?
- Are the current entry requirements for US citizens traveling to Thailand included in the response?
- Are all claims in the response supported by cited sources?
- Is the output organized as a structured list or table?

### Step 2: Dimension Scoring

#### A. Deliverable Accuracy (0.35)
Measures whether the agent correctly and completely reports the current entry requirements and updates.

5 — Reports all current entry requirements and explicitly states whether restrictions have changed, with no errors.
4 — Reports most requirements and mentions updates, but minor details are missing or slightly incorrect.
3 — Reports some requirements and mentions updates, but significant details are missing or incorrect.
2 — Reports few requirements or fails to mention updates, with major errors.
1 — Fails to report requirements or updates entirely.

#### B. Platform Coverage (0.30)
Measures whether the agent visited all required platforms and used them effectively.

5 — Successfully navigates all three platforms and extracts relevant information from each.
4 — Navigates at least two platforms and extracts most relevant information.
3 — Navigates at least one platform and extracts some relevant information.
2 — Navigates one platform but fails to extract relevant information.
1 — Fails to navigate any platform or extract information.

#### C. Specificity of Details (0.25)
Measures the depth and specificity of the response, including cited sources and detailed requirements.

5 — Provides highly detailed requirements, cites sources for all claims, and includes specific updates.
4 — Provides detailed requirements, cites sources for most claims, and includes updates.
3 — Provides basic requirements with limited detail, cites few sources, and mentions updates.
2 — Provides vague or incomplete requirements, with little or no citation of sources.
1 — Provides no specific details or citations.

#### D. Output Structure and Credibility (0.10)
Measures whether the response is well-organized and uses credible sources.

5 — Output is structured as a clear list or table, with all sources credible and clearly cited.
4 — Output is mostly structured, with minor organizational issues or unclear citations.
3 — Output is minimally structured, with some credible sources cited.
2 — Output is poorly structured, with few credible sources cited.
1 — Output is unstructured and lacks credible sources.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "deliverable_accuracy": <1-5>,
  "platform_coverage": <1-5>,
  "specificity_of_details": <1-5>,
  "output_structure_and_credibility": <1-5>,
  "dimension_reasoning": {{
    "deliverable_accuracy": "<one sentence citing specific evidence>",
    "platform_coverage": "<one sentence citing specific evidence>",
    "specificity_of_details": "<one sentence citing specific evidence>",
    "output_structure_and_credibility": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "deliverable_accuracy": 0.35,
    "platform_coverage": 0.30,
    "specificity_of_details": 0.25,
    "output_structure_and_credibility": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())