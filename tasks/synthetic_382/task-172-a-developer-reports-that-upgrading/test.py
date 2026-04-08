"""
LLM-as-judge evaluator for EvolveBench task.

Category: Software Engineering
Task: Investigate React v18.0 compatibility issues with React Router v5 and recommend a fix or workaround.
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


TASK_INSTRUCTION = """A developer reports that upgrading React to version 18.0 breaks compatibility with React Router v5. Investigate the root cause using GitHub issues, official React documentation, and community forums. Identify the affected version range and recommend a fix or workaround."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to investigate compatibility issues between React v18.0 and React Router v5. The agent must use GitHub issues, official React documentation, and community forums to identify the affected version range and recommend a fix or workaround. This is a Software Engineering task, and a successful completion involves providing a clear explanation of the root cause, version details, and actionable recommendations.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
A developer reports that upgrading React to version 18.0 breaks compatibility with React Router v5. Investigate the root cause using GitHub issues, official React documentation, and community forums. Identify the affected version range and recommend a fix or workaround.

## Task-Specific Constraints
- Must visit at least 3 of the specified platforms: reactjs.org, github.com/facebook/react, reactrouter.com, stackoverflow.com.
- Must identify the affected version range for both React and React Router.
- Must provide a clear explanation of the root cause of the compatibility issue.
- Must recommend a fix or workaround that is actionable and technically sound.
- Output must be organized as a structured list or table with version details and recommendations.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to at least 3 of the required platforms? Which ones were actually visited?
- Did the agent identify the affected version range for React and React Router?
- Is the root cause of the compatibility issue explained clearly and accurately?
- Is the recommended fix or workaround actionable and technically sound?
- Is the output organized as a structured list or table?

### Step 2: Dimension Scoring

#### A. Deliverable Accuracy (0.35)
Measures whether the agent correctly identified the version range, explained the root cause, and provided actionable recommendations.

5 — Identifies the exact version range for React and React Router, explains the root cause clearly, and provides a technically sound fix or workaround.
4 — Identifies the version range and root cause but with minor inaccuracies or omissions in the recommendations.
3 — Identifies the version range and root cause partially but lacks actionable recommendations.
2 — Provides vague or incorrect information about the version range or root cause; recommendations are unclear or impractical.
1 — Fails to identify the version range or root cause; no meaningful recommendations.

#### B. Coverage of Required Platforms (0.30)
Measures whether the agent visited at least 3 of the specified platforms and gathered relevant information.

5 — Visits all 4 specified platforms and uses information from each.
4 — Visits 3 platforms and uses information from each.
3 — Visits 2 platforms and uses information from at least one.
2 — Visits only 1 platform or gathers minimal information.
1 — Does not visit any of the required platforms.

#### C. Depth of Explanation (0.25)
Measures the level of detail in the explanation of the compatibility issue and recommendations.

5 — Provides highly detailed explanations with specific technical insights and examples.
4 — Provides detailed explanations but lacks some technical depth or examples.
3 — Provides basic explanations with minimal technical depth.
2 — Provides vague explanations with little technical insight.
1 — Provides no meaningful explanation.

#### D. Output Structure and Credibility (0.10)
Measures the organization of the output and the credibility of the sources used.

5 — Output is well-organized as a structured list or table, with credible sources cited.
4 — Output is organized but lacks full structure or source citations.
3 — Output is partially organized but difficult to follow; sources are unclear.
2 — Output is disorganized and lacks credibility.
1 — Output is completely unstructured and sources are absent.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "deliverable_accuracy": <1-5>,
  "coverage_of_required_platforms": <1-5>,
  "depth_of_explanation": <1-5>,
  "output_structure_and_credibility": <1-5>,
  "dimension_reasoning": {{
    "deliverable_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_required_platforms": "<one sentence citing specific evidence>",
    "depth_of_explanation": "<one sentence citing specific evidence>",
    "output_structure_and_credibility": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "deliverable_accuracy": 0.35,
    "coverage_of_required_platforms": 0.30,
    "depth_of_explanation": 0.25,
    "output_structure_and_credibility": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())