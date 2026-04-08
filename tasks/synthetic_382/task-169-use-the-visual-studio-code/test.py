"""
LLM-as-judge evaluator for EvolveBench task.

Category: Software Engineering
Task: Evaluate whether the agent successfully identified and reported the top five free Python development extensions on the Visual Studio Code Marketplace, meeting specific criteria.
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


TASK_INSTRUCTION = """Use the Visual Studio Code Marketplace to search for extensions compatible with Python development. Apply filters to display only free extensions and select those with at least 100,000 downloads. Report the names and descriptions of the top five results."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to use the Visual Studio Code Marketplace to identify free Python development extensions with at least 100,000 downloads. The agent must report the names and descriptions of the top five results. A successful completion involves accurate filtering, correct identification of extensions, and clear reporting of the required information.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Use the Visual Studio Code Marketplace to search for extensions compatible with Python development. Apply filters to display only free extensions and select those with at least 100,000 downloads. Report the names and descriptions of the top five results.

## Task-Specific Constraints
- Must use the Visual Studio Code Marketplace platform.
- Must apply filters to display only free extensions.
- Must correctly identify extensions with at least 100,000 downloads.
- Must report the names and descriptions of the top five results.
- Output must be organized as a structured list or table.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to the Visual Studio Code Marketplace?
- Did the agent apply the required filters (free extensions, 100,000+ downloads)?
- Are the names and descriptions of the top five extensions present in the response?
- Is the output organized as a structured list or table?
- Are the reported extensions accurate and match the filtering criteria?

### Step 2: Dimension Scoring

#### A. Filtering Accuracy (0.35)
Measures whether the agent correctly applied the required filters on the Visual Studio Code Marketplace.

5 — All filters (free, 100,000+ downloads) applied correctly, no errors.
4 — Filters mostly correct, with minor inaccuracies.
3 — Filters partially correct, with significant omissions or errors.
2 — Filters mostly incorrect or missing.
1 — No filters applied.

#### B. Extension Identification (0.30)
Measures whether the agent correctly identified the top five extensions matching the criteria.

5 — All five extensions correctly identified and match criteria.
4 — Four extensions correctly identified, one minor error.
3 — Three extensions correctly identified, significant errors for others.
2 — One or two extensions correctly identified, most incorrect.
1 — No correct extensions identified.

#### C. Output Structure (0.20)
Measures the clarity and organization of the agent's response.

5 — Output is well-organized as a structured list or table, easy to read.
4 — Output mostly organized, minor formatting issues.
3 — Output partially organized, difficult to follow in places.
2 — Output poorly organized, hard to understand.
1 — Output completely disorganized or absent.

#### D. Evidence Credibility (0.15)
Measures whether the agent's response is supported by credible evidence from the platform.

5 — All reported extensions are accurate and match the filtering criteria.
4 — Most reported extensions are accurate, with minor discrepancies.
3 — Some reported extensions are accurate, significant discrepancies for others.
2 — Few reported extensions are accurate, most are incorrect.
1 — No reported extensions are accurate.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{
  "evidence_summary": "The agent successfully navigated to the Visual Studio Code Marketplace and applied filters for free extensions with 100,000+ downloads. The response includes names and descriptions of five extensions, organized as a structured list. Minor inaccuracies were found in filtering criteria.",
  "filtering_accuracy": 4,
  "extension_identification": 4,
  "output_structure": 5,
  "evidence_credibility": 4,
  "dimension_reasoning": {
    "filtering_accuracy": "Filters were mostly correct, but one minor discrepancy was noted.",
    "extension_identification": "Four extensions matched criteria, one had a minor error.",
    "output_structure": "The response is organized as a structured list, easy to read.",
    "evidence_credibility": "Most extensions reported are accurate, with minor discrepancies."
  },
  "overall_score": 4.15,
  "passed": true
}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "filtering_accuracy": 0.35,
    "extension_identification": 0.30,
    "output_structure": 0.20,
    "evidence_credibility": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())