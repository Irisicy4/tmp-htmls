"""
LLM-as-judge evaluator for EvolveBench task.

Category: HR & Recruiting
Task: Verify whether Meta's headcount has decreased in the past quarter using evidence from specified platforms.
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


TASK_INSTRUCTION = """Verify whether Meta's headcount has decreased in the past quarter following recent news of layoffs. Check the latest Meta press release on their official website, LinkedIn company profile for updated employee counts, and the 'Layoffs' tracker section on layoffs.fyi for recent reports. Produce a verification note stating whether the headcount has decreased, including evidence from each source."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to verify whether Meta's headcount has decreased in the past quarter. The agent must consult three specific platforms: Meta's official press release page, LinkedIn's company profile for Meta, and the 'Layoffs' tracker section on layoffs.fyi. A successful completion includes a verification note stating whether the headcount has decreased, supported by evidence from all three platforms.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Verify whether Meta's headcount has decreased in the past quarter following recent news of layoffs. Check the latest Meta press release on their official website, LinkedIn company profile for updated employee counts, and the 'Layoffs' tracker section on layoffs.fyi for recent reports. Produce a verification note stating whether the headcount has decreased, including evidence from each source.

## Task-Specific Constraints
- Must visit all three specified platforms: Meta's press release page, LinkedIn, and layoffs.fyi.
- Must include specific employee count data or layoff numbers from each platform.
- Must explicitly state whether the headcount has decreased, based on the evidence.
- Output must be organized as a structured verification note with evidence cited for each platform.
- Must avoid vague or unsupported claims; all statements must be backed by platform data.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to all three required platforms? Which ones were actually visited?
- Does the response include specific employee count data or layoff numbers from each platform?
- Does the response explicitly state whether the headcount has decreased, based on the evidence?
- Is the output organized as a structured verification note with evidence cited for each platform?
- Are all claims backed by data from the platforms, avoiding unsupported statements?

### Step 2: Dimension Scoring

#### A. Primary Deliverable Accuracy (0.35)
Measures whether the agent correctly verified whether Meta's headcount has decreased, based on evidence.

5 — Correctly states whether the headcount has decreased, with evidence from all three platforms.
4 — Correctly states whether the headcount has decreased, but evidence is incomplete or partially missing.
3 — States whether the headcount has decreased, but evidence is vague or incomplete.
2 — Incorrectly states whether the headcount has decreased, or evidence is mostly missing.
1 — Fails to address the task or provides no evidence.

#### B. Coverage of Required Platforms (0.30)
Measures whether the agent used all three specified platforms and included data from each.

5 — Uses all three platforms and includes data from each.
4 — Uses all three platforms but data from one is incomplete or missing.
3 — Uses at least two platforms with partial data.
2 — Uses only one platform or data is mostly missing.
1 — Fails to use any of the specified platforms.

#### C. Depth and Specificity of Evidence (0.20)
Measures the level of detail and specificity in the evidence provided.

5 — Provides detailed, specific employee count data or layoff numbers from all platforms.
4 — Provides detailed data from most platforms but lacks some specificity.
3 — Provides general data but lacks detail or specificity.
2 — Provides vague or minimal data.
1 — Provides no meaningful data.

#### D. Output Structure and Credibility (0.15)
Measures whether the output is well-organized and cites credible evidence.

5 — Output is well-structured, with clear citations for all evidence.
4 — Output is mostly well-structured but has minor organizational issues.
3 — Output is somewhat organized but lacks clarity or proper citations.
2 — Output is poorly organized or lacks credible citations.
1 — Output is disorganized and lacks credibility.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "primary_deliverable_accuracy": <1-5>,
  "coverage_of_required_platforms": <1-5>,
  "depth_and_specificity_of_evidence": <1-5>,
  "output_structure_and_credibility": <1-5>,
  "dimension_reasoning": {{
    "primary_deliverable_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_required_platforms": "<one sentence citing specific evidence>",
    "depth_and_specificity_of_evidence": "<one sentence citing specific evidence>",
    "output_structure_and_credibility": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "primary_deliverable_accuracy": 0.35,
    "coverage_of_required_platforms": 0.30,
    "depth_and_specificity_of_evidence": 0.20,
    "output_structure_and_credibility": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())