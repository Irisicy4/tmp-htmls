"""
LLM-as-judge evaluator for EvolveBench task.

Category: Daily Activities
Task: Verify if public libraries in Boston, MA, are currently offering free access to coworking spaces or meeting rooms.
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


TASK_INSTRUCTION = """Verify if public libraries in Boston, MA, are currently offering free access to coworking spaces or meeting rooms. Use the official Boston Public Library website and community news platforms to check current policies and announcements. Provide a report on availability with evidence from official sources."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to verify whether public libraries in Boston, MA, are offering free access to coworking spaces or meeting rooms. The agent must use official sources like the Boston Public Library website and community news platforms to check current policies and announcements. A successful completion includes a clear report on availability with evidence from credible sources.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Verify if public libraries in Boston, MA, are currently offering free access to coworking spaces or meeting rooms. Use the official Boston Public Library website and community news platforms to check current policies and announcements. Provide a report on availability with evidence from official sources.

## Task-Specific Constraints
- Must visit all three specified platforms: bpl.org, patch.com, boston.gov.
- Must provide evidence from at least two official sources.
- Report must include specific details about coworking space or meeting room policies.
- Output must be organized as a structured list or table.
- Must verify the credibility of sources cited.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to all three required platforms (bpl.org, patch.com, boston.gov)?
- Are specific details about coworking spaces or meeting room policies present in the response?
- Is the output organized as a structured list or table?
- Are at least two credible sources cited in the response?
- Are all factual claims accurate and supported by evidence?

### Step 2: Dimension Scoring

#### A. Primary Deliverable Accuracy (0.35)
Measures whether the agent correctly identified and reported on coworking space or meeting room policies.

5 — Provides complete and accurate details about coworking space or meeting room policies from at least two credible sources.
4 — Provides mostly accurate details but may lack minor specifics or clarity.
3 — Provides partial or vague details, with some inaccuracies or missing elements.
2 — Provides mostly incorrect or incomplete information.
1 — Provides no relevant information.

#### B. Coverage of Required Platforms (0.30)
Measures whether the agent visited all specified platforms and used them effectively.

5 — Navigates all three required platforms and extracts relevant information from each.
4 — Navigates at least two platforms and extracts mostly relevant information.
3 — Navigates at least one platform and extracts partial information.
2 — Navigates platforms but extracts irrelevant or incorrect information.
1 — Does not navigate any required platforms.

#### C. Depth of Evidence (0.25)
Measures the specificity and detail of evidence provided in the response.

5 — Includes detailed evidence with specific policies, dates, and examples.
4 — Includes mostly detailed evidence but may lack minor specifics.
3 — Includes partial evidence with some missing details or examples.
2 — Includes vague or incomplete evidence.
1 — Includes no evidence or entirely incorrect information.

#### D. Output Structure and Source Credibility (0.10)
Measures whether the response is well-organized and cites credible sources.

5 — Response is structured as a clear list or table and cites at least two credible sources.
4 — Response is mostly well-organized and cites at least one credible source.
3 — Response is somewhat organized and cites one credible source.
2 — Response is poorly organized and cites questionable sources.
1 — Response is disorganized and cites no credible sources.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "primary_deliverable_accuracy": <1-5>,
  "coverage_of_required_platforms": <1-5>,
  "depth_of_evidence": <1-5>,
  "output_structure_and_source_credibility": <1-5>,
  "dimension_reasoning": {{
    "primary_deliverable_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_required_platforms": "<one sentence citing specific evidence>",
    "depth_of_evidence": "<one sentence citing specific evidence>",
    "output_structure_and_source_credibility": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "primary_deliverable_accuracy": 0.35,
    "coverage_of_required_platforms": 0.30,
    "depth_of_evidence": 0.25,
    "output_structure_and_source_credibility": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())