"""
LLM-as-judge evaluator for EvolveBench task.

Category: Data & ML Engineering
Task: Investigate compatibility issues between PyTorch 2.0 and CUDA 11.5, identify root cause, affected CUDA versions, and resolution path.
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


TASK_INSTRUCTION = """Investigate an error reported in the PyTorch forums where upgrading to PyTorch 2.0 caused compatibility issues with CUDA 11.5. Find the root cause, affected CUDA version ranges, and the recommended resolution path using official GitHub issues, documentation, and forums."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves investigating compatibility issues between PyTorch 2.0 and CUDA 11.5. The agent must identify the root cause, determine the affected CUDA version ranges, and provide a resolution path. The domain is Data & ML Engineering, and successful completion requires accurate identification of the issue, comprehensive coverage of relevant sources, and structured output.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Investigate an error reported in the PyTorch forums where upgrading to PyTorch 2.0 caused compatibility issues with CUDA 11.5. Find the root cause, affected CUDA version ranges, and the recommended resolution path using official GitHub issues, documentation, and forums.

## Task-Specific Constraints
- Must visit pytorch.org, github.com, and discuss.pytorch.org.
- Must identify the root cause of the compatibility issue.
- Must specify the affected CUDA version ranges.
- Must provide a recommended resolution path.
- Output must be structured as a clear list or table.
- Must reference specific GitHub issues, documentation, or forum threads.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to pytorch.org, github.com, and discuss.pytorch.org?
- Did the agent identify the root cause of the compatibility issue?
- Did the agent specify the affected CUDA version ranges?
- Did the agent provide a recommended resolution path?
- Is the output structured as a clear list or table?

### Step 2: Dimension Scoring

#### A. Root Cause Identification Accuracy (0.35)
Measures whether the agent correctly identified the root cause of the compatibility issue.

5 — Identifies the root cause with specific technical details and sources.
4 — Identifies the root cause but lacks minor technical details or sources.
3 — Identifies the root cause but is vague or incomplete.
2 — Incorrect or incomplete identification of the root cause.
1 — No attempt to identify the root cause.

#### B. Coverage of Required Platforms and Sources (0.30)
Measures whether the agent visited all required platforms and referenced relevant sources.

5 — References all required platforms and cites 3+ relevant sources.
4 — References all required platforms but cites fewer than 3 sources.
3 — References at least 2 platforms but misses key sources.
2 — References only 1 platform or sources are irrelevant.
1 — No relevant platforms or sources referenced.

#### C. Specificity of Affected CUDA Versions (0.25)
Measures whether the agent specifies the affected CUDA version ranges accurately.

5 — Specifies exact affected CUDA versions with supporting evidence.
4 — Specifies affected versions but lacks minor evidence.
3 — Specifies affected versions vaguely or incompletely.
2 — Incorrect or incomplete specification of affected versions.
1 — No attempt to specify affected versions.

#### D. Output Structure and Source Credibility (0.10)
Measures whether the output is well-organized and sources are credible.

5 — Output is structured clearly and all sources are credible.
4 — Output is structured but some sources lack credibility.
3 — Output is usable but poorly organized or sources are unclear.
2 — Output is disorganized or sources are mostly irrelevant.
1 — No structure or credible sources.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "root_cause_identification_accuracy": <1-5>,
  "coverage_of_required_platforms_and_sources": <1-5>,
  "specificity_of_affected_cuda_versions": <1-5>,
  "output_structure_and_source_credibility": <1-5>,
  "dimension_reasoning": {{
    "root_cause_identification_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_required_platforms_and_sources": "<one sentence citing specific evidence>",
    "specificity_of_affected_cuda_versions": "<one sentence citing specific evidence>",
    "output_structure_and_source_credibility": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "root_cause_identification_accuracy": 0.35,
    "coverage_of_required_platforms_and_sources": 0.30,
    "specificity_of_affected_cuda_versions": 0.25,
    "output_structure_and_source_credibility": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())