"""
LLM-as-judge evaluator for EvolveBench task.

Category: Data & ML Engineering
Task: Diagnose and resolve 'CUDA out of memory' errors for a PyTorch model on an NVIDIA RTX 3090.
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


TASK_INSTRUCTION = """Diagnose why a machine learning model running on PyTorch produces 'CUDA out of memory' errors during training when using batch size 64 on an NVIDIA RTX 3090. Investigate causes and find recommended fixes by visiting PyTorch documentation, NVIDIA forums, and GitHub issues."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves diagnosing 'CUDA out of memory' errors for a PyTorch model running on an NVIDIA RTX 3090. The agent must investigate causes and provide recommended fixes by consulting PyTorch documentation, NVIDIA forums, and GitHub issues. Successful completion requires identifying the root cause and proposing actionable solutions.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Diagnose why a machine learning model running on PyTorch produces 'CUDA out of memory' errors during training when using batch size 64 on an NVIDIA RTX 3090. Investigate causes and find recommended fixes by visiting PyTorch documentation, NVIDIA forums, and GitHub issues.

## Task-Specific Constraints
- Must visit PyTorch documentation, NVIDIA forums, and GitHub issues.
- Must identify at least one root cause for the error.
- Must provide at least two actionable solutions to fix the issue.
- Must include references or links to supporting evidence for each solution.
- Output must be organized as a structured list or table.
- Must address batch size and GPU memory considerations explicitly.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to PyTorch documentation, NVIDIA forums, and GitHub issues? Which ones were actually visited?
- Did the agent identify at least one root cause for the 'CUDA out of memory' error?
- Did the agent provide at least two actionable solutions to fix the issue?
- Are references or links to supporting evidence included for each solution?
- Is the output organized as a structured list or table?

### Step 2: Dimension Scoring

#### A. Root Cause Identification (0.35)
Measures whether the agent correctly identified the root cause(s) of the error.

5 — Identifies 2 or more root causes with detailed explanations.
4 — Identifies 1 root cause with detailed explanation.
3 — Identifies 1 root cause but lacks detail.
2 — Attempts to identify but is mostly incorrect or vague.
1 — Fails to identify any root cause.

#### B. Solution Quality (0.30)
Measures the quality and feasibility of the solutions provided.

5 — Provides 3 or more actionable solutions with supporting evidence.
4 — Provides 2 actionable solutions with supporting evidence.
3 — Provides 2 actionable solutions but lacks supporting evidence.
2 — Provides 1 solution or solutions are vague/unfeasible.
1 — Provides no solutions or completely incorrect solutions.

#### C. Coverage of Platforms (0.20)
Measures whether the agent visited all required platforms and used them effectively.

5 — Uses all 3 platforms and cites evidence from each.
4 — Uses 2 platforms and cites evidence from both.
3 — Uses 2 platforms but evidence is incomplete.
2 — Uses 1 platform or evidence is mostly missing.
1 — Fails to use any required platforms.

#### D. Output Structure and References (0.15)
Measures the organization and credibility of the output.

5 — Output is well-organized and includes references for all claims.
4 — Output is organized but references are incomplete.
3 — Output is partially organized and references are sparse.
2 — Output is poorly organized or lacks references.
1 — Output is disorganized and lacks references entirely.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{
  "evidence_summary": "The agent visited PyTorch documentation and GitHub issues but did not navigate NVIDIA forums. Identified one root cause and provided two solutions, one of which lacked supporting evidence. Output was structured as a list but references were incomplete.",
  "root_cause_identification": 3,
  "solution_quality": 3,
  "coverage_of_platforms": 2,
  "output_structure_and_references": 3,
  "dimension_reasoning": {
    "root_cause_identification": "Identified one root cause but lacked detail.",
    "solution_quality": "Provided two solutions but one lacked supporting evidence.",
    "coverage_of_platforms": "Visited PyTorch documentation and GitHub issues but missed NVIDIA forums.",
    "output_structure_and_references": "Output was structured but references were incomplete."
  },
  "overall_score": 2.85,
  "passed": false
}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "root_cause_identification": 0.35,
    "solution_quality": 0.30,
    "coverage_of_platforms": 0.20,
    "output_structure_and_references": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())