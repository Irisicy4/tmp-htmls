"""
LLM-as-judge evaluator for EvolveBench task.

Category: Data & ML Engineering
Task: Verify reproducibility of YOLOv7 benchmark results for object detection.
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


TASK_INSTRUCTION = """Verify whether the benchmark results of YOLOv7 for object detection are reproducible as claimed. Check official repositories, linked datasets, and the stated testing conditions, including hardware and configuration details."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves verifying the reproducibility of YOLOv7 benchmark results for object detection. This requires checking the official repositories, linked datasets, and stated testing conditions, including hardware and configuration details. The domain is Data & ML Engineering, and successful completion means providing a structured and evidence-backed verification of reproducibility.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Verify whether the benchmark results of YOLOv7 for object detection are reproducible as claimed. Check official repositories, linked datasets, and the stated testing conditions, including hardware and configuration details.

## Task-Specific Constraints
- Must visit github.com, paperswithcode.com, and arxiv.org.
- Must verify the presence of YOLOv7 benchmark results in the official repository.
- Must confirm dataset links and testing conditions match the claims.
- Must include hardware configuration details used for testing.
- Output must summarize reproducibility findings in a structured format (e.g., table or bullet points).

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to github.com, paperswithcode.com, and arxiv.org? Which ones were actually visited?
- Did the agent verify the benchmark results in the official repository?
- Are dataset links and testing conditions mentioned and checked against claims?
- Are hardware configuration details present and accurate?
- Is the output organized in a structured format (e.g., table or bullet points)?

### Step 2: Dimension Scoring

#### A. Benchmark Verification Accuracy (0.35)
Measures whether the agent correctly verified YOLOv7 benchmark results and testing conditions.

5 — Verifies benchmark results, datasets, and testing conditions with complete accuracy and evidence.
4 — Verifies most benchmark results and testing conditions but misses minor details.
3 — Verifies some results but lacks completeness or accuracy.
2 — Verifies few results and testing conditions; significant gaps.
1 — Fails to verify any benchmark results or testing conditions.

#### B. Platform Coverage (0.30)
Measures whether the agent visited all required platforms and used them effectively.

5 — Visits github.com, paperswithcode.com, and arxiv.org, and uses all effectively.
4 — Visits all platforms but uses one less effectively.
3 — Visits at least two platforms and uses them partially.
2 — Visits only one platform or uses none effectively.
1 — Fails to visit any required platforms.

#### C. Depth of Evidence (0.25)
Measures the level of detail and specificity in the agent's findings.

5 — Provides detailed evidence, including dataset links, hardware configurations, and testing conditions.
4 — Provides evidence but lacks minor details or specificity.
3 — Provides some evidence but misses key details.
2 — Provides minimal evidence with significant gaps.
1 — Provides no evidence or completely incorrect information.

#### D. Output Structure and Organization (0.10)
Measures whether the output is well-organized and follows the required format.

5 — Output is structured (e.g., table or bullet points) and easy to follow.
4 — Output is mostly structured but has minor organizational issues.
3 — Output is partially structured but lacks clarity.
2 — Output is poorly organized and hard to follow.
1 — Output is unstructured or incoherent.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "benchmark_verification_accuracy": <1-5>,
  "platform_coverage": <1-5>,
  "depth_of_evidence": <1-5>,
  "output_structure_and_organization": <1-5>,
  "dimension_reasoning": {{
    "benchmark_verification_accuracy": "<one sentence citing specific evidence>",
    "platform_coverage": "<one sentence citing specific evidence>",
    "depth_of_evidence": "<one sentence citing specific evidence>",
    "output_structure_and_organization": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "benchmark_verification_accuracy": 0.35,
    "platform_coverage": 0.30,
    "depth_of_evidence": 0.25,
    "output_structure_and_organization": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())