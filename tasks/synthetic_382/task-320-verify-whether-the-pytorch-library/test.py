"""
LLM-as-judge evaluator for EvolveBench task.

Category: Data & ML Engineering
Task: Compare PyTorch and TensorFlow GitHub statistics to determine which is more popular.
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


TASK_INSTRUCTION = """Verify whether the PyTorch library is still the most popular deep learning framework based on GitHub statistics. Check the repository's star count, fork count, and active contributor count. Compare these metrics with TensorFlow's GitHub statistics and report your findings."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves comparing GitHub statistics (star count, fork count, and active contributor count) for PyTorch and TensorFlow to determine which framework is more popular. This task falls within the domain of Data & ML Engineering, and a successful completion requires accurate and structured reporting of the comparison.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Verify whether the PyTorch library is still the most popular deep learning framework based on GitHub statistics. Check the repository's star count, fork count, and active contributor count. Compare these metrics with TensorFlow's GitHub statistics and report your findings.

## Task-Specific Constraints
- Must visit GitHub pages for both PyTorch and TensorFlow repositories.
- Must collect and report star count, fork count, and active contributor count for both frameworks.
- Must compare the metrics explicitly and state which framework is more popular based on the data.
- Output must be structured as a table or clearly organized list.
- Must cite GitHub as the source of the data.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to the GitHub pages for both PyTorch and TensorFlow repositories?
- Are the star count, fork count, and active contributor count for both frameworks present in the response?
- Is the output organized as a table or structured list?
- Does the response explicitly compare the metrics and state which framework is more popular?
- Are the claims sourced from GitHub?

### Step 2: Dimension Scoring

#### A. Data Accuracy (0.35)
Measures whether the reported metrics (star count, fork count, active contributor count) are correct and match GitHub data.

5 — All metrics for both frameworks are correct and match GitHub data.
4 — One metric is slightly inaccurate but does not affect the conclusion.
3 — Some metrics are missing or inaccurate but the conclusion is still usable.
2 — Most metrics are missing or incorrect, and the conclusion is unreliable.
1 — No metrics are reported or all are incorrect.

#### B. Coverage of Required Platforms (0.30)
Measures whether the agent visited both GitHub repositories and included all required metrics.

5 — Both repositories visited, and all three metrics collected for both frameworks.
4 — Both repositories visited, but one metric is missing for one framework.
3 — Both repositories visited, but multiple metrics are missing.
2 — Only one repository visited, or most metrics are missing.
1 — No repositories visited, or no metrics collected.

#### C. Depth of Comparison (0.25)
Measures whether the agent explicitly compares the metrics and provides a clear conclusion.

5 — Metrics are compared in detail, and the conclusion is clear and well-supported.
4 — Metrics are compared, but the conclusion lacks detail or clarity.
3 — Metrics are partially compared, and the conclusion is usable but incomplete.
2 — Metrics are minimally compared, and the conclusion is unclear or unsupported.
1 — No comparison or conclusion is provided.

#### D. Output Structure and Credibility (0.10)
Measures whether the output is well-organized and cites GitHub as the source.

5 — Output is structured as a table or clear list, and GitHub is explicitly cited.
4 — Output is structured but lacks explicit citation of GitHub.
3 — Output is somewhat organized but lacks clarity or citation.
2 — Output is disorganized or unclear, and no citation is provided.
1 — Output is completely disorganized and lacks citation.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{
  "evidence_summary": "The agent visited both GitHub repositories and collected star count, fork count, and active contributor count for both frameworks. The metrics were compared, and a conclusion was provided. The output was structured as a table and cited GitHub as the source.",
  "data_accuracy": 5,
  "coverage_of_required_platforms": 5,
  "depth_of_comparison": 5,
  "output_structure_and_credibility": 5,
  "dimension_reasoning": {
    "data_accuracy": "All metrics matched GitHub data and were correctly reported.",
    "coverage_of_required_platforms": "Both repositories were visited, and all metrics were collected.",
    "depth_of_comparison": "Metrics were compared in detail, and the conclusion was clear and well-supported.",
    "output_structure_and_credibility": "Output was structured as a table and explicitly cited GitHub."
  },
  "overall_score": 5.0,
  "passed": true
}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "data_accuracy": 0.35,
    "coverage_of_required_platforms": 0.30,
    "depth_of_comparison": 0.25,
    "output_structure_and_credibility": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())