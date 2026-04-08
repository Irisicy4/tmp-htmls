"""
LLM-as-judge evaluator for EvolveBench task.

Category: Data & ML Engineering
Task: Compare TensorFlow, PyTorch, and JAX on training speed, community support, and TPU compatibility.
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


TASK_INSTRUCTION = """Compare three popular open-source ML libraries — TensorFlow, PyTorch, and JAX — on training speed, community support (GitHub stars, issues), and compatibility with TPU acceleration. Gather information from their documentation, GitHub repositories, and benchmark results from MLPerf.org."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to compare TensorFlow, PyTorch, and JAX on three specific criteria: training speed, community support, and TPU compatibility. The agent must gather data from the official documentation, GitHub repositories, and MLPerf.org benchmarks. A successful completion requires accurate, sourced data presented in a structured format.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Compare three popular open-source ML libraries — TensorFlow, PyTorch, and JAX — on training speed, community support (GitHub stars, issues), and compatibility with TPU acceleration. Gather information from their documentation, GitHub repositories, and benchmark results from MLPerf.org.

## Task-Specific Constraints
- Must visit tensorflow.org, github.com, and mlperf.org.
- Must include GitHub stars and open issues for each library.
- Must provide quantitative training speed comparisons from MLPerf benchmarks.
- Must explicitly state TPU compatibility for each library.
- Output must be structured as a table or clearly labeled list.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to tensorflow.org, github.com, and mlperf.org? Which ones were actually visited?
- Are GitHub stars and open issues for all three libraries included in the response?
- Are quantitative training speed comparisons from MLPerf benchmarks present?
- Does the response explicitly state TPU compatibility for each library?
- Is the output organized as a table or clearly labeled list?

### Step 2: Dimension Scoring

#### A. Data Accuracy (0.35)
Measures whether the data provided is accurate, sourced, and relevant to the task.

5 — All data is accurate, sourced, and relevant to the task.
4 — Most data is accurate and sourced, with minor omissions or errors.
3 — Some data is accurate, but there are significant omissions or errors.
2 — Very little data is accurate or relevant.
1 — No accurate or relevant data is provided.

#### B. Coverage of Requirements (0.30)
Measures whether the agent addressed all required criteria and platforms.

5 — All criteria and platforms are fully addressed.
4 — Most criteria and platforms are addressed, with minor gaps.
3 — Some criteria and platforms are addressed, but significant gaps remain.
2 — Few criteria or platforms are addressed.
1 — No criteria or platforms are addressed.

#### C. Depth of Analysis (0.20)
Measures the level of detail and specificity in the response.

5 — Provides detailed, specific comparisons with quantitative data.
4 — Provides mostly detailed comparisons, with minor omissions.
3 — Provides some detail, but lacks depth in key areas.
2 — Provides very little detail or specificity.
1 — No detail or specificity is provided.

#### D. Output Structure and Clarity (0.15)
Measures whether the output is well-organized and easy to understand.

5 — Output is perfectly structured and easy to understand.
4 — Output is mostly well-structured, with minor issues.
3 — Output is somewhat structured, but has significant issues.
2 — Output is poorly structured and hard to understand.
1 — Output is completely unstructured or incoherent.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "data_accuracy": <1-5>,
  "coverage_of_requirements": <1-5>,
  "depth_of_analysis": <1-5>,
  "output_structure_and_clarity": <1-5>,
  "dimension_reasoning": {{
    "data_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_requirements": "<one sentence citing specific evidence>",
    "depth_of_analysis": "<one sentence citing specific evidence>",
    "output_structure_and_clarity": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "data_accuracy": 0.35,
    "coverage_of_requirements": 0.30,
    "depth_of_analysis": 0.20,
    "output_structure_and_clarity": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())