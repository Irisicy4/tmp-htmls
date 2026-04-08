"""
LLM-as-judge evaluator for EvolveBench task.

Category: Data & ML Engineering
Task: Research and compare TensorFlow, PyTorch, and JAX across multiple dimensions.
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


TASK_INSTRUCTION = """Research and compare three popular open-source machine learning framework libraries: TensorFlow, PyTorch, and JAX. Collect information on model training speed, inference speed, GPU support, community size, and licensing. Use documentation, GitHub repositories, and technical blogs to synthesize the analysis."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to research and compare TensorFlow, PyTorch, and JAX across multiple dimensions, including model training speed, inference speed, GPU support, community size, and licensing. The task is in the domain of Data & ML Engineering, and a successful completion involves synthesizing accurate and comprehensive information into a structured analysis.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Research and compare three popular open-source machine learning framework libraries: TensorFlow, PyTorch, and JAX. Collect information on model training speed, inference speed, GPU support, community size, and licensing. Use documentation, GitHub repositories, and technical blogs to synthesize the analysis.

## Task-Specific Constraints
- Must visit tensorflow.org, pytorch.org, and jax.readthedocs.io.
- Must include quantitative data for model training speed and inference speed.
- Must address GPU support explicitly for each framework.
- Must provide licensing details for each framework.
- Output must be organized as a structured table or list.
- Must cite sources used for claims.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to tensorflow.org, pytorch.org, and jax.readthedocs.io? Which ones were actually visited?
- Are model training speed and inference speed included with quantitative data?
- Is GPU support explicitly addressed for each framework?
- Are licensing details provided for each framework?
- Is the output organized as a structured table or list?
- Are sources cited for all claims made?

### Step 2: Dimension Scoring

#### A. Deliverable Accuracy (0.35)
Measures whether the main output is correct and complete.

5 — Includes accurate and complete data for all five dimensions (training speed, inference speed, GPU support, community size, licensing).
4 — Includes accurate data for at least four dimensions; minor omissions.
3 — Includes accurate data for at least three dimensions; partial completion.
2 — Includes accurate data for one or two dimensions; mostly incomplete.
1 — No accurate data provided.

#### B. Coverage of Required Platforms (0.30)
Measures whether the agent used all required sources and platforms.

5 — Navigated to tensorflow.org, pytorch.org, and jax.readthedocs.io; used additional credible sources.
4 — Navigated to all three required platforms; minimal use of additional sources.
3 — Navigated to at least two required platforms; partial coverage.
2 — Navigated to only one required platform; mostly incomplete.
1 — Did not navigate to any required platform.

#### C. Depth and Specificity (0.25)
Measures the level of detail and specificity in comparisons.

5 — Provides detailed quantitative comparisons for all five dimensions; includes nuanced insights.
4 — Provides detailed comparisons for at least four dimensions; minor omissions.
3 — Provides partial comparisons for at least three dimensions; lacks depth.
2 — Provides minimal comparisons; lacks specificity.
1 — No meaningful comparisons provided.

#### D. Source Credibility and Output Structure (0.10)
Measures whether sources are credible and output is well-organized.

5 — All claims are sourced from credible platforms; output is structured as a clear table or list.
4 — Most claims are sourced from credible platforms; output is mostly well-organized.
3 — Some claims are sourced from credible platforms; output is partially organized.
2 — Few claims are sourced from credible platforms; output is poorly organized.
1 — No credible sources used; output is unstructured.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "deliverable_accuracy": <1-5>,
  "coverage_of_required_platforms": <1-5>,
  "depth_and_specificity": <1-5>,
  "source_credibility_and_output_structure": <1-5>,
  "dimension_reasoning": {{
    "deliverable_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_required_platforms": "<one sentence citing specific evidence>",
    "depth_and_specificity": "<one sentence citing specific evidence>",
    "source_credibility_and_output_structure": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "deliverable_accuracy": 0.35,
    "coverage_of_required_platforms": 0.30,
    "depth_and_specificity": 0.25,
    "source_credibility_and_output_structure": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())