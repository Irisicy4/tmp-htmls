"""
LLM-as-judge evaluator for EvolveBench task.

Category: Data & ML Engineering
Task: Research and compare TensorFlow, PyTorch, and Scikit-learn for supervised learning tasks.
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


TASK_INSTRUCTION = """Research and compare three widely used machine learning frameworks: TensorFlow, PyTorch, and Scikit-learn. Analyze their documentation and GitHub repositories to compare features, community activity, and supported use cases for supervised learning tasks. Present findings in a comparison table highlighting advantages and disadvantages of each framework."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to research and compare TensorFlow, PyTorch, and Scikit-learn for supervised learning tasks. The agent must analyze documentation and GitHub repositories to compare features, community activity, and supported use cases. A successful completion requires presenting findings in a comparison table highlighting advantages and disadvantages of each framework.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Research and compare three widely used machine learning frameworks: TensorFlow, PyTorch, and Scikit-learn. Analyze their documentation and GitHub repositories to compare features, community activity, and supported use cases for supervised learning tasks. Present findings in a comparison table highlighting advantages and disadvantages of each framework.

## Task-Specific Constraints
- Must visit tensorflow.org, pytorch.org, scikit-learn.org, and github.com.
- Must include comparisons of features, community activity, and use cases for supervised learning.
- Output must be organized as a table.
- Must highlight advantages and disadvantages of each framework.
- Must reference specific documentation or GitHub repository data.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to tensorflow.org, pytorch.org, scikit-learn.org, and github.com? Which ones were actually visited?
- Are comparisons of features, community activity, and use cases present in the response?
- Is the output organized as a table?
- Are advantages and disadvantages of each framework clearly highlighted?
- Are references to documentation or GitHub repository data included?

### Step 2: Dimension Scoring

#### A. Comparison Accuracy (0.35)
Measures whether the comparison of features, community activity, and use cases is correct and complete.

5 — Includes accurate and complete comparisons for all three aspects across all frameworks.
4 — Includes mostly accurate comparisons but may miss minor details.
3 — Includes partially accurate comparisons but lacks significant details.
2 — Includes mostly incorrect or incomplete comparisons.
1 — No meaningful comparison provided.

#### B. Coverage of Sources (0.30)
Measures whether the agent visited all required platforms and used their data.

5 — Uses data from all four required platforms (tensorflow.org, pytorch.org, scikit-learn.org, github.com).
4 — Uses data from three platforms but misses one.
3 — Uses data from two platforms but misses key sources.
2 — Uses data from only one platform.
1 — No data from required platforms used.

#### C. Depth of Analysis (0.25)
Measures the level of detail and specificity in the analysis.

5 — Provides detailed comparisons with specific examples, numbers, or metrics.
4 — Provides moderately detailed comparisons with some examples or metrics.
3 — Provides basic comparisons with minimal detail or examples.
2 — Provides vague or shallow comparisons.
1 — No meaningful analysis provided.

#### D. Output Structure and Credibility (0.10)
Measures whether the output is well-organized and references credible sources.

5 — Output is structured as a clear table and cites credible sources.
4 — Output is mostly well-structured but has minor formatting issues or unclear citations.
3 — Output is somewhat structured but lacks clarity or credible citations.
2 — Output is poorly structured or lacks credible citations.
1 — Output is unstructured and lacks credible citations.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "comparison_accuracy": <1-5>,
  "coverage_of_sources": <1-5>,
  "depth_of_analysis": <1-5>,
  "output_structure_and_credibility": <1-5>,
  "dimension_reasoning": {{
    "comparison_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_sources": "<one sentence citing specific evidence>",
    "depth_of_analysis": "<one sentence citing specific evidence>",
    "output_structure_and_credibility": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "comparison_accuracy": 0.35,
    "coverage_of_sources": 0.30,
    "depth_of_analysis": 0.25,
    "output_structure_and_credibility": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())