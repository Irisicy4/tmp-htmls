"""
LLM-as-judge evaluator for EvolveBench task.

Category: Data & ML Engineering
Task: Compare three popular open-source machine learning frameworks based on specific criteria.
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


TASK_INSTRUCTION = """Compare three popular open-source machine learning frameworks: PyTorch, TensorFlow, and JAX. Assess them on criteria such as model training speed, GPU compatibility, ease of API use, and community support. Use sources like their official documentation, GitHub repositories, and recent blog articles."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves comparing three open-source machine learning frameworks (PyTorch, TensorFlow, and JAX) based on specific criteria like model training speed, GPU compatibility, ease of API use, and community support. The domain is Data & ML Engineering, and a successful completion requires a structured, evidence-backed comparison using credible sources.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Compare three popular open-source machine learning frameworks: PyTorch, TensorFlow, and JAX. Assess them on criteria such as model training speed, GPU compatibility, ease of API use, and community support. Use sources like their official documentation, GitHub repositories, and recent blog articles.

## Task-Specific Constraints
- Must visit pytorch.org, tensorflow.org, and github.com.
- Must provide a structured comparison table or list.
- Must include quantitative data (e.g., training speed benchmarks, GPU compatibility details).
- Must reference community metrics (e.g., GitHub stars, active contributors).
- Must cite sources explicitly for all claims.
- Must address all four criteria mentioned in the instruction.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to pytorch.org, tensorflow.org, and github.com? Which ones were actually visited?
- Does the response include a structured comparison table or list?
- Are quantitative benchmarks (e.g., training speed, GPU compatibility) present and sourced?
- Are community metrics (e.g., GitHub stars, contributors) included and sourced?
- Are all four criteria (training speed, GPU compatibility, API ease, community support) addressed?

### Step 2: Dimension Scoring

#### A. Comparison Accuracy (0.35)
Measures whether the comparison is accurate, evidence-backed, and addresses all criteria.

5 — All four criteria are addressed with accurate, evidence-backed comparisons.
4 — Three criteria are addressed with accurate comparisons; minor gaps in evidence.
3 — Two criteria are addressed; evidence is partial or incomplete.
2 — One criterion is addressed; evidence is mostly missing or incorrect.
1 — No criteria are addressed or evidence is entirely absent.

#### B. Coverage of Sources (0.30)
Measures whether the agent visited all required platforms and cited sources explicitly.

5 — All required platforms were visited, and sources are explicitly cited for all claims.
4 — Two platforms were visited, and most claims are sourced.
3 — At least one platform was visited, and some claims are sourced.
2 — No platforms were visited; few claims are sourced.
1 — No platforms were visited; no claims are sourced.

#### C. Depth of Analysis (0.25)
Measures the level of detail in the comparison, including quantitative benchmarks and community metrics.

5 — Includes detailed benchmarks and community metrics for all three frameworks.
4 — Includes benchmarks and metrics for two frameworks; minor gaps in detail.
3 — Includes benchmarks or metrics for at least one framework; lacks depth.
2 — Minimal detail; benchmarks and metrics mostly missing.
1 — No detail; benchmarks and metrics entirely absent.

#### D. Output Structure and Credibility (0.10)
Measures whether the output is well-organized and uses credible sources.

5 — Structured as a clear table or list; all sources are credible and cited.
4 — Mostly structured; minor issues with source credibility or citation.
3 — Partially structured; some sources lack credibility or citation.
2 — Poorly structured; most sources lack credibility or citation.
1 — No structure; no credible sources or citations.

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