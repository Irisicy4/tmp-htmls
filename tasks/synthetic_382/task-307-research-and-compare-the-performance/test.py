"""
LLM-as-judge evaluator for EvolveBench task.

Category: Data & ML Engineering
Task: Research and compare performance benchmarks of three leading open-source NLP models in terms of accuracy, computational efficiency, and dataset compatibility.
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


TASK_INSTRUCTION = """Research and compare the performance benchmarks of three leading open-source NLP models (such as BERT, GPT-3, and RoBERTa) in terms of accuracy, computational efficiency, and dataset compatibility. Provide details from their official papers and community benchmarks."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to research and compare performance benchmarks of three leading open-source NLP models (e.g., BERT, GPT-3, RoBERTa) in terms of accuracy, computational efficiency, and dataset compatibility. The domain is NLP model evaluation, and a successful completion requires the agent to provide structured comparisons based on official papers and community benchmarks.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Research and compare the performance benchmarks of three leading open-source NLP models (such as BERT, GPT-3, and RoBERTa) in terms of accuracy, computational efficiency, and dataset compatibility. Provide details from their official papers and community benchmarks.

## Task-Specific Constraints
- Must visit at least 3 specified platforms: paperswithcode.com, arxiv.org, huggingface.co.
- Must include accuracy, computational efficiency, and dataset compatibility for all three models.
- Output must be organized as a structured table or list for comparison.
- Must cite sources for all claims made in the response.
- Must include specific dataset names and benchmark metrics where applicable.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to the required platforms? Which ones were actually visited?
- Are accuracy, computational efficiency, and dataset compatibility included for all three models?
- Is the output organized as a structured table or list?
- Are sources cited for all claims made in the response?
- Are dataset names and benchmark metrics included where applicable?

### Step 2: Dimension Scoring

#### A. Benchmark Accuracy (0.35)
Measures whether the agent correctly identified and compared benchmark metrics for accuracy across all three models.

5 — Identifies and compares accuracy metrics for all three models with specific values and sources.
4 — Identifies accuracy metrics for all three models but lacks some specific values or sources.
3 — Identifies accuracy metrics for at least two models with partial values or sources.
2 — Identifies accuracy metrics for only one model or provides vague/incorrect values.
1 — Does not identify accuracy metrics for any model.

#### B. Platform Coverage (0.30)
Measures whether the agent visited all required platforms and utilized their information effectively.

5 — Visited all three platforms and utilized information from each in the response.
4 — Visited at least two platforms and utilized their information effectively.
3 — Visited at least one platform and utilized its information partially.
2 — Visited platforms but did not utilize their information effectively.
1 — Did not visit any of the required platforms.

#### C. Depth of Comparison (0.25)
Measures the depth and specificity of the comparisons provided, including dataset names and benchmark metrics.

5 — Provides detailed comparisons with dataset names and benchmark metrics for all three models.
4 — Provides comparisons with dataset names and metrics for at least two models.
3 — Provides partial comparisons with some dataset names or metrics.
2 — Provides vague comparisons with little to no specificity.
1 — Does not provide any meaningful comparisons.

#### D. Output Structure and Source Credibility (0.10)
Measures the organization of the output and the credibility of the sources cited.

5 — Output is well-organized (e.g., structured table or list) and all claims are sourced from credible platforms.
4 — Output is organized but lacks some sourcing or uses less credible sources.
3 — Output is partially organized and includes some credible sources.
2 — Output is disorganized and lacks credible sources.
1 — Output is completely disorganized and unsourced.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "benchmark_accuracy": <1-5>,
  "platform_coverage": <1-5>,
  "depth_of_comparison": <1-5>,
  "output_structure_and_source_credibility": <1-5>,
  "dimension_reasoning": {{
    "benchmark_accuracy": "<one sentence citing specific evidence>",
    "platform_coverage": "<one sentence citing specific evidence>",
    "depth_of_comparison": "<one sentence citing specific evidence>",
    "output_structure_and_source_credibility": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "benchmark_accuracy": 0.35,
    "platform_coverage": 0.30,
    "depth_of_comparison": 0.25,
    "output_structure_and_source_credibility": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())