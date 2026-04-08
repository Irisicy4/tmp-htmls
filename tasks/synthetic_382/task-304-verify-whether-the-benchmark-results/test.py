"""
LLM-as-judge evaluator for EvolveBench task.

Category: Data & ML Engineering
Task: Verify benchmark results of GPT-4 against official leaderboard data for reasoning tasks.
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


TASK_INSTRUCTION = """Verify whether the benchmark results of GPT-4 released by OpenAI align with results reported on official leaderboard websites for reasoning tasks in datasets like HellaSwag and ARC. Check discrepancies in top-1 accuracy scores against official claims."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves verifying benchmark results of GPT-4 for reasoning tasks against official leaderboard data from platforms like openai.com, paperswithcode.com, and arxiv.org. The agent must identify discrepancies in top-1 accuracy scores and provide structured evidence.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Verify whether the benchmark results of GPT-4 released by OpenAI align with results reported on official leaderboard websites for reasoning tasks in datasets like HellaSwag and ARC. Check discrepancies in top-1 accuracy scores against official claims.

## Task-Specific Constraints
- Must visit openai.com, paperswithcode.com, and arxiv.org.
- Must extract top-1 accuracy scores for HellaSwag and ARC from each platform.
- Must compare extracted scores against OpenAI's claims.
- Must identify and explain any discrepancies clearly.
- Output must be structured as a table or JSON object summarizing findings.
- Must cite sources for every extracted score.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to openai.com, paperswithcode.com, and arxiv.org? Which platforms were visited?
- Are top-1 accuracy scores for HellaSwag and ARC present in the response?
- Are discrepancies between OpenAI's claims and leaderboard data identified and explained?
- Is the output organized as a table or JSON object summarizing findings?
- Are all extracted scores cited with source URLs?

### Step 2: Dimension Scoring

#### A. Accuracy of Benchmark Verification (0.35)
Measures whether the agent correctly verified benchmark results and identified discrepancies.

5 — All scores verified correctly, discrepancies identified and explained.
4 — Most scores verified correctly, minor errors in discrepancy identification.
3 — Partial verification, some scores missing or incorrect.
2 — Significant errors in verification, major scores missing.
1 — No attempt to verify scores.

#### B. Platform Coverage (0.30)
Measures whether the agent visited all required platforms and extracted relevant data.

5 — Data extracted from all three platforms (openai.com, paperswithcode.com, arxiv.org).
4 — Data extracted from two platforms, minor omissions.
3 — Data extracted from one platform, significant omissions.
2 — Attempted navigation but failed to extract data.
1 — Did not navigate to required platforms.

#### C. Depth of Analysis (0.20)
Measures the level of detail in identifying and explaining discrepancies.

5 — Discrepancies explained with detailed comparisons and evidence.
4 — Discrepancies explained but missing minor details.
3 — Discrepancies partially explained, lacks depth.
2 — Minimal explanation, lacks clarity.
1 — No explanation provided.

#### D. Output Structure and Source Credibility (0.15)
Measures the organization of the output and credibility of cited sources.

5 — Output structured as a table or JSON object, all sources credible.
4 — Output structured but minor formatting issues or questionable sources.
3 — Output partially structured, some sources missing or unclear.
2 — Poorly structured output, lacks credible sources.
1 — No structure or sources provided.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "accuracy_of_benchmark_verification": <1-5>,
  "platform_coverage": <1-5>,
  "depth_of_analysis": <1-5>,
  "output_structure_and_source_credibility": <1-5>,
  "dimension_reasoning": {{
    "accuracy_of_benchmark_verification": "<one sentence citing specific evidence>",
    "platform_coverage": "<one sentence citing specific evidence>",
    "depth_of_analysis": "<one sentence citing specific evidence>",
    "output_structure_and_source_credibility": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "accuracy_of_benchmark_verification": 0.35,
    "platform_coverage": 0.30,
    "depth_of_analysis": 0.20,
    "output_structure_and_source_credibility": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())