"""
LLM-as-judge evaluator for EvolveBench task.

Category: Data & ML Engineering
Task: Verify reproducibility of benchmark claims for GPT-4, PaLM 2, and LLaMA 2 on common NLP datasets.
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


TASK_INSTRUCTION = """Check whether the benchmarks reported by OpenAI's GPT-4, Google's PaLM 2, and Meta's LLaMA 2 on common NLP datasets (e.g., SuperGLUE, SQuAD, or MMLU) are reproducible. Verify benchmark claims by cross-referencing leaderboard records, official papers, and GitHub repositories."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to verify the reproducibility of benchmark claims for GPT-4, PaLM 2, and LLaMA 2 on common NLP datasets. This involves cross-referencing leaderboard records, official papers, and GitHub repositories to confirm the accuracy of the reported results.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Check whether the benchmarks reported by OpenAI's GPT-4, Google's PaLM 2, and Meta's LLaMA 2 on common NLP datasets (e.g., SuperGLUE, SQuAD, or MMLU) are reproducible. Verify benchmark claims by cross-referencing leaderboard records, official papers, and GitHub repositories.

## Task-Specific Constraints
- Must visit leaderboard.allenai.org, github.com, and paperswithcode.com.
- Must cross-reference benchmark claims with leaderboard records and official papers.
- Must include specific dataset scores for GPT-4, PaLM 2, and LLaMA 2.
- Output must be organized as a structured table comparing scores across models and datasets.
- Must cite sources for all claims made in the response.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to leaderboard.allenai.org, github.com, and paperswithcode.com? Which platforms were actually visited?
- Are specific dataset scores for GPT-4, PaLM 2, and LLaMA 2 present in the response?
- Is the output organized as a structured table comparing scores across models and datasets?
- Are all claims in the response cited with sources?
- Are the benchmark claims verified against leaderboard records and official papers?

### Step 2: Dimension Scoring

#### A. Benchmark Verification Accuracy (0.35)
Measures whether the agent correctly verified benchmark claims using leaderboard records and official papers.

5 — Verifies all claims with leaderboard records and official papers, with no errors.
4 — Verifies most claims with minor omissions or errors.
3 — Verifies some claims but misses key benchmarks or introduces errors.
2 — Verifies few claims and introduces significant errors.
1 — Does not verify any claims or provides incorrect information.

#### B. Platform Coverage (0.30)
Measures whether the agent visited all required platforms and utilized them effectively.

5 — Utilizes leaderboard.allenai.org, github.com, and paperswithcode.com comprehensively.
4 — Utilizes at least 2 platforms comprehensively, with minor omissions.
3 — Utilizes at least 2 platforms but misses key information.
2 — Utilizes only 1 platform or misses significant information.
1 — Does not utilize any required platforms.

#### C. Depth of Analysis (0.20)
Measures the level of detail in the agent's response, including dataset scores and comparisons.

5 — Provides detailed dataset scores and comparisons for all models and datasets.
4 — Provides detailed scores for most models and datasets, with minor omissions.
3 — Provides scores for some models and datasets but lacks detail or comparisons.
2 — Provides minimal scores or comparisons with significant omissions.
1 — Provides no scores or comparisons.

#### D. Output Structure and Source Credibility (0.15)
Measures whether the response is well-organized and cites credible sources.

5 — Response is structured as a clear table and cites all sources accurately.
4 — Response is structured but has minor formatting or citation issues.
3 — Response is partially structured and cites some sources.
2 — Response is poorly structured or lacks citations.
1 — Response is unstructured and does not cite sources.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "benchmark_verification_accuracy": <1-5>,
  "platform_coverage": <1-5>,
  "depth_of_analysis": <1-5>,
  "output_structure_and_source_credibility": <1-5>,
  "dimension_reasoning": {{
    "benchmark_verification_accuracy": "<one sentence citing specific evidence>",
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
    "benchmark_verification_accuracy": 0.35,
    "platform_coverage": 0.30,
    "depth_of_analysis": 0.20,
    "output_structure_and_source_credibility": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())