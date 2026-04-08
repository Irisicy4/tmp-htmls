"""
LLM-as-judge evaluator for EvolveBench task.

Category: Data & ML Engineering
Task: Research and compare benchmarks for CIFAR-10 image classification models from arXiv papers.
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


TASK_INSTRUCTION = """Research the most recent benchmarks for image classification models using the CIFAR-10 dataset. Compare 3 models from papers on arXiv based on accuracy, training time, and hardware requirements. Create a table summarizing your findings."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves researching benchmarks for image classification models using the CIFAR-10 dataset. The agent must compare 3 models from papers on arXiv based on accuracy, training time, and hardware requirements. A successful completion requires the agent to produce a table summarizing these comparisons, with accurate and sourced data.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Research the most recent benchmarks for image classification models using the CIFAR-10 dataset. Compare 3 models from papers on arXiv based on accuracy, training time, and hardware requirements. Create a table summarizing your findings.

## Task-Specific Constraints
- Must visit arxiv.org, paperswithcode.com, and github.com.
- Must compare exactly 3 models from arXiv papers.
- Must include accuracy, training time, and hardware requirements for each model.
- Output must be organized as a table.
- Data must be sourced and referenced clearly.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to arxiv.org, paperswithcode.com, and github.com? Which ones were actually visited?
- Are accuracy, training time, and hardware requirements present for all 3 models?
- Is the output organized as a table?
- Are the data points sourced and referenced clearly?
- Are there any factual inaccuracies in the response?

### Step 2: Dimension Scoring

#### A. Deliverable Accuracy (0.35)
Measures whether the table contains correct and complete comparisons of accuracy, training time, and hardware requirements.

5 — All 3 models compared with correct accuracy, training time, and hardware requirements.
4 — 2 models fully compared; minor errors in the third.
3 — At least 2 models partially compared; significant omissions or errors.
2 — Only 1 model partially compared; major omissions or errors.
1 — No meaningful comparison attempted.

#### B. Coverage of Sources (0.30)
Measures whether the agent visited all required platforms and used them effectively.

5 — Visited arxiv.org, paperswithcode.com, and github.com; sourced data from all.
4 — Visited at least 2 platforms and sourced data from them.
3 — Visited at least 1 platform and sourced partial data.
2 — Visited platforms but failed to source meaningful data.
1 — Did not visit any required platforms.

#### C. Depth of Comparison (0.25)
Measures the level of detail and specificity in the comparisons.

5 — Includes detailed metrics (e.g., exact accuracy percentages, training times, hardware specs).
4 — Includes metrics but lacks some specificity or depth.
3 — Includes basic metrics but omits significant details.
2 — Includes vague or incomplete metrics.
1 — No meaningful metrics included.

#### D. Output Structure and Credibility (0.10)
Measures whether the output is well-organized and the sources are credible.

5 — Output is a clear, well-formatted table with credible sources cited.
4 — Output is mostly clear; minor formatting or citation issues.
3 — Output is usable but disorganized or missing citations.
2 — Output is poorly structured or lacks credibility.
1 — Output is completely disorganized or lacks sources.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "deliverable_accuracy": <1-5>,
  "coverage_of_sources": <1-5>,
  "depth_of_comparison": <1-5>,
  "output_structure_and_credibility": <1-5>,
  "dimension_reasoning": {{
    "deliverable_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_sources": "<one sentence citing specific evidence>",
    "depth_of_comparison": "<one sentence citing specific evidence>",
    "output_structure_and_credibility": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "deliverable_accuracy": 0.35,
    "coverage_of_sources": 0.30,
    "depth_of_comparison": 0.25,
    "output_structure_and_credibility": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())