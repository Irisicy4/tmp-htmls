"""
LLM-as-judge evaluator for EvolveBench task.

Category: Software Engineering
Task: Research and compare the top 3 open-source static site generators based on templating language, build performance, and plugin ecosystem.
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


TASK_INSTRUCTION = """Research the top 3 open-source static site generators and compare them based on their templating language, build performance, and plugin ecosystem. Use their official documentation and GitHub repositories along with independent technical reviews."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to research the top 3 open-source static site generators and compare them based on templating language, build performance, and plugin ecosystem. The domain is software engineering, and a successful completion requires a structured comparison with accurate and sourced information.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Research the top 3 open-source static site generators and compare them based on their templating language, build performance, and plugin ecosystem. Use their official documentation and GitHub repositories along with independent technical reviews.

## Task-Specific Constraints
- Must identify and compare exactly 3 static site generators.
- Must use official documentation and GitHub repositories for each generator.
- Must include data on templating language, build performance, and plugin ecosystem for each generator.
- Output must be structured as a table or clearly organized list.
- Must cite at least one independent technical review for each generator.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to the official documentation and GitHub repositories for all 3 generators?
- Did the agent include data on templating language, build performance, and plugin ecosystem for all 3 generators?
- Is the output structured as a table or clearly organized list?
- Did the agent cite at least one independent technical review for each generator?
- Are the comparisons accurate and supported by credible sources?

### Step 2: Dimension Scoring

#### A. Comparison Accuracy (0.35)
Measures whether the comparisons are accurate and supported by credible sources.

5 — All comparisons are accurate, sourced, and supported by credible evidence.
4 — Most comparisons are accurate and sourced, with minor omissions or errors.
3 — Some comparisons are accurate, but there are significant omissions or errors.
2 — Few comparisons are accurate or sourced; most are incorrect or missing.
1 — No accurate or sourced comparisons.

#### B. Coverage of Required Data (0.30)
Measures whether the agent included all required data points (templating language, build performance, plugin ecosystem) for all 3 generators.

5 — Includes all required data points for all 3 generators.
4 — Includes most required data points for all 3 generators, with minor omissions.
3 — Includes some required data points, but significant omissions exist.
2 — Includes few required data points; most are missing.
1 — Includes no required data points.

#### C. Depth of Analysis (0.20)
Measures the level of detail and specificity in the comparisons.

5 — Provides detailed and specific comparisons with quantitative or qualitative insights.
4 — Provides mostly detailed comparisons, with minor gaps in specificity.
3 — Provides some detail, but lacks depth or specificity in key areas.
2 — Provides minimal detail; most comparisons are vague or generic.
1 — Provides no detail or specificity.

#### D. Output Structure and Organization (0.15)
Measures whether the output is well-organized and follows the required format.

5 — Output is well-organized, follows the required format, and is easy to read.
4 — Output is mostly well-organized, with minor formatting issues.
3 — Output is somewhat organized, but formatting issues make it harder to follow.
2 — Output is poorly organized and difficult to follow.
1 — Output is not organized or formatted at all.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "comparison_accuracy": <1-5>,
  "coverage_of_required_data": <1-5>,
  "depth_of_analysis": <1-5>,
  "output_structure_and_organization": <1-5>,
  "dimension_reasoning": {{
    "comparison_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_required_data": "<one sentence citing specific evidence>",
    "depth_of_analysis": "<one sentence citing specific evidence>",
    "output_structure_and_organization": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "comparison_accuracy": 0.35,
    "coverage_of_required_data": 0.30,
    "depth_of_analysis": 0.20,
    "output_structure_and_organization": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())