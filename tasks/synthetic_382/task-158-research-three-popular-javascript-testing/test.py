"""
LLM-as-judge evaluator for EvolveBench task.

Category: Software Engineering
Task: Compare three JavaScript testing libraries (Jest, Mocha, Jasmine) based on features, configuration flexibility, and npm download statistics.
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


TASK_INSTRUCTION = """Research three popular JavaScript testing libraries (Jest, Mocha, Jasmine) by examining their official documentation and GitHub repositories. Compare them based on supported features, configuration flexibility, and the number of weekly downloads on npm."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to research three JavaScript testing libraries (Jest, Mocha, Jasmine) by visiting their official documentation and GitHub repositories. The agent must compare these libraries based on supported features, configuration flexibility, and npm download statistics. A successful completion includes accurate comparisons, sourced data, and structured output.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Research three popular JavaScript testing libraries (Jest, Mocha, Jasmine) by examining their official documentation and GitHub repositories. Compare them based on supported features, configuration flexibility, and the number of weekly downloads on npm.

## Task-Specific Constraints
- Must visit the official documentation websites for Jest, Mocha, and Jasmine.
- Must visit the GitHub repositories for Jest, Mocha, and Jasmine.
- Must include npm weekly download statistics for all three libraries.
- Output must be organized as a structured comparison (e.g., a table or bullet points).
- Must compare supported features and configuration flexibility for all three libraries.
- Must cite sources for all numerical or factual claims.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to the required platforms? Which ones were actually visited?
- Are npm weekly download statistics for Jest, Mocha, and Jasmine present in the response?
- Are comparisons of supported features and configuration flexibility included for all three libraries?
- Is the output organized as a structured comparison (e.g., table or bullet points)?
- Are all numerical or factual claims accurately sourced?

### Step 2: Dimension Scoring

#### A. Comparison Accuracy (0.35)
Measures whether the comparisons of features, configuration flexibility, and npm statistics are accurate and complete.

5 — All comparisons are accurate, complete, and sourced correctly.
4 — Comparisons are mostly accurate but slightly incomplete or missing minor details.
3 — Comparisons are partially accurate but missing significant details.
2 — Comparisons are mostly inaccurate or incomplete.
1 — Comparisons are absent or completely wrong.

#### B. Platform Coverage (0.30)
Measures whether the agent visited all required platforms (documentation websites and GitHub repositories).

5 — Agent visited all six required platforms and used data from each.
4 — Agent visited at least five platforms and used data from most.
3 — Agent visited at least three platforms and used partial data.
2 — Agent visited fewer than three platforms or used minimal data.
1 — Agent did not visit any required platforms.

#### C. Data Specificity (0.20)
Measures the inclusion of specific details such as npm download statistics and feature lists.

5 — Includes all required specific details with accurate numbers and lists.
4 — Includes most required details but lacks minor specifics.
3 — Includes some required details but lacks significant specifics.
2 — Includes minimal specific details or inaccurate data.
1 — Includes no specific details.

#### D. Output Structure and Credibility (0.15)
Measures whether the output is well-organized and sources are credible.

5 — Output is highly structured (e.g., table or bullet points) and all sources are credible.
4 — Output is mostly structured and sources are credible.
3 — Output is partially structured and sources are mostly credible.
2 — Output is poorly structured or sources lack credibility.
1 — Output is unstructured and sources are not credible.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "comparison_accuracy": <1-5>,
  "platform_coverage": <1-5>,
  "data_specificity": <1-5>,
  "output_structure_and_credibility": <1-5>,
  "dimension_reasoning": {{
    "comparison_accuracy": "<one sentence citing specific evidence>",
    "platform_coverage": "<one sentence citing specific evidence>",
    "data_specificity": "<one sentence citing specific evidence>",
    "output_structure_and_credibility": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "comparison_accuracy": 0.35,
    "platform_coverage": 0.30,
    "data_specificity": 0.20,
    "output_structure_and_credibility": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())