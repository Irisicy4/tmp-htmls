"""
LLM-as-judge evaluator for EvolveBench task.

Category: Software Engineering
Task: Compare three JavaScript testing frameworks (Jest, Mocha, Cypress) on speed, setup ease, and browser support.
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


TASK_INSTRUCTION = """Compare three popular JavaScript testing frameworks—Jest, Mocha, and Cypress—on speed of execution, ease of setup, and support for browser-based tests. Use their official documentation and GitHub repositories to gather the information."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to compare three JavaScript testing frameworks—Jest, Mocha, and Cypress—on speed of execution, ease of setup, and support for browser-based tests. The domain is software engineering, specifically testing frameworks. A successful completion requires the agent to provide accurate comparisons based on official documentation and GitHub repositories, and organize the findings clearly.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Compare three popular JavaScript testing frameworks—Jest, Mocha, and Cypress—on speed of execution, ease of setup, and support for browser-based tests. Use their official documentation and GitHub repositories to gather the information.

## Task-Specific Constraints
- Must visit the official documentation for Jest, Mocha, and Cypress.
- Must include speed of execution data for all three frameworks.
- Must evaluate ease of setup for all three frameworks.
- Must assess support for browser-based tests for all three frameworks.
- Output must be organized as a structured table or list.
- Must cite sources for all claims made.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to the official documentation for Jest, Mocha, and Cypress? Which ones were actually visited?
- Are speed of execution, ease of setup, and browser-based test support data present for all three frameworks?
- Is the output organized as a structured table or list?
- Are all claims backed by credible sources (e.g., official documentation or GitHub repositories)?
- Are there any factual inaccuracies or missing comparisons?

### Step 2: Dimension Scoring

#### A. Comparison Accuracy (0.35)
Measures whether the comparisons between Jest, Mocha, and Cypress are accurate and complete.

5 — Provides accurate and complete comparisons for speed, setup ease, and browser support across all three frameworks.
4 — Comparisons are mostly accurate but lack minor details or completeness.
3 — Comparisons are partially accurate but missing key aspects or contain minor inaccuracies.
2 — Comparisons are mostly inaccurate or incomplete.
1 — No meaningful comparisons provided.

#### B. Coverage of Required Platforms (0.30)
Measures whether the agent visited and used all required platforms and sources.

5 — Uses all three platforms (Jest, Mocha, Cypress) and cites their documentation and GitHub repositories.
4 — Uses two platforms fully and partially references the third.
3 — Uses at least one platform fully and partially references another.
2 — Uses only one platform or provides minimal references.
1 — No evidence of platform usage.

#### C. Depth of Analysis (0.25)
Measures the level of detail and specificity in the comparisons.

5 — Includes detailed metrics, examples, and specific claims for all three frameworks.
4 — Includes detailed metrics for most frameworks but lacks minor specifics.
3 — Provides basic comparisons with limited metrics or examples.
2 — Provides vague or overly general comparisons.
1 — No meaningful analysis provided.

#### D. Output Structure and Source Credibility (0.10)
Measures the organization of the response and credibility of sources.

5 — Response is well-organized (e.g., table or structured list) and all claims are sourced from credible platforms.
4 — Response is mostly organized and sources are credible but lacks minor structural clarity.
3 — Response is somewhat organized but lacks clarity or credible sourcing.
2 — Response is poorly organized or sources are unclear.
1 — Response is disorganized and sources are absent or unreliable.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "comparison_accuracy": <1-5>,
  "coverage_of_required_platforms": <1-5>,
  "depth_of_analysis": <1-5>,
  "output_structure_and_source_credibility": <1-5>,
  "dimension_reasoning": {
    "comparison_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_required_platforms": "<one sentence citing specific evidence>",
    "depth_of_analysis": "<one sentence citing specific evidence>",
    "output_structure_and_source_credibility": "<one sentence citing specific evidence>"
  },
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "comparison_accuracy": 0.35,
    "coverage_of_required_platforms": 0.30,
    "depth_of_analysis": 0.25,
    "output_structure_and_source_credibility": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())