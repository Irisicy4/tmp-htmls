"""
LLM-as-judge evaluator for EvolveBench task.

Category: Software Engineering
Task: Research and compare three JavaScript testing frameworks (Jest, Mocha, and Cypress) based on setup ease, support for mocking/stubbing, and parallel execution capabilities.
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


TASK_INSTRUCTION = """Research and compare three open-source JavaScript testing frameworks (Jest, Mocha, and Cypress) on criteria such as setup ease, support for mocking/stubbing, and parallel execution capabilities. Use their official documentation, GitHub pages, and at least one external developer blog or review site."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to research and compare three JavaScript testing frameworks (Jest, Mocha, and Cypress) based on setup ease, support for mocking/stubbing, and parallel execution capabilities. The domain is software engineering, and successful completion requires a structured comparison with evidence from official documentation, GitHub pages, and external developer blogs.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Research and compare three open-source JavaScript testing frameworks (Jest, Mocha, and Cypress) on criteria such as setup ease, support for mocking/stubbing, and parallel execution capabilities. Use their official documentation, GitHub pages, and at least one external developer blog or review site.

## Task-Specific Constraints
- Must visit jestjs.io, mochajs.org, cypress.io, and github.com.
- Must include at least one external developer blog or review site.
- Must compare setup ease, mocking/stubbing support, and parallel execution capabilities for all three frameworks.
- Output must be organized as a structured table or list.
- Must cite specific evidence from the platforms visited.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to jestjs.io, mochajs.org, cypress.io, github.com, and at least one external blog or review site? Which ones were actually visited?
- Are setup ease, mocking/stubbing support, and parallel execution capabilities compared for all three frameworks?
- Is the output organized as a structured table or list?
- Are specific claims sourced from the required platforms?

### Step 2: Dimension Scoring

#### A. Comparison Accuracy (0.35)
Measures whether the agent accurately compared the three frameworks based on the specified criteria.

5 — All three criteria (setup ease, mocking/stubbing, parallel execution) are compared accurately for all three frameworks, with evidence cited.
4 — Two criteria are compared accurately for all three frameworks, or three criteria for two frameworks.
3 — At least one criterion is compared accurately for all three frameworks.
2 — Comparisons are mostly inaccurate or incomplete.
1 — No meaningful comparisons are provided.

#### B. Coverage of Sources (0.30)
Measures whether the agent visited and used all required platforms and sources.

5 — All required platforms (jestjs.io, mochajs.org, cypress.io, github.com, and one external blog) are visited and cited.
4 — Four platforms are visited and cited.
3 — Three platforms are visited and cited.
2 — Two platforms are visited and cited.
1 — One or no platforms are visited.

#### C. Depth of Analysis (0.25)
Measures the level of detail and specificity in the comparisons.

5 — Includes detailed descriptions, numbers, or examples for all criteria and frameworks.
4 — Includes detailed descriptions for most criteria and frameworks.
3 — Includes basic descriptions for some criteria and frameworks.
2 — Includes vague or minimal descriptions.
1 — No meaningful analysis is provided.

#### D. Output Structure and Credibility (0.10)
Measures whether the output is well-organized and sourced from credible platforms.

5 — Output is structured as a clear table or list, with all claims sourced.
4 — Output is structured but some claims lack sources.
3 — Output is somewhat structured but lacks clarity or sources.
2 — Output is poorly structured or mostly unsourced.
1 — Output is unstructured and unsourced.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "comparison_accuracy": <1-5>,
  "coverage_of_sources": <1-5>,
  "depth_of_analysis": <1-5>,
  "output_structure_and_credibility": <1-5>,
  "dimension_reasoning": {
    "comparison_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_sources": "<one sentence citing specific evidence>",
    "depth_of_analysis": "<one sentence citing specific evidence>",
    "output_structure_and_credibility": "<one sentence citing specific evidence>"
  },
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "comparison_accuracy": 0.35,
    "coverage_of_sources": 0.30,
    "depth_of_analysis": 0.25,
    "output_structure_and_credibility": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())