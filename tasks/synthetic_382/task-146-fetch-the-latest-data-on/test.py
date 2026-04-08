"""
LLM-as-judge evaluator for EvolveBench task.

Category: Software Engineering
Task: Fetch the latest data on the top three database engines by market share, compare their query benchmark performance, and recommend the fastest database for large query loads.
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


TASK_INSTRUCTION = """Fetch the latest data on the top three database engines by market share from DB-Engines. Compare their query benchmark performance (SQL or NoSQL) using public benchmarks or documentation. Recommend the fastest database for handling large query loads."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to identify the top three database engines by market share from DB-Engines, compare their query benchmark performance using public benchmarks or official documentation, and recommend the fastest database for handling large query loads. This task falls under the domain of software engineering.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Fetch the latest data on the top three database engines by market share from DB-Engines. Compare their query benchmark performance (SQL or NoSQL) using public benchmarks or documentation. Recommend the fastest database for handling large query loads.

## Task-Specific Constraints
- Must visit db-engines.com to fetch market share data.
- Must compare query performance using at least one public benchmark or official documentation for each database.
- Must provide a clear recommendation for the fastest database for large query loads.
- Output must include a structured comparison (e.g., table or bullet points).
- Must specify the sources used for the comparison.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to db-engines.com to fetch market share data?
- Did the agent compare query performance using benchmarks or documentation for all three databases?
- Is the recommendation for the fastest database clearly stated and justified?
- Is the output structured as a table or bullet points?
- Are the sources for the comparison explicitly mentioned?

### Step 2: Dimension Scoring

#### A. Deliverable Accuracy (0.35)
Measures whether the agent's recommendation is correct and based on valid data.

5 — Recommendation is correct, justified, and based on valid benchmark data for all three databases.
4 — Recommendation is correct but justification is slightly incomplete or missing details.
3 — Recommendation is partially correct but lacks sufficient justification or data.
2 — Recommendation is mostly incorrect or missing key data.
1 — Recommendation is absent or completely incorrect.

#### B. Coverage of Required Platforms (0.30)
Measures whether the agent visited all required platforms and included data from them.

5 — Data from db-engines.com and benchmarks/documentation for all three databases is included.
4 — Data from db-engines.com and benchmarks/documentation for two databases is included.
3 — Data from db-engines.com is included but benchmarks/documentation for only one database is present.
2 — Data from db-engines.com is missing or benchmarks/documentation is absent.
1 — No relevant data from required platforms is included.

#### C. Depth of Comparison (0.25)
Measures the specificity and detail of the comparison provided.

5 — Includes detailed query performance metrics, comparisons, and numerical benchmarks for all three databases.
4 — Includes query performance metrics for all three databases but lacks some numerical benchmarks or comparisons.
3 — Includes basic query performance metrics but lacks depth or numerical benchmarks.
2 — Includes minimal or vague performance data.
1 — No meaningful performance data is included.

#### D. Source Credibility and Output Structure (0.10)
Measures whether the sources are credible and the output is well-organized.

5 — Sources are credible, explicitly mentioned, and the output is structured as a clear table or bullet points.
4 — Sources are credible but not explicitly mentioned; output is structured but slightly unclear.
3 — Sources are partially credible or unclear; output is minimally structured.
2 — Sources are mostly missing or not credible; output is poorly structured.
1 — Sources are absent or not credible; output is unstructured.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "deliverable_accuracy": <1-5>,
  "coverage_of_required_platforms": <1-5>,
  "depth_of_comparison": <1-5>,
  "source_credibility_and_output_structure": <1-5>,
  "dimension_reasoning": {{
    "deliverable_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_required_platforms": "<one sentence citing specific evidence>",
    "depth_of_comparison": "<one sentence citing specific evidence>",
    "source_credibility_and_output_structure": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "deliverable_accuracy": 0.35,
    "coverage_of_required_platforms": 0.30,
    "depth_of_comparison": 0.25,
    "source_credibility_and_output_structure": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())