"""
LLM-as-judge evaluator for EvolveBench task.

Category: Software Engineering
Task: Extract the top five npm packages tagged 'image-processing' with at least 1,000 weekly downloads and their latest versions.
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


TASK_INSTRUCTION = """Use the npm package search tool and filter for libraries tagged 'image-processing' with at least 1,000 weekly downloads. Extract the top five packages based on weekly download stats and note their latest published versions."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to use the npm package search tool to find libraries tagged 'image-processing' with at least 1,000 weekly downloads. The agent must extract the top five packages based on weekly download statistics and provide their latest published versions. This is a Software Engineering task requiring accurate data extraction and structured output.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Use the npm package search tool and filter for libraries tagged 'image-processing' with at least 1,000 weekly downloads. Extract the top five packages based on weekly download stats and note their latest published versions.

## Task-Specific Constraints
- Must use the npmjs.com platform to search for packages.
- Must filter results by the 'image-processing' tag.
- Must ensure packages have at least 1,000 weekly downloads.
- Must extract exactly five packages based on weekly download statistics.
- Must include the latest published version for each package.
- Output must be structured as a table or JSON list.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to npmjs.com to perform the search?
- Did the agent filter results by the 'image-processing' tag?
- Are all five packages listed in the response, with weekly download stats and latest versions?
- Is the output structured as a table or JSON list?
- Are the weekly download statistics and version numbers accurate based on the tool-call trace?

### Step 2: Dimension Scoring

#### A. Primary Deliverable Accuracy (0.35)
Measures whether the agent correctly identified the top five packages and their latest versions.

5 — All five packages are correct, with accurate weekly download stats and latest versions.
4 — Four packages are correct, or minor inaccuracies in stats/versions.
3 — Three packages are correct, with significant inaccuracies in stats/versions.
2 — One or two packages are correct, with major inaccuracies.
1 — None of the packages are correct.

#### B. Coverage of Constraints (0.30)
Measures whether the agent adhered to all task-specific constraints.

5 — All constraints satisfied (platform used, tag filtered, download threshold applied).
4 — One constraint partially missed or incorrectly applied.
3 — Two constraints missed or incorrectly applied.
2 — Three constraints missed or incorrectly applied.
1 — Four or more constraints missed.

#### C. Depth of Data Extraction (0.20)
Measures the level of detail in the extracted data (e.g., stats, versions).

5 — Each package includes detailed stats and version numbers.
4 — Minor details missing for one package.
3 — Significant details missing for two packages.
2 — Details missing for three or more packages.
1 — No meaningful details extracted.

#### D. Output Structure and Credibility (0.15)
Measures the organization and credibility of the output.

5 — Output is well-structured (table/JSON) and data is credible.
4 — Output is structured but contains minor formatting issues or credibility concerns.
3 — Output is partially structured, with moderate formatting or credibility issues.
2 — Output is poorly structured, with major formatting or credibility issues.
1 — Output is unstructured or completely unreliable.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "primary_deliverable_accuracy": <1-5>,
  "coverage_of_constraints": <1-5>,
  "depth_of_data_extraction": <1-5>,
  "output_structure_and_credibility": <1-5>,
  "dimension_reasoning": {{
    "primary_deliverable_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_constraints": "<one sentence citing specific evidence>",
    "depth_of_data_extraction": "<one sentence citing specific evidence>",
    "output_structure_and_credibility": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "primary_deliverable_accuracy": 0.35,
    "coverage_of_constraints": 0.30,
    "depth_of_data_extraction": 0.20,
    "output_structure_and_credibility": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())