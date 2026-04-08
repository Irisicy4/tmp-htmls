"""
LLM-as-judge evaluator for EvolveBench task.

Category: Design
Task: Compare three popular UI prototyping tools based on collaboration features, pricing models, and integrations with project management platforms.
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


TASK_INSTRUCTION = """Compare three popular UI prototyping tools (Adobe XD, Figma, and Sketch) based on their collaboration features, pricing models, and integrations with project management platforms. Gather the information from their respective official websites and SaaS comparison articles."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to compare three UI prototyping tools (Adobe XD, Figma, and Sketch) based on collaboration features, pricing models, and integrations with project management platforms. The domain is design and SaaS tools. A successful completion requires the agent to provide accurate, structured comparisons using credible sources.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Compare three popular UI prototyping tools (Adobe XD, Figma, and Sketch) based on their collaboration features, pricing models, and integrations with project management platforms. Gather the information from their respective official websites and SaaS comparison articles.

## Task-Specific Constraints
- Must visit the official websites of Adobe XD, Figma, and Sketch.
- Must include pricing data for all three tools.
- Must compare collaboration features explicitly (e.g., real-time editing, version control).
- Must mention integrations with at least two project management platforms for each tool.
- Output must be organized as a structured table or list.
- Must cite sources or provide evidence for claims.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to the required platforms? Which ones were actually visited?
- Are pricing details for all three tools present in the response?
- Are collaboration features compared explicitly and accurately?
- Are integrations with at least two project management platforms mentioned for each tool?
- Is the output organized as a structured table or list?

### Step 2: Dimension Scoring

#### A. Comparison Accuracy (0.35)
Measures whether the agent's comparisons of collaboration features, pricing models, and integrations are correct and complete.

5 — All comparisons are accurate, complete, and supported by evidence.
4 — Comparisons are mostly accurate and complete, with minor omissions or errors.
3 — Comparisons are partially accurate but incomplete or lack evidence.
2 — Comparisons are mostly incorrect or missing key elements.
1 — Comparisons are absent or completely incorrect.

#### B. Coverage of Required Platforms (0.30)
Measures whether the agent visited all required platforms and included data from each.

5 — Data from all three platforms is included and verified.
4 — Data from all three platforms is included but lacks verification for one.
3 — Data from two platforms is included; one is missing or incomplete.
2 — Data from only one platform is included.
1 — No data from required platforms is included.

#### C. Detail and Specificity (0.25)
Measures the depth of information provided, including pricing details, feature comparisons, and integration specifics.

5 — Includes detailed pricing, feature comparisons, and integrations for all three tools.
4 — Includes detailed information for most tools, with minor gaps.
3 — Includes basic information for all tools but lacks depth or specifics.
2 — Includes minimal information with significant gaps.
1 — Includes no meaningful details.

#### D. Output Structure and Source Credibility (0.10)
Measures whether the response is well-organized and cites credible sources.

5 — Output is structured as a clear table or list and cites credible sources.
4 — Output is structured but lacks citation for one or more claims.
3 — Output is partially structured or lacks clarity; sources are unclear.
2 — Output is poorly structured or sources are missing.
1 — Output is unstructured and sources are absent.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "comparison_accuracy": <1-5>,
  "coverage_of_required_platforms": <1-5>,
  "detail_and_specificity": <1-5>,
  "output_structure_and_source_credibility": <1-5>,
  "dimension_reasoning": {{
    "comparison_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_required_platforms": "<one sentence citing specific evidence>",
    "detail_and_specificity": "<one sentence citing specific evidence>",
    "output_structure_and_source_credibility": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "comparison_accuracy": 0.35,
    "coverage_of_required_platforms": 0.30,
    "detail_and_specificity": 0.25,
    "output_structure_and_source_credibility": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())