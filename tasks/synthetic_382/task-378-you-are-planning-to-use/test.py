"""
LLM-as-judge evaluator for EvolveBench task.

Category: Design
Task: Research and compare Figma, Sketch, and Adobe XD on features like live collaboration, version control, integrations, and pricing, and summarize findings in a comparison table.
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


TASK_INSTRUCTION = """You are planning to use a design tool for collaborative UI/UX development with remote team members. Research and compare Figma, Sketch, and Adobe XD on features like live collaboration, version control, integrations, and pricing. Summarize the findings in a comparison table."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to research and compare three design tools (Figma, Sketch, Adobe XD) on specific features: live collaboration, version control, integrations, and pricing. The domain is collaborative UI/UX design. A successful completion involves summarizing the findings in a structured comparison table that includes all required features and platforms.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
You are planning to use a design tool for collaborative UI/UX development with remote team members. Research and compare Figma, Sketch, and Adobe XD on features like live collaboration, version control, integrations, and pricing. Summarize the findings in a comparison table.

## Task-Specific Constraints
- Must visit all three specified platforms: figma.com, sketch.com, adobe.com.
- Must include data on live collaboration, version control, integrations, and pricing for each platform.
- Output must be organized as a table with clear columns and rows.
- Must provide specific pricing details for each tool.
- Must cite sources or provide evidence for claims made about features.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to all three required platforms (figma.com, sketch.com, adobe.com)?
- Are live collaboration, version control, integrations, and pricing data present for all three tools?
- Is the output organized as a table with clear columns and rows?
- Are pricing details specific and accurate for each tool?
- Are claims about features supported by evidence or citations?

### Step 2: Dimension Scoring

#### A. Deliverable Accuracy (0.35)
Measures whether the comparison table is correct, complete, and includes all required features.

5 — Includes all required features (live collaboration, version control, integrations, pricing) for all three tools, with accurate data.
4 — Includes most required features, with minor inaccuracies or omissions.
3 — Includes some required features, but incomplete or partially incorrect.
2 — Includes few required features, with significant inaccuracies.
1 — Does not include any required features or is completely incorrect.

#### B. Coverage of Platforms (0.30)
Measures whether the agent researched all three specified platforms.

5 — Researches and includes data from all three platforms (figma.com, sketch.com, adobe.com).
4 — Researches and includes data from two platforms, with minor omissions.
3 — Researches and includes data from one platform, or incomplete data from two.
2 — Minimal research, with data from one platform only.
1 — No research or data from any platform.

#### C. Depth of Comparison (0.25)
Measures the specificity and detail in the comparison (e.g., pricing breakdown, feature descriptions).

5 — Provides detailed comparisons, including specific pricing breakdowns and feature descriptions for all tools.
4 — Provides moderately detailed comparisons, with minor omissions or lack of specificity.
3 — Provides basic comparisons, with limited detail or generalizations.
2 — Provides minimal comparisons, with vague or incorrect details.
1 — Provides no meaningful comparisons.

#### D. Output Structure and Credibility (0.10)
Measures whether the output is well-organized and supported by credible evidence.

5 — Output is structured as a clear table, with citations or credible sources for all claims.
4 — Output is mostly well-structured, with minor formatting issues or limited sourcing.
3 — Output is partially structured, with some disorganization or lack of sourcing.
2 — Output is poorly structured, with significant disorganization or no sourcing.
1 — Output is completely unstructured or unsupported.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "deliverable_accuracy": <1-5>,
  "coverage_of_platforms": <1-5>,
  "depth_of_comparison": <1-5>,
  "output_structure_and_credibility": <1-5>,
  "dimension_reasoning": {{
    "deliverable_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_platforms": "<one sentence citing specific evidence>",
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
    "coverage_of_platforms": 0.30,
    "depth_of_comparison": 0.25,
    "output_structure_and_credibility": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())