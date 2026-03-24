"""
LLM-as-judge evaluator for EvolveBench task.

Category: Travel & Planning
Task: Verify whether U.S. citizens currently need a visa to visit Brazil for tourism.
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


TASK_INSTRUCTION = """Verify whether U.S. citizens currently need a visa to visit Brazil for tourism. Check the official Brazil embassy website, the U.S. State Department travel advisory page, and IATA's travel regulation map for current information on entry requirements. Provide a verification report stating the visa status and citing the evidence from the sources."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to verify whether U.S. citizens currently need a visa to visit Brazil for tourism by consulting three specific platforms: the official Brazil embassy website, the U.S. State Department travel advisory page, and IATA's travel regulation map. A successful completion requires the agent to provide a verification report stating the visa status and citing evidence from all three sources.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Verify whether U.S. citizens currently need a visa to visit Brazil for tourism. Check the official Brazil embassy website, the U.S. State Department travel advisory page, and IATA's travel regulation map for current information on entry requirements. Provide a verification report stating the visa status and citing the evidence from the sources.

## Task-Specific Constraints
- Must visit all three specified platforms: Brazil embassy website, U.S. State Department travel advisory page, and IATA's travel regulation map.
- Must explicitly state whether a visa is required for U.S. citizens visiting Brazil for tourism.
- Must cite evidence from each platform in the response.
- Output must be organized as a structured report with clear sections for each platform.
- Must ensure the information is current and accurate as of the task execution date.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to the required platforms? Which ones were actually visited?
- Does the response explicitly state whether a visa is required for U.S. citizens visiting Brazil for tourism?
- Are citations from all three platforms present in the response?
- Is the output organized as a structured report with clear sections for each platform?
- Is the information current and accurate based on the evidence?

### Step 2: Dimension Scoring

#### A. Primary Deliverable Accuracy (0.35)
Measures whether the agent correctly identifies the visa requirement status for U.S. citizens visiting Brazil.

5 — Correctly identifies visa status and provides evidence from all three platforms.
4 — Correctly identifies visa status but evidence from one platform is missing or incomplete.
3 — Partially correct visa status with incomplete or unclear evidence.
2 — Incorrect visa status or missing significant evidence.
1 — No attempt or completely incorrect response.

#### B. Coverage of Required Platforms (0.30)
Measures whether the agent visited and utilized all three specified platforms.

5 — Evidence from all three platforms is clearly cited and used.
4 — Evidence from two platforms is cited and used.
3 — Evidence from one platform is cited and used.
2 — No evidence from the specified platforms is cited.
1 — No attempt to visit any of the specified platforms.

#### C. Depth and Specificity (0.20)
Measures whether the response includes detailed and specific information from the platforms.

5 — Includes detailed information with specific citations from all platforms.
4 — Includes detailed information but lacks specificity in one platform.
3 — Includes some specific information but lacks detail across platforms.
2 — Includes vague or generic information with little specificity.
1 — No specific information included.

#### D. Output Structure and Credibility (0.15)
Measures whether the response is well-organized and cites credible sources.

5 — Response is well-organized with clear sections and credible citations.
4 — Response is mostly organized but lacks clarity in one section or citation.
3 — Response is partially organized with unclear sections or citations.
2 — Response is disorganized or lacks credible citations.
1 — Response is completely disorganized or lacks any citations.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "primary_deliverable_accuracy": <1-5>,
  "coverage_of_required_platforms": <1-5>,
  "depth_and_specificity": <1-5>,
  "output_structure_and_credibility": <1-5>,
  "dimension_reasoning": {{
    "primary_deliverable_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_required_platforms": "<one sentence citing specific evidence>",
    "depth_and_specificity": "<one sentence citing specific evidence>",
    "output_structure_and_credibility": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "primary_deliverable_accuracy": 0.35,
    "coverage_of_required_platforms": 0.30,
    "depth_and_specificity": 0.20,
    "output_structure_and_credibility": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())