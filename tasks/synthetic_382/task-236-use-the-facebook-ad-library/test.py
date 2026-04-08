"""
LLM-as-judge evaluator for EvolveBench task.

Category: (Self) Media
Task: Use the Facebook Ad Library to find and report data on active video ad campaigns related to 'content creation tools' in the United States.
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


TASK_INSTRUCTION = """Use the Facebook Ad Library to search for active video ad campaigns related to 'content creation tools'. Filter for ads currently running in the United States and extract their impressions, run dates, and advertiser names. Report data from the top 5 ads shown."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves using the Facebook Ad Library to search for active video ad campaigns related to 'content creation tools'. The agent must filter for ads currently running in the United States and extract impressions, run dates, and advertiser names. A successful completion requires reporting data from the top 5 ads shown in a structured format.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Use the Facebook Ad Library to search for active video ad campaigns related to 'content creation tools'. Filter for ads currently running in the United States and extract their impressions, run dates, and advertiser names. Report data from the top 5 ads shown.

## Task-Specific Constraints
- Must use the Facebook Ad Library platform to retrieve data.
- Must filter for ads currently running in the United States.
- Must extract impressions, run dates, and advertiser names for each ad.
- Must report data for the top 5 ads shown.
- Output must be structured as a table or structured list.
- Data must be accurate and match the platform's information.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to the Facebook Ad Library platform?
- Did the agent apply the required filter for ads running in the United States?
- Are impressions, run dates, and advertiser names present in the response?
- Is the output organized as a table or structured list?
- Are the reported data accurate and match the platform's information?

### Step 2: Dimension Scoring

#### A. Primary Deliverable Accuracy (0.35)
Measures whether the agent correctly identified and reported data for the top 5 ads.

5 — Reports data for 5 ads with accurate impressions, run dates, and advertiser names.
4 — Reports data for 4 ads with accurate details or 5 ads with minor inaccuracies.
3 — Reports data for 3 ads with some inaccuracies.
2 — Reports data for 1-2 ads or mostly inaccurate details.
1 — Fails to report any usable data.

#### B. Coverage of Required Platforms and Filters (0.30)
Measures whether the agent used the Facebook Ad Library and applied the correct filters.

5 — Uses the Facebook Ad Library and applies the filter for ads running in the United States.
4 — Uses the Facebook Ad Library but applies filters partially or incorrectly.
3 — Uses the Facebook Ad Library but fails to apply filters correctly.
2 — Navigates to unrelated platforms or applies no filters.
1 — Fails to use the required platform.

#### C. Depth and Specificity of Extracted Data (0.20)
Measures the level of detail in the extracted data.

5 — Extracts impressions, run dates, and advertiser names for all ads with high specificity.
4 — Extracts most required details but misses minor elements.
3 — Extracts some required details but lacks specificity.
2 — Extracts minimal details or mostly incorrect data.
1 — Fails to extract any meaningful data.

#### D. Output Structure and Organization (0.15)
Measures whether the output is well-organized and easy to interpret.

5 — Presents data in a clear, structured table or list format.
4 — Presents data in a mostly clear format with minor issues.
3 — Presents data in a usable but unstructured or unclear format.
2 — Presents data in a disorganized or confusing format.
1 — Fails to present data in any usable format.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "The agent navigated to the Facebook Ad Library and applied the required filters. Data for impressions, run dates, and advertiser names were extracted but partially incomplete. The output was structured as a table but lacked full accuracy.",
  "primary_deliverable_accuracy": 3,
  "coverage_of_required_platforms_and_filters": 4,
  "depth_and_specificity_of_extracted_data": 3,
  "output_structure_and_organization": 4,
  "dimension_reasoning": {{
    "primary_deliverable_accuracy": "Data for 3 ads were reported with partial accuracy.",
    "coverage_of_required_platforms_and_filters": "The agent used the Facebook Ad Library and applied filters for the United States.",
    "depth_and_specificity_of_extracted_data": "Impressions and run dates were extracted but lacked full detail.",
    "output_structure_and_organization": "The output was organized as a table but had minor formatting issues."
  }},
  "overall_score": 3.45,
  "passed": true
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "primary_deliverable_accuracy": 0.35,
    "coverage_of_required_platforms_and_filters": 0.30,
    "depth_and_specificity_of_extracted_data": 0.20,
    "output_structure_and_organization": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())