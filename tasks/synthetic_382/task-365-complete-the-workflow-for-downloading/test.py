"""
LLM-as-judge evaluator for EvolveBench task.

Category: Design
Task: Download a custom business card template from a marketplace, filtering for free options, and report the template name and URL.
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


TASK_INSTRUCTION = """Complete the workflow for downloading a custom design template on a publicly accessible marketplace. Search for a business card template, filter for free downloads, and proceed through the steps to finalize the download. Report the template name and URL."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task is in the Design domain and involves downloading a custom business card template from a publicly accessible marketplace. The agent must search for a business card template, apply a filter for free downloads, and successfully complete the download process. The agent must also report the template name and URL.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Complete the workflow for downloading a custom design template on a publicly accessible marketplace. Search for a business card template, filter for free downloads, and proceed through the steps to finalize the download. Report the template name and URL.

## Task-Specific Constraints
- Must visit at least one of the specified platforms (canva.com, freepik.com, brandpacks.com).
- Must apply a filter to ensure only free templates are considered.
- Must successfully navigate to the download page and complete the download process.
- Must report both the template name and the URL of the downloaded template.
- Output must clearly identify the template name and URL in a structured format.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to at least one of the required platforms? Which platform(s) were visited?
- Did the agent apply a filter to ensure only free templates were considered?
- Did the agent successfully complete the download process?
- Are both the template name and URL clearly identified in the response?
- Is the output structured in a clear and organized way?

### Step 2: Dimension Scoring

#### A. Primary Deliverable Accuracy (0.35)
Measures whether the agent successfully downloaded a free business card template and reported the correct name and URL.

5 — Successfully downloaded a free template, with both name and URL reported accurately.
4 — Downloaded a free template, but one minor detail (e.g., name or URL) is incomplete or slightly inaccurate.
3 — Partially completed: either the download was not finalized or the name/URL is missing.
2 — Attempted but failed to download or report key details.
1 — Did not attempt or completely failed.

#### B. Platform Coverage (0.30)
Measures whether the agent used at least one of the required platforms and applied the necessary filters.

5 — Used at least one required platform and applied the free filter correctly.
4 — Used one platform but with minor issues in applying the filter.
3 — Used a platform but did not apply the filter correctly.
2 — Attempted but failed to use a required platform or apply the filter.
1 — Did not use any required platform.

#### C. Depth of Detail (0.20)
Measures the level of detail in the agent's response, including clarity and specificity of the reported information.

5 — Response includes detailed and specific information about the template and download process.
4 — Response is mostly detailed but has minor omissions or ambiguities.
3 — Response includes basic details but lacks depth or specificity.
2 — Response is vague or missing key details.
1 — Response is completely lacking detail.

#### D. Output Structure and Clarity (0.15)
Measures how well-organized and clear the agent's response is.

5 — Response is well-structured, clear, and easy to follow.
4 — Response is mostly clear but could be better organized.
3 — Response is understandable but lacks organization or clarity.
2 — Response is poorly organized or difficult to follow.
1 — Response is completely disorganized or incoherent.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "primary_deliverable_accuracy": <1-5>,
  "platform_coverage": <1-5>,
  "depth_of_detail": <1-5>,
  "output_structure_and_clarity": <1-5>,
  "dimension_reasoning": {{
    "primary_deliverable_accuracy": "<one sentence citing specific evidence>",
    "platform_coverage": "<one sentence citing specific evidence>",
    "depth_of_detail": "<one sentence citing specific evidence>",
    "output_structure_and_clarity": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "primary_deliverable_accuracy": 0.35,
    "platform_coverage": 0.30,
    "depth_of_detail": 0.20,
    "output_structure_and_clarity": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())