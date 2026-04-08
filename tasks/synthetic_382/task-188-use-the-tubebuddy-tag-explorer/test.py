"""
LLM-as-judge evaluator for EvolveBench task.

Category: (Self) Media
Task: Identify and select the most relevant keywords for a fictional video about 'Social Media Growth in 2024' using TubeBuddy and YouTube search results, and report the top 5 recommended tags with their search volumes.
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


TASK_INSTRUCTION = """Use the TubeBuddy Tag Explorer tool and the YouTube search results page to identify and select the most relevant keywords for a fictional video about 'Social Media Growth in 2024.' Report the top 5 recommended tags and their search volumes."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

This task involves identifying and selecting the most relevant keywords for a fictional video about 'Social Media Growth in 2024' using TubeBuddy's Tag Explorer and YouTube search results. A successful completion requires the agent to report the top 5 recommended tags along with their search volumes.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Use the TubeBuddy Tag Explorer tool and the YouTube search results page to identify and select the most relevant keywords for a fictional video about 'Social Media Growth in 2024.' Report the top 5 recommended tags and their search volumes.

## Task-Specific Constraints
- Must use the TubeBuddy Tag Explorer tool to identify keywords.
- Must cross-check keyword relevance using YouTube search results.
- Must report exactly 5 tags.
- Must include search volumes for each tag.
- Output must be organized as a structured list or table.
- Tags must be relevant to the topic 'Social Media Growth in 2024.'

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to the required platforms (TubeBuddy and YouTube)?
- Are the top 5 tags reported in the response?
- Does the response include search volumes for each tag?
- Are the reported tags relevant to the topic 'Social Media Growth in 2024'?
- Is the output organized as a structured list or table?

### Step 2: Dimension Scoring

#### A. Tag Selection Accuracy (0.35)
Measures whether the agent selected the most relevant tags for the topic.

5 — All 5 tags are highly relevant to 'Social Media Growth in 2024.'
4 — 4 out of 5 tags are relevant, with minor inaccuracies.
3 — At least 3 tags are relevant, but others are generic or irrelevant.
2 — Only 1-2 tags are relevant to the topic.
1 — None of the tags are relevant.

#### B. Platform Usage Coverage (0.30)
Measures whether the agent used both TubeBuddy and YouTube as required.

5 — Evidence shows the agent used both platforms comprehensively.
4 — Evidence shows the agent used both platforms but with minor gaps.
3 — Evidence shows the agent used only one platform.
2 — Evidence shows the agent attempted but failed to use the platforms effectively.
1 — No evidence of platform usage.

#### C. Search Volume Inclusion (0.20)
Measures whether the agent included search volumes for all reported tags.

5 — Search volumes are included for all 5 tags.
4 — Search volumes are included for 4 tags.
3 — Search volumes are included for at least 3 tags.
2 — Search volumes are included for 1-2 tags.
1 — No search volumes are included.

#### D. Output Structure and Organization (0.15)
Measures whether the response is well-organized and easy to interpret.

5 — Output is structured as a clear table or list with all required elements.
4 — Output is mostly structured but has minor formatting issues.
3 — Output is partially structured but lacks clarity or completeness.
2 — Output is disorganized or difficult to interpret.
1 — Output is absent or completely unstructured.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "tag_selection_accuracy": <1-5>,
  "platform_usage_coverage": <1-5>,
  "search_volume_inclusion": <1-5>,
  "output_structure_and_organization": <1-5>,
  "dimension_reasoning": {{
    "tag_selection_accuracy": "<one sentence citing specific evidence>",
    "platform_usage_coverage": "<one sentence citing specific evidence>",
    "search_volume_inclusion": "<one sentence citing specific evidence>",
    "output_structure_and_organization": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "tag_selection_accuracy": 0.35,
    "platform_usage_coverage": 0.30,
    "search_volume_inclusion": 0.20,
    "output_structure_and_organization": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())