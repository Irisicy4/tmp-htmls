"""
LLM-as-judge evaluator for EvolveBench task.

Category: Travel & Planning
Task: Investigate why Hyperdia may no longer be accessible or updated for Japan Rail Pass users by consulting forums, FAQs, and travel blogs.
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


TASK_INSTRUCTION = """A user reports difficulty accessing train schedules for Japan Rail Pass users on the Hyperdia website. Investigate why the site may no longer be accessible or updated for these services by consulting forums like Japan Rail Pass Reddit, Hyperdia FAQ sections, and travel blogs."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to investigate why Hyperdia may no longer be accessible or updated for Japan Rail Pass users. This involves consulting forums, FAQ sections, and travel blogs to gather information about the issue. A successful completion requires identifying credible reasons for the site's inaccessibility or lack of updates, citing sources, and presenting findings in a structured format.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
A user reports difficulty accessing train schedules for Japan Rail Pass users on the Hyperdia website. Investigate why the site may no longer be accessible or updated for these services by consulting forums like Japan Rail Pass Reddit, Hyperdia FAQ sections, and travel blogs.

## Task-Specific Constraints
- Must visit at least 3 of the specified platforms (reddit.com, hyperdia.com, japan-guide.com).
- Must identify credible reasons for the site's inaccessibility or lack of updates.
- Must cite sources for all claims made.
- Output must be organized as a structured summary or bullet points.
- Must address whether alternative tools or resources are available for train schedules.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to the required platforms? Which ones were actually visited?
- Are credible reasons for Hyperdia's inaccessibility or lack of updates present in the response?
- Are sources cited for all claims made by the agent?
- Is the output organized as a structured summary or bullet points?
- Does the response address whether alternative tools or resources are available for train schedules?

### Step 2: Dimension Scoring

#### A. Primary Deliverable Accuracy (0.35)
Measures whether the agent correctly identified credible reasons for Hyperdia's inaccessibility or lack of updates.

5 — Identifies 3 or more credible reasons with specific evidence cited.
4 — Identifies 2 credible reasons with evidence cited.
3 — Identifies 1 credible reason with partial evidence.
2 — Identifies reasons but lacks evidence or credibility.
1 — No credible reasons identified.

#### B. Coverage of Required Sources (0.30)
Measures whether the agent visited and utilized the required platforms.

5 — Utilizes all 3 specified platforms and cites information from each.
4 — Utilizes 2 specified platforms and cites information from each.
3 — Utilizes 1 specified platform and cites information.
2 — Visits platforms but fails to cite information.
1 — Does not visit or utilize specified platforms.

#### C. Depth and Specificity (0.25)
Measures the level of detail, specificity, and presence of alternative tools or resources.

5 — Provides detailed explanations with alternative tools/resources identified.
4 — Provides explanations with some detail; mentions alternatives.
3 — Provides basic explanations with minimal detail; alternatives partially addressed.
2 — Explanations lack detail; alternatives not addressed.
1 — No explanations or alternatives provided.

#### D. Source Quality and Output Structure (0.10)
Measures the credibility of sources and the organization of the response.

5 — Sources are credible and response is well-organized in structured format.
4 — Sources are mostly credible and response is organized.
3 — Sources are partially credible; response is somewhat organized.
2 — Sources lack credibility; response is poorly organized.
1 — No credible sources; response is disorganized.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "primary_deliverable_accuracy": <1-5>,
  "coverage_of_required_sources": <1-5>,
  "depth_and_specificity": <1-5>,
  "source_quality_and_output_structure": <1-5>,
  "dimension_reasoning": {{
    "primary_deliverable_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_required_sources": "<one sentence citing specific evidence>",
    "depth_and_specificity": "<one sentence citing specific evidence>",
    "source_quality_and_output_structure": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "primary_deliverable_accuracy": 0.35,
    "coverage_of_required_sources": 0.30,
    "depth_and_specificity": 0.25,
    "source_quality_and_output_structure": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())