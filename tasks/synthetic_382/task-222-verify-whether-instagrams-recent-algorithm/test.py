"""
LLM-as-judge evaluator for EvolveBench task.

Category: (Self) Media
Task: Verify whether Instagram's recent algorithm change favors video content over images using official announcements and engagement metrics.
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


TASK_INSTRUCTION = """Verify whether Instagram's recent algorithm change (reported in tech blogs) favors video content over images. Collect evidence from Instagram's official announcements and the engagement metrics of recent posts to confirm or refute the claim."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to investigate whether Instagram's algorithm change prioritizes video content over images. The agent must gather evidence from Instagram's official announcements and analyze engagement metrics of recent posts. A successful completion includes a clear conclusion supported by evidence from both sources.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Verify whether Instagram's recent algorithm change (reported in tech blogs) favors video content over images. Collect evidence from Instagram's official announcements and the engagement metrics of recent posts to confirm or refute the claim.

## Task-Specific Constraints
- Must visit Instagram's official announcements or blog posts to gather evidence.
- Must analyze engagement metrics (e.g., likes, comments, shares) of recent Instagram posts.
- Must compare engagement metrics for videos and images.
- Must provide a clear conclusion on whether video content is favored over images.
- Output must include specific data points or examples to support the conclusion.
- Sources must be explicitly cited.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Instagram's official announcements or blog posts?
- Did the agent analyze engagement metrics for both video and image content?
- Did the agent provide specific data points or examples to support their conclusion?
- Did the agent provide a clear and logical conclusion about whether video content is favored over images?
- Were all sources explicitly cited and credible?

### Step 2: Dimension Scoring

#### A. Accuracy of Conclusion (0.35)
Measures whether the agent's final conclusion is correct and supported by evidence.

5 — Conclusion is correct and fully supported by evidence from both official announcements and engagement metrics.
4 — Conclusion is mostly correct but lacks minor supporting evidence or details.
3 — Conclusion is partially correct but lacks significant supporting evidence.
2 — Conclusion is mostly incorrect or poorly supported.
1 — Conclusion is completely incorrect or unsupported.

#### B. Coverage of Sources (0.30)
Measures whether the agent used all required sources and platforms.

5 — Agent used Instagram's official announcements and analyzed engagement metrics for both videos and images.
4 — Agent used both sources but missed minor details or analyzed only one type of content.
3 — Agent used only one source or partially analyzed the required data.
2 — Agent attempted to use sources but failed to extract relevant information.
1 — Agent did not use any required sources or platforms.

#### C. Depth of Analysis (0.20)
Measures the level of detail and specificity in the agent's analysis.

5 — Provides detailed engagement metrics with specific examples for both videos and images.
4 — Provides engagement metrics but with less detail or fewer examples.
3 — Provides some metrics but lacks specificity or examples.
2 — Provides minimal or vague metrics with no examples.
1 — Provides no metrics or analysis.

#### D. Source Credibility and Output Structure (0.15)
Measures whether the sources are credible and the response is well-organized.

5 — All sources are credible, and the response is well-structured and easy to follow.
4 — Sources are mostly credible, and the response is mostly well-structured.
3 — Some sources are credible, but the response is poorly structured or unclear.
2 — Sources are mostly not credible, and the response is disorganized.
1 — Sources are not credible, and the response is incoherent.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "accuracy_of_conclusion": <1-5>,
  "coverage_of_sources": <1-5>,
  "depth_of_analysis": <1-5>,
  "source_credibility_and_output_structure": <1-5>,
  "dimension_reasoning": {{
    "accuracy_of_conclusion": "<one sentence citing specific evidence>",
    "coverage_of_sources": "<one sentence citing specific evidence>",
    "depth_of_analysis": "<one sentence citing specific evidence>",
    "source_credibility_and_output_structure": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "accuracy_of_conclusion": 0.35,
    "coverage_of_sources": 0.30,
    "depth_of_analysis": 0.20,
    "source_credibility_and_output_structure": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())