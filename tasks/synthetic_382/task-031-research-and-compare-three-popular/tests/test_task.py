"""
LLM-as-judge evaluator for EvolveBench task.

Category: Marketing & Analytics
Task: Research and compare Google Ads, Facebook Ads, and TikTok Ads based on CPM, audience targeting, and ease of use for small businesses.
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


TASK_INSTRUCTION = """Research and compare three popular ad platforms — Google Ads, Facebook Ads, and TikTok Ads — based on their average CPM (cost per thousand impressions), audience targeting capabilities, and ease of use for small businesses. Extract this information from their respective official documentation pages and recent third-party marketing blogs."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to research and compare Google Ads, Facebook Ads, and TikTok Ads based on CPM, audience targeting capabilities, and ease of use for small businesses. The domain is marketing and analytics. A successful completion requires extracting accurate data from official documentation and credible third-party sources, and presenting the findings in a structured format.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Research and compare three popular ad platforms — Google Ads, Facebook Ads, and TikTok Ads — based on their average CPM (cost per thousand impressions), audience targeting capabilities, and ease of use for small businesses. Extract this information from their respective official documentation pages and recent third-party marketing blogs.

## Task-Specific Constraints
- Must visit the official documentation pages of Google Ads, Facebook Ads, and TikTok Ads.
- Must include CPM data for all three platforms.
- Must compare audience targeting capabilities for each platform.
- Must evaluate ease of use specifically for small businesses.
- Output must be organized as a structured table or list.
- Must cite sources for all claims made.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to the official documentation pages of Google Ads, Facebook Ads, and TikTok Ads?
- Are CPM values for all three platforms present in the response?
- Are audience targeting capabilities compared for all three platforms?
- Is ease of use for small businesses evaluated for all three platforms?
- Is the output organized as a structured table or list?
- Are all claims sourced from credible documentation or blogs?

### Step 2: Dimension Scoring

#### A. Deliverable Accuracy (0.35)
Measures whether the main deliverable (comparison of CPM, audience targeting, and ease of use) is correct and complete.

5 — All three criteria (CPM, audience targeting, ease of use) are fully addressed for all platforms with accurate data.
4 — Two criteria are fully addressed, and the third is partially addressed.
3 — At least one criterion is fully addressed, and others are partially addressed.
2 — Only one criterion is partially addressed.
1 — None of the criteria are addressed.

#### B. Coverage of Platforms and Sources (0.30)
Measures whether the agent included all required platforms and used credible sources.

5 — All three platforms are covered, and all claims are sourced from credible documentation or blogs.
4 — All three platforms are covered, but one or more claims lack credible sourcing.
3 — At least two platforms are covered, with credible sourcing for claims.
2 — Only one platform is covered, or sourcing is mostly missing.
1 — No platforms are covered, or sourcing is absent.

#### C. Depth of Analysis (0.25)
Measures the specificity and detail of comparisons.

5 — Includes specific CPM values, detailed audience targeting features, and ease-of-use evaluations for all platforms.
4 — Includes specific CPM values and detailed comparisons for at least two platforms.
3 — Includes CPM values and general comparisons for at least two platforms.
2 — Includes vague or incomplete comparisons for one platform.
1 — No specific comparisons are made.

#### D. Output Structure and Credibility (0.10)
Measures whether the output is well-organized and sources are credible.

5 — Output is organized as a structured table or list, with all claims sourced.
4 — Output is structured, but some claims lack sourcing.
3 — Output is partially structured, with minimal sourcing.
2 — Output is poorly structured, with few credible sources.
1 — Output is unstructured and unsourced.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "deliverable_accuracy": <1-5>,
  "coverage_of_platforms_and_sources": <1-5>,
  "depth_of_analysis": <1-5>,
  "output_structure_and_credibility": <1-5>,
  "dimension_reasoning": {{
    "deliverable_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_platforms_and_sources": "<one sentence citing specific evidence>",
    "depth_of_analysis": "<one sentence citing specific evidence>",
    "output_structure_and_credibility": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "deliverable_accuracy": 0.35,
    "coverage_of_platforms_and_sources": 0.30,
    "depth_of_analysis": 0.25,
    "output_structure_and_credibility": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())