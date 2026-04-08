"""
LLM-as-judge evaluator for EvolveBench task.

Category: (Self) Media
Task: Verify whether TikTok's algorithm policy changes announced in 2023 have impacted small creator reach.
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


TASK_INSTRUCTION = """Verify whether TikTok's algorithm policy changes announced in 2023 have impacted small creator reach. Check official TikTok blog posts, and search for relevant news articles or user-submitted posts on Reddit communities like 'r/TikTokHelp' or 'r/SocialMediaNews'."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to investigate whether TikTok's algorithm policy changes in 2023 have impacted small creator reach. The agent must gather evidence from official TikTok blog posts, relevant news articles, and user-submitted posts on Reddit communities such as 'r/TikTokHelp' or 'r/SocialMediaNews'. A successful completion involves synthesizing findings from these sources into a structured and accurate response.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Verify whether TikTok's algorithm policy changes announced in 2023 have impacted small creator reach. Check official TikTok blog posts, and search for relevant news articles or user-submitted posts on Reddit communities like 'r/TikTokHelp' or 'r/SocialMediaNews'.

## Task-Specific Constraints
- Must visit tiktok.com, reddit.com, and at least one news platform (e.g., techcrunch.com).
- Must include evidence from at least two Reddit posts and one news article.
- Must reference at least one official TikTok blog post.
- Output must summarize findings in a structured format (e.g., bullet points or a table).
- Must address whether small creator reach has increased, decreased, or remained unchanged.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to tiktok.com, reddit.com, and at least one news platform? Which ones were actually visited?
- Are there references to at least two Reddit posts and one news article in the response?
- Does the response include evidence from an official TikTok blog post?
- Is the output organized in a structured format (e.g., bullet points or a table)?
- Does the response address whether small creator reach has increased, decreased, or remained unchanged?

### Step 2: Dimension Scoring

#### A. Primary Deliverable Accuracy (0.35)
Measures whether the agent accurately assessed the impact of TikTok's algorithm changes on small creator reach.

5 — Clearly identifies impact (increase/decrease/unchanged) with evidence from all required sources.
4 — Identifies impact but misses minor details or uses incomplete evidence.
3 — Partially identifies impact but lacks clarity or misses significant evidence.
2 — Provides unclear or incorrect assessment with minimal evidence.
1 — Fails to assess impact or provides no evidence.

#### B. Coverage of Required Sources (0.30)
Measures whether the agent included evidence from all required platforms and sources.

5 — Includes evidence from tiktok.com, reddit.com, and at least one news platform, with references to all required posts/articles.
4 — Includes evidence from most platforms and sources but misses minor requirements.
3 — Includes evidence from some platforms and sources but misses significant requirements.
2 — Includes minimal evidence from required platforms and sources.
1 — Fails to include evidence from required platforms and sources.

#### C. Depth and Specificity of Evidence (0.25)
Measures the level of detail and specificity in the agent's response.

5 — Provides detailed evidence with specific examples, numbers, or comparisons.
4 — Provides evidence with moderate detail but lacks some specificity.
3 — Provides basic evidence but lacks significant detail or specificity.
2 — Provides minimal evidence with little detail or specificity.
1 — Provides no meaningful evidence.

#### D. Source Quality and Output Structure (0.10)
Measures the credibility of sources used and the organization of the response.

5 — Uses credible sources and organizes the response in a clear, structured format (e.g., table or bullet points).
4 — Uses mostly credible sources and organizes the response moderately well.
3 — Uses some credible sources but response lacks clear organization.
2 — Uses few credible sources and response is poorly organized.
1 — Uses no credible sources and response is unstructured.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "primary_deliverable_accuracy": <1-5>,
  "coverage_of_required_sources": <1-5>,
  "depth_and_specificity_of_evidence": <1-5>,
  "source_quality_and_output_structure": <1-5>,
  "dimension_reasoning": {{
    "primary_deliverable_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_required_sources": "<one sentence citing specific evidence>",
    "depth_and_specificity_of_evidence": "<one sentence citing specific evidence>",
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
    "depth_and_specificity_of_evidence": 0.25,
    "source_quality_and_output_structure": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())