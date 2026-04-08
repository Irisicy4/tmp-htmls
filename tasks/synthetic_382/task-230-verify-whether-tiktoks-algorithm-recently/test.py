"""
LLM-as-judge evaluator for EvolveBench task.

Category: (Self) Media
Task: Verify whether TikTok's algorithm recently deprioritized content with external links.
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


TASK_INSTRUCTION = """Verify whether TikTok's algorithm recently deprioritized content with external links (e.g., links to YouTube or external e-commerce sites). Check TikTok's official updates, recent blog posts, and public creator discussions on Reddit."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to verify whether TikTok's algorithm recently deprioritized content with external links. This involves checking TikTok's official updates, recent blog posts, and public creator discussions on Reddit. A successful completion requires accurate identification of any algorithm changes, supported by credible sources from the specified platforms.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Verify whether TikTok's algorithm recently deprioritized content with external links (e.g., links to YouTube or external e-commerce sites). Check TikTok's official updates, recent blog posts, and public creator discussions on Reddit.

## Task-Specific Constraints
- Must visit TikTok's official blog or update page.
- Must review at least one relevant Reddit discussion thread.
- Must provide evidence from at least one recent blog post or article.
- Output must clearly state whether the algorithm deprioritization is confirmed, unconfirmed, or inconclusive.
- Must cite sources explicitly with URLs or platform names.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to TikTok's official blog or update page?
- Did the agent review at least one relevant Reddit discussion thread?
- Did the agent provide evidence from a recent blog post or article?
- Does the output clearly state whether the algorithm deprioritization is confirmed, unconfirmed, or inconclusive?
- Are all cited sources explicitly mentioned with URLs or platform names?

### Step 2: Dimension Scoring

#### A. Primary Deliverable Accuracy (0.35)
Measures whether the agent accurately identified and explained TikTok's algorithm behavior.

5 — Clearly confirms, denies, or states inconclusive results based on credible evidence.
4 — Mostly accurate but lacks clarity or misses minor details.
3 — Partially accurate but incomplete or unclear.
2 — Poor accuracy with significant errors or missing key conclusions.
1 — Completely inaccurate or absent.

#### B. Coverage of Required Sources (0.30)
Measures whether the agent used all specified platforms and sources.

5 — Uses TikTok's official blog, Reddit, and at least one external blog/article.
4 — Uses two of the required platforms and sources.
3 — Uses only one required platform/source.
2 — Attempts but fails to use any required platform/source.
1 — No attempt to use required platforms or sources.

#### C. Depth and Specificity (0.20)
Measures the level of detail and specificity in the agent's response.

5 — Provides detailed evidence, including URLs, quotes, or specific claims.
4 — Provides evidence but lacks some specificity or minor details.
3 — Provides general evidence but lacks depth or specificity.
2 — Provides vague or minimal evidence.
1 — No evidence provided.

#### D. Source Credibility and Output Structure (0.15)
Measures the credibility of sources and organization of the response.

5 — Sources are credible, and response is well-organized and easy to follow.
4 — Sources are mostly credible, and response is organized but slightly unclear.
3 — Sources are partially credible, and response is somewhat disorganized.
2 — Sources are questionable, and response is poorly organized.
1 — Sources are not credible, and response is completely disorganized.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "primary_deliverable_accuracy": <1-5>,
  "coverage_of_required_sources": <1-5>,
  "depth_and_specificity": <1-5>,
  "source_credibility_and_output_structure": <1-5>,
  "dimension_reasoning": {{
    "primary_deliverable_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_required_sources": "<one sentence citing specific evidence>",
    "depth_and_specificity": "<one sentence citing specific evidence>",
    "source_credibility_and_output_structure": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "primary_deliverable_accuracy": 0.35,
    "coverage_of_required_sources": 0.30,
    "depth_and_specificity": 0.20,
    "source_credibility_and_output_structure": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())