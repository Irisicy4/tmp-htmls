"""
LLM-as-judge evaluator for EvolveBench task.

Category: Design
Task: Find a minimalistic business card design resource that is free to download and compatible with Adobe Illustrator.
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


TASK_INSTRUCTION = """Find a design resource for creating modern business cards that matches the following criteria: minimalistic style, free download, and compatible with Adobe Illustrator. Navigate popular resource libraries to extract the link to the best-matching template."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to find a design resource for modern business cards that meets specific criteria: minimalistic style, free download, and compatibility with Adobe Illustrator. The agent must navigate popular design resource platforms and provide a link to the best-matching template.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Find a design resource for creating modern business cards that matches the following criteria: minimalistic style, free download, and compatible with Adobe Illustrator. Navigate popular resource libraries to extract the link to the best-matching template.

## Task-Specific Constraints
- Must visit at least 3 of the specified platforms (freepik.com, behance.net, dribbble.com).
- Must verify that the resource is free to download.
- Must confirm compatibility with Adobe Illustrator.
- Must provide a direct link to the resource.
- Must describe why the selected resource matches the criteria.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to at least 3 of the required platforms? Which ones were visited?
- Did the agent verify that the resource is free to download?
- Did the agent confirm compatibility with Adobe Illustrator?
- Is the provided link functional and does it lead to the correct resource?
- Does the response explain why the selected resource matches the criteria?

### Step 2: Dimension Scoring

#### A. Deliverable Accuracy (0.35)
Measures whether the agent provided a correct and complete resource link that matches the criteria.

5 — Provides a valid link to a resource that is minimalistic, free, and compatible with Adobe Illustrator.
4 — Provides a valid link but misses one minor criterion (e.g., lacks confirmation of compatibility).
3 — Provides a link but misses multiple criteria or lacks verification.
2 — Provides a link but it is incorrect or fails to meet most criteria.
1 — No valid link provided.

#### B. Platform Coverage (0.30)
Measures whether the agent visited all required platforms and utilized them effectively.

5 — Visits all 3 specified platforms and uses them to compare resources.
4 — Visits at least 2 platforms and uses them effectively.
3 — Visits at least 1 platform but with limited comparison.
2 — Visits platforms but does not use them effectively.
1 — Does not visit any specified platforms.

#### C. Detail and Specificity (0.20)
Measures whether the agent provides detailed explanations and verifications for the selected resource.

5 — Provides detailed reasoning and verifies all criteria (style, free, compatibility).
4 — Provides reasoning but misses minor details or verifications.
3 — Provides basic reasoning but lacks depth or misses key verifications.
2 — Provides reasoning but it is vague or incomplete.
1 — No reasoning provided.

#### D. Output Organization and Credibility (0.15)
Measures whether the response is well-structured and uses credible sources.

5 — Response is well-organized, clear, and uses credible sources.
4 — Response is clear but has minor organizational issues.
3 — Response is usable but lacks clarity or structure.
2 — Response is poorly organized or unclear.
1 — Response is disorganized and lacks credibility.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "deliverable_accuracy": <1-5>,
  "platform_coverage": <1-5>,
  "detail_and_specificity": <1-5>,
  "output_organization_and_credibility": <1-5>,
  "dimension_reasoning": {{
    "deliverable_accuracy": "<one sentence citing specific evidence>",
    "platform_coverage": "<one sentence citing specific evidence>",
    "detail_and_specificity": "<one sentence citing specific evidence>",
    "output_organization_and_credibility": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "deliverable_accuracy": 0.35,
    "platform_coverage": 0.30,
    "detail_and_specificity": 0.20,
    "output_organization_and_credibility": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())