"""
LLM-as-judge evaluator for EvolveBench task.

Category: (Self) Media
Task: Extract engagement rate, audience demographics, and campaign collaborations for TikTok creators in the 'Food & Cooking' category with over 1 million followers.
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


TASK_INSTRUCTION = """Go to TikTok's Creator Marketplace, search for creators with over 1 million followers in the 'Food & Cooking' category, and extract their engagement rate, audience demographics, and recent campaign collaborations."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

This task asks the agent to identify TikTok creators in the 'Food & Cooking' category with over 1 million followers, extract their engagement rate, audience demographics, and recent campaign collaborations, and present the findings in a structured format. The domain involves social media analytics and creator marketing.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Go to TikTok's Creator Marketplace, search for creators with over 1 million followers in the 'Food & Cooking' category, and extract their engagement rate, audience demographics, and recent campaign collaborations.

## Task-Specific Constraints
- Must visit TikTok Creator Marketplace, TikTok.com, and SocialBlade.com.
- Must extract engagement rate, audience demographics, and campaign collaborations for at least 3 creators.
- Output must be organized as a structured table or JSON list.
- Must verify follower count exceeds 1 million for all creators included.
- Must include specific demographic details such as age and location.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to TikTok Creator Marketplace, TikTok.com, and SocialBlade.com? Which platforms were actually visited?
- Are engagement rate, audience demographics, and campaign collaborations present for at least 3 creators?
- Is the output organized as a structured table or JSON list?
- Are follower counts verified to exceed 1 million for all creators included?
- Are specific demographic details such as age and location present?

### Step 2: Dimension Scoring

#### A. Deliverable Accuracy (0.35)
Measures whether the extracted data (engagement rate, audience demographics, campaign collaborations) is correct and complete.

5 — Data for all 3+ creators is accurate and complete, including engagement rate, demographics, and collaborations.
4 — Data for 3+ creators is mostly accurate but has minor omissions or errors.
3 — Data for at least 2 creators is partially complete but lacks key elements.
2 — Data for fewer than 2 creators or mostly incorrect.
1 — No usable data extracted.

#### B. Coverage of Platforms and Sources (0.30)
Measures whether the agent visited all required platforms and used them appropriately.

5 — All 3 required platforms were visited and used correctly.
4 — At least 2 platforms were visited and used correctly.
3 — At least 1 platform was visited and partially used.
2 — Platforms visited but not used correctly.
1 — No required platforms visited.

#### C. Specificity of Extracted Data (0.20)
Measures the level of detail in the extracted data, such as demographic breakdowns and collaboration specifics.

5 — Includes detailed demographic breakdowns (age, location) and collaboration specifics for all creators.
4 — Includes demographic breakdowns and collaboration specifics for most creators.
3 — Includes partial demographic or collaboration details for at least 2 creators.
2 — Minimal demographic or collaboration details provided.
1 — No specific details provided.

#### D. Output Structure and Credibility (0.15)
Measures whether the output is well-organized and sourced from credible platforms.

5 — Output is structured as a clear table or JSON list, with credible sourcing.
4 — Output is mostly structured and credible but has minor formatting issues.
3 — Output is partially structured and credible but lacks clarity.
2 — Output is poorly structured or lacks credibility.
1 — Output is unstructured and lacks credibility.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "deliverable_accuracy": <1-5>,
  "coverage_of_platforms_and_sources": <1-5>,
  "specificity_of_extracted_data": <1-5>,
  "output_structure_and_credibility": <1-5>,
  "dimension_reasoning": {{
    "deliverable_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_platforms_and_sources": "<one sentence citing specific evidence>",
    "specificity_of_extracted_data": "<one sentence citing specific evidence>",
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
    "specificity_of_extracted_data": 0.20,
    "output_structure_and_credibility": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())