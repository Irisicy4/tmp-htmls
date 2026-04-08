"""
LLM-as-judge evaluator for EvolveBench task.

Category: Shopping
Task: Create a Google Sheets tracker comparing subscription box services under $50/month.
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


TASK_INSTRUCTION = """Go to Google Sheets and create a tracker to compare monthly subscription box services. Research on Cratejoy.com, Birchbox.com, and FabFitFun.com to find three subscription boxes under $50/month. Populate the tracker with details about price, included items, user ratings, and shipping options."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves creating a Google Sheets tracker comparing subscription box services under $50/month. The agent must research Cratejoy.com, Birchbox.com, and FabFitFun.com to gather details about price, included items, user ratings, and shipping options. A successful completion requires accurate and structured data presented in a table format.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Go to Google Sheets and create a tracker to compare monthly subscription box services. Research on Cratejoy.com, Birchbox.com, and FabFitFun.com to find three subscription boxes under $50/month. Populate the tracker with details about price, included items, user ratings, and shipping options.

## Task-Specific Constraints
- Must visit Cratejoy.com, Birchbox.com, and FabFitFun.com.
- Must include price data for all three subscription boxes.
- Must include details about included items, user ratings, and shipping options.
- Output must be organized as a table in Google Sheets.
- All subscription boxes must cost under $50/month.
- Data must be accurate and sourced from the specified platforms.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Cratejoy.com, Birchbox.com, and FabFitFun.com? Which ones were actually visited?
- Are price, included items, user ratings, and shipping options present for all three boxes?
- Is the output organized as a table in Google Sheets?
- Are all subscription boxes under $50/month, as required?
- Is the data accurate and sourced from the specified platforms?

### Step 2: Dimension Scoring

#### A. Deliverable Accuracy (0.35)
Measures whether the tracker contains accurate and complete data for all required fields.

5 — All required fields (price, included items, user ratings, shipping options) are accurate and complete for all three boxes.
4 — Most fields are accurate and complete, with minor omissions or inaccuracies.
3 — Some fields are accurate and complete, but others are missing or incorrect.
2 — Few fields are accurate or complete; major omissions or inaccuracies.
1 — No accurate or complete fields; entirely incorrect or missing.

#### B. Platform Coverage (0.30)
Measures whether the agent visited all required platforms and sourced data from them.

5 — Data is sourced from all three platforms (Cratejoy.com, Birchbox.com, FabFitFun.com).
4 — Data is sourced from two platforms, with minor omissions.
3 — Data is sourced from one platform or partially from two platforms.
2 — Minimal platform coverage; data is mostly missing or incomplete.
1 — No platform coverage; data is entirely missing.

#### C. Detail Specificity (0.20)
Measures the depth and specificity of the information provided.

5 — Includes highly detailed descriptions (e.g., specific items, numerical ratings, shipping terms).
4 — Includes moderately detailed descriptions, with minor omissions.
3 — Includes basic descriptions, but lacks depth or specificity.
2 — Includes minimal descriptions; vague or incomplete.
1 — No descriptions provided; entirely missing.

#### D. Output Organization (0.15)
Measures the structure and presentation of the tracker.

5 — Tracker is well-organized as a table in Google Sheets, with clear labels and formatting.
4 — Tracker is organized as a table, but formatting is inconsistent or unclear.
3 — Tracker is minimally organized, with basic table structure but lacks clarity.
2 — Tracker is poorly organized; difficult to interpret.
1 — No tracker provided; entirely missing.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "deliverable_accuracy": <1-5>,
  "platform_coverage": <1-5>,
  "detail_specificity": <1-5>,
  "output_organization": <1-5>,
  "dimension_reasoning": {{
    "deliverable_accuracy": "<one sentence citing specific evidence>",
    "platform_coverage": "<one sentence citing specific evidence>",
    "detail_specificity": "<one sentence citing specific evidence>",
    "output_organization": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "deliverable_accuracy": 0.35,
    "platform_coverage": 0.30,
    "detail_specificity": 0.20,
    "output_organization": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())