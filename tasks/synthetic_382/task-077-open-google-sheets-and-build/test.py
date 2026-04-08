"""
LLM-as-judge evaluator for EvolveBench task.

Category: Shopping
Task: Build a shopping list for home office essentials, including average prices and hyperlinks from Amazon, Staples, and Office Depot.
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


TASK_INSTRUCTION = """Open Google Sheets and build a shopping list for setting up a home office with essentials such as a desk, ergonomic chair, monitor, and desk lamp. Find average prices for each item from Amazon, Staples, and Office Depot, and include hyperlinks to specific product pages."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to create a shopping list for home office essentials, including items like a desk, ergonomic chair, monitor, and desk lamp. The agent must find average prices for each item from Amazon, Staples, and Office Depot, and provide hyperlinks to specific product pages. A successful completion includes a structured shopping list with accurate price data and working links.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Open Google Sheets and build a shopping list for setting up a home office with essentials such as a desk, ergonomic chair, monitor, and desk lamp. Find average prices for each item from Amazon, Staples, and Office Depot, and include hyperlinks to specific product pages.

## Task-Specific Constraints
- Must visit Amazon, Staples, and Office Depot to gather price data.
- Must include average prices for all four items: desk, ergonomic chair, monitor, and desk lamp.
- Must provide hyperlinks to specific product pages for each item.
- Output must be organized as a structured table or list.
- Must calculate average prices correctly based on gathered data.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Amazon, Staples, and Office Depot? Which platforms were actually visited?
- Are all four required items (desk, ergonomic chair, monitor, desk lamp) present in the response?
- Is the output organized as a structured table or list?
- Are average prices calculated correctly based on gathered data?
- Are hyperlinks to product pages present and functional?

### Step 2: Dimension Scoring

#### A. Deliverable Accuracy (0.35)
Measures whether the shopping list is correct, complete, and includes all required elements.

5 — All items are present with accurate average prices and working hyperlinks.
4 — All items are present, but minor errors in prices or links.
3 — Most items are present, but some are missing or incorrect.
2 — Few items are present, with significant errors in prices or links.
1 — No usable shopping list provided.

#### B. Platform Coverage (0.30)
Measures whether the agent visited all required platforms and gathered data from them.

5 — Data gathered from Amazon, Staples, and Office Depot for all items.
4 — Data gathered from two platforms, with minor omissions.
3 — Data gathered from one platform, or incomplete data from multiple platforms.
2 — Minimal data gathered from platforms.
1 — No platform data gathered.

#### C. Detail and Specificity (0.20)
Measures the depth of information, including accurate averages and detailed comparisons.

5 — Accurate averages calculated with detailed comparisons between platforms.
4 — Accurate averages calculated, but comparisons lack depth.
3 — Averages calculated, but with errors or insufficient detail.
2 — Minimal detail or incorrect averages.
1 — No averages or comparisons provided.

#### D. Output Structure and Credibility (0.15)
Measures whether the output is well-organized and sources are credible.

5 — Output is a well-structured table or list with credible sources.
4 — Output is structured but lacks polish or minor credibility issues.
3 — Output is usable but poorly organized or unclear.
2 — Output is disorganized or lacks credibility.
1 — No usable output provided.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "deliverable_accuracy": <1-5>,
  "platform_coverage": <1-5>,
  "detail_and_specificity": <1-5>,
  "output_structure_and_credibility": <1-5>,
  "dimension_reasoning": {{
    "deliverable_accuracy": "<one sentence citing specific evidence>",
    "platform_coverage": "<one sentence citing specific evidence>",
    "detail_and_specificity": "<one sentence citing specific evidence>",
    "output_structure_and_credibility": "<one sentence citing specific evidence>"
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
    "output_structure_and_credibility": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())