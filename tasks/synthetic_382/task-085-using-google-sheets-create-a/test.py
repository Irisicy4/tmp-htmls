"""
LLM-as-judge evaluator for EvolveBench task.

Category: Shopping
Task: Create a product comparison table for gaming monitors under $400 using data from Amazon, Newegg, and Walmart, and share a publicly accessible Google Sheets link.
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


TASK_INSTRUCTION = """Using Google Sheets, create a product comparison table for gaming monitors under $400. Include data on resolution, refresh rate, screen size, and customer reviews extracted from Amazon, Newegg, and Walmart. Share the link to the publicly accessible spreadsheet along with the data filled in."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to collect and compare data on gaming monitors under $400 from Amazon, Newegg, and Walmart, and organize this information into a Google Sheets table. The table must include resolution, refresh rate, screen size, and customer reviews for each product. The agent must provide a publicly accessible link to the completed spreadsheet.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Using Google Sheets, create a product comparison table for gaming monitors under $400. Include data on resolution, refresh rate, screen size, and customer reviews extracted from Amazon, Newegg, and Walmart. Share the link to the publicly accessible spreadsheet along with the data filled in.

## Task-Specific Constraints
- Must visit Amazon, Newegg, and Walmart to collect data.
- Must include resolution, refresh rate, screen size, and customer reviews for each product.
- Must ensure all products listed are under $400.
- Must organize the data into a structured table format.
- Must provide a publicly accessible Google Sheets link.
- Must include at least 5 products in the comparison.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Amazon, Newegg, and Walmart? Which platforms were actually visited?
- Are resolution, refresh rate, screen size, and customer reviews present for all listed products?
- Is the data organized into a structured table format in Google Sheets?
- Are all products listed under $400?
- Was a publicly accessible Google Sheets link provided?

### Step 2: Dimension Scoring

#### A. Data Accuracy and Completeness (0.35)
Measures whether the data in the table is accurate, complete, and includes all required fields.

5 — All required fields (resolution, refresh rate, screen size, customer reviews) are present and accurate for all products.
4 — Most required fields are present and accurate, with minor omissions or inaccuracies.
3 — Some required fields are present, but significant omissions or inaccuracies exist.
2 — Few required fields are present, with major omissions or inaccuracies.
1 — No required fields are present or the data is completely inaccurate.

#### B. Platform Coverage (0.30)
Measures whether the agent visited all required platforms and collected data from them.

5 — Data was collected from Amazon, Newegg, and Walmart, and all platforms were used effectively.
4 — Data was collected from at least two platforms, with minor omissions.
3 — Data was collected from only one platform, or significant omissions exist.
2 — Attempted to collect data but failed to use platforms effectively.
1 — No data was collected from any platform.

#### C. Depth and Specificity (0.20)
Measures whether the agent provided detailed and specific comparisons for each product.

5 — Includes detailed comparisons with specific numerical values for all required fields.
4 — Includes detailed comparisons for most products, with minor omissions.
3 — Includes basic comparisons, but lacks depth or specificity.
2 — Comparisons are vague or incomplete.
1 — No meaningful comparisons are provided.

#### D. Output Structure and Accessibility (0.15)
Measures whether the output is well-organized and the Google Sheets link is accessible.

5 — The table is well-organized, and the Google Sheets link is publicly accessible without issues.
4 — The table is organized, but the link has minor accessibility issues.
3 — The table is partially organized, and the link is accessible.
2 — The table is disorganized, or the link has major accessibility issues.
1 — No table or accessible link is provided.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "data_accuracy_and_completeness": <1-5>,
  "platform_coverage": <1-5>,
  "depth_and_specificity": <1-5>,
  "output_structure_and_accessibility": <1-5>,
  "dimension_reasoning": {{
    "data_accuracy_and_completeness": "<one sentence citing specific evidence>",
    "platform_coverage": "<one sentence citing specific evidence>",
    "depth_and_specificity": "<one sentence citing specific evidence>",
    "output_structure_and_accessibility": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "data_accuracy_and_completeness": 0.35,
    "platform_coverage": 0.30,
    "depth_and_specificity": 0.20,
    "output_structure_and_accessibility": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())