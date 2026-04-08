"""
LLM-as-judge evaluator for EvolveBench task.

Category: (Self) Media
Task: Configure a content calendar for 8 weeks of Instagram posts using Google Sheets, including example data sourced from trending fitness hashtags.
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


TASK_INSTRUCTION = """Open Google Sheets and configure a content calendar for the next 8 weeks of Instagram posts. Create columns for post date, caption, hashtags, image source, and engagement goal. Populate the first two rows with example data sourced from trending Instagram hashtags for 'Fitness'."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves creating a content calendar for Instagram posts using Google Sheets. The agent must set up columns for post date, caption, hashtags, image source, and engagement goal, and populate the first two rows with example data sourced from trending fitness hashtags. Success requires the agent to use relevant platforms (Google Sheets, Instagram, best-hashtags.com) and produce a structured, accurate table.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Open Google Sheets and configure a content calendar for the next 8 weeks of Instagram posts. Create columns for post date, caption, hashtags, image source, and engagement goal. Populate the first two rows with example data sourced from trending Instagram hashtags for 'Fitness'.

## Task-Specific Constraints
- Must create a table with the specified columns: post date, caption, hashtags, image source, engagement goal.
- Must populate the first two rows with example data sourced from trending fitness hashtags.
- Must use Google Sheets to create the content calendar.
- Must visit Instagram or best-hashtags.com to source trending fitness hashtags.
- Output must be structured and organized as a table.
- Example data must be relevant to the fitness domain.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Google Sheets and create a table with the required columns?
- Did the agent visit Instagram or best-hashtags.com to source trending fitness hashtags?
- Are the first two rows of the table populated with relevant example data?
- Is the output structured and organized as a table?
- Are the example hashtags relevant to the fitness domain?

### Step 2: Dimension Scoring

#### A. Content Calendar Accuracy (0.35)
Measures whether the content calendar is correctly configured and includes all required columns.

5 — All columns are present and correctly labeled; table is fully configured.
4 — One column is missing or mislabeled; table is mostly configured.
3 — Two columns are missing or mislabeled; table is partially configured.
2 — Most columns are missing or mislabeled; table is poorly configured.
1 — No table or columns are present.

#### B. Example Data Completeness (0.30)
Measures whether the first two rows are populated with relevant example data sourced from trending fitness hashtags.

5 — Both rows are fully populated with accurate, relevant data.
4 — One row is fully populated; the other is incomplete or slightly inaccurate.
3 — Both rows are partially populated; data is minimally usable.
2 — One row is partially populated; the other is missing.
1 — No example data is present.

#### C. Platform Usage Coverage (0.20)
Measures whether the agent visited the required platforms (Google Sheets, Instagram, best-hashtags.com).

5 — All required platforms were visited and used appropriately.
4 — Two required platforms were visited and used appropriately.
3 — One required platform was visited and used appropriately.
2 — Platforms were visited but not used appropriately.
1 — No required platforms were visited.

#### D. Output Structure and Organization (0.15)
Measures whether the output is structured and organized as a table.

5 — Output is fully structured and organized as a table.
4 — Output is mostly structured, with minor formatting issues.
3 — Output is minimally structured but usable.
2 — Output is poorly structured and barely usable.
1 — Output is unstructured or missing.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "content_calendar_accuracy": <1-5>,
  "example_data_completeness": <1-5>,
  "platform_usage_coverage": <1-5>,
  "output_structure_and_organization": <1-5>,
  "dimension_reasoning": {{
    "content_calendar_accuracy": "<one sentence citing specific evidence>",
    "example_data_completeness": "<one sentence citing specific evidence>",
    "platform_usage_coverage": "<one sentence citing specific evidence>",
    "output_structure_and_organization": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "content_calendar_accuracy": 0.35,
    "example_data_completeness": 0.30,
    "platform_usage_coverage": 0.20,
    "output_structure_and_organization": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())