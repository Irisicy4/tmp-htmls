"""
LLM-as-judge evaluator for EvolveBench task.

Category: Shopping
Task: Create a grocery shopping list template in Google Sheets and populate it with data from Target's website.
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


TASK_INSTRUCTION = """Create a grocery shopping list template in Google Sheets that includes columns for item name, category, store name, price, and quantity. Use Target's website to look up three common grocery items with their prices and add them to the sheet to demonstrate usability."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves creating a grocery shopping list template in Google Sheets with specific columns for item name, category, store name, price, and quantity. The agent must use Target's website to find prices for three common grocery items and add them to the sheet to demonstrate usability.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Create a grocery shopping list template in Google Sheets that includes columns for item name, category, store name, price, and quantity. Use Target's website to look up three common grocery items with their prices and add them to the sheet to demonstrate usability.

## Task-Specific Constraints
- Must create a Google Sheets document with the specified columns.
- Must visit Target's website to look up prices for three grocery items.
- Must include item name, category, store name, price, and quantity for each item in the sheet.
- Must demonstrate usability by populating the sheet with accurate data from Target's website.
- Output must be organized as a structured table in Google Sheets.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Google Sheets and create a document with the required columns?
- Did the agent visit Target's website and look up prices for three grocery items?
- Are the item name, category, store name, price, and quantity fields populated correctly in the sheet?
- Is the sheet organized as a structured table?
- Are the prices and item details sourced accurately from Target's website?

### Step 2: Dimension Scoring

#### A. Template Creation Accuracy (0.35)
Measures whether the Google Sheets template includes all required columns.

5 — All required columns (item name, category, store name, price, quantity) are present and correctly labeled.
4 — One column is missing or mislabeled.
3 — Two columns are missing or mislabeled.
2 — More than two columns are missing or mislabeled.
1 — No template created or completely incorrect.

#### B. Data Retrieval Completeness (0.30)
Measures whether the agent retrieved price data for three grocery items from Target's website.

5 — Prices for three grocery items are retrieved and accurately recorded.
4 — Prices for two grocery items are retrieved and accurately recorded.
3 — Price for one grocery item is retrieved and accurately recorded.
2 — No price data retrieved, or all data is inaccurate.
1 — No attempt to retrieve price data.

#### C. Usability Demonstration (0.25)
Measures whether the sheet is populated with data demonstrating usability.

5 — All fields (item name, category, store name, price, quantity) are populated for three items.
4 — All fields are populated for two items.
3 — All fields are populated for one item.
2 — Some fields are populated, but the sheet is incomplete.
1 — No fields are populated.

#### D. Output Structure and Organization (0.10)
Measures whether the sheet is well-organized and structured as a table.

5 — Sheet is well-organized, structured as a table, and visually clear.
4 — Sheet is mostly organized but has minor formatting issues.
3 — Sheet is somewhat organized but lacks clarity or structure.
2 — Sheet is poorly organized and difficult to interpret.
1 — Sheet is completely disorganized or absent.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "template_creation_accuracy": <1-5>,
  "data_retrieval_completeness": <1-5>,
  "usability_demonstration": <1-5>,
  "output_structure_and_organization": <1-5>,
  "dimension_reasoning": {{
    "template_creation_accuracy": "<one sentence citing specific evidence>",
    "data_retrieval_completeness": "<one sentence citing specific evidence>",
    "usability_demonstration": "<one sentence citing specific evidence>",
    "output_structure_and_organization": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "template_creation_accuracy": 0.35,
    "data_retrieval_completeness": 0.30,
    "usability_demonstration": 0.25,
    "output_structure_and_organization": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())