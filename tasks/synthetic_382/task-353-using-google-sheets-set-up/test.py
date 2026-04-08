"""
LLM-as-judge evaluator for EvolveBench task.

Category: Design
Task: Set up a design token system in Google Sheets for a mobile app theme using Material Design resources.
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


TASK_INSTRUCTION = """Using Google Sheets, set up a design token system for a mobile app theme. Include columns for token name, token type (color, spacing, typography), and token value (e.g., hex codes for colors). Populate the sheet with at least ten sample tokens found from Material Design resources."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to create a design token system in Google Sheets for a mobile app theme. The system must include columns for token name, token type, and token value, and be populated with at least ten sample tokens sourced from Material Design resources. This task is in the domain of design and requires structured output in a spreadsheet format.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Using Google Sheets, set up a design token system for a mobile app theme. Include columns for token name, token type (color, spacing, typography), and token value (e.g., hex codes for colors). Populate the sheet with at least ten sample tokens found from Material Design resources.

## Task-Specific Constraints
- Must create a Google Sheet with the specified column structure.
- Must include at least ten design tokens sourced from Material Design resources.
- Tokens must cover at least three types: color, spacing, and typography.
- Token values must be accurate and match Material Design guidelines.
- The sheet must be shared or saved in a way that demonstrates completion.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent create a Google Sheet with the required column structure?
- Are there at least ten design tokens present in the sheet?
- Do the tokens cover at least three types: color, spacing, and typography?
- Are the token values accurate and sourced from Material Design resources?
- Is the sheet shared or saved in a way that demonstrates completion?

### Step 2: Dimension Scoring

#### A. Deliverable Accuracy (0.35)
Measures whether the design token system is correctly structured and populated.

5 — Contains all required columns and at least ten tokens with accurate values.
4 — Contains all required columns but fewer than ten tokens or minor inaccuracies.
3 — Contains partial columns and tokens but usable.
2 — Contains major structural issues or very few tokens.
1 — No usable output.

#### B. Coverage of Token Types (0.30)
Measures whether the tokens cover the required types: color, spacing, typography.

5 — Includes tokens for all three types with sufficient examples.
4 — Includes tokens for two types with sufficient examples.
3 — Includes tokens for one type or insufficient examples.
2 — Includes tokens but lacks type diversity.
1 — No tokens or completely wrong types.

#### C. Token Value Specificity (0.20)
Measures whether token values are detailed and sourced accurately.

5 — All token values are accurate and match Material Design resources.
4 — Most token values are accurate with minor discrepancies.
3 — Some token values are accurate but lacks detail.
2 — Few token values are accurate or detailed.
1 — No accurate token values.

#### D. Output Structure and Presentation (0.15)
Measures whether the sheet is well-organized and shared appropriately.

5 — Sheet is well-organized, clearly labeled, and shared appropriately.
4 — Sheet is mostly organized but lacks clarity or sharing proof.
3 — Sheet is partially organized but usable.
2 — Sheet is disorganized or incomplete.
1 — No usable sheet.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "The agent created a Google Sheet with the required columns and populated it with design tokens. Tokens covered three types (color, spacing, typography), and values matched Material Design resources.",
  "deliverable_accuracy": 5,
  "coverage_of_token_types": 5,
  "token_value_specificity": 5,
  "output_structure_and_presentation": 5,
  "dimension_reasoning": {{
    "deliverable_accuracy": "The sheet contained all required columns and ten tokens with accurate values.",
    "coverage_of_token_types": "Tokens covered all three types with sufficient examples.",
    "token_value_specificity": "All token values matched Material Design resources accurately.",
    "output_structure_and_presentation": "The sheet was well-organized and shared appropriately."
  }},
  "overall_score": 5.0,
  "passed": true
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "deliverable_accuracy": 0.35,
    "coverage_of_token_types": 0.30,
    "token_value_specificity": 0.20,
    "output_structure_and_presentation": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())