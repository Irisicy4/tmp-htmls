"""
LLM-as-judge evaluator for EvolveBench task.

Category: Design
Task: Locate and extract a tropical-themed color palette from three specified color palette websites.
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


TASK_INSTRUCTION = """Go to three public websites that offer color palette resources (Coolors, Color Hunt, and Adobe Color) and locate a palette suitable for a travel website focusing on tropical destinations. Extract the hex codes of the top 5 colors from your chosen palette and ensure they relate to a tropical theme (e.g., ocean blues, palm greens, sandy beiges)."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to visit three specific websites (Coolors, Color Hunt, and Adobe Color) and extract a tropical-themed color palette suitable for a travel website. The agent must provide the top 5 hex codes from the selected palette, ensuring the colors align with a tropical theme (e.g., ocean blues, palm greens, sandy beiges).

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Go to three public websites that offer color palette resources (Coolors, Color Hunt, and Adobe Color) and locate a palette suitable for a travel website focusing on tropical destinations. Extract the hex codes of the top 5 colors from your chosen palette and ensure they relate to a tropical theme (e.g., ocean blues, palm greens, sandy beiges).

## Task-Specific Constraints
- Must visit all three specified platforms: Coolors, Color Hunt, and Adobe Color.
- Must select a palette that aligns with a tropical theme (e.g., ocean blues, palm greens, sandy beiges).
- Must extract exactly 5 hex codes from the chosen palette.
- Must ensure the hex codes are presented in a structured format (e.g., a list or table).
- Must provide evidence of platform usage in the tool-call trace.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to all three required platforms (Coolors, Color Hunt, Adobe Color)?
- Did the agent select a palette that aligns with a tropical theme?
- Are exactly 5 hex codes extracted and presented in a structured format?
- Is there evidence in the tool-call trace that the platforms were actually used?
- Are the hex codes relevant to the tropical theme (e.g., ocean blues, palm greens, sandy beiges)?

### Step 2: Dimension Scoring

#### A. Palette Relevance (0.35)
Measures how well the selected palette aligns with the tropical theme.

5 — All 5 colors are highly relevant to a tropical theme.
4 — 4 colors are relevant, 1 is slightly off-theme.
3 — At least 3 colors are relevant to the tropical theme.
2 — Only 1-2 colors are relevant to the tropical theme.
1 — None of the colors are relevant to the tropical theme.

#### B. Platform Coverage (0.30)
Measures whether the agent visited all three specified platforms.

5 — All three platforms were visited, with evidence in the tool-call trace.
4 — Two platforms were visited, with evidence in the tool-call trace.
3 — At least one platform was visited, with evidence in the tool-call trace.
2 — No evidence of visiting the required platforms, but partial attempt made.
1 — No attempt to visit the required platforms.

#### C. Output Structure (0.20)
Measures the clarity and organization of the output.

5 — Hex codes are presented in a clear, structured format (e.g., list or table).
4 — Hex codes are mostly clear, with minor formatting issues.
3 — Hex codes are present but lack clear structure.
2 — Hex codes are incomplete or disorganized.
1 — Hex codes are missing or completely unstructured.

#### D. Evidence Credibility (0.15)
Measures the credibility and sufficiency of evidence in the tool-call trace.

5 — Strong evidence of platform usage and task completion in the trace.
4 — Evidence is mostly sufficient, with minor gaps.
3 — Evidence is present but incomplete or unclear.
2 — Minimal evidence of platform usage or task completion.
1 — No credible evidence of platform usage or task completion.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "palette_relevance": <1-5>,
  "platform_coverage": <1-5>,
  "output_structure": <1-5>,
  "evidence_credibility": <1-5>,
  "dimension_reasoning": {{
    "palette_relevance": "<one sentence citing specific evidence>",
    "platform_coverage": "<one sentence citing specific evidence>",
    "output_structure": "<one sentence citing specific evidence>",
    "evidence_credibility": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "palette_relevance": 0.35,
    "platform_coverage": 0.30,
    "output_structure": 0.20,
    "evidence_credibility": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())