"""
LLM-as-judge evaluator for EvolveBench task.

Category: Design
Task: Select and preview a font pairing for a modern design project using Adobe Fonts.
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
    aggregated["evidence_summary"] = valid[0].get("evidence_summary", "")
    aggregated["dimension_reasoning"] = valid[0].get("dimension_reasoning", {})
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


TASK_INSTRUCTION = """Use the Adobe Fonts website to set up a font pairing for a modern design project. Select one sans-serif font for headings and one serif font for body text, then preview the pairing on a sample paragraph using the interactive tool. Provide the font names and weights used in the final pairing."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to use Adobe Fonts to select and preview a font pairing for a modern design project. The agent must choose one sans-serif font for headings and one serif font for body text, preview the pairing using the interactive tool on the website, and provide the font names and weights used in the final pairing.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Use the Adobe Fonts website to set up a font pairing for a modern design project. Select one sans-serif font for headings and one serif font for body text, then preview the pairing on a sample paragraph using the interactive tool. Provide the font names and weights used in the final pairing.

## Task-Specific Constraints
- Must use the Adobe Fonts website to select the fonts.
- Must choose one sans-serif font for headings and one serif font for body text.
- Must preview the font pairing using the interactive tool on Adobe Fonts.
- Must provide the font names and weights in the response.
- The response must clearly specify which font is for headings and which is for body text.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to the Adobe Fonts website and use the interactive tool?
- Did the agent select one sans-serif font for headings and one serif font for body text?
- Are the font names and weights clearly specified in the response?
- Is it clear which font is for headings and which is for body text?
- Does the response align with the task instruction and constraints?

### Step 2: Dimension Scoring

#### A. Font Pairing Accuracy (0.35)
Measures whether the agent correctly selected and paired one sans-serif font for headings and one serif font for body text.

5 — Both fonts are correctly paired, and their roles (headings/body) are clearly specified.
4 — Both fonts are correctly paired, but roles are unclear or partially specified.
3 — Fonts are partially correct (e.g., wrong typeface for one role) but usable.
2 — Fonts are mostly incorrect or missing key details.
1 — No correct font pairing provided.

#### B. Platform Usage (0.30)
Measures whether the agent correctly used the Adobe Fonts website and its interactive tool.

5 — Agent fully used the Adobe Fonts website, including the interactive tool, to preview the pairing.
4 — Agent used the website but did not fully utilize the interactive tool.
3 — Agent partially used the website but missed key steps.
2 — Agent barely used the website or skipped critical steps.
1 — Agent did not use the website at all.

#### C. Detail Specificity (0.20)
Measures whether the agent provided specific font names and weights in the response.

5 — Font names and weights are fully specified and accurate.
4 — Font names are specified, but weights are partially missing or unclear.
3 — Font names are present, but weights are missing or incorrect.
2 — Font names are vague or incorrect, and weights are missing.
1 — No font names or weights are provided.

#### D. Output Clarity (0.15)
Measures whether the agent's response is well-organized and easy to understand.

5 — Response is clear, well-structured, and easy to follow.
4 — Response is mostly clear but could be better organized.
3 — Response is somewhat clear but contains ambiguities.
2 — Response is poorly organized and hard to follow.
1 — Response is completely unclear or incoherent.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "font_pairing_accuracy": <1-5>,
  "platform_usage": <1-5>,
  "detail_specificity": <1-5>,
  "output_clarity": <1-5>,
  "dimension_reasoning": {{
    "font_pairing_accuracy": "<one sentence citing specific evidence>",
    "platform_usage": "<one sentence citing specific evidence>",
    "detail_specificity": "<one sentence citing specific evidence>",
    "output_clarity": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "font_pairing_accuracy": 0.35,
    "platform_usage": 0.30,
    "detail_specificity": 0.20,
    "output_clarity": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())