"""
LLM-as-judge evaluator for EvolveBench task-82.

Category: (Self) Media
Task: Change this image to have a deeper blue sky background. Use Photopea at https://www.photopea.com to open an image from W
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
    m = re.search(r"<Answer>(.*?)</Answer>", text, re.DOTALL | re.IGNORECASE)
    if m:
        try: return json.loads(m.group(1).strip())
        except Exception: pass
    try: return json.loads(text.strip())
    except Exception: pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try: return json.loads(m.group())
        except Exception: pass
    return None

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


TASK_INSTRUCTION = """Change this image to have a deeper blue sky background. Use Photopea at https://www.photopea.com to open an image from Wikipedia (e.g. a landscape photo) and deepen the sky to a richer blue."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves using Photopea (browser-based image editor) to open an image and modify the sky color to be a deeper, richer blue.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Tool: Photopea at photopea.com
- Action: deepen/intensify the blue sky — not replace it entirely
- Method: selection tool + hue/saturation or color balance adjustment
- Output: edited image saved as file

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent open Photopea?
- Was an image loaded?
- What method was used to deepen the sky blue (selection, adjustment layers)?
- Was the edit executed and result saved?
- Does the output have a visibly deeper blue sky?

### Step 2: Dimension Scoring

#### A. Tool Usage (0.25)
Did the agent use Photopea correctly?

5 — Agent opened Photopea, loaded an image, and used editing tools.
4 — Photopea opened but with some setup difficulty.
3 — Agent described Photopea workflow without evidence of execution.
2 — Agent used a different image editor.
1 — No image editor used.

#### B. Sky Selection (0.25)
Was the sky area properly selected for editing?

5 — Sky selected using Magic Wand, Select by Color, or Quick Select; selection refined.
4 — Sky approximately selected but imprecisely.
3 — Whole image edited instead of just sky.
2 — No selection — edits applied globally.
1 — No selection attempted.

#### C. Color Adjustment (0.35)
Was the sky color deepened effectively?

5 — Sky is visibly deeper blue using Hue/Saturation, Curves, or Color Balance adjustment.
4 — Sky deepened but effect is subtle or inconsistent.
3 — Color adjustment applied but to wrong area or with wrong method.
2 — Adjustment attempted but no visible change.
1 — No color adjustment.

#### D. File Output (0.15)
Was the edited image saved?

5 — Edited image exported and saved as PNG/JPG file.
4 — Save initiated but format or location unclear.
3 — Image edited but not saved.
2 — Agent described saving without evidence.
1 — No output file.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "tool_usage": <1-5>,
  "sky_selection": <1-5>,
  "color_adjustment": <1-5>,
  "file_output": <1-5>,
  "dimension_reasoning": {{
    "tool_usage": "<one sentence citing specific evidence>",
    "sky_selection": "<one sentence citing specific evidence>",
    "color_adjustment": "<one sentence citing specific evidence>",
    "file_output": "<one sentence citing specific evidence>",
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "tool_usage": 0.25,
    "sky_selection": 0.25,
    "color_adjustment": 0.35,
    "file_output": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())