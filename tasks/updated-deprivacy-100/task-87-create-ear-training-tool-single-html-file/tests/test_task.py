"""
LLM-as-judge evaluator for EvolveBench task-87.

Category: Software Engineering
Task: Please create an ear training (ear copy) tool as a web application. Build it as a single HTML file using Web Audio API t
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


TASK_INSTRUCTION = """Please create an ear training (ear copy) tool as a web application. Build it as a single HTML file using Web Audio API that plays musical intervals or chords and lets the user identify them."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves building a functional single-file HTML ear training web app using the Web Audio API.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Format: single HTML file with embedded CSS and JavaScript
- API: Web Audio API for sound generation (no external audio files)
- Features: plays musical intervals or chords; user can identify them; feedback provided
- Quality: must be functional and usable — not just a mockup

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent produce a single HTML file?
- Does it use Web Audio API for sound generation?
- What musical content is included (intervals, chords, both)?
- Is there an identification UI (buttons, input) with feedback?
- Does the app function correctly?

### Step 2: Dimension Scoring

#### A. Web Audio Implementation (0.3)
Is Web Audio API used correctly?

5 — AudioContext, OscillatorNode, frequency calculations for musical intervals all correctly implemented.
4 — Web Audio API used but with minor errors.
3 — Web Audio API attempted but fundamental issues.
2 — External audio files or HTML audio elements used instead.
1 — No audio implementation.

#### B. Musical Content (0.25)
Is the musical content accurate and varied?

5 — Correct frequencies for intervals (unison through octave) and/or chords; covers major and minor at minimum.
4 — Mostly correct but missing some intervals/chords.
3 — Basic intervals only (major/minor/perfect).
2 — Inaccurate frequencies or very limited content.
1 — No musical content.

#### C. Ui Completeness (0.25)
Is the UI functional for ear training?

5 — Play button, answer selection (buttons/dropdown), feedback (correct/wrong), score tracking.
4 — Play + answer selection + feedback but no score.
3 — Play + answer but no feedback.
2 — Play only without identification UI.
1 — No functional UI.

#### D. Code Quality (0.2)
Is the single-file HTML well-written?

5 — Clean, readable HTML/CSS/JS in one file; no external dependencies; works on open.
4 — Functional but slightly messy code.
3 — Works but has notable quality issues.
2 — Significant bugs; partially functional.
1 — Does not work.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "web_audio_implementation": <1-5>,
  "musical_content": <1-5>,
  "ui_completeness": <1-5>,
  "code_quality": <1-5>,
  "dimension_reasoning": {{
    "web_audio_implementation": "<one sentence citing specific evidence>",
    "musical_content": "<one sentence citing specific evidence>",
    "ui_completeness": "<one sentence citing specific evidence>",
    "code_quality": "<one sentence citing specific evidence>",
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "web_audio_implementation": 0.3,
    "musical_content": 0.25,
    "ui_completeness": 0.25,
    "code_quality": 0.2,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())