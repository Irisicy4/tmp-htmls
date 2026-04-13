"""
LLM-as-judge evaluator for EvolveBench task-47.

Category: Software Engineering
Task: Please rebuild the reservation program.
"""

import os, json, re

TASK_INSTRUCTION = "Please rebuild the reservation program."
PASS_THRESHOLD = 3.0

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully rebuilt a reservation web application.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- The task is intentionally vague ("rebuild the reservation program") — the agent must interpret this reasonably and build a functional reservation/booking web app
- A working UI is required — not just a description or plan
- Core reservation features expected: ability to make/view/cancel a reservation with date/time selection
- Code must be runnable (HTML file, React app, or similar)

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent produce actual code or just a description?
- What type of application was built (HTML, React, etc.)?
- What reservation features are present (booking form, date picker, list view, cancellation)?
- Was the app tested or previewed in a browser?
- Was a file saved?

### Step 2: Dimension Scoring

#### A. Application Completeness
Does the app cover core reservation functionality?

5 — App has booking form with date/time, reservation list view, and cancellation functionality.
4 — App has booking form and list view but no cancellation, or one feature is incomplete.
3 — App has only a booking form or only a list view — core loop is incomplete.
2 — App skeleton exists but no functional reservation logic.
1 — No working application produced.

#### B. Code Quality
Is the code well-structured and runnable?

5 — Clean, well-structured code that runs without errors; uses appropriate framework (React, HTML/JS).
4 — Code runs but has minor issues or is poorly organized.
3 — Code has bugs or incomplete sections but the intent is clear.
2 — Code exists but has major errors that prevent it from running.
1 — No code produced or code is clearly non-functional.

#### C. UI/UX Quality
Is the interface usable and reasonably well-designed?

5 — Clean, intuitive UI with clear labels, responsive layout, and good user experience.
4 — Functional UI but aesthetically basic or with minor UX issues.
3 — UI works but is hard to use or poorly laid out.
2 — UI exists but is broken or barely usable.
1 — No UI produced.

#### D. Delivery
Was the application properly saved and deliverable?

5 — File saved locally and/or dev server launched; agent confirmed it works.
4 — File saved but not tested or confirmed.
3 — Code provided in response but not saved as a file.
2 — Only partial code provided.
1 — Nothing delivered.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "application_completeness": <1-5>,
  "code_quality": <1-5>,
  "ui_ux_quality": <1-5>,
  "delivery": <1-5>,
  "dimension_reasoning": {{
    "application_completeness": "<one sentence citing specific evidence>",
    "code_quality": "<one sentence citing specific evidence>",
    "ui_ux_quality": "<one sentence citing specific evidence>",
    "delivery": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "application_completeness": 0.35,
    "code_quality":             0.25,
    "ui_ux_quality":            0.25,
    "delivery":                 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())


def _extract_response(result):
    task_result = result.get("task_result") or ""
    if isinstance(task_result, str) and task_result.strip():
        return task_result
    for message in reversed(result.get("conversation") or []):
        if not isinstance(message, dict): continue
        if message.get("role") == "assistant":
            content = message.get("content") or ""
            if isinstance(content, str) and len(content) > 20:
                return content
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
    except Exception as e:
        return {"error": str(e)}

def _vote(votes):
    valid = [v for v in votes if v and "error" not in v and all(d in v for d in DIMENSIONS)]
    if not valid: return votes[0] if votes else {"error": "All judge calls failed"}
    aggregated = {dim: sorted([v[dim] for v in valid])[len(valid) // 2] for dim in DIMENSIONS}
    overall = sum(aggregated[d] * DIMENSION_WEIGHTS[d] for d in DIMENSIONS)
    aggregated["overall_score"] = round(overall, 2)
    aggregated["passed"] = overall >= PASS_THRESHOLD
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