"""
LLM-as-judge evaluator for EvolveBench task-22-send-message-to-my-manager.

Category: Professional Communication
Task: Compose an email to my manager (manager@example.com) telling them I will be over an hour late due to traffic conditions. Write it politely. Stop just before pressing send — do not actually send it.
"""

import os, json, re

TASK_INSTRUCTION = """Compose an email to my manager (manager@example.com) telling them I will be over an hour late due to traffic conditions. Write it politely. Stop just before pressing send — do not actually send it."""
PASS_THRESHOLD = 3.0

SYSTEM_PROMPT = """The judge is evaluating the agent's ability to compose a polite and professional email to a manager, clearly explaining the reason for being late due to traffic conditions. The evaluation focuses on clarity, tone, adherence to instructions, and grammatical correctness.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- The email must be addressed to manager@example.com.
- The reason for being late must explicitly reference traffic conditions.
- The tone must be polite and professional.
- The agent must stop before pressing send.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Does the email include the correct recipient address (manager@example.com)?
- Does the email clearly explain the reason for being late as traffic conditions?
- Is the tone of the email polite and professional?
- Did the agent stop before pressing send?

### Step 2: Dimension Scoring

#### A. Clarity Of Reason
How clearly the email explains the reason for being late.

5 — The email explicitly states traffic conditions as the reason for being late in a clear and concise manner.
4 — The email states traffic conditions as the reason for being late, but with minor ambiguity or unnecessary details.
3 — The email mentions traffic conditions but lacks clarity or includes irrelevant information.
2 — The email vaguely references being late without clearly attributing it to traffic conditions.
1 — The email does not mention traffic conditions as the reason for being late.

#### B. Tone And Politeness
How polite and professional the tone of the email is.

5 — The email uses consistently polite and professional language throughout.
4 — The email is mostly polite and professional, with minor lapses in tone.
3 — The email is somewhat polite but includes noticeable lapses in professionalism.
2 — The email is minimally polite and lacks professionalism in tone.
1 — The email is rude or unprofessional in tone.

#### C. Adherence To Instructions
How well the agent follows the task instructions.

5 — The agent fully adheres to all instructions, including stopping before pressing send.
4 — The agent adheres to most instructions but with minor deviations.
3 — The agent partially adheres to instructions, missing key details or steps.
2 — The agent minimally adheres to instructions, ignoring significant aspects.
1 — The agent does not adhere to the instructions at all.

#### D. Grammar And Spelling
The grammatical correctness and spelling accuracy of the email.

5 — The email is free of grammatical errors and spelling mistakes.
4 — The email has one or two minor grammatical errors or spelling mistakes.
3 — The email contains several noticeable grammatical errors or spelling mistakes.
2 — The email has frequent grammatical errors or spelling mistakes that affect readability.
1 — The email is riddled with grammatical errors and spelling mistakes, making it difficult to understand.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "clarity_of_reason": <1-5>,
  "tone_and_politeness": <1-5>,
  "adherence_to_instructions": <1-5>,
  "grammar_and_spelling": <1-5>,
  "dimension_reasoning": {{
    "clarity_of_reason": "<one sentence citing specific evidence>",
    "tone_and_politeness": "<one sentence citing specific evidence>",
    "adherence_to_instructions": "<one sentence citing specific evidence>",
    "grammar_and_spelling": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "clarity_of_reason":   0.3,
    "tone_and_politeness": 0.3,
    "adherence_to_instructions": 0.2,
    "grammar_and_spelling": 0.2,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())

def _extract_response(result):
    t = result.get("task_result") or ""
    if isinstance(t, str) and t.strip(): return t
    for m in reversed(result.get("conversation") or []):
        if isinstance(m, dict) and m.get("role") == "assistant":
            c = m.get("content") or ""
            if isinstance(c, str) and len(c) > 20: return c
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
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": USER_PROMPT_TEMPLATE.format(task_instruction=TASK_INSTRUCTION, agent_response=agent_response, execution_summary=execution_summary or "Not available.")}],
            max_tokens=1024)
        return _parse(completion.choices[0].message.content)
    except Exception as e: return {"error": str(e)}

def _vote(votes):
    valid = [v for v in votes if v and "error" not in v and all(d in v for d in DIMENSIONS)]
    if not valid: return votes[0] if votes else {"error": "All calls failed"}
    agg = {d: sorted([v[d] for v in valid])[len(valid)//2] for d in DIMENSIONS}
    overall = sum(agg[d] * DIMENSION_WEIGHTS[d] for d in DIMENSIONS)
    agg["overall_score"] = round(overall, 2); agg["passed"] = overall >= PASS_THRESHOLD
    median = sorted(valid, key=lambda v: abs(v.get("overall_score",0)-overall))[0]
    agg["evidence_summary"] = median.get("evidence_summary",""); agg["dimension_reasoning"] = median.get("dimension_reasoning",{}); agg["_votes_used"] = len(valid)
    return agg

def test(result):
    agent_response = _extract_response(result)
    execution_summary = result.get("execution_summary", "")
    if not agent_response.strip():
        return {"passed": False, "feedback": "No response found from agent.", "details": {"task_completed": result.get("status") == "success"}}
    first = _call(agent_response, execution_summary)
    if first and "error" not in first:
        overall = first.get("overall_score", 0)
        scores = _vote([first, _call(agent_response, execution_summary), _call(agent_response, execution_summary)]) if abs(float(overall) - PASS_THRESHOLD) <= 0.5 else (first.__setitem__("_votes_used", 1) or first)
    else:
        scores = first or {"error": "Judge call failed", "overall_score": 0}
    overall = scores.get("overall_score", 0); passed = scores.get("passed", float(overall) >= PASS_THRESHOLD)
    lines = [f"Overall score: {overall}/5  (threshold: {PASS_THRESHOLD})"] + [f"  {d}: {scores[d]}/5" for d in DIMENSIONS if d in scores]
    if scores.get("evidence_summary"): lines.append(f"\nEvidence summary: {scores['evidence_summary']}")
    if scores.get("dimension_reasoning"):
        lines.append("\nDimension reasoning:")
        for d, r in scores["dimension_reasoning"].items(): lines.append(f"  {d}: {r}")
    if scores.get("_votes_used", 1) > 1: lines.append(f"\n(Borderline: {scores['_votes_used']} calls, majority vote)")
    return {"passed": bool(passed), "feedback": "\n".join(lines), "details": {"task_completed": result.get("status") == "success", "overall_score": overall, "dimension_scores": {d: scores.get(d) for d in DIMENSIONS}, "evidence_summary": scores.get("evidence_summary"), "dimension_reasoning": scores.get("dimension_reasoning"), "pass_threshold": PASS_THRESHOLD, "votes_used": scores.get("_votes_used", 1)}}