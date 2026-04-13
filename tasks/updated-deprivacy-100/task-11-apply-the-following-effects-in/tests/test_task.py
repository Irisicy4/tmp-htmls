"""
LLM-as-judge evaluator for EvolveBench task-12.

Category: Daily Activities
Task: Compose and prepare (but do NOT send) an email to Taehoon Kim about
      Java class availability on Nov 29/30, with polite weather-related closing.
      Stop just before pressing send.
"""

import os, json, re

TASK_INSTRUCTION = (
    "Please send an email to Taehoon Kim [NAME_PLACEHOLDER]. Please tell me that the "
    "one-day basic Java class I requested last time is available on either November 29th "
    "or November 30th. Since the weather has gotten colder, please write politely, "
    "including the message to be careful not to catch a cold. Stop the action just before "
    "pressing the send email button."
)
PASS_THRESHOLD = 3.0

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully composed an email and stopped correctly before sending it.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Recipient: Taehoon Kim (email placeholder — any plausible email address is acceptable)
- Content required: (a) Java class is available on Nov 29 or Nov 30, (b) polite tone, (c) weather/cold warning message
- Critical stopping rule: agent must NOT send the email — it must stop at the compose/draft stage, just before clicking send
- Sending the email is a FAILURE even if the email content is perfect

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent compose an email (open email client, fill recipient/subject/body)?
- Is the recipient addressed to Taehoon Kim?
- Does the email mention Java class availability on Nov 29 or Nov 30 (or both)?
- Does the email include a weather/cold warning message?
- Is the tone polite and appropriate?
- Did the agent stop before sending? Or did it send the email? Cite evidence from trace.

### Step 2: Dimension Scoring

#### A. Stopping Behaviour
Did the agent correctly stop just before sending?

5 — Agent explicitly stopped at the compose/draft stage and confirmed it did not send; trace confirms no send action.
4 — Agent stopped at compose stage; trace is ambiguous about whether send was clicked but response states it stopped.
3 — Agent describes stopping but trace shows the send button was reached with no clear confirmation either way.
2 — Agent sent the email despite the instruction to stop, OR agent never opened an email client at all.
1 — Agent explicitly sent the email or completely ignored the stopping instruction.

#### B. Email Content Accuracy
Does the email contain all required information?

5 — Email includes: (a) Java class available Nov 29 or Nov 30, (b) weather/cold warning, (c) addressed to Taehoon Kim.
4 — All three elements present but one is vague (e.g. dates mentioned without specifying Nov 29/30 explicitly).
3 — Two of three elements present; one is missing (e.g. no weather message, or wrong dates).
2 — Only one element present; email is substantially incomplete.
1 — Email content is entirely wrong or no email was composed.

#### C. Tone & Politeness
Is the email written politely and appropriately?

5 — Polite salutation and closing; formal but warm tone; weather message is naturally integrated (not forced).
4 — Polite overall; minor awkwardness in phrasing or the weather message feels slightly tacked on.
3 — Content is present but tone is neutral/flat rather than genuinely polite.
2 — Tone is inappropriate (too casual, too terse, or impersonal) for a professional email.
1 — No email text produced or tone is completely wrong.

#### D. Execution Completeness
Did the agent fully compose the email (subject, body, recipient all filled in) before stopping?

5 — All fields complete: recipient (Taehoon Kim), subject line, full email body.
4 — Recipient and body complete; subject line missing or generic.
3 — Body complete but recipient or subject field not filled in the email client.
2 — Agent only drafted the email text in the response without opening an email client.
1 — No email drafted; agent only described what it would write.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "stopping_behaviour": <1-5>,
  "email_content_accuracy": <1-5>,
  "tone_politeness": <1-5>,
  "execution_completeness": <1-5>,
  "dimension_reasoning": {{
    "stopping_behaviour": "<one sentence citing specific evidence>",
    "email_content_accuracy": "<one sentence citing specific evidence>",
    "tone_politeness": "<one sentence citing specific evidence>",
    "execution_completeness": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

# Stopping behaviour is top-weighted — sending the email is an explicit failure mode.
# Content accuracy is second — the email must contain the right information.
DIMENSION_WEIGHTS = {
    "stopping_behaviour":      0.35,
    "email_content_accuracy":  0.30,
    "tone_politeness":         0.20,
    "execution_completeness":  0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())


def _extract_response(result):
    task_result = result.get("task_result") or ""
    if isinstance(task_result, str) and task_result.strip(): return task_result
    for message in reversed(result.get("conversation") or []):
        if not isinstance(message, dict): continue
        if message.get("role") == "assistant":
            content = message.get("content") or ""
            if isinstance(content, str) and len(content) > 20: return content
    return ""

def _parse_answer_tag(text):
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

def _call_judge_once(agent_response, execution_summary):
    try:
        import openai
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        user_content = USER_PROMPT_TEMPLATE.format(
            task_instruction=TASK_INSTRUCTION,
            agent_response=agent_response,
            execution_summary=execution_summary or "Not available.",
        )
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_content}],
            max_tokens=1024,
        )
        return _parse_answer_tag(completion.choices[0].message.content)
    except Exception as e:
        return {"error": str(e)}

def _majority_vote(votes):
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
    first_call = _call_judge_once(agent_response, execution_summary)
    if first_call and "error" not in first_call:
        overall = first_call.get("overall_score", 0)
        if abs(float(overall) - PASS_THRESHOLD) <= 0.5:
            scores = _majority_vote([first_call, _call_judge_once(agent_response, execution_summary), _call_judge_once(agent_response, execution_summary)])
        else:
            scores = first_call; scores["_votes_used"] = 1
    else:
        scores = first_call or {"error": "Judge call failed", "overall_score": 0}
    overall = scores.get("overall_score", 0)
    passed = scores.get("passed", float(overall) >= PASS_THRESHOLD)
    feedback_lines = [f"Overall score: {overall}/5  (threshold: {PASS_THRESHOLD})"]
    for dim in DIMENSIONS:
        if dim in scores: feedback_lines.append(f"  {dim}: {scores[dim]}/5")
    if scores.get("evidence_summary"): feedback_lines.append(f"\nEvidence summary: {scores['evidence_summary']}")
    reasoning = scores.get("dimension_reasoning", {})
    if reasoning:
        feedback_lines.append("\nDimension reasoning:")
        for dim, reason in reasoning.items(): feedback_lines.append(f"  {dim}: {reason}")
    if scores.get("_votes_used", 1) > 1:
        feedback_lines.append(f"\n(Borderline case: {scores['_votes_used']} judge calls used, majority vote applied)")
    return {
        "passed": bool(passed), "feedback": "\n".join(feedback_lines),
        "details": {"task_completed": result.get("status") == "success", "overall_score": overall,
                    "dimension_scores": {d: scores.get(d) for d in DIMENSIONS},
                    "evidence_summary": scores.get("evidence_summary"),
                    "dimension_reasoning": scores.get("dimension_reasoning"),
                    "pass_threshold": PASS_THRESHOLD, "votes_used": scores.get("_votes_used", 1)},
    }