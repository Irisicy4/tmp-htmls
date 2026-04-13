"""
LLM-as-judge evaluator for EvolveBench task-59.

Category: Software Engineering
Task: Please update my N8N installation on Hostinger....
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


TASK_INSTRUCTION = """Please update my N8N installation on Hostinger."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves updating N8N (a workflow automation tool) on a Hostinger VPS or hosting account.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Platform: Hostinger (VPS or shared hosting)
- Action: update/upgrade N8N to latest version
- Method: agent may use SSH terminal, Hostinger admin panel, or Docker depending on installation type
- Confirmation: agent should verify the updated version number

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent access Hostinger or an SSH terminal?
- What N8N update method was used (npm, Docker, hPanel)?
- Was the update command executed successfully?
- Was the new N8N version confirmed?

### Step 2: Dimension Scoring

#### A. Platform Access (0.25)
Did the agent access Hostinger or a terminal?

5 — Agent accessed Hostinger hPanel or SSH terminal and authenticated successfully.
4 — Agent accessed the platform but with some difficulty.
3 — Agent described update steps without accessing the platform.
2 — Agent gave generic N8N update instructions without Hostinger context.
1 — No platform access attempted.

#### B. Update Execution (0.4)
Was the N8N update command executed?

5 — Update command run (e.g. npm update -g n8n, docker pull n8nio/n8n, or equivalent) with output shown.
4 — Update initiated but completion unclear.
3 — Correct command identified but not executed.
2 — Wrong update method attempted.
1 — No update attempt.

#### C. Version Verification (0.25)
Was the updated version verified?

5 — New N8N version confirmed (e.g. n8n --version output shown after update).
4 — Update appeared successful but version not explicitly verified.
3 — Agent mentioned version check but result unclear.
2 — No version verification.
1 — Update failed or not attempted.

#### D. Error Handling (0.1)
Did the agent handle any errors gracefully?

5 — Any errors encountered were addressed and resolved.
4 — Errors acknowledged with reasonable workaround.
3 — Errors noted but not resolved.
2 — Errors ignored.
1 — Task failed due to unhandled errors.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "platform_access": <1-5>,
  "update_execution": <1-5>,
  "version_verification": <1-5>,
  "error_handling": <1-5>,
  "dimension_reasoning": {{
    "platform_access": "<one sentence citing specific evidence>",
    "update_execution": "<one sentence citing specific evidence>",
    "version_verification": "<one sentence citing specific evidence>",
    "error_handling": "<one sentence citing specific evidence>",
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "platform_access": 0.25,
    "update_execution": 0.4,
    "version_verification": 0.25,
    "error_handling": 0.1,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())