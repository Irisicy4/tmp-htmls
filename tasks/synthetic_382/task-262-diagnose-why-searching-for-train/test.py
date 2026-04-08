"""
LLM-as-judge evaluator for EvolveBench task.

Category: Travel & Planning
Task: Diagnose why searching for train tickets on SNCF’s English website for Paris to Lyon on December 15, 2024, is showing errors during checkout.
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


TASK_INSTRUCTION = """Diagnose why searching for train tickets on SNCF’s English website (Oui.sncf) for Paris to Lyon on December 15, 2024, is showing errors during checkout. Check confirmation from SNCF's help section, forums like TripAdvisor, and technical announcements on their site."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to diagnose checkout errors when searching for train tickets on SNCF’s English website for Paris to Lyon on December 15, 2024. The agent must investigate using SNCF's help section, forums like TripAdvisor, and technical announcements on SNCF's site. Successful completion requires identifying the root cause and providing a clear explanation.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Diagnose why searching for train tickets on SNCF’s English website (Oui.sncf) for Paris to Lyon on December 15, 2024, is showing errors during checkout. Check confirmation from SNCF's help section, forums like TripAdvisor, and technical announcements on their site.

## Task-Specific Constraints
- Must visit Oui.sncf, TripAdvisor, and SNCF's English website.
- Must identify the root cause of the checkout error.
- Must confirm findings using credible sources or technical announcements.
- Output must clearly explain the issue and resolution steps.
- Must include direct references to the platforms visited.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Oui.sncf, TripAdvisor, and SNCF's English website? Which ones were actually visited?
- Did the agent identify the root cause of the checkout error?
- Are the findings confirmed using credible sources or technical announcements?
- Is the output organized clearly with explanation and resolution steps?
- Are direct references to the platforms visited included in the response?

### Step 2: Dimension Scoring

#### A. Root Cause Identification (0.35)
Measures whether the agent accurately identified the root cause of the checkout error.

5 — Clearly identifies the root cause with supporting evidence from multiple sources.
4 — Identifies the root cause but lacks supporting evidence from all required sources.
3 — Partially identifies the root cause but lacks clarity or evidence.
2 — Incorrect or vague identification of the root cause.
1 — Fails to identify the root cause entirely.

#### B. Platform Coverage (0.30)
Measures whether the agent visited all required platforms and used them effectively.

5 — Visits all required platforms and uses them effectively to gather information.
4 — Visits most required platforms but misses one or uses one ineffectively.
3 — Visits some required platforms but misses key ones or gathers minimal information.
2 — Visits only one platform or gathers insufficient information.
1 — Fails to visit any required platforms.

#### C. Evidence Depth (0.20)
Measures the depth and specificity of the evidence provided.

5 — Provides detailed evidence with specific references and examples.
4 — Provides evidence but lacks some specificity or examples.
3 — Provides minimal evidence with limited specificity.
2 — Provides vague or unsupported evidence.
1 — Provides no evidence or completely unsupported claims.

#### D. Output Organization (0.15)
Measures the clarity and structure of the agent's response.

5 — Response is well-organized, clear, and easy to follow.
4 — Response is mostly clear but has minor organizational issues.
3 — Response is partially organized but lacks clarity in some areas.
2 — Response is poorly organized and difficult to follow.
1 — Response is completely disorganized or incoherent.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "root_cause_identification": <1-5>,
  "platform_coverage": <1-5>,
  "evidence_depth": <1-5>,
  "output_organization": <1-5>,
  "dimension_reasoning": {{
    "root_cause_identification": "<one sentence citing specific evidence>",
    "platform_coverage": "<one sentence citing specific evidence>",
    "evidence_depth": "<one sentence citing specific evidence>",
    "output_organization": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "root_cause_identification": 0.35,
    "platform_coverage": 0.30,
    "evidence_depth": 0.20,
    "output_organization": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())