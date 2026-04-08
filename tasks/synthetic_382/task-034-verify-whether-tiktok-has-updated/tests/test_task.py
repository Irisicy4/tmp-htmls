"""
LLM-as-judge evaluator for EvolveBench task.

Category: Marketing & Analytics
Task: Verify whether TikTok has updated its ad policy regarding targeting underage users by consulting specified sources and summarizing the findings.
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


TASK_INSTRUCTION = """Verify whether TikTok has updated its ad policy regarding targeting underage users by checking TikTok's official policy page, any recent announcements on their newsroom, and coverage on a reputable marketing news site like Marketing Dive. State whether the policy has changed, when it was updated, and what the change entails."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to determine if TikTok has updated its ad policy regarding targeting underage users. The agent must consult TikTok's official policy page, their newsroom for recent announcements, and a reputable marketing news site like Marketing Dive. A successful completion includes a clear statement on whether the policy has changed, the date of the update, and a summary of the change.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Verify whether TikTok has updated its ad policy regarding targeting underage users by checking TikTok's official policy page, any recent announcements on their newsroom, and coverage on a reputable marketing news site like Marketing Dive. State whether the policy has changed, when it was updated, and what the change entails.

## Task-Specific Constraints
- Must visit TikTok's official policy page.
- Must check TikTok's newsroom for recent announcements.
- Must check a reputable marketing news site like Marketing Dive.
- Must explicitly state whether the policy has changed.
- Must include the date of the policy update if applicable.
- Must summarize the specific changes to the policy.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to TikTok's official policy page?
- Did the agent check TikTok's newsroom for recent announcements?
- Did the agent consult a reputable marketing news site like Marketing Dive?
- Did the agent clearly state whether the policy has changed?
- Did the agent include the date of the policy update if applicable?
- Did the agent summarize the specific changes to the policy?

### Step 2: Dimension Scoring

#### A. Policy Update Accuracy (0.35)
Measures whether the agent correctly identified if TikTok's ad policy has changed and provided the correct details.

5 — Accurately identifies if the policy changed, includes the correct date and a detailed summary of the changes.
4 — Accurately identifies if the policy changed, includes the correct date, but the summary is missing minor details.
3 — Accurately identifies if the policy changed but lacks the date or provides an incomplete summary.
2 — Incorrectly identifies if the policy changed or provides incorrect details.
1 — Fails to address whether the policy changed or provides completely inaccurate information.

#### B. Source Coverage (0.30)
Measures whether the agent consulted all required sources (TikTok's policy page, newsroom, and Marketing Dive).

5 — Consults all three required sources and provides evidence from each.
4 — Consults all three sources but provides incomplete evidence from one.
3 — Consults at least two sources and provides some evidence from each.
2 — Consults only one source or provides minimal evidence.
1 — Does not consult any of the required sources.

#### C. Specificity of Findings (0.20)
Measures the level of detail in the agent's findings about the policy update.

5 — Provides highly specific details, including exact policy language or direct quotes.
4 — Provides specific details but lacks direct quotes or exact language.
3 — Provides general details but lacks specificity.
2 — Provides vague or minimal details.
1 — Provides no meaningful details.

#### D. Output Structure and Credibility (0.15)
Measures whether the response is well-organized and cites credible sources.

5 — Response is well-organized, clearly written, and cites credible sources for all claims.
4 — Response is mostly well-organized and cites credible sources for most claims.
3 — Response is somewhat organized but may lack citations or clarity in places.
2 — Response is poorly organized or lacks credible sources.
1 — Response is disorganized and lacks credible sources.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "policy_update_accuracy": <1-5>,
  "source_coverage": <1-5>,
  "specificity_of_findings": <1-5>,
  "output_structure_and_credibility": <1-5>,
  "dimension_reasoning": {{
    "policy_update_accuracy": "<one sentence citing specific evidence>",
    "source_coverage": "<one sentence citing specific evidence>",
    "specificity_of_findings": "<one sentence citing specific evidence>",
    "output_structure_and_credibility": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "policy_update_accuracy": 0.35,
    "source_coverage": 0.30,
    "specificity_of_findings": 0.20,
    "output_structure_and_credibility": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())