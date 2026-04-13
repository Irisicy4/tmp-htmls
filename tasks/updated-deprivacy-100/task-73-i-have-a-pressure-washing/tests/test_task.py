"""
LLM-as-judge evaluator for EvolveBench task-73.

Category: Marketing & Analytics
Task: Find best practices, content ideas, and growth strategies for starting a local newsletter
      for a pressure washing business in Jacksonville, FL, following the Naptown Scoop format.
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


TASK_INSTRUCTION = """I have a pressure washing business that operates in jacksonvile florida. i want to start a local newsletter to own my distristribution, and market my business essentially for free. Please find best practices to start a local newsletter here, what content ideas people would like best, and outline real growth strategies to start the newsletter. I want to follow the naptown scoop format. find what is currently working in the local newsletter idustry and outline strategies"""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to research and compile a practical guide for starting a local newsletter for a pressure washing business in Jacksonville, Florida. The agent must find best practices, content ideas suited to the local market, and growth strategies — specifically referencing the Naptown Scoop as a format model.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Business context: pressure washing company in Jacksonville, Florida
- Goal: start a local newsletter for free distribution/marketing
- Must cover: best practices for launching a local newsletter
- Must cover: specific content ideas (ideally relevant to Jacksonville community or home services)
- Must cover: concrete growth strategies for building the subscriber base
- Format reference: Naptown Scoop model (local-focused, community-driven newsletter)
- Research requirement: findings from what is currently working in the local newsletter industry

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent research local newsletter best practices? What sources or examples were found?
- Are content ideas provided and are they relevant to Jacksonville or home services?
- Are growth strategies specific and actionable (e.g. platforms, referral tactics, cross-promotions)?
- Did the agent reference or analyze the Naptown Scoop or a comparable local newsletter model?

### Step 2: Dimension Scoring

#### A. Best Practices Coverage (0.30)
Did the agent find specific, actionable best practices for launching a local newsletter?

5 — 5 or more concrete best practices with specific tools, platforms (e.g. Beehiiv, Substack, Mailchimp), or examples drawn from the local newsletter industry.
4 — 3-4 solid best practices with some platform-level or operational specifics.
3 — General best practices mentioned with limited actionable detail.
2 — Very generic newsletter advice not tailored to local or small-business context.
1 — No best practices provided.

#### B. Content Strategy (0.25)
Are content ideas specific and relevant to Jacksonville, FL and/or the pressure washing business?

5 — 5 or more content ideas, at least some tied to Jacksonville community (local events, neighborhoods, home maintenance tips, seasonal demand) or the pressure washing trade.
4 — 3-4 relevant content ideas with some local or business-specific angle.
3 — Generic content ideas applicable to any local newsletter without Jacksonville or trade specificity.
2 — Very few (1-2) or entirely irrelevant content ideas.
1 — No content strategy provided.

#### C. Growth Strategies (0.25)
Are growth strategies concrete and actionable?

5 — 3 or more specific growth tactics with implementation detail (e.g. referral programs with incentives, partnerships with local real estate agents or HOAs, cross-promotion with complementary businesses, newsletter aggregator listings).
4 — 2-3 growth strategies with reasonable implementation detail.
3 — General growth tactics (e.g. "post on social media") without specifics.
2 — Vague growth mentions without strategies.
1 — No growth strategies provided.

#### D. Naptown Scoop Format Reference (0.20)
Did the agent reference and apply the Naptown Scoop format model?

5 — Specifically describes key Naptown Scoop format elements (e.g. local event roundups, sponsor model, community-first tone, consistent cadence) and directly adapts them to the Jacksonville pressure washing context.
4 — Mentions Naptown Scoop by name and draws at least one relevant structural or strategic lesson.
3 — Naptown Scoop referenced but analysis is superficial (name-drop only without explanation).
2 — No mention of Naptown Scoop but analyzes another successful local newsletter as a model.
1 — No reference to Naptown Scoop or any comparable local newsletter example.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "best_practices_coverage": <1-5>,
  "content_strategy": <1-5>,
  "growth_strategies": <1-5>,
  "naptown_scoop_reference": <1-5>,
  "dimension_reasoning": {{
    "best_practices_coverage": "<one sentence citing specific evidence>",
    "content_strategy": "<one sentence citing specific evidence>",
    "growth_strategies": "<one sentence citing specific evidence>",
    "naptown_scoop_reference": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "best_practices_coverage": 0.30,
    "content_strategy": 0.25,
    "growth_strategies": 0.25,
    "naptown_scoop_reference": 0.20,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())