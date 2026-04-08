"""
LLM-as-judge evaluator for EvolveBench task-13.

Category: Shopping
Task: Find best eSIM plans for Osaka Japan Dec 22 2025 – Jan 12 2026.
      At least 50GB (75GB preferred), 5G preferred over LTE unless big price difference.
"""

import os, json, re

TASK_INSTRUCTION = (
    "I need to purchase ESIM for my vacation to Osaka, Japan next week. I will be there "
    "from December 22nd 2025, until January 12th, 2026. I need to find an ESIM that has a "
    "lot of data (preferably 5G, but LTE will do), and is cheap. Please find the best plans "
    "for me. I need at least 50GB over the duration of my stay, but 75GB would be better. "
    "If there's a big price difference, feel free to choose LTE. If not, go 5G. I would "
    "prefer 5G over LTE though"
)
PASS_THRESHOLD = 3.0

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully found appropriate eSIM plans matching a traveller's specific requirements.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Destination: Japan (Osaka) — eSIM must work in Japan
- Duration: December 22, 2025 – January 12, 2026 (21 days) — plan must cover full duration
- Data minimum: 50GB (hard minimum); 75GB preferred
- Network: 5G preferred; LTE acceptable if significantly cheaper
- Price decision rule: if 5G and LTE plans are similarly priced, choose 5G; if LTE is significantly cheaper, LTE is fine
- Agent must search for actual current plans, not rely on prior knowledge

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent search for eSIM plans? What platform or site did it use?
- What specific plans were found? List them with data allowance, network type, price, and duration.
- Does each plan meet the 50GB minimum? Does it cover the full 21-day period?
- Did the agent reason about the 5G vs LTE tradeoff as instructed?
- Did the agent make a clear recommendation with justification?

### Step 2: Dimension Scoring

#### A. Search Execution
Did the agent actively search for current eSIM plans?

5 — Agent searched a real eSIM platform or comparison site (e.g. Airalo, Holafly, Nomad, Esimdb) and retrieved current listings.
4 — Agent searched but retrieved limited results or from only one provider without comparison.
3 — Agent retrieved eSIM information but from a general web search rather than a dedicated eSIM platform.
2 — Agent described what to look for without actually searching.
1 — No search performed; response is from prior knowledge or fabricated.

#### B. Constraint Satisfaction
Do the recommended plans meet all hard requirements?

5 — Recommended plan(s): (a) work in Japan, (b) cover full 21-day period, (c) provide ≥50GB data.
4 — All constraints met but one is weakly supported (e.g. plan says "30 days" but exact start date not verified).
3 — Two of three constraints clearly met; one is ambiguous or missing (e.g. data amount not specified).
2 — Only one constraint clearly met; plan may not cover the full duration or data is below 50GB.
1 — Constraints ignored; recommended plans clearly don't meet the requirements.

#### C. 5G/LTE Decision Quality
Did the agent reason correctly about the 5G vs LTE tradeoff?

5 — Agent explicitly compared 5G and LTE plan prices, applied the user's stated rule (choose 5G unless significant price difference), and justified the recommendation.
4 — Agent recommended 5G or LTE with brief price comparison; reasoning is present but not fully explicit.
3 — Agent mentioned both options but did not compare prices or apply the decision rule.
2 — Agent defaulted to one option without considering the other.
1 — 5G/LTE preference completely ignored.

#### D. Recommendation Quality
Is the final recommendation specific, actionable, and well-justified?

5 — Clear winner recommended with: plan name/provider, price, data, network type, duration, and why it's the best choice given the constraints.
4 — Good recommendation with most details; one element missing (e.g. no price or no comparison to alternatives).
3 — A plan is identified but the recommendation is thin (name only, or no justification).
2 — Multiple options listed without a clear recommendation.
1 — No recommendation made; agent only described the search process.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "search_execution": <1-5>,
  "constraint_satisfaction": <1-5>,
  "network_decision_quality": <1-5>,
  "recommendation_quality": <1-5>,
  "dimension_reasoning": {{
    "search_execution": "<one sentence citing specific evidence>",
    "constraint_satisfaction": "<one sentence citing specific evidence>",
    "network_decision_quality": "<one sentence citing specific evidence>",
    "recommendation_quality": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "search_execution":        0.20,
    "constraint_satisfaction": 0.35,
    "network_decision_quality":0.20,
    "recommendation_quality":  0.25,
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
    match = re.search(r"<Answer>(.*?)</Answer>", text, re.DOTALL | re.IGNORECASE)
    if not match: return None
    try: return json.loads(match.group(1).strip())
    except json.JSONDecodeError: return None

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