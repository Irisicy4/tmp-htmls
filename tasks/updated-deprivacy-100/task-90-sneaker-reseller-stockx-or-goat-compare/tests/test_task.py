"""
LLM-as-judge evaluator for EvolveBench task-90.

Category: Shopping
Task: I am a sneaker reseller. Should I choose StockX or GOAT? Please research and compare the two platforms and give me a rec
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


TASK_INSTRUCTION = """I am a sneaker reseller. Should I choose StockX or GOAT? Please research and compare the two platforms and give me a recommendation."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves researching and comparing StockX and GOAT sneaker resale platforms from a seller's perspective to give a concrete recommendation.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Perspective: sneaker reseller (seller, not buyer) — fees, payout speed, authentication, seller protections
- Platforms: StockX AND GOAT — both must be covered
- Recommendation: must give a clear recommendation with reasoning
- Data: current fee structures and platform features

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent research both platforms?
- What seller fees were found for each platform?
- Were payout timelines, authentication processes, and seller protections compared?
- Was a clear recommendation made with reasoning?

### Step 2: Dimension Scoring

#### A. Platform Research (0.25)
Did the agent research both platforms?

5 — Both StockX and GOAT researched with current seller fee data from official sources.
4 — Both researched but one less thoroughly.
3 — Both covered but data may be outdated.
2 — Only one platform researched.
1 — No research.

#### B. Comparison Depth (0.35)
Were relevant seller factors compared?

5 — Fees, payout speed, authentication process, seller protections, geographic availability, and user base all compared.
4 — 4-5 factors compared.
3 — 2-3 factors compared.
2 — Only fees compared.
1 — No comparison.

#### C. Data Accuracy (0.25)
Is the comparison data accurate?

5 — Fee percentages match current published rates (StockX ~9-10%, GOAT ~15-25% depending on tier); other facts verifiable.
4 — Mostly accurate with minor discrepancies.
3 — Approximately correct but imprecise.
2 — Significant inaccuracies.
1 — Clearly wrong data.

#### D. Recommendation Quality (0.15)
Is the recommendation clear and justified?

5 — Clear recommendation with specific reasoning tailored to a reseller's priorities.
4 — Recommendation made but reasoning is somewhat generic.
3 — Balanced 'it depends' without a clear winner.
2 — Recommendation without justification.
1 — No recommendation.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "platform_research": <1-5>,
  "comparison_depth": <1-5>,
  "data_accuracy": <1-5>,
  "recommendation_quality": <1-5>,
  "dimension_reasoning": {{
    "platform_research": "<one sentence citing specific evidence>",
    "comparison_depth": "<one sentence citing specific evidence>",
    "data_accuracy": "<one sentence citing specific evidence>",
    "recommendation_quality": "<one sentence citing specific evidence>",
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "platform_research": 0.25,
    "comparison_depth": 0.35,
    "data_accuracy": 0.25,
    "recommendation_quality": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())