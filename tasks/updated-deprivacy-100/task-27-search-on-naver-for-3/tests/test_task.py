"""
LLM-as-judge evaluator for EvolveBench task-27.

Category: Daily Activities
Task: Search Naver for 3 famous hair salons in Gangnam, Seoul. Book the best-value one for a men's perm.
"""

import os, json, re

TASK_INSTRUCTION = "Search on Naver for 3 famous hair salons in Gangnam, Seoul, and book the one that offers the best value. I want to get a men's perm."
PASS_THRESHOLD = 3.0

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. The task involves using Naver (a Korean search engine) to find hair salons in Gangnam, Seoul. Evaluate whether the agent finds salon information, makes a value assessment, and attempts booking."""

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Platform: Naver (Korean search engine — naver.com)
- Location: Gangnam, Seoul (강남, 서울)
- Required count: 3 salons found
- Service: men's perm specifically
- Action: must attempt to book the best-value salon (not just list them)
- Value assessment: agent must compare options (price/reputation) and justify its choice

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis
- Did the agent navigate Naver? Cite evidence.
- How did the agent search for salons in Gangnam, Seoul on Naver?
- Were 3 salon options identified (from any source)?
- Did the agent compare options and select best value?
- Was a booking attempted? What method (phone, website, app)?

### Step 2: Dimension Scoring

#### A. Platform Execution
Did the agent attempt to use Naver as instructed?

5 — Agent navigated Naver, searched for Gangnam, Seoul hair salons, and found results or clearly described the process.
4 — Agent used Naver but found limited results; supplemented with another platform.
3 — Agent attempted Naver but immediately switched to a different platform without explanation.
2 — Agent used a completely different platform (not Naver) without acknowledgement.
1 — No search performed.

#### B. Salon Discovery
Did the agent identify 3 named hair salons relevant to the task?

5 — 3 specific, named salons in Gangnam, Seoul found with location and men's perm pricing or service info.
4 — 3 salons found but one lacks specific location or pricing detail.
3 — 2 salons found with detail; third is vague or unverified.
2 — Only 1 salon found, or salons found are generic (not named establishments).
1 — No specific salons identified.

#### C. Value Assessment & Selection
Did the agent compare salons and select the best value with justification?

5 — Agent compared 2+ salons on price and/or reputation, selected one, and justified the choice (e.g. "salon X offers men's perm at ¥X with good reviews").
4 — Agent selected a salon with brief justification; comparison is implicit.
3 — Agent selected a salon without comparing alternatives or without justification.
2 — Agent listed salons but did not make a selection.
1 — No selection made.

#### D. Booking Attempt
Did the agent attempt to book the selected salon?

5 — Booking attempted via appropriate channel (website, phone, reservation app) with confirmation or next steps described.
4 — Booking attempted but outcome unclear or channel is indirect (e.g. found contact info but did not call/submit).
3 — Agent identified booking method but did not execute it.
2 — Agent noted that booking is required but took no action.
1 — No booking attempt of any kind.

### Step 3: Output
<Answer>
{{
  "evidence_summary": "<2-3 sentences>",
  "platform_execution": <1-5>,
  "salon_discovery": <1-5>,
  "value_assessment": <1-5>,
  "booking_attempt": <1-5>,
  "dimension_reasoning": {{"platform_execution": "<one sentence>", "salon_discovery": "<one sentence>", "value_assessment": "<one sentence>", "booking_attempt": "<one sentence>"}},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {"platform_execution": 0.20, "salon_discovery": 0.25, "value_assessment": 0.25, "booking_attempt": 0.30}
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
    match = re.search(r"<Answer>(.*?)</Answer>", text, re.DOTALL | re.IGNORECASE)
    if not match: return None
    try: return json.loads(match.group(1).strip())
    except: return None

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