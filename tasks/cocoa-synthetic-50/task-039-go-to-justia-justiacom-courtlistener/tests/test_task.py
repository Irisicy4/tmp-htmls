"""
LLM-as-judge evaluator for EvolveBench task.

Category: Legal
Task: Search for recent court cases involving Section 230 of the Communications Decency Act decided within the last two years, and compile case details into a summary list.
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


TASK_INSTRUCTION = """Go to Justia (justia.com), CourtListener (courtlistener.com), and the U.S. Department of Justice website (justice.gov), and search for recent court cases involving Section 230 of the Communications Decency Act decided within the last two years. For each case you find, record the case name, court, decision date, and a brief summary of the ruling. Extract this information for at least three cases and compile the details into a summary list."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to search three specific legal websites for recent court cases involving Section 230 of the Communications Decency Act decided within the last two years. The agent must extract case names, courts, decision dates, and brief summaries of the rulings for at least three cases. A successful completion requires accurate and complete information compiled into a structured summary list.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Go to Justia (justia.com), CourtListener (courtlistener.com), and the U.S. Department of Justice website (justice.gov), and search for recent court cases involving Section 230 of the Communications Decency Act decided within the last two years. For each case you find, record the case name, court, decision date, and a brief summary of the ruling. Extract this information for at least three cases and compile the details into a summary list.

## Task-Specific Constraints
- Must visit all three specified platforms: Justia, CourtListener, and justice.gov.
- Must identify at least three distinct court cases involving Section 230 within the last two years.
- Must include case name, court, decision date, and a brief summary of the ruling for each case.
- Output must be organized as a structured summary list (e.g., JSON or table format).
- Summaries must be concise, accurate, and relevant to Section 230 rulings.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to all three required platforms (Justia, CourtListener, justice.gov)?
- Are at least three distinct court cases involving Section 230 included in the response?
- Does the response include case name, court, decision date, and a brief summary for each case?
- Is the output organized as a structured summary list (e.g., JSON or table)?
- Are the summaries accurate and relevant to Section 230 rulings?

### Step 2: Dimension Scoring

#### A. Deliverable Accuracy (0.35)
Measures whether the extracted case details (name, court, date, summary) are accurate and complete.

5 — All case details are accurate, complete, and relevant for at least three cases.
4 — Minor inaccuracies or omissions in one case, but overall usable.
3 — Partial completion (e.g., missing details for one or more cases).
2 — Significant inaccuracies or missing details for most cases.
1 — No usable case details provided.

#### B. Coverage of Sources (0.30)
Measures whether the agent visited all required platforms and extracted relevant cases.

5 — Cases extracted from all three platforms (Justia, CourtListener, justice.gov).
4 — Cases extracted from two platforms; one platform missing.
3 — Cases extracted from only one platform; others ignored.
2 — Attempted but failed to extract cases from any platform.
1 — No evidence of platform navigation or case extraction.

#### C. Depth of Summaries (0.25)
Measures the quality and specificity of the case summaries provided.

5 — Summaries are concise, accurate, and include key details about Section 230 rulings.
4 — Summaries are mostly accurate but lack depth or specificity in one case.
3 — Summaries are overly vague or missing key details for multiple cases.
2 — Summaries are mostly incorrect or irrelevant to Section 230.
1 — No summaries provided.

#### D. Output Structure and Credibility (0.10)
Measures whether the response is well-organized and sources are credible.

5 — Output is well-structured (e.g., JSON or table) and sources are clearly credible.
4 — Output is mostly well-structured but has minor formatting issues.
3 — Output is usable but poorly structured or lacks clarity.
2 — Output is disorganized or sources are unclear.
1 — No structured output or credible sources provided.

DIMENSION_WEIGHTS = {
    "deliverable_accuracy": 0.35,
    "coverage_of_sources": 0.30,
    "depth_of_summaries": 0.25,
    "output_structure_and_credibility": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())