"""
LLM-as-judge evaluator for EvolveBench task.

Category: Legal
Task: Navigate the U.S. Copyright Office website to initiate the registration process for a fictional book, and report the registration fee and review screen details.
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


TASK_INSTRUCTION = """Go to the U.S. Copyright Office website (copyright.gov) and navigate to the registration portal for literary works. 
Begin the registration process for a fictional book titled 'The Artificial Harp,' authored by 'Jane Doe' in the year 2023. 
Fill out the public-facing form with placeholder details provided in the instruction. 
Progress through the steps up to the fee payment screen (but do not pay). 
Report back the total registration fee and the summary of the input details shown on the final review screen."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves navigating the U.S. Copyright Office website to initiate the registration process for a fictional literary work. 
The agent must report the total registration fee and summarize the input details shown on the final review screen. 
A successful completion requires accurate navigation, correct input of details, and reporting the required information.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Go to the U.S. Copyright Office website (copyright.gov) and navigate to the registration portal for literary works. 
Begin the registration process for a fictional book titled 'The Artificial Harp,' authored by 'Jane Doe' in the year 2023. 
Fill out the public-facing form with placeholder details provided in the instruction. 
Progress through the steps up to the fee payment screen (but do not pay). 
Report back the total registration fee and the summary of the input details shown on the final review screen.

## Task-Specific Constraints
- Must navigate to the U.S. Copyright Office website (copyright.gov).
- Must locate and access the registration portal for literary works.
- Must correctly input the provided details for the fictional book.
- Must progress through the registration steps up to the fee payment screen without completing payment.
- Must accurately report the total registration fee displayed on the fee payment screen.
- Must provide a clear summary of the input details shown on the final review screen.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to the U.S. Copyright Office website?
- Did the agent locate and access the registration portal for literary works?
- Did the agent correctly input the provided details for the fictional book?
- Did the agent progress to the fee payment screen without completing payment?
- Did the agent accurately report the total registration fee and summarize the input details?

### Step 2: Dimension Scoring

#### A. Navigation Accuracy (0.35)
Measures whether the agent successfully navigated to the correct website and registration portal.

5 — Successfully navigated to the correct website and registration portal without errors.
4 — Navigated to the correct website and portal but with minor delays or extra steps.
3 — Reached the correct website but struggled to locate the portal.
2 — Attempted navigation but did not reach the correct portal.
1 — Did not navigate to the correct website or portal.

#### B. Input Accuracy (0.30)
Measures whether the agent correctly input all the provided details for the fictional book.

5 — All details were input correctly without any errors.
4 — Most details were input correctly, with only minor errors.
3 — Some details were input correctly, but key information was missing or incorrect.
2 — Few details were input correctly, with significant errors or omissions.
1 — Did not input any details correctly.

#### C. Fee Reporting Accuracy (0.20)
Measures whether the agent accurately reported the total registration fee displayed on the fee payment screen.

5 — Fee was reported accurately and clearly.
4 — Fee was reported with minor inaccuracies or formatting issues.
3 — Fee was reported but with significant inaccuracies or omissions.
2 — Fee was mentioned but mostly incorrect or unclear.
1 — Fee was not reported at all.

#### D. Output Organization (0.15)
Measures whether the agent's final response was well-structured and easy to understand.

5 — Response was well-structured, clear, and easy to follow.
4 — Response was mostly clear with minor formatting issues.
3 — Response was understandable but poorly organized or unclear in parts.
2 — Response was disorganized and difficult to follow.
1 — Response was completely incoherent or absent.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "navigation_accuracy": <1-5>,
  "input_accuracy": <1-5>,
  "fee_reporting_accuracy": <1-5>,
  "output_organization": <1-5>,
  "dimension_reasoning": {{
    "navigation_accuracy": "<one sentence citing specific evidence>",
    "input_accuracy": "<one sentence citing specific evidence>",
    "fee_reporting_accuracy": "<one sentence citing specific evidence>",
    "output_organization": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "navigation_accuracy": 0.35,
    "input_accuracy": 0.30,
    "fee_reporting_accuracy": 0.20,
    "output_organization": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())