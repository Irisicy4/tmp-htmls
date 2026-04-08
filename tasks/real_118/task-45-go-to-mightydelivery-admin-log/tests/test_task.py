"""
LLM-as-judge evaluator for EvolveBench task-45.

Category: Daily Activities
Task: Go to https://mightydelivery.meetmighty.com/admin and log in, navigate
      to the deliveries section and extract only today's scheduled items,
      presenting them in a structured table.
"""

import os, json, re

TASK_INSTRUCTION = (
    "Go to https://mightydelivery.meetmighty.com/admin and log in with email: "
    "admin@mightydelivery.com and password: 12345678. Navigate to the deliveries "
    "or orders section and extract only the items scheduled for today. Present them "
    "in a structured table with order ID, recipient, address, and scheduled delivery time."
)
PASS_THRESHOLD = 3.0

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully logged into an admin panel, filtered for today's deliveries, and presented the results clearly.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Platform: mightydelivery.meetmighty.com/admin — must log in with provided credentials
- Filter: today's deliveries only — not all orders, not past/future orders
- Output format: structured table with order ID, recipient, address, and delivery time
- If no deliveries are scheduled for today, the agent should explicitly state this

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to the admin panel and log in? Cite evidence.
- Did the agent locate the deliveries/orders section?
- Was a date filter applied to show only today's items?
- What data was returned? Is it clearly for today's date?
- Is the output structured as a table with the required fields?

### Step 2: Dimension Scoring

#### A. Login & Navigation
Did the agent successfully log in and reach the correct section?

5 — Agent logged in with provided credentials and navigated to deliveries/orders section.
4 — Agent logged in but had difficulty finding the correct section.
3 — Agent reached the site but could not log in or reached the wrong section.
2 — Agent attempted to navigate but encountered login or access errors.
1 — No navigation attempted.

#### B. Date Filtering
Did the agent correctly filter for today's deliveries only?

5 — Agent explicitly filtered for today's date and confirmed the results are for today.
4 — Agent filtered but the date boundary is approximate or not explicitly confirmed.
3 — Agent retrieved some orders but did not clearly filter for today only.
2 — Agent retrieved all orders without any date filtering.
1 — No filtering attempted.

#### C. Data Completeness
Did the agent extract the required fields for each delivery?

5 — All four fields present: order ID, recipient, address, scheduled delivery time.
4 — Three of four fields present.
3 — Two of four fields present.
2 — Only one field present or data is very incomplete.
1 — No structured data extracted.

#### D. Output Clarity
Is the output presented clearly as a structured table?

5 — Clear table format with headers and one row per delivery item.
4 — Table-like format but minor formatting issues.
3 — Data present but in paragraph or list form rather than a table.
2 — Data present but very hard to parse.
1 — No structured output.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "login_navigation": <1-5>,
  "date_filtering": <1-5>,
  "data_completeness": <1-5>,
  "output_clarity": <1-5>,
  "dimension_reasoning": {{
    "login_navigation": "<one sentence citing specific evidence>",
    "date_filtering": "<one sentence citing specific evidence>",
    "data_completeness": "<one sentence citing specific evidence>",
    "output_clarity": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "login_navigation":  0.25,
    "date_filtering":    0.30,
    "data_completeness": 0.25,
    "output_clarity":    0.20,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())


def _extract_response(result):
    task_result = result.get("task_result") or ""
    if isinstance(task_result, str) and task_result.strip():
        return task_result
    for message in reversed(result.get("conversation") or []):
        if not isinstance(message, dict): continue
        if message.get("role") == "assistant":
            content = message.get("content") or ""
            if isinstance(content, str) and len(content) > 20:
                return content
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
    except Exception as e:
        return {"error": str(e)}

def _vote(votes):
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