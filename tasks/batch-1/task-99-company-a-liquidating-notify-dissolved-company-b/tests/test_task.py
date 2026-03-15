"""
LLM-as-judge evaluator for EvolveBench task-99.

Category: Daily Activities
Task: Company A is preparing for liquidation and dissolution. It has discovered that it still has accounts payable to Company 
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


TASK_INSTRUCTION = """Company A is preparing for liquidation and dissolution. It has discovered that it still has accounts payable to Company B on its books. However, Company B has already been dissolved. After Company A begins its liquidation process, does it still have an obligation to notify Company B? Please answer based on Chinese law."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves answering a specific Chinese corporate law question about liquidation obligations when a creditor company has already been dissolved.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Jurisdiction: Chinese law specifically
- Topic: liquidation notification obligations when creditor is dissolved
- Required: cite specific Chinese legal provisions (Company Law, Liquidation regulations)
- Answer: must address the notification obligation and what Company A should do instead

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent research Chinese law on this specific scenario?
- What specific legal provisions were cited (Company Law articles)?
- Is the notification obligation clearly addressed?
- What practical guidance is provided for Company A?

### Step 2: Dimension Scoring

#### A. Legal Research (0.25)
Did the agent research the relevant Chinese legal framework?

5 — Researched Chinese Company Law, liquidation regulations, and relevant judicial interpretations.
4 — Found relevant law but search was less comprehensive.
3 — General Chinese corporate law found without specific liquidation provisions.
2 — Generic legal answer without Chinese law research.
1 — No legal research.

#### B. Legal Accuracy (0.35)
Is the legal analysis accurate under Chinese law?

5 — Correct analysis: under Chinese Company Law, notification obligation exists but must be directed to known creditors; when creditor is dissolved, obligation typically transforms to public announcement; specific articles cited.
4 — Mostly correct with minor gaps.
3 — Generally correct direction but missing key provisions.
2 — Partially correct but significant errors.
1 — Incorrect or no legal analysis.

#### C. Citation Quality (0.25)
Are specific legal provisions cited?

5 — Specific articles from Chinese Company Law, liquidation procedures, or relevant regulations cited.
4 — Some articles cited.
3 — Legal framework referenced without specific articles.
2 — Vague reference to Chinese law.
1 — No citations.

#### D. Practical Guidance (0.15)
Is practical guidance provided for Company A?

5 — Clear next steps: what Company A should do (public announcement, hold funds, court guidance).
4 — Some guidance provided.
3 — Legal conclusion reached without practical next steps.
2 — Very vague guidance.
1 — No guidance.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "legal_research": <1-5>,
  "legal_accuracy": <1-5>,
  "citation_quality": <1-5>,
  "practical_guidance": <1-5>,
  "dimension_reasoning": {{
    "legal_research": "<one sentence citing specific evidence>",
    "legal_accuracy": "<one sentence citing specific evidence>",
    "citation_quality": "<one sentence citing specific evidence>",
    "practical_guidance": "<one sentence citing specific evidence>",
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "legal_research": 0.25,
    "legal_accuracy": 0.35,
    "citation_quality": 0.25,
    "practical_guidance": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())