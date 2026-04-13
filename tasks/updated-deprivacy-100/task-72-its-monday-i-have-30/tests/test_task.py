"""
LLM-as-judge evaluator for EvolveBench task-72.

Category: Finance & Economics
Task: Summarize the major tech shifts from last week and simulate 3 tough investor questions
      based on those trends, to be ready before an all-hands meeting.
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


TASK_INSTRUCTION = """It's Monday. I have 30 minutes before the all-hands meeting. Summarize the major tech shifts from last week and simulate 3 tough questions investors might ask us today based on those trends."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to research and summarize major technology news and developments from the past week, then generate 3 challenging investor-style questions grounded in those real developments — all concisely enough for a 30-minute pre-meeting briefing.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Must identify major tech developments from the past week (recent, not evergreen content)
- Must produce exactly 3 simulated investor questions
- Questions must directly reference the identified tech shifts (not generic questions)
- Output must be concise and structured for a 30-minute pre-meeting scan
- Questions should be genuinely challenging (probing assumptions, risks, or strategic implications)

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent search for recent tech news? What sources or topics were identified?
- Are the tech developments named specifically (company names, product launches, regulatory actions, etc.)?
- How many investor questions were generated, and are they directly tied to the identified trends?
- Is the output structured and brief enough for a 30-minute pre-meeting read?

### Step 2: Dimension Scoring

#### A. Tech Shift Identification (0.35)
Did the agent identify specific, recent major tech developments from the past week?

5 — 3 or more distinct, named tech developments with specific companies, dates, or events (e.g. "OpenAI released GPT-X on [date]", "EU passed AI regulation", "Apple acquired...").
4 — 2-3 tech developments identified with reasonable specificity.
3 — Tech topics mentioned but vague, or clearly not from the past week (evergreen topics).
2 — Only one development identified, or developments are generic without specifics.
1 — No tech shifts identified or researched.

#### B. Investor Question Quality (0.30)
Are exactly 3 tough, specific investor questions provided that are grounded in the identified trends?

5 — Exactly 3 questions, each directly referencing a named tech development; questions probe risks, competitive dynamics, or strategic implications in a way that challenges the presenter.
4 — 3 questions provided, mostly tied to the trends but some are slightly generic.
3 — 3 questions provided but generic, or not clearly tied to the specific tech shifts identified.
2 — Fewer than 3 questions, or questions are low-quality/softball.
1 — No investor questions generated.

#### C. Recency and Specificity (0.20)
Is the content clearly from the past week with concrete details?

5 — Content references specific dates, events, or announcements from the past week; shows evidence of actual web research.
4 — Recent content with implicit timeframe (e.g. references to recent announcements without specific dates).
3 — Some recent content mixed with older or evergreen material.
2 — Content appears outdated or could apply to any time period.
1 — No evidence of recent research; entirely generic.

#### D. Briefing Readability (0.15)
Is the output well-organized and scannable within 30 minutes?

5 — Clear sections (tech summary + investor questions), concise bullet points or headers, actionable takeaways; digestible in under 5 minutes.
4 — Organized output with some structure, but slightly verbose.
3 — Information present but dense or disorganized.
2 — Poorly structured wall of text.
1 — No structure; output unusable as a briefing.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "tech_shift_identification": <1-5>,
  "investor_question_quality": <1-5>,
  "recency_and_specificity": <1-5>,
  "briefing_readability": <1-5>,
  "dimension_reasoning": {{
    "tech_shift_identification": "<one sentence citing specific evidence>",
    "investor_question_quality": "<one sentence citing specific evidence>",
    "recency_and_specificity": "<one sentence citing specific evidence>",
    "briefing_readability": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "tech_shift_identification": 0.35,
    "investor_question_quality": 0.30,
    "recency_and_specificity": 0.20,
    "briefing_readability": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())