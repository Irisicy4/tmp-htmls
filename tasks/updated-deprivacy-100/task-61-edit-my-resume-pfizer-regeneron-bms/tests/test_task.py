"""
LLM-as-judge evaluator for EvolveBench task-61.

Category: Daily Activities
Task: Can you edit my resume? I don't get any interviews. I usually apply to Pfizer, Regeneron, BMS, and E...
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


TASK_INSTRUCTION = """Can you edit my resume? I don't get any interviews. I usually apply to Pfizer, Regeneron, BMS, and Eli Lilly."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves reviewing and editing a user's resume to improve interview success rates for pharmaceutical/biotech companies (Pfizer, Regeneron, BMS, Eli Lilly).

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Target companies: Pfizer, Regeneron, Bristol Myers Squibb, Eli Lilly — all pharma/biotech
- The agent must work with an actual resume (either uploaded or obtained from the user)
- Edits must be substantive — not just surface formatting changes
- Agent should align resume language with pharma industry keywords and job requirements

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent access the user's resume?
- What specific edits were made (or proposed)?
- Were pharma-specific keywords and requirements addressed?
- Was the resume tailored to Pfizer/Regeneron/BMS/Eli Lilly specifically?
- Did the agent provide actionable, substantive changes?

### Step 2: Dimension Scoring

#### A. Resume Access (0.2)
Did the agent access and work with the actual resume?

5 — Agent accessed and read the resume content and made specific edits.
4 — Agent referenced resume content but edits were partial.
3 — Agent provided generic advice without accessing resume content.
2 — Agent asked for the resume without proceeding.
1 — No resume engagement.

#### B. Edit Quality (0.35)
Are the edits substantive and high-quality?

5 — Specific rewrites of bullet points, summary, and skills sections with strong action verbs and quantified achievements.
4 — Good edits but not all sections addressed.
3 — Surface-level edits (formatting, grammar) without substantive content improvement.
2 — Generic advice rather than actual edits.
1 — No edits made.

#### C. Pharma Alignment (0.3)
Is the resume tailored to pharma/biotech companies?

5 — Incorporates pharma-specific keywords (GMP, clinical trials, regulatory, drug development) and company culture alignment.
4 — Some industry-specific language but not fully targeted.
3 — Generic corporate resume improvements without pharma focus.
2 — Minimal industry awareness.
1 — No pharma/biotech tailoring.

#### D. Interview Strategy (0.15)
Does the agent explain why changes help get interviews?

5 — Clear explanation of how each major change addresses ATS keywords or recruiter priorities at target companies.
4 — Some explanation provided.
3 — Changes made without explanation.
2 — Vague rationale.
1 — No rationale.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "resume_access": <1-5>,
  "edit_quality": <1-5>,
  "pharma_alignment": <1-5>,
  "interview_strategy": <1-5>,
  "dimension_reasoning": {{
    "resume_access": "<one sentence citing specific evidence>",
    "edit_quality": "<one sentence citing specific evidence>",
    "pharma_alignment": "<one sentence citing specific evidence>",
    "interview_strategy": "<one sentence citing specific evidence>",
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "resume_access": 0.2,
    "edit_quality": 0.35,
    "pharma_alignment": 0.3,
    "interview_strategy": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())