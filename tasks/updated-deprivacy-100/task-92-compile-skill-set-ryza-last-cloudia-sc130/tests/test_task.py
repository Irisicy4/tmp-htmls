"""
LLM-as-judge evaluator for EvolveBench task-92.

Category: Daily Activities
Task: For the Last Cloudia x Atelier Ryza collaboration, please compile the recommended skill set for Ryza with an SC limit of
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


TASK_INSTRUCTION = """For the Last Cloudia x Atelier Ryza collaboration, please compile the recommended skill set for Ryza with an SC limit of 130, focusing on break specialization."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves researching the mobile game Last Cloudia's collaboration with Atelier Ryza to find optimal skill builds for the Ryza character under specific constraints (SC 130, break spec).

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Game: Last Cloudia (ラストクラウディア) collaboration with Atelier Ryza
- Character: Ryza
- Constraint: SC (Skill Cost) limit of 130
- Specialization: break (ブレイク) focus
- Source: community guides, wikis, or YouTube — not just general advice

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent search for Last Cloudia Ryza build guides?
- What specific skills were recommended?
- Do the recommended skills fit within SC 130?
- Is break specialization the focus?
- Were community sources (wiki, Reddit, YouTube) used?

### Step 2: Dimension Scoring

#### A. Source Research (0.25)
Did the agent find relevant community sources?

5 — Found wiki, forum, or YouTube guide specifically for Last Cloudia Ryza collab build.
4 — Found partial sources.
3 — Found general Last Cloudia build guides without Ryza-specific focus.
2 — Only general game info without build data.
1 — No research.

#### B. Skill Specificity (0.35)
Are specific skills listed?

5 — Named skills with SC costs listed; total SC within 130 confirmed.
4 — Skills named but SC costs not totaled.
3 — Skill categories suggested without specific skill names.
2 — Very vague 'use attack skills' without specifics.
1 — No skill information.

#### C. Sc Compliance (0.25)
Do the recommended skills fit within SC 130?

5 — Total SC calculated and confirmed ≤130.
4 — Skills appropriate for SC 130 but total not explicitly confirmed.
3 — SC constraint acknowledged but compliance unclear.
2 — SC constraint ignored.
1 — Skills clearly exceed SC 130.

#### D. Break Focus (0.15)
Is break specialization the focus?

5 — Break-focused skills prioritized with explanation of why they synergize with Ryza.
4 — Break skills included but focus not clearly stated.
3 — Mix of break and general skills without prioritization.
2 — Generic build not focused on break.
1 — No break focus.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "source_research": <1-5>,
  "skill_specificity": <1-5>,
  "sc_compliance": <1-5>,
  "break_focus": <1-5>,
  "dimension_reasoning": {{
    "source_research": "<one sentence citing specific evidence>",
    "skill_specificity": "<one sentence citing specific evidence>",
    "sc_compliance": "<one sentence citing specific evidence>",
    "break_focus": "<one sentence citing specific evidence>",
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "source_research": 0.25,
    "skill_specificity": 0.35,
    "sc_compliance": 0.25,
    "break_focus": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())