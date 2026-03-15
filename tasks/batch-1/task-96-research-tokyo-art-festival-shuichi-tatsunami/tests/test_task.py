"""
LLM-as-judge evaluator for EvolveBench task-96.

Category: Daily Activities
Task: Research both the Art Festival at Tokyo University's affiliated secondary school and Shuichi Tatsunami, then create a Go
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


TASK_INSTRUCTION = """Research both the Art Festival at Tokyo University's affiliated secondary school and Shuichi Tatsunami, then create a Google Document summarizing each. In a separate tab of the same document, draft a concrete proposal for hosting a lecture or workshop by Shuichi Tatsunami at the Art Festival, including specific content and a schedule."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves researching two specific topics, creating a Google Doc with summaries, and drafting a formal event proposal in the same document.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Research targets: (1) Tokyo University's affiliated secondary school Art Festival, (2) Shuichi Tatsunami (artist)
- Output: Google Doc with two sections/tabs: summaries + proposal
- Proposal: must be concrete with specific content, schedule, and logistics
- Language: likely Japanese context

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent research both the Art Festival and Shuichi Tatsunami?
- Was a Google Doc created?
- Are both summary sections present?
- Is the proposal concrete with schedule and content?
- Are the two sections clearly separated?

### Step 2: Dimension Scoring

#### A. Research Quality (0.25)
Was both the Art Festival and Tatsunami researched?

5 — Specific findings on both: Art Festival dates/format, Tatsunami's work and background.
4 — Both researched but one less thoroughly.
3 — One thoroughly researched; other vague.
2 — Only one topic researched.
1 — No research.

#### B. Google Doc Creation (0.2)
Was a Google Doc created?

5 — Google Doc created with confirmed URL or sharing link.
4 — Google Doc created but link not confirmed.
3 — Doc described but creation unclear.
2 — Content written in response instead of Google Doc.
1 — No Google Doc.

#### C. Summary Completeness (0.25)
Are both summaries well-written?

5 — Both summaries present with specific facts, dates, and relevant details.
4 — Both present but one is thin.
3 — One summary complete; other minimal.
2 — Only one summary.
1 — No summaries.

#### D. Proposal Concreteness (0.3)
Is the proposal concrete with schedule and content?

5 — Specific workshop/lecture content, duration, schedule, equipment needs, and audience defined.
4 — Good proposal but one element (e.g. schedule) vague.
3 — Proposal present but generic without specifics.
2 — Vague proposal without schedule.
1 — No proposal.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "research_quality": <1-5>,
  "google_doc_creation": <1-5>,
  "summary_completeness": <1-5>,
  "proposal_concreteness": <1-5>,
  "dimension_reasoning": {{
    "research_quality": "<one sentence citing specific evidence>",
    "google_doc_creation": "<one sentence citing specific evidence>",
    "summary_completeness": "<one sentence citing specific evidence>",
    "proposal_concreteness": "<one sentence citing specific evidence>",
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "research_quality": 0.25,
    "google_doc_creation": 0.2,
    "summary_completeness": 0.25,
    "proposal_concreteness": 0.3,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())