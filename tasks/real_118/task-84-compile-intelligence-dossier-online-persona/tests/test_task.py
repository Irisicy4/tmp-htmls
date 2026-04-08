"""
LLM-as-judge evaluator for EvolveBench task-84.

Category: Marketing & Analytics
Task: Compile a comprehensive intelligence dossier on the target online persona, covering identity mapping, content forensics,
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


TASK_INSTRUCTION = """Compile a comprehensive intelligence dossier on the target online persona, covering identity mapping, content forensics, platform and network audit, psychological profiling, and asset valuation."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves researching a specific online persona across platforms and compiling a structured intelligence report. This is a legitimate competitive intelligence or OSINT research task for marketing/analytics purposes.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Scope: public information only — social media profiles, public posts, published content
- Coverage: identity mapping, content analysis, platform presence, network connections, estimated reach/value
- Structure: organized report with all five sections
- Ethics: public information only; no private data, no hacking

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent research the target persona across multiple platforms?
- Were all five sections covered (identity, content, platform, network, valuation)?
- Is the information clearly sourced from public data?
- Is the report well-organized with each section distinct?
- Is asset/reach valuation grounded in methodology?

### Step 2: Dimension Scoring

#### A. Research Breadth (0.25)
Did the agent research across multiple platforms?

5 — Searched 4+ platforms (Twitter/X, LinkedIn, Instagram, YouTube, etc.) with specific findings from each.
4 — 2-3 platforms researched.
3 — 1-2 platforms with limited findings.
2 — Single source research.
1 — No research.

#### B. Section Completeness (0.35)
Were all five dossier sections covered?

5 — All five: identity mapping, content forensics, platform audit, network analysis, asset valuation — all present with substance.
4 — Four of five sections.
3 — Three of five sections.
2 — One or two sections.
1 — No structured sections.

#### C. Insight Depth (0.25)
Is the analysis insightful beyond surface facts?

5 — Deep analysis: content patterns, audience psychology, influence network structure, monetization signals.
4 — Good depth in most sections.
3 — Surface-level facts without analytical insight.
2 — Mostly a list of profile details.
1 — No analytical depth.

#### D. Report Structure (0.15)
Is the dossier well-organized and professional?

5 — Clear sections with headers, findings, and methodology notes; professional tone.
4 — Well-organized but missing some formatting polish.
3 — Content present but loosely organized.
2 — Wall of text.
1 — No structure.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "research_breadth": <1-5>,
  "section_completeness": <1-5>,
  "insight_depth": <1-5>,
  "report_structure": <1-5>,
  "dimension_reasoning": {{
    "research_breadth": "<one sentence citing specific evidence>",
    "section_completeness": "<one sentence citing specific evidence>",
    "insight_depth": "<one sentence citing specific evidence>",
    "report_structure": "<one sentence citing specific evidence>",
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "research_breadth": 0.25,
    "section_completeness": 0.35,
    "insight_depth": 0.25,
    "report_structure": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())