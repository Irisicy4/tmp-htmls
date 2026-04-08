"""
LLM-as-judge evaluator for EvolveBench task.

Category: Daily Activities
Task: Investigate and diagnose Google Maps loading issue on Safari, determining whether it is due to browser compatibility or a temporary outage, and provide resolution options.
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


TASK_INSTRUCTION = """A user reports that their Google Maps directions fail to load when trying to navigate from Seattle to Portland, OR, on Safari. Investigate whether this issue is related to browser compatibility or a temporary outage. Use Google Maps Help Center, Safari support documentation, and Downdetector. Provide a diagnosis, affected browser versions, and recommended resolution options."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to investigate a reported issue with Google Maps directions failing to load on Safari. The agent must determine whether the issue is caused by browser compatibility problems or a temporary outage. The task requires the agent to use credible sources (Google Maps Help Center, Safari support documentation, and Downdetector) and provide a diagnosis, affected browser versions, and resolution options.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
A user reports that their Google Maps directions fail to load when trying to navigate from Seattle to Portland, OR, on Safari. Investigate whether this issue is related to browser compatibility or a temporary outage. Use Google Maps Help Center, Safari support documentation, and Downdetector. Provide a diagnosis, affected browser versions, and recommended resolution options.

## Task-Specific Constraints
- Must visit all three specified platforms: Google Maps Help Center, Safari support documentation, and Downdetector.
- Must identify whether the issue is browser compatibility-related or a temporary outage.
- Must specify affected Safari browser versions (if applicable).
- Must provide at least two resolution options.
- Output must be organized as a structured list or table.
- Must cite sources for all claims made.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to all three required platforms? Which ones were actually visited?
- Did the agent identify whether the issue is browser compatibility-related or a temporary outage?
- Are affected Safari browser versions clearly specified?
- Are at least two resolution options provided?
- Is the output organized as a structured list or table?
- Are all claims sourced and credible?

### Step 2: Dimension Scoring

#### A. Diagnosis Accuracy (0.35)
Measures whether the agent correctly identified the root cause of the issue.

5 — Clearly identifies the issue as browser compatibility-related or a temporary outage, with supporting evidence.
4 — Identifies the issue correctly but lacks supporting evidence or specificity.
3 — Partially identifies the issue but is incomplete or unclear.
2 — Incorrect diagnosis or minimal effort.
1 — No diagnosis provided.

#### B. Platform Coverage (0.30)
Measures whether the agent used all required platforms and incorporated relevant information.

5 — Uses all three specified platforms and incorporates relevant data from each.
4 — Uses two platforms thoroughly but misses one.
3 — Uses one platform thoroughly or all three superficially.
2 — Minimal use of platforms; misses key sources.
1 — No use of required platforms.

#### C. Specificity of Details (0.20)
Measures whether the agent provides detailed information, including affected browser versions and resolution options.

5 — Specifies affected browser versions and provides at least two detailed resolution options.
4 — Specifies browser versions but resolution options lack detail or vice versa.
3 — Provides minimal detail on browser versions and resolution options.
2 — Vague or incomplete details provided.
1 — No details provided.

#### D. Output Structure and Credibility (0.15)
Measures whether the output is well-organized and sources are credible.

5 — Output is structured as a clear list or table, with all claims sourced.
4 — Output is structured but some claims lack sourcing or clarity.
3 — Output is minimally organized; sourcing is incomplete.
2 — Poorly structured output; minimal sourcing.
1 — No structure or credible sourcing.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "diagnosis_accuracy": <1-5>,
  "platform_coverage": <1-5>,
  "specificity_of_details": <1-5>,
  "output_structure_and_credibility": <1-5>,
  "dimension_reasoning": {{
    "diagnosis_accuracy": "<one sentence citing specific evidence>",
    "platform_coverage": "<one sentence citing specific evidence>",
    "specificity_of_details": "<one sentence citing specific evidence>",
    "output_structure_and_credibility": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "diagnosis_accuracy": 0.35,
    "platform_coverage": 0.30,
    "specificity_of_details": 0.20,
    "output_structure_and_credibility": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())