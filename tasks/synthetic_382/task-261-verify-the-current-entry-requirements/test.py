"""
LLM-as-judge evaluator for EvolveBench task.

Category: Travel & Planning
Task: Verify entry requirements for U.S. citizens traveling to Australia using official sources.
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


TASK_INSTRUCTION = """Verify the current entry requirements for U.S. citizens traveling to Australia. Check official government sources for visa requirements, vaccination rules, and any COVID-19 restrictions using Australia.gov.au, SmartTraveller.gov.au, and CDC.gov."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to verify entry requirements for U.S. citizens traveling to Australia. This includes visa requirements, vaccination rules, and COVID-19 restrictions. The agent must use official government sources (Australia.gov.au, SmartTraveller.gov.au, and CDC.gov) and provide accurate, structured information.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Verify the current entry requirements for U.S. citizens traveling to Australia. Check official government sources for visa requirements, vaccination rules, and any COVID-19 restrictions using Australia.gov.au, SmartTraveller.gov.au, and CDC.gov.

## Task-Specific Constraints
- Must visit all three specified platforms: Australia.gov.au, SmartTraveller.gov.au, and CDC.gov.
- Must provide visa requirements for U.S. citizens traveling to Australia.
- Must include vaccination rules and any COVID-19 restrictions.
- Output must be organized as a structured list or table.
- Information must be sourced and accurate, with references to the platforms used.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to the required platforms? Which ones were actually visited?
- Are visa requirements for U.S. citizens clearly stated and accurate?
- Are vaccination rules and COVID-19 restrictions included and sourced?
- Is the output organized as a structured list or table?
- Are all claims backed by credible references to the specified platforms?

### Step 2: Dimension Scoring

#### A. Deliverable Accuracy (0.35)
Measures whether the agent's output is correct and complete.

5 — All required information (visa, vaccination, COVID-19 restrictions) is accurate and complete.
4 — Minor omissions or inaccuracies in one area.
3 — Partial information provided; significant omissions or inaccuracies.
2 — Mostly incorrect or missing information.
1 — No relevant information provided.

#### B. Coverage of Sources (0.30)
Measures whether the agent used all required platforms and included their information.

5 — All three platforms were used, and their information is included.
4 — Two platforms were used, with minor omissions from one.
3 — At least one platform was used, but significant omissions exist.
2 — Platforms were visited but not used effectively.
1 — No platforms were visited or used.

#### C. Specificity and Detail (0.20)
Measures the depth and specificity of the information provided.

5 — Detailed information with specific rules, numbers, and examples.
4 — Mostly detailed, with minor omissions or lack of specificity.
3 — General information provided, but lacks depth or examples.
2 — Vague or superficial information.
1 — No meaningful details provided.

#### D. Output Structure and Credibility (0.15)
Measures whether the output is well-organized and sourced from credible platforms.

5 — Information is structured as a clear list or table, with sources cited.
4 — Mostly well-organized, with minor structural issues or missing citations.
3 — Acceptable structure, but lacks clarity or citations.
2 — Poorly organized or unclear output.
1 — No structure or credible sourcing.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "deliverable_accuracy": <1-5>,
  "coverage_of_sources": <1-5>,
  "specificity_and_detail": <1-5>,
  "output_structure_and_credibility": <1-5>,
  "dimension_reasoning": {{
    "deliverable_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_sources": "<one sentence citing specific evidence>",
    "specificity_and_detail": "<one sentence citing specific evidence>",
    "output_structure_and_credibility": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "deliverable_accuracy": 0.35,
    "coverage_of_sources": 0.30,
    "specificity_and_detail": 0.20,
    "output_structure_and_credibility": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())