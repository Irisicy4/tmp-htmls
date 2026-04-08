"""
LLM-as-judge evaluator for EvolveBench task.

Category: Design
Task: Verify whether WCAG 2.1 still recommends high-contrast text in web design by cross-checking across three reputable sources.
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


TASK_INSTRUCTION = """Verify whether the latest accessibility guidelines (WCAG 2.1) still recommend using high-contrast text in web design as of today. Cross-check this information across the W3C official website and two additional reputable sources."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to verify whether WCAG 2.1 guidelines still recommend high-contrast text in web design. This involves cross-checking information from the W3C official website and two additional reputable sources (webaim.org and smashingmagazine.com). A successful completion requires the agent to provide accurate, sourced information and confirm consistency across the platforms.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Verify whether the latest accessibility guidelines (WCAG 2.1) still recommend using high-contrast text in web design as of today. Cross-check this information across the W3C official website and two additional reputable sources.

## Task-Specific Constraints
- Must visit w3.org, webaim.org, and smashingmagazine.com.
- Must explicitly confirm whether high-contrast text is recommended in WCAG 2.1.
- Must provide sourced evidence from all three platforms.
- Output must summarize findings in a structured format (e.g., bullet points or table).
- Must address whether the recommendation aligns across all platforms.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to w3.org, webaim.org, and smashingmagazine.com? Which ones were actually visited?
- Does the response explicitly confirm whether high-contrast text is recommended in WCAG 2.1?
- Are sourced references from all three platforms present in the response?
- Is the output organized in a structured format (e.g., bullet points or table)?
- Are the findings consistent across the sources?

### Step 2: Dimension Scoring

#### A. Recommendation Accuracy (0.35)
Measures whether the agent correctly identifies WCAG 2.1's recommendation on high-contrast text.

5 — Clearly confirms WCAG 2.1 recommendation with sourced evidence from all platforms.
4 — Confirms recommendation but lacks complete sourcing or minor inconsistencies.
3 — Partially confirms recommendation but misses key details or sources.
2 — Incorrect or incomplete confirmation with major gaps.
1 — Fails to confirm or completely wrong.

#### B. Platform Coverage (0.30)
Measures whether the agent visited all required platforms and used their information.

5 — Uses sourced evidence from all three platforms (w3.org, webaim.org, smashingmagazine.com).
4 — Uses evidence from two platforms but misses one.
3 — Uses evidence from only one platform.
2 — Attempts but fails to use any platform effectively.
1 — Does not use any platform.

#### C. Depth of Evidence (0.25)
Measures the level of detail and specificity in the agent's response.

5 — Provides detailed evidence, including quotes or specific references from each platform.
4 — Provides evidence but lacks depth or specificity in some areas.
3 — Provides minimal evidence with limited detail.
2 — Provides vague or incomplete evidence.
1 — Provides no evidence.

#### D. Output Structure and Credibility (0.10)
Measures the organization and credibility of the response.

5 — Response is well-structured and cites credible sources clearly.
4 — Response is structured but lacks clarity or credibility in some areas.
3 — Response is usable but poorly organized or unclear.
2 — Response is disorganized or lacks credibility.
1 — Response is completely unstructured or not credible.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "recommendation_accuracy": <1-5>,
  "platform_coverage": <1-5>,
  "depth_of_evidence": <1-5>,
  "output_structure_and_credibility": <1-5>,
  "dimension_reasoning": {{
    "recommendation_accuracy": "<one sentence citing specific evidence>",
    "platform_coverage": "<one sentence citing specific evidence>",
    "depth_of_evidence": "<one sentence citing specific evidence>",
    "output_structure_and_credibility": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "recommendation_accuracy": 0.35,
    "platform_coverage": 0.30,
    "depth_of_evidence": 0.25,
    "output_structure_and_credibility": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())