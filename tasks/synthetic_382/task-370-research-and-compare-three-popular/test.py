"""
LLM-as-judge evaluator for EvolveBench task.

Category: Design
Task: Research and compare three popular UI design tools on collaborative workflows, exporting options, and pricing tiers.
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


TASK_INSTRUCTION = """Research and compare three popular UI design tools (e.g., Figma, Sketch, Adobe XD) on their features relevant to collaborative workflows, exporting options, and pricing tiers. Gather your information from their respective official websites and a third-party review site for unbiased comparisons."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to research and compare three popular UI design tools (Figma, Sketch, Adobe XD) on collaborative workflows, exporting options, and pricing tiers. The domain is UI/UX design tools, and a successful completion requires gathering accurate, structured, and unbiased information from official websites and third-party review sources.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Research and compare three popular UI design tools (e.g., Figma, Sketch, Adobe XD) on their features relevant to collaborative workflows, exporting options, and pricing tiers. Gather your information from their respective official websites and a third-party review site for unbiased comparisons.

## Task-Specific Constraints
- Must visit the official websites of Figma, Sketch, and Adobe XD.
- Must include pricing tier data for all three tools.
- Must compare features related to collaborative workflows and exporting options.
- Output must be organized as a structured table or list.
- Must include at least one source from a third-party review site for unbiased comparisons.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to the required platforms? Which ones were actually visited?
- Are pricing tier details included for all three tools?
- Are features related to collaborative workflows and exporting options compared?
- Is the output organized as a structured table or list?
- Is there evidence of using a third-party review site for unbiased comparisons?

### Step 2: Dimension Scoring

#### A. Deliverable Accuracy (0.35)
Measures whether the agent's output correctly and completely addresses the task requirements.

5 — Includes accurate and complete comparisons for all three tools across collaborative workflows, exporting options, and pricing tiers.
4 — Includes mostly accurate comparisons but may miss minor details.
3 — Includes partial comparisons but lacks significant details.
2 — Includes mostly incorrect or incomplete comparisons.
1 — No meaningful comparisons provided.

#### B. Coverage of Sources (0.30)
Measures whether the agent used all required platforms and included third-party sources.

5 — Navigated all three official websites and included at least one third-party review source.
4 — Navigated at least two official websites and included a third-party review source.
3 — Navigated at least two official websites but missed third-party sources.
2 — Navigated only one official website or missed key sources.
1 — Did not navigate any required sources.

#### C. Depth of Information (0.25)
Measures the specificity and depth of the comparisons.

5 — Provides detailed feature descriptions, pricing tiers, and specific exporting/collaboration details for all tools.
4 — Provides mostly detailed descriptions but lacks minor specifics.
3 — Provides basic descriptions but lacks depth or specificity.
2 — Provides vague or incomplete descriptions.
1 — Provides no meaningful descriptions.

#### D. Output Structure and Credibility (0.10)
Measures the organization and credibility of the output.

5 — Output is well-organized (e.g., table or structured list) and cites credible sources.
4 — Output is mostly organized but may lack minor structural elements or clear citations.
3 — Output is partially organized but lacks clarity or credible citations.
2 — Output is poorly organized or lacks credibility.
1 — Output is completely disorganized or not credible.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "deliverable_accuracy": <1-5>,
  "coverage_of_sources": <1-5>,
  "depth_of_information": <1-5>,
  "output_structure_and_credibility": <1-5>,
  "dimension_reasoning": {{
    "deliverable_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_sources": "<one sentence citing specific evidence>",
    "depth_of_information": "<one sentence citing specific evidence>",
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
    "depth_of_information": 0.25,
    "output_structure_and_credibility": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())