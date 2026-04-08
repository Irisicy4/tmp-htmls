"""
LLM-as-judge evaluator for EvolveBench task.

Category: Medical & Clinical & Bio
Task: Research and compare clinical guidelines for acute ischemic stroke treatment from three organizations and present findings in a comparison table.
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


TASK_INSTRUCTION = """Research and compare the current clinical guidelines for the treatment of acute ischemic stroke from the American Heart Association (https://www.heart.org/), the European Stroke Organisation (https://eso-stroke.org/), and the World Health Organization (https://www.who.int/). For each source, document the recommended treatment protocol for the hyperacute phase (within 24 hours of onset), including the use of thrombolytics, thrombectomy, antiplatelet therapy, and blood pressure management. Present your findings in a comparison table with columns: Source, Criteria for Thrombolysis, Criteria for Thrombectomy, Antiplatelet Guidance, Blood Pressure Management Strategies."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to research and compare clinical guidelines for the hyperacute treatment of acute ischemic stroke from three organizations: the American Heart Association, the European Stroke Organisation, and the World Health Organization. The deliverable is a comparison table summarizing the guidelines for thrombolysis, thrombectomy, antiplatelet therapy, and blood pressure management.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Research and compare the current clinical guidelines for the treatment of acute ischemic stroke from the American Heart Association (https://www.heart.org/), the European Stroke Organisation (https://eso-stroke.org/), and the World Health Organization (https://www.who.int/). For each source, document the recommended treatment protocol for the hyperacute phase (within 24 hours of onset), including the use of thrombolytics, thrombectomy, antiplatelet therapy, and blood pressure management. Present your findings in a comparison table with columns: Source, Criteria for Thrombolysis, Criteria for Thrombectomy, Antiplatelet Guidance, Blood Pressure Management Strategies.

## Task-Specific Constraints
- Must visit all three specified platforms: heart.org, eso-stroke.org, who.int.
- Must include treatment protocols for thrombolysis, thrombectomy, antiplatelet therapy, and blood pressure management.
- Output must be organized as a comparison table with the specified columns.
- Must accurately summarize hyperacute treatment guidelines within 24 hours of onset.
- Must cite specific criteria or thresholds for each treatment modality.
- Must ensure the information aligns with the latest guidelines published by each organization.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to the required platforms? Which ones were actually visited?
- Are treatment protocols for thrombolysis, thrombectomy, antiplatelet therapy, and blood pressure management present in the response?
- Is the output organized as a comparison table with the specified columns?
- Are specific thresholds or criteria for each treatment modality included and accurate?
- Does the information align with the latest guidelines published by each organization?

### Step 2: Dimension Scoring

#### A. Deliverable Accuracy (0.35)
Measures whether the comparison table is correct, complete, and accurately summarizes the guidelines.

5 — All treatment protocols are correctly summarized with specific criteria for thrombolysis, thrombectomy, antiplatelet therapy, and blood pressure management.
4 — Most treatment protocols are correctly summarized, but minor details or criteria are missing.
3 — Some treatment protocols are summarized, but significant details or criteria are missing.
2 — Few treatment protocols are summarized, with major inaccuracies or omissions.
1 — No treatment protocols are summarized or completely inaccurate.

#### B. Coverage of Sources (0.30)
Measures whether the agent visited all required platforms and included guidelines from all three organizations.

5 — Guidelines from all three organizations (AHA, ESO, WHO) are included.
4 — Guidelines from two organizations are included, with minor omissions.
3 — Guidelines from one organization are included, or guidelines are incomplete.
2 — Minimal coverage of sources, with major omissions.
1 — No guidelines from any organization are included.

#### C. Depth and Specificity (0.20)
Measures whether the response includes detailed criteria, thresholds, or specific guidance for each treatment modality.

5 — Includes detailed criteria and thresholds for all four treatment modalities.
4 — Includes detailed criteria for most modalities, with minor omissions.
3 — Includes some criteria, but lacks depth or specificity for multiple modalities.
2 — Minimal criteria or thresholds are included, with significant gaps.
1 — No detailed criteria or thresholds are included.

#### D. Output Structure and Credibility (0.15)
Measures whether the response is well-organized, formatted as a table, and cites credible sources.

5 — Output is well-organized as a table, with credible citations for all data.
4 — Output is mostly well-organized, with minor formatting or citation issues.
3 — Output is partially organized, with noticeable formatting or credibility issues.
2 — Output is poorly organized, with major formatting or credibility issues.
1 — Output is completely disorganized or lacks credibility.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{
  "evidence_summary": "The agent visited all required platforms and included guidelines from AHA, ESO, and WHO. The comparison table is mostly complete but lacks specific thresholds for thrombectomy and antiplatelet therapy.",
  "deliverable_accuracy": 4,
  "coverage_of_sources": 5,
  "depth_and_specificity": 3,
  "output_structure_and_credibility": 4,
  "dimension_reasoning": {
    "deliverable_accuracy": "Most treatment protocols are summarized correctly, but minor details are missing.",
    "coverage_of_sources": "All three platforms were visited and guidelines included.",
    "depth_and_specificity": "Some criteria are detailed, but thresholds for thrombectomy and antiplatelet therapy are incomplete.",
    "output_structure_and_credibility": "The response is well-organized as a table with credible citations."
  },
  "overall_score": 4.0,
  "passed": true
}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "deliverable_accuracy": 0.35,
    "coverage_of_sources": 0.30,
    "depth_and_specificity": 0.20,
    "output_structure_and_credibility": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())