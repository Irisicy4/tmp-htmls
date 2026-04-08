"""
LLM-as-judge evaluator for EvolveBench task.

Category: Medical & Clinical & Bio
Task: Investigate and document whether an herbal supplement interacts with warfarin, including the mechanism, severity, and recommended clinical action.
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


TASK_INSTRUCTION = """A patient using warfarin reports increased bruising and bleeding after starting a new herbal supplement. Investigate whether the reported symptoms could be caused by a known interaction between warfarin and the supplement. Use the FDA's drug interaction checker (https://www.fda.gov/), a clinical pharmacology database like Drugs.com (https://www.drugs.com/), and PubMed (https://pubmed.ncbi.nlm.nih.gov/) to determine if the herbal supplement inhibits or enhances warfarin metabolism. Document the mechanism of the interaction (e.g., via CYP enzyme interference), the severity of the interaction, and any recommended clinical action."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to investigate and document whether an herbal supplement interacts with warfarin, including the mechanism of interaction, severity, and recommended clinical action. This is a Medical & Clinical & Bio task requiring the agent to use multiple credible sources and provide a structured, evidence-based response.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
A patient using warfarin reports increased bruising and bleeding after starting a new herbal supplement. Investigate whether the reported symptoms could be caused by a known interaction between warfarin and the supplement. Use the FDA's drug interaction checker (https://www.fda.gov/), a clinical pharmacology database like Drugs.com (https://www.drugs.com/), and PubMed (https://pubmed.ncbi.nlm.nih.gov/) to determine if the herbal supplement inhibits or enhances warfarin metabolism. Document the mechanism of the interaction (e.g., via CYP enzyme interference), the severity of the interaction, and any recommended clinical action.

## Task-Specific Constraints
- Must visit at least 3 of the specified platforms (FDA, Drugs.com, PubMed).
- Must identify the mechanism of interaction (e.g., CYP enzyme inhibition or enhancement).
- Must classify the severity of the interaction (e.g., mild, moderate, severe).
- Must recommend a clinical action based on findings.
- Output must be structured as a clear, evidence-based explanation.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to the required platforms? Which ones were actually visited?
- Did the agent identify the mechanism of interaction (e.g., CYP enzyme interference)?
- Did the agent classify the severity of the interaction?
- Did the agent recommend a clinical action based on findings?
- Is the output structured as a clear, evidence-based explanation?

### Step 2: Dimension Scoring

#### A. Primary Deliverable Accuracy (0.35)
Measures whether the agent correctly identified the mechanism, severity, and clinical action.

5 — All three elements (mechanism, severity, clinical action) are correct and well-supported.
4 — Two elements are correct and well-supported; the third is partially correct.
3 — At least one element is correct and well-supported; others are partially correct or missing.
2 — Minimal accuracy; only one element is partially correct.
1 — No correct elements or completely incorrect.

#### B. Coverage of Required Platforms (0.30)
Measures whether the agent used all required platforms (FDA, Drugs.com, PubMed).

5 — All three platforms are used, and relevant data is extracted from each.
4 — Two platforms are used, and relevant data is extracted.
3 — At least one platform is used, with relevant data extracted.
2 — Platforms are visited, but no relevant data is extracted.
1 — No platforms are visited or used.

#### C. Depth of Explanation (0.20)
Measures the level of detail and specificity in the response.

5 — Response includes detailed mechanisms, specific severity classifications, and precise clinical recommendations.
4 — Response includes some detail but lacks full specificity in one area.
3 — Response is general but usable; lacks detail in multiple areas.
2 — Response is vague or incomplete; lacks significant detail.
1 — Response is entirely vague or absent.

#### D. Output Structure and Credibility (0.15)
Measures whether the response is well-structured and cites credible sources.

5 — Response is well-organized and cites all sources explicitly.
4 — Response is organized but cites only some sources explicitly.
3 — Response is somewhat organized but lacks explicit citations.
2 — Response is poorly organized and lacks citations.
1 — Response is disorganized and lacks credibility.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "primary_deliverable_accuracy": <1-5>,
  "coverage_of_required_platforms": <1-5>,
  "depth_of_explanation": <1-5>,
  "output_structure_and_credibility": <1-5>,
  "dimension_reasoning": {{
    "primary_deliverable_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_required_platforms": "<one sentence citing specific evidence>",
    "depth_of_explanation": "<one sentence citing specific evidence>",
    "output_structure_and_credibility": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "primary_deliverable_accuracy": 0.35,
    "coverage_of_required_platforms": 0.30,
    "depth_of_explanation": 0.20,
    "output_structure_and_credibility": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())