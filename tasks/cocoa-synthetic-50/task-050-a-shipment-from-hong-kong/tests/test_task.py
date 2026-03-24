"""
LLM-as-judge evaluator for EvolveBench task.

Category: Logistics & Supply Chain
Task: Diagnose a customs clearance issue for a shipment and provide corrective steps using CBP and WCO guidelines.
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


TASK_INSTRUCTION = """A shipment from Hong Kong to the Port of Los Angeles is delayed due to a customs clearance issue flagged as 'documentation discrepancy.' Research the requirements for customs clearance for imports into the USA using CBP (Customs and Border Protection) guidelines and the World Customs Organization (WCO) framework. Diagnose the most likely cause of the discrepancy and state the corrective steps to resolve it."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves diagnosing a customs clearance issue for a shipment delayed at the Port of Los Angeles due to a 'documentation discrepancy.' The agent must research CBP (Customs and Border Protection) guidelines and the World Customs Organization (WCO) framework to identify the likely cause of the issue and recommend corrective steps. A successful completion includes accurate diagnosis, clear corrective steps, and evidence from the required platforms.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
A shipment from Hong Kong to the Port of Los Angeles is delayed due to a customs clearance issue flagged as 'documentation discrepancy.' Research the requirements for customs clearance for imports into the USA using CBP (Customs and Border Protection) guidelines and the World Customs Organization (WCO) framework. Diagnose the most likely cause of the discrepancy and state the corrective steps to resolve it.

## Task-Specific Constraints
- Must visit cbp.gov and wcoomd.org to gather information.
- Must identify at least one likely cause of the documentation discrepancy.
- Must provide at least two corrective steps to resolve the issue.
- Output must include references to the guidelines or frameworks consulted.
- Response must be structured as a clear, step-by-step explanation.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to cbp.gov and wcoomd.org? Were these platforms used effectively?
- Does the response identify at least one likely cause of the documentation discrepancy?
- Are at least two corrective steps provided, and are they actionable?
- Are references to CBP and WCO guidelines included in the response?
- Is the output structured as a clear, step-by-step explanation?

### Step 2: Dimension Scoring

#### A. Diagnosis Accuracy (0.35)
Measures whether the agent correctly identifies the most likely cause of the documentation discrepancy.

5 — Accurately identifies the primary cause with evidence from CBP and WCO guidelines.
4 — Identifies a plausible cause but lacks full evidence or specificity.
3 — Identifies a cause but with limited relevance or evidence.
2 — Provides an incorrect or vague diagnosis.
1 — No diagnosis provided.

#### B. Corrective Steps Quality (0.30)
Measures the quality and feasibility of the corrective steps provided.

5 — Provides at least two clear, actionable, and evidence-based corrective steps.
4 — Provides two steps, but one is less clear or actionable.
3 — Provides one actionable step or two vague steps.
2 — Provides unclear or irrelevant steps.
1 — No corrective steps provided.

#### C. Platform Usage and Coverage (0.20)
Measures whether the agent effectively used the required platforms (cbp.gov and wcoomd.org).

5 — Effectively uses both platforms and references them in the response.
4 — Uses both platforms but references are incomplete or unclear.
3 — Uses one platform effectively or both minimally.
2 — Uses only one platform with minimal relevance.
1 — No evidence of platform usage.

#### D. Output Structure and Clarity (0.15)
Measures the organization and clarity of the response.

5 — Response is well-structured, step-by-step, and easy to follow.
4 — Response is mostly clear but slightly disorganized.
3 — Response is partially clear but lacks structure.
2 — Response is unclear or poorly organized.
1 — Response is incoherent or absent.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "diagnosis_accuracy": <1-5>,
  "corrective_steps_quality": <1-5>,
  "platform_usage_and_coverage": <1-5>,
  "output_structure_and_clarity": <1-5>,
  "dimension_reasoning": {{
    "diagnosis_accuracy": "<one sentence citing specific evidence>",
    "corrective_steps_quality": "<one sentence citing specific evidence>",
    "platform_usage_and_coverage": "<one sentence citing specific evidence>",
    "output_structure_and_clarity": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "diagnosis_accuracy": 0.35,
    "corrective_steps_quality": 0.30,
    "platform_usage_and_coverage": 0.20,
    "output_structure_and_clarity": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())