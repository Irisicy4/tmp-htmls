"""
LLM-as-judge evaluator for EvolveBench task.

Category: Travel & Planning
Task: Verify whether tourists still need a visa to enter Singapore for a stay of up to 14 days using official resources.
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


TASK_INSTRUCTION = """Verify whether tourists still need a visa to enter Singapore for a stay of up to 14 days, as of today. Use information from the official Immigration and Checkpoints Authority of Singapore website and IATA's Travel Center resources."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to verify visa requirements for tourists entering Singapore for up to 14 days. The agent must use official resources, including the Immigration and Checkpoints Authority of Singapore website and IATA's Travel Center. A successful completion includes accurate and up-to-date visa information, sourced from the required platforms.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Verify whether tourists still need a visa to enter Singapore for a stay of up to 14 days, as of today. Use information from the official Immigration and Checkpoints Authority of Singapore website and IATA's Travel Center resources.

## Task-Specific Constraints
- Must visit both ica.gov.sg and iatatravelcentre.com.
- Must provide visa requirements for tourists explicitly.
- Must confirm information is up-to-date as of today.
- Must cite sources clearly in the response.
- Output must be organized in a structured format (e.g., bullet points or table).

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to both required platforms (ica.gov.sg and iatatravelcentre.com)?
- Does the response explicitly state visa requirements for tourists entering Singapore for up to 14 days?
- Is the information confirmed to be up-to-date as of today?
- Are sources cited clearly in the response?
- Is the output organized in a structured format (e.g., bullet points or table)?

### Step 2: Dimension Scoring

#### A. Primary Deliverable Accuracy (0.35)
Measures whether the visa requirement information is correct and complete.

5 — Provides accurate visa requirements for tourists entering Singapore for up to 14 days, with no errors or omissions.
4 — Provides mostly accurate visa requirements, with minor omissions or errors.
3 — Provides partially accurate visa requirements, with significant omissions or errors.
2 — Provides mostly incorrect or incomplete visa requirements.
1 — Provides no accurate visa requirement information.

#### B. Coverage of Required Platforms (0.30)
Measures whether the agent used both required platforms (ica.gov.sg and iatatravelcentre.com).

5 — Uses both platforms and explicitly references them in the response.
4 — Uses both platforms but references only one explicitly.
3 — Uses only one platform or references both vaguely.
2 — Attempts to use platforms but fails to retrieve relevant information.
1 — Does not use any required platform.

#### C. Depth of Information (0.20)
Measures the level of detail and specificity in the response.

5 — Includes detailed visa requirements, conditions, and exceptions if applicable.
4 — Includes visa requirements with some detail but lacks conditions or exceptions.
3 — Includes basic visa requirements with minimal detail.
2 — Includes vague or incomplete visa requirements.
1 — Includes no meaningful detail.

#### D. Source Citation and Output Structure (0.15)
Measures the credibility of sources cited and the organization of the response.

5 — Cites sources clearly and organizes the response in a structured format (e.g., bullet points or table).
4 — Cites sources but organization is slightly unclear or inconsistent.
3 — Cites sources vaguely and organization is minimally structured.
2 — Does not cite sources clearly or response is poorly organized.
1 — Does not cite sources and response is unstructured.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "primary_deliverable_accuracy": <1-5>,
  "coverage_of_required_platforms": <1-5>,
  "depth_of_information": <1-5>,
  "source_citation_and_output_structure": <1-5>,
  "dimension_reasoning": {{
    "primary_deliverable_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_required_platforms": "<one sentence citing specific evidence>",
    "depth_of_information": "<one sentence citing specific evidence>",
    "source_citation_and_output_structure": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "primary_deliverable_accuracy": 0.35,
    "coverage_of_required_platforms": 0.30,
    "depth_of_information": 0.20,
    "source_citation_and_output_structure": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())