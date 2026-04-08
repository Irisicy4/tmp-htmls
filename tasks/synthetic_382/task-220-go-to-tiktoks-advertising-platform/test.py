"""
LLM-as-judge evaluator for EvolveBench task.

Category: (Self) Media
Task: Navigate TikTok's advertising platform to set up a simulated campaign targeting content creators aged 18–30 promoting video editing tools, and report the estimated reach and CPM.
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


TASK_INSTRUCTION = """Go to TikTok's advertising platform, navigate the campaign setup workflow, and fill in details to simulate a campaign targeting content creators aged 18–30 promoting video editing tools. Report the final estimated reach and CPM displayed on the summary screen."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to navigate TikTok's advertising platform, simulate a campaign targeting content creators aged 18–30 promoting video editing tools, and report the final estimated reach and CPM. Success depends on the agent correctly using the platform, filling in campaign details, and providing accurate and structured output.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Go to TikTok's advertising platform, navigate the campaign setup workflow, and fill in details to simulate a campaign targeting content creators aged 18–30 promoting video editing tools. Report the final estimated reach and CPM displayed on the summary screen.

## Task-Specific Constraints
- Must navigate TikTok's advertising platform and complete the campaign setup workflow.
- Must specify the target audience as content creators aged 18–30.
- Must include campaign details promoting video editing tools.
- Must report the final estimated reach and CPM from the summary screen.
- Output must be structured and include both reach and CPM values clearly.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate TikTok's advertising platform and complete the campaign setup workflow?
- Did the agent specify the target audience as content creators aged 18–30?
- Did the agent include campaign details promoting video editing tools?
- Did the agent report both the estimated reach and CPM values from the summary screen?
- Is the output structured and clearly organized?

### Step 2: Dimension Scoring

#### A. Deliverable Accuracy (0.35)
Measures whether the agent correctly reported the estimated reach and CPM values.

5 — Both reach and CPM values are accurate and clearly reported.
4 — Both values are reported but one is slightly inaccurate or unclear.
3 — Both values are reported but lack accuracy or clarity.
2 — Only one value is reported or both are highly inaccurate.
1 — Neither value is reported.

#### B. Platform Navigation (0.30)
Measures whether the agent successfully navigated TikTok's advertising platform and completed the campaign setup workflow.

5 — Successfully navigated the platform and completed the workflow with all required details.
4 — Navigated the platform but missed minor details in the workflow.
3 — Navigated the platform but missed significant details or steps.
2 — Attempted navigation but failed to complete the workflow.
1 — Did not navigate the platform.

#### C. Campaign Specificity (0.20)
Measures whether the campaign details are specific to content creators aged 18–30 and video editing tools.

5 — Campaign details are highly specific and tailored to the target audience and tools.
4 — Campaign details are specific but slightly generic or incomplete.
3 — Campaign details are present but lack specificity or relevance.
2 — Campaign details are vague or mostly irrelevant.
1 — No campaign details provided.

#### D. Output Structure (0.15)
Measures whether the agent's output is well-organized and clearly structured.

5 — Output is highly structured and easy to read, with clear labeling of reach and CPM.
4 — Output is structured but slightly unclear or inconsistent.
3 — Output is minimally structured but usable.
2 — Output is poorly structured and difficult to interpret.
1 — Output is unstructured or unreadable.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "deliverable_accuracy": <1-5>,
  "platform_navigation": <1-5>,
  "campaign_specificity": <1-5>,
  "output_structure": <1-5>,
  "dimension_reasoning": {{
    "deliverable_accuracy": "<one sentence citing specific evidence>",
    "platform_navigation": "<one sentence citing specific evidence>",
    "campaign_specificity": "<one sentence citing specific evidence>",
    "output_structure": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "deliverable_accuracy": 0.35,
    "platform_navigation": 0.30,
    "campaign_specificity": 0.20,
    "output_structure": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())