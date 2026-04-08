"""
LLM-as-judge evaluator for EvolveBench task.

Category: (Self) Media
Task: Set up a Patreon creator profile for a fictional content creator producing weekly photography tips, including creator category, three membership tiers, and basic rewards for each tier.
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


TASK_INSTRUCTION = """Use Patreon to set up a creator profile for a fictional content creator producing weekly photography tips. Complete the workflow to specify the creator category, set three membership tiers, and configure basic rewards for each tier. Report the final setup screen details."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves setting up a Patreon creator profile for a fictional content creator producing weekly photography tips. The agent must specify the creator category, configure three membership tiers, and define basic rewards for each tier. Successful completion requires accurate reporting of the final setup screen details.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Use Patreon to set up a creator profile for a fictional content creator producing weekly photography tips. Complete the workflow to specify the creator category, set three membership tiers, and configure basic rewards for each tier. Report the final setup screen details.

## Task-Specific Constraints
- Must specify the creator category accurately.
- Must configure exactly three membership tiers with distinct names and pricing.
- Must define at least one reward for each membership tier.
- Must include final setup screen details in the response.
- Must use Patreon as the primary platform for setup.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Patreon and complete the setup workflow?
- Are three membership tiers present, each with distinct names and pricing?
- Are rewards defined for each membership tier?
- Is the creator category specified correctly?
- Are the final setup screen details included and accurate?

### Step 2: Dimension Scoring

#### A. Deliverable Accuracy (0.35)
Measures whether the final Patreon setup screen details are correct and complete.

5 — All required details (creator category, three membership tiers, rewards) are present and accurate.
4 — Minor inaccuracies or omissions in one detail.
3 — Partial completion: at least two membership tiers and some rewards defined.
2 — Major omissions or inaccuracies in multiple details.
1 — No meaningful attempt to complete the setup.

#### B. Coverage of Requirements (0.30)
Measures whether all required elements (category, tiers, rewards) were addressed.

5 — All required elements are addressed completely.
4 — One required element is partially addressed.
3 — At least two required elements are addressed.
2 — Only one required element is addressed.
1 — No required elements are addressed.

#### C. Depth of Rewards Configuration (0.20)
Measures the specificity and thoughtfulness of the rewards defined for each tier.

5 — Rewards are detailed, unique, and appropriate for all tiers.
4 — Rewards are appropriate but lack detail or uniqueness.
3 — Rewards are defined but generic or incomplete.
2 — Rewards are mostly missing or inappropriate.
1 — No rewards are defined.

#### D. Output Structure and Clarity (0.15)
Measures whether the final response is well-organized and easy to understand.

5 — Response is structured clearly with all details easy to verify.
4 — Response is mostly clear but contains minor formatting issues.
3 — Response is somewhat clear but lacks organization or has errors.
2 — Response is poorly structured and difficult to follow.
1 — Response is completely disorganized or incomprehensible.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "deliverable_accuracy": <1-5>,
  "coverage_of_requirements": <1-5>,
  "depth_of_rewards_configuration": <1-5>,
  "output_structure_and_clarity": <1-5>,
  "dimension_reasoning": {{
    "deliverable_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_requirements": "<one sentence citing specific evidence>",
    "depth_of_rewards_configuration": "<one sentence citing specific evidence>",
    "output_structure_and_clarity": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "deliverable_accuracy": 0.35,
    "coverage_of_requirements": 0.30,
    "depth_of_rewards_configuration": 0.20,
    "output_structure_and_clarity": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())