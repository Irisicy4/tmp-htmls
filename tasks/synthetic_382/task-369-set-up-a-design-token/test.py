"""
LLM-as-judge evaluator for EvolveBench task.

Category: Design
Task: Set up a design token framework using a public Figma community file, configure tokens for color, typography, and spacing, export the configuration, and summarize the structure.
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


TASK_INSTRUCTION = """Set up a design token framework using a public Figma community file. Configure tokens for color (primary, secondary, neutral), typography (headings, body), and spacing scales (small, medium, large). Export the final configuration and summarize the structure."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to set up a design token framework using a public Figma community file. The framework must include tokens for color (primary, secondary, neutral), typography (headings, body), and spacing scales (small, medium, large). The agent must export the configuration and provide a summary of the structure. A successful completion involves correctly setting up the tokens, using the required platforms, and providing a clear and organized summary.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Set up a design token framework using a public Figma community file. Configure tokens for color (primary, secondary, neutral), typography (headings, body), and spacing scales (small, medium, large). Export the final configuration and summarize the structure.

## Task-Specific Constraints
- Must use a public Figma community file as the starting point.
- Must configure tokens for all three categories: color, typography, and spacing.
- Must include at least three color tokens (primary, secondary, neutral).
- Must include at least two typography tokens (headings, body).
- Must include at least three spacing tokens (small, medium, large).
- Must export the final configuration and provide a structured summary.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to a public Figma community file and use it as the starting point?
- Did the agent configure tokens for all three categories: color, typography, and spacing?
- Are at least three color tokens, two typography tokens, and three spacing tokens present in the response?
- Did the agent export the final configuration and provide a structured summary?
- Is the summary clear, organized, and reflective of the configuration?

### Step 2: Dimension Scoring

#### A. Token Configuration Accuracy (0.35)
Measures whether the agent correctly configured the required tokens.

5 — All required tokens (color, typography, spacing) are configured correctly and completely.
4 — Most tokens are configured correctly, with minor omissions or errors.
3 — Some tokens are configured correctly, but others are missing or incorrect.
2 — Few tokens are configured correctly; most are missing or incorrect.
1 — No tokens are configured.

#### B. Coverage of Required Categories (0.30)
Measures whether the agent addressed all required categories (color, typography, spacing).

5 — All three categories are fully addressed with the required tokens.
4 — All three categories are addressed, but one is incomplete.
3 — Two categories are addressed, but one is missing.
2 — Only one category is addressed.
1 — None of the categories are addressed.

#### C. Summary Clarity and Structure (0.20)
Measures the clarity and organization of the summary provided by the agent.

5 — The summary is clear, well-structured, and fully reflects the token configuration.
4 — The summary is mostly clear and organized, with minor issues.
3 — The summary is somewhat clear but lacks detail or organization.
2 — The summary is unclear or poorly organized.
1 — No summary is provided.

#### D. Use of Required Platform (0.15)
Measures whether the agent used a public Figma community file as the starting point.

5 — The agent clearly used a public Figma community file as the starting point.
4 — The agent used Figma but did not clearly specify a public community file.
3 — The agent used Figma but the source is unclear or partially incorrect.
2 — The agent did not use Figma but attempted an alternative.
1 — The agent did not use Figma or any alternative.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "token_configuration_accuracy": <1-5>,
  "coverage_of_required_categories": <1-5>,
  "summary_clarity_and_structure": <1-5>,
  "use_of_required_platform": <1-5>,
  "dimension_reasoning": {{
    "token_configuration_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_required_categories": "<one sentence citing specific evidence>",
    "summary_clarity_and_structure": "<one sentence citing specific evidence>",
    "use_of_required_platform": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "token_configuration_accuracy": 0.35,
    "coverage_of_required_categories": 0.30,
    "summary_clarity_and_structure": 0.20,
    "use_of_required_platform": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())