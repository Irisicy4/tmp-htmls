"""
LLM-as-judge evaluator for EvolveBench task.

Category: Design
Task: Extract a trending color palette from Adobe Color and two complementary font pairings from FontPair.com.
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


TASK_INSTRUCTION = """Go to the Adobe Color website and extract a trending color palette from the 'Explore' section. Then, visit FontPair.com and extract two font pairings that complement your chosen palette. Collect the extracted palette and font pairs into a brief summary."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves extracting a trending color palette from the 'Explore' section of Adobe Color and identifying two complementary font pairings from FontPair.com. The deliverable is a brief summary containing the palette and font pairings.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Go to the Adobe Color website and extract a trending color palette from the 'Explore' section. Then, visit FontPair.com and extract two font pairings that complement your chosen palette. Collect the extracted palette and font pairs into a brief summary.

## Task-Specific Constraints
- Must extract a trending color palette from Adobe Color's 'Explore' section.
- Must extract two font pairings from FontPair.com that complement the chosen palette.
- The summary must include the palette (color names or hex codes) and font pairings (font names).
- The response must be concise and well-organized.
- The agent must visit both specified platforms.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Adobe Color and FontPair.com?
- Did the agent extract a trending color palette from Adobe Color's 'Explore' section?
- Did the agent extract two complementary font pairings from FontPair.com?
- Is the summary organized and does it include both the palette and font pairings?
- Are the extracted palette and font pairings appropriate and relevant?

### Step 2: Dimension Scoring

#### A. Deliverable Accuracy (0.35)
Measures whether the extracted palette and font pairings are correct and complete.

5 — Includes a valid trending palette and two complementary font pairings.
4 — Includes a valid palette and one complementary font pairing.
3 — Includes a valid palette but no font pairings or incomplete data.
2 — Palette or font pairings are incorrect or missing.
1 — No valid palette or font pairings provided.

#### B. Platform Coverage (0.30)
Measures whether the agent visited both required platforms and extracted data.

5 — Successfully visited both platforms and extracted required data.
4 — Visited both platforms but extracted incomplete data.
3 — Visited one platform and extracted partial data.
2 — Only visited one platform with no valid data extracted.
1 — Did not visit either platform.

#### C. Specificity of Output (0.20)
Measures the level of detail and appropriateness of the extracted palette and font pairings.

5 — Palette includes hex codes and font pairings are clearly named and relevant.
4 — Palette includes hex codes but font pairings lack clarity or relevance.
3 — Palette lacks hex codes or font pairings are vague.
2 — Palette or font pairings are generic or irrelevant.
1 — No specific details provided.

#### D. Organization and Clarity (0.15)
Measures how well-organized and readable the summary is.

5 — Summary is concise, well-structured, and easy to understand.
4 — Summary is mostly clear but lacks some structure.
3 — Summary is readable but disorganized.
2 — Summary is poorly organized or unclear.
1 — Summary is incomprehensible.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{
  "evidence_summary": "The agent visited Adobe Color and FontPair.com, extracted a trending palette, and provided font pairings. The summary includes the required elements but lacks some specificity.",
  "deliverable_accuracy": 4,
  "platform_coverage": 5,
  "specificity_of_output": 3,
  "organization_and_clarity": 4,
  "dimension_reasoning": {
    "deliverable_accuracy": "The palette and font pairings are mostly correct but one pairing is missing.",
    "platform_coverage": "Both platforms were visited and data was extracted.",
    "specificity_of_output": "Palette lacks hex codes and font pairings are somewhat vague.",
    "organization_and_clarity": "Summary is clear but could be better structured."
  },
  "overall_score": 4.0,
  "passed": true
}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "deliverable_accuracy": 0.35,
    "platform_coverage": 0.30,
    "specificity_of_output": 0.20,
    "organization_and_clarity": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())