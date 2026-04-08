"""
LLM-as-judge evaluator for EvolveBench task.

Category: (Self) Media
Task: Verify whether Instagram's algorithm favors video content over image posts as of today.
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


TASK_INSTRUCTION = """Verify whether Instagram's algorithm favors video content over image posts as of today. Check Instagram's official blog, creator guidelines, and recent media reports to confirm the current algorithm's stance. Present evidence supporting or refuting this claim."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to determine whether Instagram's algorithm currently favors video content over image posts. The agent must gather evidence from Instagram's official blog, creator guidelines, and recent media reports. Successful completion requires presenting clear evidence supporting or refuting the claim, based on credible sources.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Verify whether Instagram's algorithm favors video content over image posts as of today. Check Instagram's official blog, creator guidelines, and recent media reports to confirm the current algorithm's stance. Present evidence supporting or refuting this claim.

## Task-Specific Constraints
- Must visit Instagram's official blog, creator guidelines, and at least one media report from theverge.com or techcrunch.com.
- Must explicitly state whether video content is favored, with supporting evidence.
- Must include direct quotes or summaries from the sources visited.
- Must address any conflicting evidence and explain the reasoning behind the conclusion.
- Output must be organized as a structured summary or list.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Instagram's official blog, creator guidelines, and at least one media report from theverge.com or techcrunch.com?
- Are direct quotes or summaries from the sources present in the response?
- Does the response explicitly state whether video content is favored, with supporting evidence?
- Are conflicting pieces of evidence addressed, and is the reasoning behind the conclusion explained?
- Is the output organized as a structured summary or list?

### Step 2: Dimension Scoring

#### A. Primary Deliverable Accuracy (0.35)
Measures whether the agent correctly determined Instagram's algorithm stance and provided evidence.

5 — Correctly identifies the algorithm's stance with 3+ pieces of supporting evidence from required sources.
4 — Correctly identifies the stance with 2 pieces of evidence; minor gaps in reasoning.
3 — Partially correct; identifies stance but lacks sufficient evidence or reasoning.
2 — Incorrect or incomplete; minimal evidence provided.
1 — No attempt or completely wrong.

#### B. Coverage of Required Sources (0.30)
Measures whether the agent visited all required platforms and included evidence from them.

5 — Evidence from Instagram's blog, creator guidelines, and both media reports (theverge.com and techcrunch.com).
4 — Evidence from Instagram's blog, creator guidelines, and one media report.
3 — Evidence from Instagram's blog and creator guidelines; media report missing.
2 — Minimal source coverage; only one platform visited.
1 — No sources visited or cited.

#### C. Depth and Specificity of Evidence (0.25)
Measures the level of detail and specificity in the evidence presented.

5 — Includes detailed quotes, summaries, and analysis from all sources; addresses conflicting evidence.
4 — Includes quotes and summaries; minor gaps in detail or analysis.
3 — Includes basic summaries; lacks depth or specificity.
2 — Minimal evidence; vague or generic statements.
1 — No evidence or entirely generic.

#### D. Output Structure and Credibility (0.10)
Measures whether the response is well-organized and uses credible sources.

5 — Structured summary or list format; all sources are credible and cited.
4 — Mostly structured; minor formatting issues; sources credible.
3 — Basic structure; some formatting issues or unclear citations.
2 — Poorly organized; credibility of sources unclear.
1 — No structure; sources not credible or absent.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "primary_deliverable_accuracy": <1-5>,
  "coverage_of_required_sources": <1-5>,
  "depth_and_specificity_of_evidence": <1-5>,
  "output_structure_and_credibility": <1-5>,
  "dimension_reasoning": {{
    "primary_deliverable_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_required_sources": "<one sentence citing specific evidence>",
    "depth_and_specificity_of_evidence": "<one sentence citing specific evidence>",
    "output_structure_and_credibility": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "primary_deliverable_accuracy": 0.35,
    "coverage_of_required_sources": 0.30,
    "depth_and_specificity_of_evidence": 0.25,
    "output_structure_and_credibility": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())