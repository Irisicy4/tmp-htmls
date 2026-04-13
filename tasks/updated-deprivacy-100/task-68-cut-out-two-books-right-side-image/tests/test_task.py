"""
LLM-as-judge evaluator for EvolveBench task-68.

Category: (Self) Media
Task: This is my product main image. Please cut out the 2 books on the right side of the image separately....
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
    m = re.search(r"<Answer>(.*?)</Answer>", text, re.DOTALL | re.IGNORECASE)
    if m:
        try: return json.loads(m.group(1).strip())
        except Exception: pass
    try: return json.loads(text.strip())
    except Exception: pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try: return json.loads(m.group())
        except Exception: pass
    return None

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


TASK_INSTRUCTION = """This is my product main image. Please cut out the 2 books on the right side of the image separately."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves using an image editing tool (likely Photopea) to isolate and extract two book objects from a product image as separate files.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Tool: Photopea or equivalent browser-based image editor
- Action: extract/cut out 2 specific books from the right side of the image
- Output: 2 separate image files (one per book) with transparent or clean backgrounds
- Precision: cutouts should be clean without significant artifacts

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent open an image editing tool?
- Were the two books on the right side identified and selected?
- Were separate cutout files produced for each book?
- Is the background clean (transparent or white) around the cutouts?

### Step 2: Dimension Scoring

#### A. Tool Usage (0.25)
Did the agent use an appropriate image editing tool?

5 — Agent opened Photopea or equivalent and loaded the product image.
4 — Agent used a tool but with setup difficulties.
3 — Agent described using a tool without clear evidence of execution.
2 — Agent suggested a tool without using it.
1 — No tool used.

#### B. Object Identification (0.2)
Were the correct objects (2 books on right side) identified?

5 — Both books on the right side correctly identified and selected.
4 — Books identified but selection slightly off.
3 — One book correctly identified.
2 — Wrong objects selected.
1 — No object identification.

#### C. Extraction Quality (0.4)
Were clean cutouts produced?

5 — Both books extracted as separate files with clean transparent/white background and minimal artifacts.
4 — Both extracted but with minor artifacts.
3 — One book extracted cleanly; other has issues.
2 — Both extracted but with significant artifacts.
1 — No cutouts produced.

#### D. File Output (0.15)
Were separate files saved for each book?

5 — Two separate image files saved (e.g. book1.png, book2.png).
4 — Files saved but both in one file or wrong format.
3 — One file saved.
2 — Files described but not saved.
1 — No output files.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "tool_usage": <1-5>,
  "object_identification": <1-5>,
  "extraction_quality": <1-5>,
  "file_output": <1-5>,
  "dimension_reasoning": {{
    "tool_usage": "<one sentence citing specific evidence>",
    "object_identification": "<one sentence citing specific evidence>",
    "extraction_quality": "<one sentence citing specific evidence>",
    "file_output": "<one sentence citing specific evidence>",
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "tool_usage": 0.25,
    "object_identification": 0.2,
    "extraction_quality": 0.4,
    "file_output": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())