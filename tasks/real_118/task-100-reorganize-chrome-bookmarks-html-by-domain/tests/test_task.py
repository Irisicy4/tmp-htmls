"""
LLM-as-judge evaluator for EvolveBench task-100.

Category: Daily Activities
Task: The file /Users/user/Downloads/bookmarks.html is my exported Chrome bookmarks. Please reorganize them, automatically cat
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


TASK_INSTRUCTION = """The file /Users/user/Downloads/bookmarks.html is my exported Chrome bookmarks. Please reorganize them, automatically categorizing by domain, and generate a new file that I can import back into Chrome."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves parsing an exported Chrome bookmarks HTML file, reorganizing bookmarks by domain category, and generating a new valid Chrome-importable bookmarks HTML file.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Input: Chrome bookmarks HTML file (Netscape bookmark format)
- Processing: categorize by domain automatically
- Output: new HTML file in Chrome-importable format (Netscape Bookmark File Format)
- Validity: output must be importable into Chrome without errors

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent read/parse the bookmarks file?
- Were bookmarks categorized by domain?
- Was a new HTML file generated in correct format?
- Would the output file be importable into Chrome?
- How many bookmarks were processed?

### Step 2: Dimension Scoring

#### A. File Parsing (0.2)
Did the agent parse the bookmarks file?

5 — Bookmarks HTML correctly parsed; bookmark titles, URLs, and existing folders extracted.
4 — Parsing mostly correct with minor misses.
3 — Partial parsing.
2 — Agent read the file but parsing is incomplete.
1 — No parsing.

#### B. Domain Categorization (0.3)
Were bookmarks categorized by domain?

5 — All bookmarks grouped into domain-based folders (e.g. GitHub, Google, YouTube); ungrouped items in 'Other'.
4 — Most bookmarks categorized; some missed.
3 — Categorization present but inconsistent.
2 — Minimal categorization.
1 — No categorization.

#### C. Output Validity (0.35)
Is the output a valid Chrome-importable bookmarks file?

5 — Valid Netscape Bookmark File Format with correct DOCTYPE, DL/DT structure, and ADD_DATE attributes.
4 — Mostly valid but minor format issues.
3 — HTML produced but format deviates from Netscape standard.
2 — HTML produced but not importable.
1 — No output file.

#### D. Completeness (0.15)
Were all bookmarks preserved in the output?

5 — All input bookmarks present in output with no loss.
4 — Most bookmarks preserved; a few dropped.
3 — Significant bookmark loss.
2 — Only sample bookmarks in output.
1 — No bookmarks in output.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "file_parsing": <1-5>,
  "domain_categorization": <1-5>,
  "output_validity": <1-5>,
  "completeness": <1-5>,
  "dimension_reasoning": {{
    "file_parsing": "<one sentence citing specific evidence>",
    "domain_categorization": "<one sentence citing specific evidence>",
    "output_validity": "<one sentence citing specific evidence>",
    "completeness": "<one sentence citing specific evidence>",
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "file_parsing": 0.2,
    "domain_categorization": 0.3,
    "output_validity": 0.35,
    "completeness": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())