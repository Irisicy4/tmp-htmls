"""
LLM-as-judge evaluator for EvolveBench task-95.

Category: Daily Activities
Task: Please help me compile a table of car models that currently support automated parking into mechanical or multi-level par
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


TASK_INSTRUCTION = """Please help me compile a table of car models that currently support automated parking into mechanical or multi-level parking spaces via parking assist. Include any height restrictions for the parking spaces they support, as well as the vehicle's own height dimensions. Cite your sources."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves researching which car models have automated/remote parking assist that works with mechanical or multi-level (立体駐車場) parking structures, including height constraint compatibility.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Feature: automated parking assist compatible with mechanical/multi-level parking structures
- Data: both vehicle height AND parking space height restriction required
- Coverage: multiple car brands/models
- Output: table format with sources cited

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent search for cars with mechanical parking space compatibility?
- How many car models were found?
- Are vehicle heights and parking space height limits both included?
- Are sources cited?
- Is output in table format?

### Step 2: Dimension Scoring

#### A. Research Quality (0.25)
Did the agent find relevant car models?

5 — Multiple car models found with specific parking assist feature names (e.g. Toyota Advanced Park, BMW Parking Assistant Plus).
4 — Several models found but feature details thin.
3 — Some models identified without feature specifics.
2 — Very limited findings.
1 — No research.

#### B. Height Data (0.35)
Are both vehicle and parking height data included?

5 — Vehicle height AND supported parking space max height both present for each model.
4 — Vehicle height present; parking space limit for most models.
3 — One height dimension present for most models.
2 — Heights mentioned without specific values.
1 — No height data.

#### C. Table Format (0.25)
Is output in table format?

5 — Clear table: car model, brand, vehicle height, max parking height supported, source.
4 — Table present but missing one column.
3 — List format instead of table.
2 — Narrative without table.
1 — No structured output.

#### D. Source Citation (0.15)
Are sources cited for each entry?

5 — Each model's data attributed to a specific source (manufacturer spec page, review, news).
4 — Most entries sourced.
3 — Some sources cited.
2 — General sources without per-model attribution.
1 — No sources.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "research_quality": <1-5>,
  "height_data": <1-5>,
  "table_format": <1-5>,
  "source_citation": <1-5>,
  "dimension_reasoning": {{
    "research_quality": "<one sentence citing specific evidence>",
    "height_data": "<one sentence citing specific evidence>",
    "table_format": "<one sentence citing specific evidence>",
    "source_citation": "<one sentence citing specific evidence>",
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "research_quality": 0.25,
    "height_data": 0.35,
    "table_format": 0.25,
    "source_citation": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())