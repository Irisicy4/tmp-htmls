"""
LLM-as-judge evaluator for EvolveBench task-07-save-todays-top-traded-etf.

Category: Data Collection and Documentation
Task: Save today’s top traded ETF stocks from Naver Securities to Google Docs
"""

import os, json, re

TASK_INSTRUCTION = """Save today’s top traded ETF stocks from Naver Securities to Google Docs"""
PASS_THRESHOLD = 3.0

SYSTEM_PROMPT = """The judge is evaluating the agent's ability to accurately retrieve today's top traded ETF stocks from Naver Securities and document them in Google Docs. The evaluation focuses on the correctness, completeness, formatting, and adherence to the task constraints. The judge should ensure the agent followed the specified workflow and produced a usable output.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Retrieve data specifically from Naver Securities.
- Focus only on ETF stocks and identify the top traded ones.
- Save the data in a Google Docs document with clear formatting.
- Ensure the document is accessible and properly titled.
- Avoid including irrelevant or outdated information.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent retrieve the data from Naver Securities as instructed?
- Does the document include only ETF stocks and identify the top traded ones?
- Is the Google Docs document properly formatted and titled?
- Is the information accurate and complete?
- Is the document accessible and free of irrelevant content?

### Step 2: Dimension Scoring

#### A. Data Accuracy
Measures whether the retrieved ETF stock data is accurate and matches Naver Securities.

5 — All ETF stock data is accurate and matches Naver Securities perfectly.
4 — Most ETF stock data is accurate, with minor discrepancies.
3 — Some ETF stock data is accurate, but there are noticeable errors.
2 — Significant inaccuracies in the ETF stock data retrieved.
1 — ETF stock data is entirely inaccurate or missing.

#### B. Data Completeness
Evaluates whether all top traded ETF stocks are included in the document.

5 — All top traded ETF stocks are included without omissions.
4 — Most top traded ETF stocks are included, with minor omissions.
3 — Some top traded ETF stocks are included, but key entries are missing.
2 — Few top traded ETF stocks are included, with major omissions.
1 — No top traded ETF stocks are included or the list is irrelevant.

#### C. Document Formatting
Assesses the clarity and organization of the Google Docs document.

5 — The document is well-organized, clearly formatted, and easy to read.
4 — The document is mostly well-organized, with minor formatting issues.
3 — The document has noticeable formatting issues but is still readable.
2 — The document is poorly formatted and difficult to read.
1 — The document is completely disorganized and unreadable.

#### D. Task Adherence
Checks whether the agent followed all task constraints and instructions.

5 — All task constraints and instructions were followed perfectly.
4 — Most task constraints and instructions were followed, with minor deviations.
3 — Some task constraints and instructions were followed, but there are noticeable deviations.
2 — Few task constraints and instructions were followed, with major deviations.
1 — None of the task constraints or instructions were followed.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "data_accuracy": <1-5>,
  "data_completeness": <1-5>,
  "document_formatting": <1-5>,
  "task_adherence": <1-5>,
  "dimension_reasoning": {{
    "data_accuracy": "<one sentence citing specific evidence>",
    "data_completeness": "<one sentence citing specific evidence>",
    "document_formatting": "<one sentence citing specific evidence>",
    "task_adherence": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "data_accuracy":       0.35,
    "data_completeness":   0.25,
    "document_formatting": 0.2,
    "task_adherence":      0.2,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())

def _extract_response(result):
    task_result = result.get("task_result") or ""
    if isinstance(task_result, str) and task_result.strip(): return task_result
    for message in reversed(result.get("conversation") or []):
        if not isinstance(message, dict): continue
        if message.get("role") == "assistant":
            content = message.get("content") or ""
            if isinstance(content, str) and len(content) > 20: return content
    return ""

def _parse_answer_tag(text):
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

def _call_judge_once(agent_response, execution_summary):
    try:
        import openai
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        user_content = USER_PROMPT_TEMPLATE.format(
            task_instruction=TASK_INSTRUCTION,
            agent_response=agent_response,
            execution_summary=execution_summary or "Not available.",
        )
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_content}],
            max_tokens=1024,
        )
        return _parse_answer_tag(completion.choices[0].message.content)
    except Exception as e:
        return {"error": str(e)}

def _majority_vote(votes):
    valid = [v for v in votes if v and "error" not in v and all(d in v for d in DIMENSIONS)]
    if not valid: return votes[0] if votes else {"error": "All judge calls failed"}
    aggregated = {dim: sorted([v[dim] for v in valid])[len(valid) // 2] for dim in DIMENSIONS}
    overall = sum(aggregated[d] * DIMENSION_WEIGHTS[d] for d in DIMENSIONS)
    aggregated["overall_score"] = round(overall, 2)
    aggregated["passed"] = overall >= PASS_THRESHOLD
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
    first_call = _call_judge_once(agent_response, execution_summary)
    if first_call and "error" not in first_call:
        overall = first_call.get("overall_score", 0)
        if abs(float(overall) - PASS_THRESHOLD) <= 0.5:
            scores = _majority_vote([first_call, _call_judge_once(agent_response, execution_summary), _call_judge_once(agent_response, execution_summary)])
        else:
            scores = first_call; scores["_votes_used"] = 1
    else:
        scores = first_call or {"error": "Judge call failed", "overall_score": 0}
    overall = scores.get("overall_score", 0)
    passed = scores.get("passed", float(overall) >= PASS_THRESHOLD)
    feedback_lines = [f"Overall score: {overall}/5  (threshold: {PASS_THRESHOLD})"]
    for dim in DIMENSIONS:
        if dim in scores: feedback_lines.append(f"  {dim}: {scores[dim]}/5")
    if scores.get("evidence_summary"): feedback_lines.append(f"\nEvidence summary: {scores['evidence_summary']}")
    reasoning = scores.get("dimension_reasoning", {})
    if reasoning:
        feedback_lines.append("\nDimension reasoning:")
        for dim, reason in reasoning.items(): feedback_lines.append(f"  {dim}: {reason}")
    if scores.get("_votes_used", 1) > 1:
        feedback_lines.append(f"\n(Borderline case: {scores['_votes_used']} judge calls used, majority vote applied)")
    return {
        "passed": bool(passed), "feedback": "\n".join(feedback_lines),
        "details": {"task_completed": result.get("status") == "success", "overall_score": overall,
                    "dimension_scores": {d: scores.get(d) for d in DIMENSIONS},
                    "evidence_summary": scores.get("evidence_summary"),
                    "dimension_reasoning": scores.get("dimension_reasoning"),
                    "pass_threshold": PASS_THRESHOLD, "votes_used": scores.get("_votes_used", 1)},
    }