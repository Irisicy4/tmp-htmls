"""
LLM-as-judge evaluator for EvolveBench task-71.

Category: HR & Recruiting
Task: Search for and organize jobs that women aged 50 and above can pursue.
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


TASK_INSTRUCTION = """Search for and organize jobs that women aged 50 and above can do."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to search for and organize jobs that women aged 50 and above can pursue. The agent should research suitable career options considering factors like physical demands, experience leverage, re-entry feasibility, and flexible scheduling relevant to this demographic.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Target demographic: women aged 50 or older
- Must search the web for relevant job options (not rely solely on internal knowledge)
- Should cover diverse job categories (full-time, part-time, self-employed, remote, etc.)
- Jobs should be realistically accessible to women in this age group
- Output should be organized/structured, not a flat unformatted list

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent search the web for jobs suited to women over 50? What sources or keywords were used?
- How many distinct job types or career categories were identified?
- Are the jobs appropriate for women in this age group (e.g. leverages experience, manageable physical demands, accessible entry)?
- Is the output organized into categories or structured in a useful way?
- Are practical details (qualifications, pay range, how to get started) included?

### Step 2: Dimension Scoring

#### A. Job Coverage (0.30)
How many distinct job types or career categories were identified?

5 — 8 or more distinct job titles or categories with specific examples (e.g. tutoring, bookkeeping, healthcare aide, real estate agent, consultant, social worker, etc.).
4 — 5-7 distinct jobs or categories identified.
3 — 3-4 job types mentioned.
2 — Only 1-2 job types.
1 — No jobs identified or response is not about jobs.

#### B. Age-Group Relevance (0.30)
Are the jobs specifically appropriate and accessible for women aged 50 and older?

5 — Jobs explicitly vetted for the 50+ demographic: leverages decades of experience, manageable physical requirements, flexible or part-time options highlighted, re-entry or upskilling paths noted.
4 — Most jobs are suitable for 50+ women but some may have high physical demands or tech barriers without accommodation notes.
3 — General job list not specifically filtered or tailored for women over 50.
2 — Several jobs are poorly suited for this age group (heavy physical labor, highly competitive entry-level roles, etc.).
1 — Jobs are not relevant to women over 50.

#### C. Organization Quality (0.25)
Is the information well-organized and easy to navigate?

5 — Clear categories or groupings (e.g. "Healthcare", "Education", "Remote Work") with job titles and brief descriptions for each.
4 — Organized list with most categories or descriptions present.
3 — Simple list with job names but minimal structure or description.
2 — Unorganized or incomplete output.
1 — No structure; jobs mentioned in passing without a list or categories.

#### D. Practical Information (0.15)
Are actionable details provided to help women pursue these jobs?

5 — Qualification requirements, estimated pay range or demand outlook, and getting-started tips included for most jobs.
4 — At least two of the three types of practical info (qualifications, pay, getting started) for most jobs.
3 — Some practical info included for a few jobs only.
2 — Very limited practical info (e.g. only job titles with no supporting detail).
1 — No practical information provided.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "job_coverage": <1-5>,
  "age_group_relevance": <1-5>,
  "organization_quality": <1-5>,
  "practical_information": <1-5>,
  "dimension_reasoning": {{
    "job_coverage": "<one sentence citing specific evidence>",
    "age_group_relevance": "<one sentence citing specific evidence>",
    "organization_quality": "<one sentence citing specific evidence>",
    "practical_information": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "job_coverage": 0.30,
    "age_group_relevance": 0.30,
    "organization_quality": 0.25,
    "practical_information": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())