"""
LLM-as-judge evaluator for EvolveBench task-05.

Category: Daily Activities / Sports
Task: "go to nba.com and check for Jayson Tatum's current 3-point status"
"""

import os, json, re

TASK_INSTRUCTION = "go to nba.com and check for Jayson Tatum's current 3-point status"
PASS_THRESHOLD = 3.0

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based sports statistics lookup task.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Platform: nba.com specifically — not ESPN, Basketball-Reference, or general web search
- Player: Jayson Tatum specifically
- Stat: 3-point shooting status — this could mean current season 3PT stats (attempts, makes, percentage), all-time 3PT record progress, or both; the agent should surface whatever is most current and relevant
- Recency: the data must be current (from this season or the most recent game), not career averages alone

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to nba.com? Cite evidence from the trace or response.
- What 3-point statistics did the agent report for Jayson Tatum? List them explicitly (e.g. 3PM, 3PA, 3P%).
- Are the stats current (this season or recent game), or are they career/historical only?
- Did the agent interpret "3-point status" meaningfully (e.g. ranking, record chase, season performance)?
- Did the agent fabricate or estimate stats, or retrieve them from the live site?

### Step 2: Dimension Scoring

#### A. Platform Execution
Did the agent actually navigate nba.com as instructed?

5 — Clear evidence agent visited nba.com and retrieved live stats from the page.
4 — Agent used nba.com but retrieved limited data (e.g. only landed on player profile without drilling into stats).
3 — Agent retrieved Tatum stats but from a different source (ESPN, Basketball-Reference, etc.).
2 — Agent searched the web generally without visiting a specific stats platform.
1 — No navigation performed; stats appear fabricated or from prior knowledge.

#### B. Stat Accuracy & Recency
Are the reported 3-point statistics accurate and current?

5 — Stats are clearly sourced from nba.com, include current season figures, and are specific (3PM, 3PA, 3P% with game/season context).
4 — Stats are current but one figure is missing or slightly vague (e.g. percentage given but not attempts).
3 — Stats are present but only career averages or outdated; no current season breakdown.
2 — Stats are mentioned but vague (e.g. "he's a great 3-point shooter") without specific numbers for Tatum.
1 — No stats provided, or stats are clearly incorrect.

#### C. Interpretive Completeness
Did the agent interpret "3-point status" in a meaningful, useful way?

5 — Agent reports current season 3PT stats AND contextualises them (e.g. league ranking, record chase, recent game performance).
4 — Agent reports current stats with minimal context (numbers only, no ranking or significance).
3 — Agent reports only one aspect (e.g. all-time record progress but no season stats, or vice versa).
2 — Agent reports generic information about Tatum being a good 3-point shooter without specific current status.
1 — Response does not address 3-point status at all.

#### D. Response Clarity
Is the output concise and directly useful?

5 — Stats are clearly presented, labelled, and easy to read at a glance.
4 — Stats are present but formatting is slightly cluttered or requires parsing.
3 — Stats are buried in a long paragraph; requires effort to extract the key numbers.
2 — Response is mostly narrative with stats hard to find.
1 — No useful stats present.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "platform_execution": <1-5>,
  "stat_accuracy_recency": <1-5>,
  "interpretive_completeness": <1-5>,
  "response_clarity": <1-5>,
  "dimension_reasoning": {{
    "platform_execution": "<one sentence citing specific evidence>",
    "stat_accuracy_recency": "<one sentence citing specific evidence>",
    "interpretive_completeness": "<one sentence citing specific evidence>",
    "response_clarity": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "platform_execution":        0.30,
    "stat_accuracy_recency":     0.35,
    "interpretive_completeness": 0.20,
    "response_clarity":          0.15,
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