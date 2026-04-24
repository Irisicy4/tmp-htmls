"""
LLM-as-judge evaluator for EvolveBench task-14-download-the-100-most-liked.

Category: Web Research
Task: Download the 100 most liked videos on tiktok about diving mask and diving snorkel
"""

import os, json, re

TASK_INSTRUCTION = """Download the 100 most liked videos on tiktok about diving mask and diving snorkel"""
PASS_THRESHOLD = 3.0

SYSTEM_PROMPT = """The judge is evaluating the agent's ability to accurately identify, download, and organize the 100 most liked TikTok videos related to diving masks and diving snorkels. The evaluation focuses on the agent's adherence to the task constraints, the relevance of the videos to the specified topic, and the completeness of the dataset. The judge should also assess whether the agent followed ethical and legal guidelines while performing the task.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Only videos explicitly related to diving masks and diving snorkels should be included.
- The dataset must consist of exactly 100 videos.
- The videos must be ranked by the number of likes, starting from the most liked.
- The agent must ensure compliance with TikTok's terms of service and copyright laws.
- The videos must be downloaded in a format that preserves their original quality.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent collect exactly 100 videos?
- Are all the videos directly related to diving masks and diving snorkels?
- Are the videos ranked correctly by the number of likes?
- Did the agent provide evidence of compliance with TikTok's terms of service and copyright laws?
- Is the quality of the downloaded videos consistent with their original format?

### Step 2: Dimension Scoring

#### A. Relevance To Topic
Measures how well the videos align with the topic of diving masks and diving snorkels.

5 — All 100 videos are directly and clearly related to diving masks and diving snorkels.
4 — At least 90% of the videos are related to diving masks and diving snorkels, with minor deviations.
3 — At least 75% of the videos are related to diving masks and diving snorkels, but some are off-topic.
2 — Fewer than 75% of the videos are related to diving masks and diving snorkels.
1 — The majority of the videos are unrelated to diving masks and diving snorkels.

#### B. Ranking Accuracy
Assesses whether the videos are correctly ranked by the number of likes.

5 — All 100 videos are perfectly ranked by the number of likes in descending order.
4 — The ranking is mostly accurate, with minor errors affecting fewer than 5 videos.
3 — The ranking has noticeable errors affecting 5-15 videos.
2 — The ranking has significant errors affecting more than 15 videos.
1 — The ranking is largely incorrect or absent.

#### C. Dataset Completeness
Evaluates whether the dataset contains exactly 100 videos.

5 — The dataset contains exactly 100 videos with no omissions or duplicates.
4 — The dataset contains 95-99 videos, with no duplicates.
3 — The dataset contains 90-94 videos, or includes a few duplicates.
2 — The dataset contains 80-89 videos, or includes several duplicates.
1 — The dataset contains fewer than 80 videos, or is riddled with duplicates.

#### D. Legal And Ethical Compliance
Checks whether the agent adhered to TikTok's terms of service and copyright laws.

5 — The agent provided clear evidence of full compliance with TikTok's terms of service and copyright laws.
4 — The agent provided some evidence of compliance, with minor gaps in documentation.
3 — The agent provided limited evidence of compliance, with noticeable gaps in documentation.
2 — The agent provided little evidence of compliance, with major gaps in documentation.
1 — The agent provided no evidence of compliance or clearly violated terms of service or copyright laws.

#### E. Video Quality Preservation
Assesses whether the downloaded videos maintain their original quality.

5 — All videos are downloaded in their original quality with no noticeable degradation.
4 — Most videos are downloaded in their original quality, with minor degradation in a few.
3 — Some videos show noticeable degradation in quality.
2 — Many videos show significant degradation in quality.
1 — The majority of the videos are of poor quality compared to their originals.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "relevance_to_topic": <1-5>,
  "ranking_accuracy": <1-5>,
  "dataset_completeness": <1-5>,
  "legal_and_ethical_compliance": <1-5>,
  "video_quality_preservation": <1-5>,
  "dimension_reasoning": {{
    "relevance_to_topic": "<one sentence citing specific evidence>",
    "ranking_accuracy": "<one sentence citing specific evidence>",
    "dataset_completeness": "<one sentence citing specific evidence>",
    "legal_and_ethical_compliance": "<one sentence citing specific evidence>",
    "video_quality_preservation": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "relevance_to_topic":  0.25,
    "ranking_accuracy":    0.25,
    "dataset_completeness": 0.2,
    "legal_and_ethical_compliance": 0.15,
    "video_quality_preservation": 0.15,
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