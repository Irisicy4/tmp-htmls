"""
LLM-as-judge evaluator for EvolveBench task-06.

Category: (Self) Media
Task: "help me download a song named happy birthday from youtube"

Note: This task involves a download action. Success requires the agent to
identify a specific YouTube video AND produce or initiate an actual download,
not merely provide a link or instructions.
"""

import os, json, re

TASK_INSTRUCTION = "help me download a song named happy birthday from youtube"
PASS_THRESHOLD = 3.0

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a media download task from YouTube.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Platform: YouTube specifically
- Content: a "Happy Birthday" song — any widely recognised version is acceptable
- Action required: an actual download must be attempted or completed, not just a link provided or instructions given
- File output: ideally an audio file (mp3, m4a, wav) or video file; the agent should confirm what was downloaded

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to YouTube? Cite evidence.
- Did the agent identify a specific "Happy Birthday" video (title, channel, URL)?
- Did the agent attempt or complete a download? What method was used (yt-dlp, browser download, third-party tool)?
- Is there evidence the file was actually saved (file path, confirmation message, download completion)?
- Did the agent only provide instructions or a link instead of executing the download?

### Step 2: Dimension Scoring

#### A. Platform Execution
Did the agent actually navigate YouTube to find the content?

5 — Agent navigated YouTube, identified a specific Happy Birthday video with title/URL.
4 — Agent found a YouTube video but with limited detail (e.g. URL only, no title or channel).
3 — Agent searched YouTube but did not access a specific video page.
2 — Agent used a non-YouTube source or only searched generally without finding a specific video.
1 — No search performed; response is instructions only or from prior knowledge.

#### B. Download Execution
Did the agent actually attempt or complete the download?

5 — Download completed; file saved with confirmation (file path, size, or format mentioned).
4 — Download initiated and likely completed based on trace; no explicit confirmation in response.
3 — Download attempted but outcome uncertain (e.g. tool invoked but no success confirmation).
2 — Agent provided a download link or instructions but did not execute the download itself.
1 — No download attempted; agent only described how to download or refused.

#### C. Content Match
Is the downloaded or identified content a "Happy Birthday" song as requested?

5 — Content is clearly a Happy Birthday song (title confirms it; well-known version).
4 — Content is likely a Happy Birthday song but title is ambiguous or version is obscure.
3 — Agent found Happy Birthday content but it may be a cover, instrumental, or unrelated remix.
2 — Content is loosely related (birthday music) but not "Happy Birthday" specifically.
1 — Content is unrelated or agent could not find any matching content.

#### D. Response Clarity
Does the agent clearly communicate what was found and downloaded?

5 — Response states: video title, source URL, download method, and file format/location.
4 — Most details present; one piece missing (e.g. no file path or no format mentioned).
3 — Agent confirms a download but with minimal detail.
2 — Response is vague; unclear whether anything was actually downloaded.
1 — No useful output; response is a refusal, error, or generic instructions.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "platform_execution": <1-5>,
  "download_execution": <1-5>,
  "content_match": <1-5>,
  "response_clarity": <1-5>,
  "dimension_reasoning": {{
    "platform_execution": "<one sentence citing specific evidence>",
    "download_execution": "<one sentence citing specific evidence>",
    "content_match": "<one sentence citing specific evidence>",
    "response_clarity": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "platform_execution":  0.20,
    "download_execution":  0.40,
    "content_match":       0.25,
    "response_clarity":    0.15,
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