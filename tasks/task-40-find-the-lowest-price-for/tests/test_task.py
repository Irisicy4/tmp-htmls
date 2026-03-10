"""
LLM-as-judge evaluator for an evolvebench task.
Auto-generated from spreadsheet row.
"""

import os
import json

TASK_INSTRUCTION = """Find the lowest price for Numi organic tea flower tea gift set across the web."""

RUBRIC = """
You are evaluating an AI agent's response to this task:
"Find the lowest price for Numi organic tea flower tea gift set across the web."

Score on each dimension from 1 to 5:

1. task_completion (weight 0.36):
   Did the agent complete the core request?
   5 = fully completed all required steps
   3 = partially completed or missing minor details
   1 = failed or gave up without completing the main task

2. information_quality (weight 0.27):
   Is the retrieved information accurate, current, and from credible sources?
   5 = accurate, up-to-date, with credible sources cited
   3 = mostly accurate but vague or missing some sources
   1 = hallucinated, inaccurate, or no sources

3. response_quality (weight 0.18):
   Is the response comprehensive, well-organised, and useful?
   5 = thorough, well-structured, actionable
   3 = adequate but could be more detailed
   1 = sparse, disorganised, or not useful

4. completeness (weight 0.18):
   Did the agent address all aspects and requirements of the task?
   5 = all requirements fully addressed
   3 = most requirements met, minor gaps
   1 = significant requirements missed

Respond ONLY with valid JSON:
{
  "task_completion": <1-5>,
  "information_quality": <1-5>,
  "response_quality": <1-5>,
  "completeness": <1-5>,
  "reasoning": "<one paragraph explaining the scores>",
  "overall_score": <weighted average as a single decimal number>
}
"""

PASS_THRESHOLD = 3.0
DIMENSIONS = ["task_completion", "information_quality", "response_quality", "completeness"]


def _extract_response(result: dict) -> str:
    task_result = result.get("task_result") or ""
    if task_result.strip():
        return task_result
    for message in reversed(result.get("conversation") or []):
        if not isinstance(message, dict):
            continue
        if message.get("role") == "assistant":
            content = message.get("content") or ""
            if isinstance(content, str) and len(content) > 20:
                return content
    return ""


def _call_judge(agent_response: str, execution_summary: str = "") -> dict:
    try:
        import openai
        api_key = os.environ.get("OPENAI_API_KEY")
        base_url = os.environ.get("OPENAI_BASE_URL") or None
        if not api_key:
            return {"error": "OPENAI_API_KEY not set (required for LLM judge)", "overall_score": 0}
        client = openai.OpenAI(api_key=api_key, base_url=base_url)
        content = f"{RUBRIC}\n\nAgent response to evaluate:\n\n{agent_response}"
        if execution_summary:
            content += f"\n\nVerified agent tool-call trace (ground truth of what the agent actually did):\n{execution_summary}"
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": content}],
            response_format={"type": "json_object"},
            max_tokens=512,
        )
        return json.loads(completion.choices[0].message.content)
    except Exception as e:
        return {"error": str(e), "overall_score": 0}


def test(result: dict) -> dict:
    agent_response = _extract_response(result)
    execution_summary = result.get("execution_summary", "")

    if not agent_response.strip():
        return {
            "passed": False,
            "feedback": "No response found from agent.",
            "details": {"task_completed": result.get("status") == "success"},
        }

    scores = _call_judge(agent_response, execution_summary)
    overall = scores.get("overall_score", 0)
    passed = float(overall) >= PASS_THRESHOLD

    feedback_lines = [f"Overall score: {overall}/5"]
    for dim in DIMENSIONS:
        if dim in scores:
            feedback_lines.append(f"  {dim}: {scores[dim]}/5")
    if "reasoning" in scores:
        feedback_lines.append(f"\nJudge reasoning: {scores['reasoning']}")

    return {
        "passed": passed,
        "feedback": "\n".join(feedback_lines),
        "details": {
            "task_completed": result.get("status") == "success",
            "overall_score": overall,
            "dimension_scores": {k: scores.get(k) for k in DIMENSIONS},
            "judge_reasoning": scores.get("reasoning"),
            "pass_threshold": PASS_THRESHOLD,
        },
    }
