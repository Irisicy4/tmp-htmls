"""
LLM-as-judge evaluator for EvolveBench task-19.

Category: Travel & Planning
Task: "I am now in central London. Please tell me 10 places with a good view of Big Ben,
       their addresses, and for each place provide directions from the city center, nearby
       attractions, and nearby restaurants."
"""

import os
import json

TASK_INSTRUCTION = """I am now in central London. Please tell me 10 places with a good view of Big Ben, their addresses, and for each place provide directions from the city center, nearby attractions, and nearby restaurants."""

RUBRIC = """
You are evaluating an AI agent's response to this task:
"I am now in central London. Please tell me 10 places with a good view of Big Ben, their addresses, and for each place provide directions from the city center, nearby attractions, and nearby restaurants."

The agent should identify London locations known for views of Big Ben (e.g. Westminster Bridge, Jubilee Gardens, St James's Park, Southbank, Lambeth Bridge, Victoria Tower Gardens, Parliament Square, Hungerford Bridge, etc.) and for each provide address, directions from central London, nearby English/British attractions, and nearby restaurants.

Score on each dimension from 1 to 5:

1. task_completion (weight 0.36):
   Did the agent complete the core request?
   5 = fully completed all required steps: 10 London locations with Big Ben views, all four sub-details per location
   3 = partially completed or missing minor details (e.g. fewer than 10 locations, or missing one sub-detail for several)
   1 = failed or gave up without completing the main task

2. information_quality (weight 0.27):
   Is the retrieved information accurate, current, and from credible sources?
   5 = accurate London geography, correct addresses, credible sources cited
   3 = mostly accurate but vague or missing some sources; minor London geography errors
   1 = hallucinated locations, wrong city, inaccurate, or no sources

3. response_quality (weight 0.18):
   Is the response comprehensive, well-organised, and useful?
   5 = thorough, well-structured, actionable — each of the 10 locations clearly laid out with all four required elements
   3 = adequate but could be more detailed; structure present but inconsistent
   1 = sparse, disorganised, or not useful

4. completeness (weight 0.18):
   Did the agent address all aspects and requirements of the task?
   5 = all requirements fully addressed: 10 places, addresses, directions from city centre, nearby attractions, nearby restaurants
   3 = most requirements met, minor gaps (e.g. missing restaurants for a few locations)
   1 = significant requirements missed (fewer than 7 locations, or directions/restaurants absent for most)

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
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
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
