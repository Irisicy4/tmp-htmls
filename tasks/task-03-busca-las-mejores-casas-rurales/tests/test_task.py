"""
LLM-as-judge evaluator for an evolvebench task.
Auto-generated from spreadsheet row.
"""

import os
import json

TASK_INSTRUCTION = """busca las mejores casas rurales donde hospedarse en Asturias. Que tengan al menos 4 estrellas. Buscalo en Tripavisor"""

RUBRIC_GENERIC = """
You are evaluating an AI agent's response to this task:
"busca las mejores casas rurales donde hospedarse en Asturias. Que tengan al menos 4 estrellas. Buscalo en Tripavisor"

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

RUBRIC = """
You are evaluating an AI agent's response to this travel research task (originally in Spanish):
"Search Tripadvisor for the best rural houses (casas rurales) to stay in Asturias, Spain, with at least 4 stars."

Context: The task requires the agent to navigate to Tripadvisor specifically and find rural accommodation (not hotels or apartments) in Asturias rated 4+ stars. Tripadvisor may block automated access via CAPTCHA — partial credit should be given if the agent clearly attempted navigation but was blocked by the platform.

Score on each dimension from 1 to 5:

1. platform_navigation (weight 0.30):
   Did the agent specifically attempt to use Tripadvisor?
   5 = agent visited tripadvisor.com and searched for rural accommodation in Asturias
   3 = agent reached Tripadvisor but was blocked (CAPTCHA/bot detection) after a genuine attempt, and clearly documented this
   1 = agent used a different platform (Booking.com, Google Maps, etc.) without attempting Tripadvisor, or made no navigation attempt

2. rating_compliance (weight 0.30):
   Do all listed properties meet the 4+ star requirement?
   5 = all listed properties have an explicit 4+ star rating from Tripadvisor
   3 = properties listed but ratings are approximate, unverified, or sourced from a different rating system
   1 = no ratings provided, properties below 4 stars included, or access was blocked before any ratings could be retrieved

3. listing_completeness (weight 0.25):
   Does each listed property include essential booking information?
   5 = each property includes name, star rating, location within Asturias, and price range or direct booking link
   3 = properties listed with name and rating but missing location detail or price
   1 = properties listed without ratings or booking-relevant detail, or no properties listed at all

4. result_quantity (weight 0.15):
   Did the agent find a useful number of qualifying rural houses?
   5 = 5+ distinct rural houses listed, all meeting the 4+ star and Asturias criteria
   3 = 3-4 qualifying properties found
   1 = fewer than 3 properties found, or no specific properties listed

Respond ONLY with valid JSON:
{
  "platform_navigation": <1-5>,
  "rating_compliance": <1-5>,
  "listing_completeness": <1-5>,
  "result_quantity": <1-5>,
  "reasoning": "<one paragraph explaining the scores>",
  "overall_score": <weighted average 0.30*platform_navigation + 0.30*rating_compliance + 0.25*listing_completeness + 0.15*result_quantity as a single decimal>
}
"""

PASS_THRESHOLD = 3.0
DIMENSIONS = ["platform_navigation", "rating_compliance", "listing_completeness", "result_quantity"]
DIMENSIONS_GENERIC = ["task_completion", "information_quality", "response_quality", "completeness"]


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


def _call_judge(agent_response: str, execution_summary: str = "", rubric: str = None) -> dict:
    if rubric is None:
        rubric = RUBRIC
    try:
        import openai
        api_key = os.environ.get("OPENAI_API_KEY")
        base_url = os.environ.get("OPENAI_BASE_URL") or None
        if not api_key:
            return {"error": "OPENAI_API_KEY not set (required for LLM judge)", "overall_score": 0}
        client = openai.OpenAI(api_key=api_key, base_url=base_url)
        content = f"{rubric}\n\nAgent response to evaluate:\n\n{agent_response}"
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

    scores = _call_judge(agent_response, execution_summary, RUBRIC)
    scores_generic = _call_judge(agent_response, execution_summary, RUBRIC_GENERIC)

    overall = scores.get("overall_score", 0)
    overall_generic = scores_generic.get("overall_score", 0)
    passed = float(overall) >= PASS_THRESHOLD

    feedback_lines = [f"=== Customized Rubric Score: {overall}/5 ==="]
    if "error" in scores:
        feedback_lines.append(f"  [ERROR: {scores['error']}]")
    for dim in DIMENSIONS:
        if dim in scores:
            feedback_lines.append(f"  {dim}: {scores[dim]}/5")
    if "reasoning" in scores:
        feedback_lines.append(f"\nCustomized reasoning: {scores['reasoning']}")

    feedback_lines.append(f"\n=== Generic Rubric Score: {overall_generic}/5 ===")
    if "error" in scores_generic:
        feedback_lines.append(f"  [ERROR: {scores_generic['error']}]")
    for dim in DIMENSIONS_GENERIC:
        if dim in scores_generic:
            feedback_lines.append(f"  {dim}: {scores_generic[dim]}/5")
    if "reasoning" in scores_generic:
        feedback_lines.append(f"\nGeneric reasoning: {scores_generic['reasoning']}")

    return {
        "passed": passed,
        "feedback": "\n".join(feedback_lines),
        "details": {
            "task_completed": result.get("status") == "success",
            "overall_score": overall,
            "dimension_scores": {k: scores.get(k) for k in DIMENSIONS},
            "judge_reasoning": scores.get("reasoning"),
            "pass_threshold": PASS_THRESHOLD,
            "generic_score": overall_generic,
            "generic_dimension_scores": {k: scores_generic.get(k) for k in DIMENSIONS_GENERIC},
            "generic_reasoning": scores_generic.get("reasoning"),
        },
    }
