"""
LLM-as-judge evaluator for task-01 (backpack research).
"""

import os
import json

TASK_INSTRUCTION = """im looking for a backpack under $75 that has all the features of this one: https://www.amazon.com/dp/B09YRC9Y3G please do some research and find 3-5 optionsand summarize their key features and prices, comparing them to the original."""

RUBRIC_GENERIC = """
You are evaluating an AI agent's response to this task:
"im looking for a backpack under $75 that has all the features of this one: https://www.amazon.com/dp/B09YRC9Y3G please do some research and find 3-5 optionsand summarize their key features and prices, comparing them to the original."

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
You are evaluating an AI agent's response to this shopping research task:
"Find 3-5 backpacks under $75 with features similar to https://www.amazon.com/dp/B09YRC9Y3G (a travel laptop backpack). Summarize key features and prices, comparing alternatives to the original."

The original product (B09YRC9Y3G) is a travel laptop backpack. Key expected features include: laptop compartment, USB charging port, multiple pockets/compartments, water resistance, travel-friendly design.

Score on each dimension from 1 to 5:

1. source_navigation (weight 0.20):
   Did the agent visit the original Amazon URL to extract the reference product's specific features?
   5 = clearly accessed the URL and listed specific features (compartments, USB port, dimensions, material, etc.)
   3 = referenced some plausible features of the product but details are vague or generic
   1 = made no attempt to check the original URL or invented the features

2. budget_compliance (weight 0.25):
   Are all recommended alternatives explicitly priced under $75?
   5 = all 3-5 alternatives have stated prices under $75
   3 = most alternatives are priced under $75, or some prices are missing
   1 = prices missing for most alternatives, or alternatives exceed $75

3. recommendation_count (weight 0.15):
   Did the agent find the required 3-5 distinct alternatives?
   5 = exactly 3-5 distinct product names or links provided
   3 = 2 alternatives found, or products lack sufficient identification
   1 = fewer than 2 alternatives, or no specific products named

4. feature_comparison (weight 0.25):
   Are alternatives compared against the original on specific features?
   5 = each alternative compared on 3+ features (e.g., laptop size, USB port, compartments, material, capacity)
   3 = comparison present but limited to 1-2 features (e.g., price only or general category)
   1 = no meaningful feature comparison with the original product

5. actionability (weight 0.15):
   Is the response structured for easy purchase decision-making?
   5 = clear list or table format with product identifiers (name, link, or ASIN) and a brief recommendation
   3 = structured output but lacking product links or a concluding recommendation
   1 = unstructured prose with no product identifiers or navigation cues

Respond ONLY with valid JSON:
{
  "source_navigation": <1-5>,
  "budget_compliance": <1-5>,
  "recommendation_count": <1-5>,
  "feature_comparison": <1-5>,
  "actionability": <1-5>,
  "reasoning": "<one paragraph explaining the scores>",
  "overall_score": <weighted average 0.20*source_navigation + 0.25*budget_compliance + 0.15*recommendation_count + 0.25*feature_comparison + 0.15*actionability as a single decimal>
}
"""

PASS_THRESHOLD = 3.0
DIMENSIONS = ["source_navigation", "budget_compliance", "recommendation_count", "feature_comparison", "actionability"]
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
