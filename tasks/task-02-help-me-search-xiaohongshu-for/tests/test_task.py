"""
LLM-as-judge evaluator for an evolvebench task.
Auto-generated from spreadsheet row.
"""

import os
import json

TASK_INSTRUCTION = """Help me search Xiaohongshu for all interview questions related to the Chinese University of Hong Kong (Shenzhen)/Hong Kong-China-Shenzhen Artificial Intelligence and Robotics (MAIR) and summarize them into a list"""

RUBRIC_GENERIC = """
You are evaluating an AI agent's response to this task:
"Help me search Xiaohongshu for all interview questions related to the Chinese University of Hong Kong (Shenzhen)/Hong Kong-China-Shenzhen Artificial Intelligence and Robotics (MAIR) and summarize them into a list"

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
You are evaluating an AI agent's response to this social media research task:
"Search Xiaohongshu (小红书, xiaohongshu.com) for interview questions about the CUHK-Shenzhen (港中深) MAIR (Artificial Intelligence and Robotics) graduate program, and summarize them into a list."

Context: Xiaohongshu is a Chinese social platform. The target program is CUHK(SZ) MAIR — a competitive graduate program where users share interview experiences. The agent must navigate to Xiaohongshu and search for this specific content, which may be in Chinese.

Score on each dimension from 1 to 5:

1. platform_specificity (weight 0.30):
   Did the agent specifically navigate to and search on Xiaohongshu?
   5 = agent visited xiaohongshu.com (or app), searched for MAIR/CUHK-Shenzhen interview content in Chinese or English
   3 = agent attempted Xiaohongshu but may have fallen back to web search, or reached the site but couldn't extract content
   1 = agent used generic web search only, never attempted Xiaohongshu, or used a completely different platform

2. content_coverage (weight 0.30):
   How many distinct interview questions or question topics were found and listed?
   5 = 5+ distinct interview questions or question categories from Xiaohongshu posts
   3 = 2-4 distinct questions or topics found, or a general description of question types without specific examples
   1 = 0-1 specific questions found, or content is fabricated without source attribution

3. program_relevance (weight 0.25):
   Are the collected questions specifically about CUHK-Shenzhen MAIR, not generic AI/ML interviews?
   5 = questions clearly reference CUHK-Shenzhen MAIR specifics (e.g., program structure, application process, specific interview style, faculty)
   3 = mix of MAIR-specific context and generic AI/robotics interview questions
   1 = questions are entirely generic AI interview prep with no CUHK-Shenzhen connection

4. list_format (weight 0.15):
   Is the output presented as an organized list?
   5 = clearly numbered or bulleted list, optionally grouped by topic (technical, English interview, motivation, etc.)
   3 = list-like structure present but inconsistent formatting, or items lumped in prose
   1 = no list structure; questions buried in unformatted paragraphs

Respond ONLY with valid JSON:
{
  "platform_specificity": <1-5>,
  "content_coverage": <1-5>,
  "program_relevance": <1-5>,
  "list_format": <1-5>,
  "reasoning": "<one paragraph explaining the scores>",
  "overall_score": <weighted average 0.30*platform_specificity + 0.30*content_coverage + 0.25*program_relevance + 0.15*list_format as a single decimal>
}
"""

PASS_THRESHOLD = 3.0
DIMENSIONS = ["platform_specificity", "content_coverage", "program_relevance", "list_format"]
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
