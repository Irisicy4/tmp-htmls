import os
import json
import re

PASS_THRESHOLD = 3.0
DEFAULT_MODEL = "gpt-4o"
DEFAULT_TEMPERATURE = 0.0

SYSTEM_PROMPT = (
    "You are an expert evaluator for AI agent benchmarks. "
    "You will receive a task instruction, the agent's response, an execution summary, "
    "and a set of evaluation instructions. "
    "You must follow the evaluation instructions exactly and return your assessment "
    "as a JSON object wrapped in <Answer>...</Answer> tags."
)


def _extract_response(result: dict) -> str:
    task_result = result.get("task_result")
    if isinstance(task_result, str) and task_result.strip():
        return task_result

    for message in reversed(result.get("conversation", [])):
        if message.get("role") == "assistant":
            content = message.get("content", "")
            if isinstance(content, str) and len(content) > 20:
                return content

    return ""


def _parse_answer_tag(text: str) -> dict | None:
    match = re.search(r"<Answer>(.*?)</Answer>", text, re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(1).strip())
    except (json.JSONDecodeError, ValueError):
        return None


def _call_judge(prompt: str, temperature: float, model: str) -> dict | None:
    import openai

    try:
        client = openai.OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=os.environ.get("OPENAI_BASE_URL"),
        )
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=1500,
        )
        raw = response.choices[0].message.content or ""
        return _parse_answer_tag(raw)
    except Exception as e:
        return {"error": str(e)}


def _failed_result(feedback: str, agent_response: str = "", model: str = DEFAULT_MODEL,
                   temperature: float = DEFAULT_TEMPERATURE, call_index: int = 1,
                   task_completed: bool = False) -> dict:
    return {
        "passed": False,
        "feedback": feedback,
        "details": {
            "overall_score": 0.0,
            "dimension_scores": {},
            "evidence_summary": "",
            "dimension_reasoning": {},
            "pass_threshold": PASS_THRESHOLD,
            "agent_response": agent_response,
            "model": model,
            "temperature": temperature,
            "call_index": call_index,
            "task_completed": task_completed,
        },
    }


def evaluate(
    result: dict,
    task_instruction: str,
    prompt_template: str,
    dimension_weights: dict,
    call_index: int = 1,
    temperature: float = DEFAULT_TEMPERATURE,
    model: str = DEFAULT_MODEL,
) -> dict:
    agent_response = _extract_response(result)
    if not agent_response:
        return _failed_result(
            "No response found from agent.",
            model=model,
            temperature=temperature,
            call_index=call_index,
            task_completed=result.get("status") == "success",
        )

    execution_summary = result.get("execution_summary", "Not available.")
    prompt = prompt_template.format(
        task_instruction=task_instruction,
        agent_response=agent_response,
        execution_summary=execution_summary,
    )

    judge_output = _call_judge(prompt, temperature, model)

    if judge_output is None or "error" in judge_output:
        error_msg = judge_output.get("error", "Judge returned no parseable output.") if judge_output else "Judge returned no parseable output."
        return _failed_result(
            f"Judge call failed: {error_msg}",
            agent_response=agent_response,
            model=model,
            temperature=temperature,
            call_index=call_index,
            task_completed=result.get("status") == "success",
        )

    dimension_scores = {dim: judge_output[dim] for dim in dimension_weights if dim in judge_output}
    evidence_summary = judge_output.get("evidence_summary", "")
    dimension_reasoning = judge_output.get("dimension_reasoning", {})

    overall_score = round(
        sum(dimension_scores[d] * dimension_weights[d] for d in dimension_scores),
        2,
    )

    passed = overall_score >= PASS_THRESHOLD

    score_lines = "\n".join(
        f"  {dim}: {dimension_scores.get(dim, 'N/A')}" for dim in dimension_weights
    )
    reasoning_lines = "\n".join(
        f"  {dim}: {dimension_reasoning.get(dim, '')}" for dim in dimension_weights
    )
    feedback = (
        f"Overall score: {overall_score} ({'PASS' if passed else 'FAIL'})\n"
        f"Dimension scores:\n{score_lines}\n"
        f"Evidence summary: {evidence_summary}\n"
        f"Dimension reasoning:\n{reasoning_lines}"
    )

    return {
        "passed": passed,
        "feedback": feedback,
        "details": {
            "overall_score": overall_score,
            "dimension_scores": dimension_scores,
            "evidence_summary": evidence_summary,
            "dimension_reasoning": dimension_reasoning,
            "pass_threshold": PASS_THRESHOLD,
            "agent_response": agent_response,
            "model": model,
            "temperature": temperature,
            "call_index": call_index,
            "task_completed": result.get("status") == "success",
        },
    }
