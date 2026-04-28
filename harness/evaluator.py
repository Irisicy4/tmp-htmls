import os
import json
import re

PASS_THRESHOLD = 3.0
DEFAULT_MODEL = os.environ.get("LLM_MODEL", "gpt-4.1-mini")
DEFAULT_TEMPERATURE = 0.0

SYSTEM_PROMPT = (
    "You are an expert evaluator for AI agent benchmarks. "
    "You will receive a task instruction, the agent's response, an execution summary, "
    "and a set of evaluation instructions. "
    "Return your assessment as a JSON object wrapped in <Answer>...</Answer> tags."
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


# ---------------------------------------------------------------------------
# Structured-rubric API used by updated-deprivacy-100 tasks.
#
# Task files supply only the task-specific pieces (instruction, constraints,
# evidence questions, dimension rubrics, dimension weights). All shared
# scaffolding (system prompt, section headers, JSON output schema, helper
# logic, borderline majority-vote) lives here.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_HEADER = "You are an expert evaluator for AI agent benchmarks."

SYSTEM_PROMPT_FOOTER = (
    "You will be given:\n"
    "1. The task instruction\n"
    "2. The agent's final response\n"
    "3. A summary of the agent's tool-call trace (what it actually did, if available)\n"
    "\n"
    "Follow the evaluation steps exactly in order."
)


def build_system_prompt(extra: str = "") -> str:
    extra = (extra or "").strip()
    head = f"{SYSTEM_PROMPT_HEADER} {extra}".strip() if extra else SYSTEM_PROMPT_HEADER
    return f"{head}\n\n{SYSTEM_PROMPT_FOOTER}"


def build_user_prompt(
    task_instruction: str,
    agent_response: str,
    execution_summary: str,
    constraints: str,
    evidence_questions: str,
    dimension_rubrics: str,
    dimension_weights: dict,
) -> str:
    dim_names = list(dimension_weights.keys())
    json_score_fields = ",\n  ".join(f'"{d}": <1-5>' for d in dim_names)
    json_reasoning_fields = ",\n    ".join(
        f'"{d}": "<one sentence citing specific evidence>"' for d in dim_names
    )
    return (
        f"## Task Instruction\n{task_instruction}\n\n"
        f"## Task-Specific Constraints\n{constraints}\n\n"
        f"## Agent Final Response\n{agent_response}\n\n"
        f"## Agent Tool-Call Trace (what the agent actually did)\n{execution_summary or 'Not available.'}\n\n"
        f"---\n\n"
        f"## Evaluation Instructions\n\n"
        f"### Step 1: Evidence Analysis (do this before scoring)\n{evidence_questions}\n\n"
        f"### Step 2: Dimension Scoring\n\n{dimension_rubrics}\n\n"
        f"### Step 3: Output\n"
        f"Respond ONLY with valid JSON inside <Answer></Answer> tags:\n\n"
        f"<Answer>\n"
        f"{{\n"
        f'  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",\n'
        f"  {json_score_fields},\n"
        f'  "dimension_reasoning": {{\n'
        f"    {json_reasoning_fields}\n"
        f"  }},\n"
        f'  "overall_score": <weighted average, one decimal>,\n'
        f'  "passed": <true or false based on overall_score >= 3.0>\n'
        f"}}\n"
        f"</Answer>\n"
    )


def _judge_call(system_prompt: str, user_prompt: str, model: str, temperature: float) -> dict | None:
    import openai
    try:
        client = openai.OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=os.environ.get("OPENAI_BASE_URL"),
        )
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=1500,
        )
        return _parse_answer_tag(resp.choices[0].message.content or "")
    except Exception as e:
        return {"error": str(e)}


def _aggregate_votes(votes: list, dim_names: list, weights: dict) -> dict:
    valid = [v for v in votes if v and "error" not in v and all(d in v for d in dim_names)]
    if not valid:
        return votes[0] if votes else {"error": "All judge calls failed"}
    agg = {d: sorted([v[d] for v in valid])[len(valid) // 2] for d in dim_names}
    overall = round(sum(agg[d] * weights[d] for d in dim_names), 2)
    median_call = sorted(valid, key=lambda v: abs(v.get("overall_score", 0) - overall))[0]
    return {
        **agg,
        "overall_score": overall,
        "passed": overall >= PASS_THRESHOLD,
        "evidence_summary": median_call.get("evidence_summary", ""),
        "dimension_reasoning": median_call.get("dimension_reasoning", {}),
        "_votes_used": len(valid),
    }


def run_judge(
    result: dict,
    *,
    task_instruction: str,
    constraints: str,
    evidence_questions: str,
    dimension_rubrics: str,
    dimension_weights: dict,
    system_prompt_extra: str = "",
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
) -> dict:
    """One-shot judge with borderline majority-vote (3 calls when |score - 3.0| <= 0.5)."""
    agent_response = _extract_response(result)
    execution_summary = result.get("execution_summary", "")
    if not agent_response.strip():
        return {
            "passed": False,
            "feedback": "No response found from agent.",
            "details": {"task_completed": result.get("status") == "success"},
        }

    system_prompt = build_system_prompt(system_prompt_extra)
    user_prompt = build_user_prompt(
        task_instruction, agent_response, execution_summary,
        constraints, evidence_questions, dimension_rubrics, dimension_weights,
    )
    dim_names = list(dimension_weights.keys())

    first = _judge_call(system_prompt, user_prompt, model, temperature)
    if first and "error" not in first:
        overall = float(first.get("overall_score", 0))
        if abs(overall - PASS_THRESHOLD) <= 0.5:
            extra_calls = [
                _judge_call(system_prompt, user_prompt, model, temperature),
                _judge_call(system_prompt, user_prompt, model, temperature),
            ]
            scores = _aggregate_votes([first] + extra_calls, dim_names, dimension_weights)
        else:
            scores = first
            scores["_votes_used"] = 1
    else:
        scores = first or {"error": "Judge call failed", "overall_score": 0}

    overall = scores.get("overall_score", 0)
    passed = scores.get("passed", float(overall) >= PASS_THRESHOLD)
    lines = [f"Overall score: {overall}/5  (threshold: {PASS_THRESHOLD})"]
    for dim in dim_names:
        if dim in scores:
            lines.append(f"  {dim}: {scores[dim]}/5")
    if scores.get("evidence_summary"):
        lines.append(f"\nEvidence summary: {scores['evidence_summary']}")
    reasoning = scores.get("dimension_reasoning", {})
    if reasoning:
        lines.append("\nDimension reasoning:")
        for dim, reason in reasoning.items():
            lines.append(f"  {dim}: {reason}")
    if scores.get("_votes_used", 1) > 1:
        lines.append(f"\n(Borderline case: {scores['_votes_used']} judge calls used, majority vote applied)")

    return {
        "passed": bool(passed),
        "feedback": "\n".join(lines),
        "details": {
            "task_completed": result.get("status") == "success",
            "overall_score": overall,
            "dimension_scores": {d: scores.get(d) for d in dim_names},
            "evidence_summary": scores.get("evidence_summary"),
            "dimension_reasoning": scores.get("dimension_reasoning"),
            "pass_threshold": PASS_THRESHOLD,
            "votes_used": scores.get("_votes_used", 1),
            "model": model,
        },
    }
