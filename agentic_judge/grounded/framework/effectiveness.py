"""Effectiveness judge — LLM-as-judge over the soft rubric dimensions.

This is the MAIN judge.  It preserves the gpt-4o-style anchor scoring
the v1 tasks already define, but is given three extra signals:

  1. The agent's structured summary JSON (already extracted).
  2. A hard-constraint report (machine-checkable predicates against the JSON).
  3. A faithfulness report (URL fetches verifying the agent's cited claims).

In addition, it can act agentically: if it is uncertain about whether a
cited URL actually supports a claim, OR whether the agent's "I couldn't
find X" claims are genuinely unanswerable from the cited source, it can
emit `pending_faith_questions` — a list of `{url, question}` items that
get dispatched to the Faithfulness LLM sub-agent.  The runner then calls
the Effectiveness judge a second time with the answers, so it can revise
its dimension scores with the new evidence.

This file exports:
  * call_llm_judge(...)            — first-pass scoring
  * revise_with_faith_answers(...) — second-pass scoring after Faith answers
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Any


_SYSTEM_PROMPT = """You are the MAIN evaluator scoring an AI agent's work
on a research/web task.  You have FOUR pre-computed signals:

  1. The agent's structured summary JSON (already extracted).
  2. A hard-constraint report (machine-checkable predicates against the JSON).
  3. A faithfulness report (URL fetches + claim verification on the URLs
     the agent put in its summary JSON).
  4. The agent's full stdout / trace — tool calls, fetched-page snippets,
     intermediate reasoning, error traces, follow-up searches.

You should READ THE STDOUT CAREFULLY.  Often the agent visits URLs that
do NOT appear in the final summary JSON — sources it consulted, pages it
opened to verify an answer, dead ends it explored.  These are
load-bearing for the verdict but are NOT in signal (3) by default.

You are also AGENTIC: you ORCHESTRATE the Webpage Grounding sub-agent
by emitting a `pending_faith_questions` list.  Use it for:

  - URLs the agent visited in stdout but did not include in the summary
    JSON — was the page content actually consistent with what the agent
    concluded?
  - the agent claims an item is unavailable/unanswerable on a specific
    source; confirm the source really lacks it
  - a citation's claim was not in the substring-match faithfulness
    report but might be paraphrased on the page
  - the agent cites a stat that is plausible but not verbatim in the
    page text shown in the faithfulness report

Each pending question is dispatched to the sub-agent (which fetches the
URL and replies with `{supported, evidence_quote, answer, reasoning}`);
we then call you a second time with the answers so you can revise dim
scores.

Use pending questions sparingly: at most 3 per task.  Only ask when the
answer would actually move a dim score.

REQUIRED USES (you MUST emit at least one pending question in these cases):

  R1.  If the initial faithfulness report shows a claim was NOT
       supported but you think the substring matcher may have missed a
       paraphrase, you MUST emit a pending question on that URL asking
       the sub-agent to confirm.  Do not silently override the faith
       signal with your prior — verify it.

  R2.  If the agent's stdout shows a URL that is NOT covered by the
       initial faith report and the answer to "does this page support
       what the agent inferred from it?" would change a dim score, you
       MUST emit a pending question on that URL.

Score honestly.  Anchoring discipline:

  A1.  When a hard constraint failed, the relevant dimension cannot be 5.

  A2.  When a URL fetch failed or a claim was unsupported by the live
       page, "specificity" / "data accuracy" / similar dimensions
       cannot be 5.

  A3.  HARD 5/5 DOES NOT MEAN ANY SOFT DIM IS 5.  Hard predicates check
       shape (count, range, schema, presence) — not semantic quality.
       An option priced $74.99 satisfies a budget cap even if its
       features are unrelated to the reference; the agent can satisfy
       every required field and still have produced a shallow answer.
       Read the agent's reasoning in stdout before assigning a 5; if you
       did not see real engagement with the task's substance, the soft
       dims should be 3 or 4, not 5."""


def _build_user_prompt(*,
                        task_instruction: str,
                        agent_response: str,
                        summary_json: dict | None,
                        hard_constraint_report: list[dict],
                        faithfulness_report: dict,
                        dimensions: list[str],
                        dimension_weights: dict[str, float],
                        task_specific_rubric: str,
                        agent_stdout: str = "",
                        answered_faith_questions: list[dict] | None = None,
                        prior_judgment: dict | None = None) -> str:
    summary_dump = json.dumps(summary_json, indent=2) if summary_json else "(no summary JSON parsed)"
    hard_dump = json.dumps(hard_constraint_report, indent=2)
    faith_dump = json.dumps({k: v for k, v in faithfulness_report.items() if k != "details"}, indent=2)
    faith_details = json.dumps(faithfulness_report.get("details", [])[:10], indent=2)
    dim_lines = "\n".join(f"- {d} (weight {dimension_weights[d]:.2f})" for d in dimensions)

    revise_block = ""
    if answered_faith_questions or prior_judgment:
        revise_block = (
            "\n\n## SECOND PASS — REVISE YOUR SCORES\n"
            "Your earlier scores are below, followed by the Faithfulness "
            "sub-agent's answers to the questions you raised.  Re-score "
            "with this new evidence; you may keep, lower, or raise any "
            "dimension.\n\n"
            f"### Your prior judgment\n{json.dumps(prior_judgment or {}, indent=2)}\n\n"
            f"### Answers from the Faithfulness sub-agent\n"
            f"{json.dumps(answered_faith_questions or [], indent=2)}\n"
        )

    stdout_section = ""
    if agent_stdout and agent_stdout.strip():
        # Show the tail of stdout — that's where tool-call results, fetched
        # pages, and the final answer live.  Earlier scrollback is usually
        # setup noise.
        s = agent_stdout.strip()
        tail = s[-12000:] if len(s) > 12000 else s
        stdout_section = (
            "\n\n## Agent stdout / trace (last "
            f"{len(tail)} of {len(s)} chars — includes tool calls, "
            "fetched-page snippets, intermediate reasoning, error traces)\n"
            f"{tail}"
        )

    return f"""## Task Instruction
{task_instruction}

## Agent Final Response (raw, truncated)
{(agent_response or "")[:8000]}{stdout_section}

## Agent Summary JSON (already extracted)
{summary_dump}

## Hard-constraint report
{hard_dump}

## Faithfulness report (URL fetches)
Headline: {faith_dump}
Details: {faith_details}

## Task-specific rubric (anchors)
{task_specific_rubric}

## Dimensions to score (1-5 each)
{dim_lines}{revise_block}

## Instructions
Score each dimension 1-5.  Use the hard-constraint and faithfulness
reports as ground truth.  If you are unsure about a citation or an
"unanswerable" claim, emit pending_faith_questions instead of guessing.

Return ONLY:

<Answer>
{{
  "evidence_summary": "<2-3 sentences>",
  {", ".join(f'"{d}": <1-5>' for d in dimensions)},
  "dimension_reasoning": {{
    {", ".join(f'"{d}": "<one sentence citing evidence>"' for d in dimensions)}
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true|false based on overall_score >= 3.0>,
  "pending_faith_questions": [
    {{"url": "<url to check>", "question": "<natural-language question>"}}
  ]
}}
</Answer>
"""


def _call_chat(model: str, system: str, user: str, max_retries: int = 4) -> dict:
    try:
        import openai
    except ImportError:
        return {"error": "openai library not available"}
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"),
                            base_url=os.environ.get("OPENAI_BASE_URL") or None)
    last_err = ""
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=1800,
            )
            return _parse_answer_tag(completion.choices[0].message.content or "")
        except Exception as e:
            last_err = f"{type(e).__name__}: {str(e)[:200]}"
            time.sleep(2 ** attempt)
    return {"error": last_err}


def call_llm_judge(*,
                    task_instruction: str,
                    agent_response: str,
                    summary_json: dict | None,
                    hard_constraint_report: list[dict],
                    faithfulness_report: dict,
                    dimensions: list[str],
                    dimension_weights: dict[str, float],
                    task_specific_rubric: str,
                    agent_stdout: str = "",
                    model: str | None = None,
                    max_retries: int = 4) -> dict:
    """First-pass judge call.  Returns the parsed dict from `<Answer>`
    tags, including any `pending_faith_questions`.  Or {"error": ...}."""
    model = model or os.environ.get("EVOLVEBENCH_JUDGE_MODEL", "gpt-4o")
    user = _build_user_prompt(
        task_instruction=task_instruction,
        agent_response=agent_response,
        summary_json=summary_json,
        hard_constraint_report=hard_constraint_report,
        faithfulness_report=faithfulness_report,
        dimensions=dimensions,
        dimension_weights=dimension_weights,
        task_specific_rubric=task_specific_rubric,
        agent_stdout=agent_stdout,
    )
    return _call_chat(model, _SYSTEM_PROMPT, user, max_retries=max_retries)


def revise_with_faith_answers(*,
                                task_instruction: str,
                                agent_response: str,
                                summary_json: dict | None,
                                hard_constraint_report: list[dict],
                                faithfulness_report: dict,
                                dimensions: list[str],
                                dimension_weights: dict[str, float],
                                task_specific_rubric: str,
                                prior_judgment: dict,
                                answered_faith_questions: list[dict],
                                agent_stdout: str = "",
                                model: str | None = None,
                                max_retries: int = 3) -> dict:
    """Second-pass judge call after the Faithfulness sub-agent has
    answered the pending questions.  Same shape as call_llm_judge."""
    model = model or os.environ.get("EVOLVEBENCH_JUDGE_MODEL", "gpt-4o")
    user = _build_user_prompt(
        task_instruction=task_instruction,
        agent_response=agent_response,
        summary_json=summary_json,
        hard_constraint_report=hard_constraint_report,
        faithfulness_report=faithfulness_report,
        dimensions=dimensions,
        dimension_weights=dimension_weights,
        task_specific_rubric=task_specific_rubric,
        agent_stdout=agent_stdout,
        answered_faith_questions=answered_faith_questions,
        prior_judgment=prior_judgment,
    )
    return _call_chat(model, _SYSTEM_PROMPT, user, max_retries=max_retries)


def _parse_answer_tag(text: str) -> dict:
    m = re.search(r"<Answer>(.*?)</Answer>", text, re.S | re.I)
    raw = (m.group(1) if m else text).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        return {"error": f"json parse: {e}", "raw": raw[:500]}
