"""
LLM-as-judge evaluator for EvolveBench task-02.

Category: HR & Recruiting
Approach: Hybrid — category-level rubric axes (HR & Recruiting) + task-specific constraint injection.
Design grounded in:
  - AutoMetrics: axes-from-examples pipeline, observable 5-point rubric anchors
  - SWE-ABS: forced analysis before verdict, orthogonal sub-dimensions, structured output tag

Task:
  "Help me search Xiaohongshu for all interview questions related to the National University
   of Singapore (NUS) and summarize them into a list"
"""

import os
import json
import re

# ---------------------------------------------------------------------------
# Task metadata
# ---------------------------------------------------------------------------

TASK_INSTRUCTION = (
    "Help me search Xiaohongshu for all interview questions related to the National University "
    "of Singapore (NUS) and summarize them into a list"
)

PASS_THRESHOLD = 3.0

# ---------------------------------------------------------------------------
# Judge prompt
#
# Structure (from SWE-ABS pattern):
#   1. Forced analysis section — prevents jumping to verdict
#   2. Per-dimension scoring — orthogonal axes, each with observable anchors
#   3. Structured output tag — enables reliable parsing + majority voting
#
# Axes (HR & Recruiting category):
#   A. Platform Execution        — did the agent actually use Xiaohongshu as instructed?
#   B. Source Coverage           — how many posts/sources were consulted and synthesized?
#   C. Content Relevance         — are the questions specifically about NUS?
#   D. Synthesis Quality         — is the output a clean, usable list of interview questions?
#
# Task-specific constraints injected:
#   - Platform: Xiaohongshu (小红书) specifically — not Zhihu, Baidu, or general web
#   - Program: NUS specifically — not generic Singapore universities or unrelated programs
#   - Output format: summarized list of interview questions
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based research task on a Chinese social platform (Xiaohongshu / 小红书).

You will be given:
1. The task instruction the agent was given
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

You must evaluate the agent's performance across four dimensions. Follow the evaluation steps exactly in order."""


USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Platform: The agent must search Xiaohongshu (小红书) specifically — not Zhihu, Baidu, general web search, or any other platform
- Program specificity: Questions must be about NUS (National University of Singapore) — not generic Singapore university admissions or unrelated programs
- Output format: A summarized list of interview questions (not a narrative description, not a single paragraph)
- Language note: Source content may be in Chinese; the agent may respond in English or Chinese — both are acceptable as long as the questions are clearly conveyed

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
Carefully read the agent response and trace. Then answer each of the following questions with a direct observation — do not infer or assume:

- Did the agent navigate to or search on Xiaohongshu? Cite evidence from the trace or response.
- How many distinct interview questions or question topics did the agent surface? List them briefly.
- Are the questions specifically about NUS, or are they generic interview questions?
- Did the agent synthesize from multiple posts/sources, or just one?
- Is the output structured as a list, or presented as prose/narrative?
- Did the agent acknowledge any limitations (e.g. login required, few results found)?

### Step 2: Dimension Scoring
Score each dimension from 1 to 5 using the rubrics below. Each score must be justified by specific evidence from Step 1.

#### A. Platform Execution
Did the agent actually use Xiaohongshu as instructed, rather than substituting another source?

5 — Clear evidence in trace and/or response that the agent searched Xiaohongshu; multiple posts accessed.
4 — Agent used Xiaohongshu but accessed only 1–2 posts, or trace confirms platform but response doesn't cite it.
3 — Ambiguous: response mentions Xiaohongshu but trace shows no platform navigation, or agent used Xiaohongshu alongside other platforms without distinguishing sources.
2 — Agent used a different platform (e.g. Zhihu, Baidu) and did not use Xiaohongshu at all.
1 — Agent did not perform any search; response is hallucinated or generated from prior knowledge only.

#### B. Source Coverage
How thoroughly did the agent search and synthesize across multiple posts?

5 — Agent consulted 3+ distinct Xiaohongshu posts and synthesized questions across all of them into a unified list.
4 — Agent consulted 2–3 posts; synthesis is present but one source may be underrepresented.
3 — Agent consulted 1–2 posts; questions are drawn from a narrow source base.
2 — Agent found results but only skimmed or summarized 1 post superficially.
1 — No sources consulted; content appears fabricated or copied from a single non-Xiaohongshu source.

#### C. Content Relevance
Are the surfaced questions specifically relevant to NUS interviews?

5 — All or nearly all questions are clearly specific to NUS (e.g. reference the university name, specific programs, NUS-specific topics, program-specific logistics).
4 — Most questions are relevant to NUS; 1–2 are generic postgraduate interview questions that could apply anywhere.
3 — Questions are relevant to graduate programs generally but lack NUS specificity.
2 — Questions are generic interview questions (e.g. "why do you want to study here?") with no program-specific content.
1 — Questions are off-topic, fabricated, or unrelated to the program.

#### D. Synthesis Quality
Is the final output a clean, usable list of interview questions?

5 — Output is a well-organized list with 5+ distinct questions; questions are clearly worded and directly usable as interview prep material.
4 — Output is a list with 3–5 questions; minor formatting issues or slight redundancy between items.
3 — Output contains interview question content but is presented as prose or a loosely structured summary rather than a clean list.
2 — Output mentions interview topics but does not present actual questions; or list has fewer than 3 items.
1 — Output is a refusal, an error message, or contains no interview question content.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags, exactly as shown:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising your Step 1 findings>",
  "platform_execution": <1-5>,
  "source_coverage": <1-5>,
  "content_relevance": <1-5>,
  "synthesis_quality": <1-5>,
  "dimension_reasoning": {{
    "platform_execution": "<one sentence citing specific evidence>",
    "source_coverage": "<one sentence citing specific evidence>",
    "content_relevance": "<one sentence citing specific evidence>",
    "synthesis_quality": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

# Dimension weights — Platform Execution is highest because using the wrong
# platform is a fundamental failure of the task regardless of output quality.
# Content Relevance is second because generic questions are not useful for
# NUS interview prep specifically.
DIMENSION_WEIGHTS = {
    "platform_execution": 0.35,
    "source_coverage":    0.20,
    "content_relevance":  0.30,
    "synthesis_quality":  0.15,
}

DIMENSIONS = list(DIMENSION_WEIGHTS.keys())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_response(result: dict) -> str:
    task_result = result.get("task_result") or ""
    if isinstance(task_result, str) and task_result.strip():
        return task_result
    for message in reversed(result.get("conversation") or []):
        if not isinstance(message, dict):
            continue
        if message.get("role") == "assistant":
            content = message.get("content") or ""
            if isinstance(content, str) and len(content) > 20:
                return content
    return ""


def _parse_answer_tag(text: str) -> dict | None:
    """Extract JSON from inside <Answer>...</Answer> tags."""
    match = re.search(r"<Answer>(.*?)</Answer>", text, re.DOTALL | re.IGNORECASE)
    if not match:
        return None
    try:
        return json.loads(match.group(1).strip())
    except json.JSONDecodeError:
        return None


def _call_judge_once(agent_response: str, execution_summary: str) -> dict | None:
    """Single judge call. Returns parsed dict or None on failure."""
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
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_content},
            ],
            max_tokens=1024,
        )
        raw = completion.choices[0].message.content
        return _parse_answer_tag(raw)
    except Exception as e:
        return {"error": str(e)}


def _majority_vote(votes: list[dict]) -> dict:
    """
    Aggregate up to 3 judge calls via majority vote on each dimension.
    Overall score is recomputed from voted dimension scores using weights.
    Used for borderline cases (overall_score within 0.5 of threshold).
    """
    valid = [v for v in votes if v and "error" not in v and all(d in v for d in DIMENSIONS)]
    if not valid:
        return votes[0] if votes else {"error": "All judge calls failed"}

    aggregated = {}
    for dim in DIMENSIONS:
        scores = [v[dim] for v in valid]
        aggregated[dim] = sorted(scores)[len(scores) // 2]

    overall = sum(aggregated[d] * DIMENSION_WEIGHTS[d] for d in DIMENSIONS)
    aggregated["overall_score"] = round(overall, 2)
    aggregated["passed"] = overall >= PASS_THRESHOLD

    median_call = sorted(valid, key=lambda v: abs(v.get("overall_score", 0) - overall))[0]
    aggregated["evidence_summary"] = median_call.get("evidence_summary", "")
    aggregated["dimension_reasoning"] = median_call.get("dimension_reasoning", {})
    aggregated["_votes_used"] = len(valid)

    return aggregated


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def test(result: dict) -> dict:
    agent_response = _extract_response(result)
    execution_summary = result.get("execution_summary", "")

    if not agent_response.strip():
        return {
            "passed": False,
            "feedback": "No response found from agent.",
            "details": {"task_completed": result.get("status") == "success"},
        }

    first_call = _call_judge_once(agent_response, execution_summary)

    if first_call and "error" not in first_call:
        overall = first_call.get("overall_score", 0)
        borderline = abs(float(overall) - PASS_THRESHOLD) <= 0.5

        if borderline:
            call2 = _call_judge_once(agent_response, execution_summary)
            call3 = _call_judge_once(agent_response, execution_summary)
            scores = _majority_vote([first_call, call2, call3])
        else:
            scores = first_call
            scores["_votes_used"] = 1
    else:
        scores = first_call or {"error": "Judge call failed", "overall_score": 0}

    overall = scores.get("overall_score", 0)
    passed = scores.get("passed", float(overall) >= PASS_THRESHOLD)

    feedback_lines = [f"Overall score: {overall}/5  (threshold: {PASS_THRESHOLD})"]
    for dim in DIMENSIONS:
        if dim in scores:
            feedback_lines.append(f"  {dim}: {scores[dim]}/5")
    if scores.get("evidence_summary"):
        feedback_lines.append(f"\nEvidence summary: {scores['evidence_summary']}")
    reasoning = scores.get("dimension_reasoning", {})
    if reasoning:
        feedback_lines.append("\nDimension reasoning:")
        for dim, reason in reasoning.items():
            feedback_lines.append(f"  {dim}: {reason}")
    if scores.get("_votes_used", 1) > 1:
        feedback_lines.append(f"\n(Borderline case: {scores['_votes_used']} judge calls used, majority vote applied)")

    return {
        "passed": bool(passed),
        "feedback": "\n".join(feedback_lines),
        "details": {
            "task_completed": result.get("status") == "success",
            "overall_score": overall,
            "dimension_scores": {d: scores.get(d) for d in DIMENSIONS},
            "evidence_summary": scores.get("evidence_summary"),
            "dimension_reasoning": scores.get("dimension_reasoning"),
            "pass_threshold": PASS_THRESHOLD,
            "votes_used": scores.get("_votes_used", 1),
        },
    }