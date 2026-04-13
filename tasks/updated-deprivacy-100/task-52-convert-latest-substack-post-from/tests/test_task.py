"""
LLM-as-judge evaluator for EvolveBench task-52.

Category: (Self) Media
Task: Convert latest Substack post from The Biblical Man into Instagram/TikTok carousels and Instagram Stories.
"""

import os, json, re
PASS_THRESHOLD = 3.0

def _extract_response(result):
    task_result = result.get("task_result") or ""
    if isinstance(task_result, str) and task_result.strip(): return task_result
    for message in reversed(result.get("conversation") or []):
        if not isinstance(message, dict): continue
        if message.get("role") == "assistant":
            content = message.get("content") or ""
            if isinstance(content, str) and len(content) > 20: return content
    return ""

def _parse(text):
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

def _call(agent_response, execution_summary, system_prompt, user_prompt_template, task_instruction):
    try:
        import openai
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt_template.format(
                    task_instruction=task_instruction,
                    agent_response=agent_response,
                    execution_summary=execution_summary or "Not available.",
                )}
            ],
            max_tokens=1024,
        )
        return _parse(completion.choices[0].message.content)
    except Exception as e: return {"error": str(e)}

def _vote(votes, dimensions, weights, pass_threshold):
    valid = [v for v in votes if v and "error" not in v and all(d in v for d in dimensions)]
    if not valid: return votes[0] if votes else {"error": "All judge calls failed"}
    aggregated = {dim: sorted([v[dim] for v in valid])[len(valid) // 2] for dim in dimensions}
    overall = sum(aggregated[d] * weights[d] for d in dimensions)
    aggregated["overall_score"] = round(overall, 2); aggregated["passed"] = overall >= pass_threshold
    median_call = sorted(valid, key=lambda v: abs(v.get("overall_score", 0) - overall))[0]
    aggregated["evidence_summary"] = median_call.get("evidence_summary", "")
    aggregated["dimension_reasoning"] = median_call.get("dimension_reasoning", {})
    aggregated["_votes_used"] = len(valid)
    return aggregated


TASK_INSTRUCTION = """Convert the latest Substack post from The Biblical Man (biblicalman.substack.com) into: (1) an Instagram Carousel with 10 slides, caption, and hashtags, (2) a TikTok Carousel with 7 slides, caption, and hashtags, and (3) 5 Instagram Story frames. Follow the hook-build-payoff-CTA arc and use a confrontational, declarative voice."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Assess whether an AI agent successfully converted a Substack post into properly formatted social media carousel content."""

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Source: must fetch actual content from biblicalman.substack.com — not invented content
- Instagram Carousel: exactly 10 slides + caption + hashtags
- TikTok Carousel: exactly 7 slides + caption + hashtags
- Instagram Stories: 5 frames
- Arc: Hook → Build → Build → Build → Bridge → Payoff → CTA
- Voice: confrontational, declarative, no therapeutic language

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent fetch the Substack post from biblicalman.substack.com?
- Were all three output formats produced (IG carousel, TikTok carousel, IG Stories)?
- Do slide counts match (10, 7, 5)?
- Does the content follow the hook-build-payoff-CTA arc?
- Is the voice confrontational and declarative?

### Step 2: Dimension Scoring

#### A. Content Sourcing (0.25)
Did the agent fetch actual content from biblicalman.substack.com?

5 — Agent navigated to biblicalman.substack.com, fetched the latest post, and based content on it.
4 — Agent fetched content but from a cached or slightly outdated version.
3 — Agent referenced the Substack but content seems partially invented.
2 — Agent created generic Biblical Man-style content without fetching actual post.
1 — No sourcing from biblicalman.substack.com.

#### B. Format Completeness (0.3)
Were all three output formats produced with correct slide counts?

5 — All three formats present: 10-slide IG carousel, 7-slide TikTok carousel, 5 IG Story frames.
4 — Two of three formats present, or slide counts slightly off.
3 — One format present and complete.
2 — Formats present but significantly incomplete (e.g. 3 slides instead of 10).
1 — No carousel content produced.

#### C. Narrative Arc (0.25)
Does the content follow the hook-build-payoff-CTA arc?

5 — Clear arc: scroll-stopping hook → tension-building slides → bridge → payoff → CTA.
4 — Arc mostly followed but one element is weak or missing.
3 — Some arc structure but not consistently applied.
2 — Content is good but arc is not intentionally structured.
1 — No arc — slides are random or disjointed.

#### D. Voice Quality (0.2)
Is the voice confrontational, declarative, and on-brand?

5 — Short punchy sentences, declarative statements, physical/visceral language, no emojis in slides, no therapeutic language.
4 — Mostly on-brand but a few slides drift soft.
3 — Mix of confrontational and soft voice.
2 — Mostly motivational-poster tone rather than confrontational.
1 — Wrong voice entirely — too soft, too casual, or too corporate.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "content_sourcing": <1-5>,
  "format_completeness": <1-5>,
  "narrative_arc": <1-5>,
  "voice_quality": <1-5>,
  "dimension_reasoning": {{
    "content_sourcing": "<one sentence citing specific evidence>",
    "format_completeness": "<one sentence citing specific evidence>",
    "narrative_arc": "<one sentence citing specific evidence>",
    "voice_quality": "<one sentence citing specific evidence>",
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "content_sourcing": 0.25,
    "format_completeness": 0.3,
    "narrative_arc": 0.25,
    "voice_quality": 0.2,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())

def test(result):
    agent_response = _extract_response(result)
    execution_summary = result.get("execution_summary", "")
    if not agent_response.strip():
        return {"passed": False, "feedback": "No response found from agent.",
                "details": {"task_completed": result.get("status") == "success"}}
    first = _call(agent_response, execution_summary, SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, TASK_INSTRUCTION)
    if first and "error" not in first:
        overall = first.get("overall_score", 0)
        if abs(float(overall) - PASS_THRESHOLD) <= 0.5:
            scores = _vote([first,
                            _call(agent_response, execution_summary, SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, TASK_INSTRUCTION),
                            _call(agent_response, execution_summary, SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, TASK_INSTRUCTION)],
                           DIMENSIONS, DIMENSION_WEIGHTS, PASS_THRESHOLD)
        else:
            scores = first; scores["_votes_used"] = 1
    else:
        scores = first or {"error": "Judge call failed", "overall_score": 0}
    overall = scores.get("overall_score", 0)
    passed = scores.get("passed", float(overall) >= PASS_THRESHOLD)
    lines = [f"Overall score: {overall}/5  (threshold: {PASS_THRESHOLD})"]
    for dim in DIMENSIONS:
        if dim in scores: lines.append(f"  {dim}: {scores[dim]}/5")
    if scores.get("evidence_summary"): lines.append(f"\nEvidence summary: {scores['evidence_summary']}")
    reasoning = scores.get("dimension_reasoning", {})
    if reasoning:
        lines.append("\nDimension reasoning:")
        for dim, reason in reasoning.items(): lines.append(f"  {dim}: {reason}")
    if scores.get("_votes_used", 1) > 1:
        lines.append(f"\n(Borderline case: {scores['_votes_used']} judge calls used, majority vote applied)")
    return {
        "passed": bool(passed), "feedback": "\n".join(lines),
        "details": {"task_completed": result.get("status") == "success", "overall_score": overall,
                    "dimension_scores": {d: scores.get(d) for d in DIMENSIONS},
                    "evidence_summary": scores.get("evidence_summary"),
                    "dimension_reasoning": scores.get("dimension_reasoning"),
                    "pass_threshold": PASS_THRESHOLD, "votes_used": scores.get("_votes_used", 1)},
    }