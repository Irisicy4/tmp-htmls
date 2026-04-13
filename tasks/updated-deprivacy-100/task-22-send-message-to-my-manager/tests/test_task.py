"""
LLM-as-judge evaluator for EvolveBench task-23.

Category: Daily Activities
Task: Design a Web 3.0 learning plan — but first list the steps, do NOT act yet.
      This is a planning/meta task: the agent must describe its approach, not execute it.
"""

import os, json, re

TASK_INSTRUCTION = "Please design a comprehensive learning plan for Web 3.0. Do not take any action yet—first list the steps you will use to help me create the plan."
PASS_THRESHOLD = 3.0

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. This is a planning/meta task: the agent must list its intended steps WITHOUT actually executing the learning plan. Executing the plan (e.g. outputting a full learning plan) instead of listing steps is a failure mode."""

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Critical: agent must list steps/approach ONLY — not produce the actual learning plan
- Producing the full plan when asked to only list steps is a violation of the task constraint
- Steps should be specific to Web 3.0 (not generic learning plan steps)
- Steps should be logical, ordered, and comprehensive enough to lead to a good learning plan

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis
- Did the agent list steps/approach rather than producing the full plan?
- How many steps were listed? Are they specific to Web 3.0?
- Did the agent accidentally produce a full learning plan instead of just steps?
- Are the steps logically ordered and comprehensive?

### Step 2: Dimension Scoring

#### A. Constraint Adherence
Did the agent correctly list steps WITHOUT producing the full plan?

5 — Agent listed steps only; explicitly stated it will not act yet; no full plan produced.
4 — Agent listed steps; briefly elaborated on one or two but did not produce a full plan.
3 — Agent listed steps but also began producing partial plan content (borderline violation).
2 — Agent produced a partial or full learning plan instead of listing steps.
1 — Agent ignored the constraint entirely and produced a full learning plan.

#### B. Step Specificity
Are the steps specific to Web 3.0 rather than generic learning plan steps?

5 — Steps reference Web 3.0-specific topics (blockchain, smart contracts, DeFi, NFTs, DAOs, wallets, Solidity, etc.) and how they would be covered.
4 — Most steps are Web 3.0-specific; 1–2 are generic (e.g. "assess current knowledge").
3 — Steps are relevant to technology learning broadly but Web 3.0-specific content is mentioned vaguely.
2 — Steps are completely generic (could apply to learning any technology).
1 — No steps listed or steps are irrelevant.

#### C. Step Completeness & Logic
Are the steps comprehensive and logically ordered?

5 — 5+ distinct steps covering: assessment, topic sequencing, resource identification, practice, and review; logical progression.
4 — 4–5 steps with clear logic; one phase (e.g. practice or review) missing.
3 — 3–4 steps; logical but missing key phases.
2 — 1–2 steps; far too sparse to constitute a plan for creating a plan.
1 — No structured steps.

#### D. Response Clarity
Is the step list clear, well-formatted, and easy to follow?

5 — Numbered or bulleted list; each step has a clear title and brief description.
4 — List format with step titles; descriptions are minimal but understandable.
3 — Steps present but loosely formatted (e.g. paragraph form).
2 — Steps are buried in prose; hard to extract.
1 — No structured output.

### Step 3: Output
<Answer>
{{
  "evidence_summary": "<2-3 sentences>",
  "constraint_adherence": <1-5>,
  "step_specificity": <1-5>,
  "step_completeness": <1-5>,
  "response_clarity": <1-5>,
  "dimension_reasoning": {{"constraint_adherence": "<one sentence>", "step_specificity": "<one sentence>", "step_completeness": "<one sentence>", "response_clarity": "<one sentence>"}},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {"constraint_adherence": 0.40, "step_specificity": 0.25, "step_completeness": 0.20, "response_clarity": 0.15}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())

def _extract_response(result):
    t = result.get("task_result") or ""
    if isinstance(t, str) and t.strip(): return t
    for m in reversed(result.get("conversation") or []):
        if isinstance(m, dict) and m.get("role") == "assistant":
            c = m.get("content") or ""
            if isinstance(c, str) and len(c) > 20: return c
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

def _call(agent_response, execution_summary):
    try:
        import openai
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": USER_PROMPT_TEMPLATE.format(task_instruction=TASK_INSTRUCTION, agent_response=agent_response, execution_summary=execution_summary or "Not available.")}],
            max_tokens=1024)
        return _parse(completion.choices[0].message.content)
    except Exception as e: return {"error": str(e)}

def _vote(votes):
    valid = [v for v in votes if v and "error" not in v and all(d in v for d in DIMENSIONS)]
    if not valid: return votes[0] if votes else {"error": "All calls failed"}
    agg = {d: sorted([v[d] for v in valid])[len(valid)//2] for d in DIMENSIONS}
    overall = sum(agg[d] * DIMENSION_WEIGHTS[d] for d in DIMENSIONS)
    agg["overall_score"] = round(overall, 2); agg["passed"] = overall >= PASS_THRESHOLD
    median = sorted(valid, key=lambda v: abs(v.get("overall_score",0)-overall))[0]
    agg["evidence_summary"] = median.get("evidence_summary",""); agg["dimension_reasoning"] = median.get("dimension_reasoning",{}); agg["_votes_used"] = len(valid)
    return agg

def test(result):
    agent_response = _extract_response(result)
    execution_summary = result.get("execution_summary", "")
    if not agent_response.strip():
        return {"passed": False, "feedback": "No response found from agent.", "details": {"task_completed": result.get("status") == "success"}}
    first = _call(agent_response, execution_summary)
    if first and "error" not in first:
        overall = first.get("overall_score", 0)
        scores = _vote([first, _call(agent_response, execution_summary), _call(agent_response, execution_summary)]) if abs(float(overall) - PASS_THRESHOLD) <= 0.5 else (first.__setitem__("_votes_used", 1) or first)
    else:
        scores = first or {"error": "Judge call failed", "overall_score": 0}
    overall = scores.get("overall_score", 0); passed = scores.get("passed", float(overall) >= PASS_THRESHOLD)
    lines = [f"Overall score: {overall}/5  (threshold: {PASS_THRESHOLD})"] + [f"  {d}: {scores[d]}/5" for d in DIMENSIONS if d in scores]
    if scores.get("evidence_summary"): lines.append(f"\nEvidence summary: {scores['evidence_summary']}")
    if scores.get("dimension_reasoning"):
        lines.append("\nDimension reasoning:")
        for d, r in scores["dimension_reasoning"].items(): lines.append(f"  {d}: {r}")
    if scores.get("_votes_used", 1) > 1: lines.append(f"\n(Borderline: {scores['_votes_used']} calls, majority vote)")
    return {"passed": bool(passed), "feedback": "\n".join(lines), "details": {"task_completed": result.get("status") == "success", "overall_score": overall, "dimension_scores": {d: scores.get(d) for d in DIMENSIONS}, "evidence_summary": scores.get("evidence_summary"), "dimension_reasoning": scores.get("dimension_reasoning"), "pass_threshold": PASS_THRESHOLD, "votes_used": scores.get("_votes_used", 1)}}