"""
LLM-as-judge evaluator for EvolveBench task-25.

Category: Finance & Economics
Task: Collect US stock market info Dec 15-16 2025, explain why market was down,
      predict Dec 16 trend (5 hours before market open).
"""

import os, json, re

TASK_INSTRUCTION = (
    "Collect all information about the US stock market from 2025-12-15 to 2025-12-16. "
    "Explain why the market was largely down (mostly in the red). Using all collected "
    "information, think deeply and give your own conclusion and reasoning for your "
    "prediction of the market trend for 2025-12-16 (5 hours before market open)."
)
PASS_THRESHOLD = 3.0

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. You are assessing a financial research and prediction task covering specific dates (Dec 15-16, 2025).

Important: The task asks for a prediction "5 hours before market open on Dec 16" — this means the agent was operating in a pre-market context. Evaluate the quality of the reasoning and data collection, not whether the prediction turned out to be correct."""

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Date range: December 15–16, 2025 specifically
- Three required components: (1) data collection, (2) explanation of why market was down, (3) prediction for Dec 16
- Prediction context: 5 hours before Dec 16 market open (pre-market reasoning)
- Sources: must use credible financial sources (not just prior knowledge)
- Note: task assumes market was "largely down" on these dates — agent should find evidence for this

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis
- Did the agent search for Dec 15-16 2025 market data? What sources were used?
- What specific market data was collected (indices, % changes, volumes)?
- Did the agent identify reasons for the market being down?
- Did the agent make a prediction for Dec 16 with reasoning?
- Is the reasoning analytical and grounded in the collected data?

### Step 2: Dimension Scoring

#### A. Data Collection
Did the agent collect specific, dated market data for Dec 15-16 2025?

5 — Specific data collected: index levels or % changes for Dec 15 (S&P 500, NASDAQ, or Dow); sources cited; data is from Dec 15-16 2025 specifically.
4 — Data collected but less specific (e.g. general market direction without exact figures) or one date missing.
3 — Market data mentioned but vague or not clearly from the specified dates.
2 — Generic market commentary without specific Dec 15-16 data.
1 — No market data collected; response from prior knowledge or refused.

#### B. Explanation Quality
Does the agent provide a credible explanation for why the market was down?

5 — 2+ specific causal factors identified (e.g. Fed decision, economic data release, geopolitical event) with evidence linking them to the Dec 15 selloff.
4 — 1–2 factors identified with some evidence; reasoning is plausible.
3 — Explanation is directionally plausible but vague (e.g. "interest rate concerns" without specifics).
2 — Explanation is generic (e.g. "markets go up and down") without specific factors.
1 — No explanation provided or explanation is clearly wrong.

#### C. Prediction & Reasoning
Does the agent make a specific Dec 16 prediction with analytical reasoning?

5 — Clear directional prediction (up/down/flat) with 2+ reasoned factors drawn from collected data; reasoning is coherent and data-driven.
4 — Prediction made with 1–2 supporting reasons; reasoning present but not deeply analytical.
3 — Prediction made but reasoning is thin or generic (e.g. "likely to continue downward trend" without justification).
2 — Prediction mentioned without any reasoning.
1 — No prediction made.

#### D. Analytical Depth
Does the response demonstrate genuine analytical thinking beyond surface-level summary?

5 — Agent synthesises multiple data points, identifies market sentiment, considers catalysts, and provides nuanced pre-market analysis.
4 — Good analysis with some synthesis; one dimension (e.g. sentiment or catalysts) underexplored.
3 — Adequate summary of facts with basic analysis; limited synthesis.
2 — Mostly a data dump without analytical commentary.
1 — No analysis; pure narrative or refusal.

### Step 3: Output
<Answer>
{{
  "evidence_summary": "<2-3 sentences>",
  "data_collection": <1-5>,
  "explanation_quality": <1-5>,
  "prediction_reasoning": <1-5>,
  "analytical_depth": <1-5>,
  "dimension_reasoning": {{"data_collection": "<one sentence>", "explanation_quality": "<one sentence>", "prediction_reasoning": "<one sentence>", "analytical_depth": "<one sentence>"}},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {"data_collection": 0.25, "explanation_quality": 0.30, "prediction_reasoning": 0.30, "analytical_depth": 0.15}
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