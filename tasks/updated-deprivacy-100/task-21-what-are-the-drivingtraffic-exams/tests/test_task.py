"""
LLM-as-judge evaluator for EvolveBench task-21.

Category: Daily Activities
Task: Compare driving/traffic exams in China, US, and France — structure and differences.
"""

import os, json, re

TASK_INSTRUCTION = "What are the driving/traffic exams like in China, the US, and France? What are the differences between them?"
PASS_THRESHOLD = 3.0

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a comparative research task about driving examinations across three countries."""

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- All three countries must be covered: China, US, and France
- Both description AND comparison are required — not just listing each country's system separately
- Research expected: agent should search for current, accurate exam structures rather than relying solely on prior knowledge
- Key aspects to cover: written/theory test, practical/driving test, required hours, pass rates, notable differences

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Are all three countries covered (China, US, France)?
- What specific exam components are described for each country?
- Does the agent explicitly compare and contrast the systems?
- Did the agent search for this information or rely on prior knowledge?
- Are any facts clearly wrong or outdated?

### Step 2: Dimension Scoring

#### A. Country Coverage
Are all three countries covered with sufficient detail?

5 — All three countries covered with: exam structure, components (theory/practical), and key requirements.
4 — All three covered but one country has noticeably less detail.
3 — Two countries covered well; one is only briefly mentioned.
2 — Only one or two countries covered.
1 — No country-specific detail provided.

#### B. Comparative Analysis
Does the agent explicitly compare the three systems rather than just listing them separately?

5 — Structured comparison of key differences (e.g. number of required hours, difficulty, pass rates, unique features); highlights what makes each system distinctive.
4 — Comparison present but focuses on 1–2 dimensions only (e.g. only compares written tests).
3 — Implicit comparison (e.g. "China requires X while the US does not") but no dedicated comparison section.
2 — Three systems described separately with no explicit comparison.
1 — No comparison; single-country description or generic overview.

#### C. Information Accuracy
Is the information accurate and reasonably current?

5 — Key facts are correct (e.g. China's 科目一–四 structure, US state-by-state system, French code de la route + conduite); no clearly wrong claims.
4 — Mostly accurate; 1–2 minor inaccuracies or outdated details.
3 — Generally correct direction but several vague or unverifiable claims.
2 — Significant inaccuracies or information appears fabricated.
1 — Clearly wrong or completely generic.

#### D. Response Organisation
Is the output well-structured and easy to read?

5 — Clear structure: per-country sections + dedicated comparison; headers or table used effectively.
4 — Well-organised but comparison is embedded in narrative rather than structured separately.
3 — Content is present but formatting makes it hard to extract comparative information quickly.
2 — Wall of text with no clear organisation.
1 — Unstructured or non-responsive.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences>",
  "country_coverage": <1-5>,
  "comparative_analysis": <1-5>,
  "information_accuracy": <1-5>,
  "response_organisation": <1-5>,
  "dimension_reasoning": {{
    "country_coverage": "<one sentence>",
    "comparative_analysis": "<one sentence>",
    "information_accuracy": "<one sentence>",
    "response_organisation": "<one sentence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {"country_coverage": 0.25, "comparative_analysis": 0.35, "information_accuracy": 0.25, "response_organisation": 0.15}
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