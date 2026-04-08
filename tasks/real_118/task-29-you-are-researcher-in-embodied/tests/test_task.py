"""
LLM-as-judge evaluator for EvolveBench task-29.

Category: Data & ML Engineering
Task: Act as embodied AI research collaborator — scan latest frontier progress
      across specific sub-topics, summarise weekly, generate a report.

Note: This is a single-session instantiation of a "long-term" task.
      The agent cannot truly do weekly monitoring in one session.
      Evaluate whether the agent (a) does a genuine research scan NOW and
      (b) establishes a credible framework for ongoing monitoring.
"""

import os, json, re

TASK_INSTRUCTION = (
    "You are a researcher in embodied AI. Please act as a long-term collaborator to help "
    "with technical insight: continuously scan the latest frontier progress in embodied "
    "intelligence, including but not limited to new technical demos, blogs, papers, and "
    "technical interpretations. Summarize key findings weekly and generate reports, with "
    "focus on fast-slow systems, VLA, embodied reinforcement learning, world models, "
    "teleoperation, force feedback, robot configurations, and related technologies."
)
PASS_THRESHOLD = 3.0

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. This is a research monitoring task. Because a single agent session cannot span weeks, evaluate: (1) the quality of the agent's immediate research scan and report, and (2) whether the agent provides a credible, specific framework for ongoing weekly monitoring."""

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Sub-topics required: fast-slow systems, VLA (Vision-Language-Action), embodied RL, world models, teleoperation, force feedback, robot configurations
- Expected output: a structured report with findings per sub-topic
- Sources: papers (arXiv), demos, blogs, technical writeups — must be recent
- Long-term framework: agent should describe how it would set up ongoing weekly scanning
- Single-session scope: agent cannot do weeks of monitoring, but should do one thorough scan now

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis
- Did the agent search for recent embodied AI papers/demos? What sources?
- Which of the 7+ sub-topics are covered in the report?
- Are specific papers, demos, or findings cited with dates?
- Did the agent produce a structured report (not just a literature overview)?
- Did the agent describe an ongoing monitoring framework?

### Step 2: Dimension Scoring

#### A. Research Scan Quality
Did the agent actively search for and retrieve recent embodied AI content?

5 — Agent searched arXiv, blogs, or demo sites; found 3+ recent (within last few months) specific papers or demos; citations with titles and dates.
4 — Agent found 2–3 specific recent items; some sub-topics covered from search, others from prior knowledge.
3 — Agent provided a research overview but few specific recent citations; mostly relies on prior knowledge.
2 — No active search; response is a general overview of embodied AI from training knowledge.
1 — No research content; refused or completely off-topic.

#### B. Sub-topic Coverage
How many of the required sub-topics are substantively addressed?

5 — All 7 sub-topics covered: fast-slow systems, VLA, embodied RL, world models, teleoperation, force feedback, robot configurations.
4 — 5–6 sub-topics covered with substance; 1–2 briefly mentioned.
3 — 4–5 sub-topics covered; remainder missing.
2 — 2–3 sub-topics covered in any depth.
1 — Fewer than 2 sub-topics, or generic overview without sub-topic structure.

#### C. Report Structure & Quality
Is the output a genuine structured report with actionable findings?

5 — Clearly structured by sub-topic with: key finding, source, significance, and connection to broader trends.
4 — Structured by sub-topic with findings; source attribution or significance analysis is thin for some.
3 — Report-like structure but findings are shallow summaries rather than technical insights.
2 — Unstructured narrative with some relevant content.
1 — No report; just a list of topics or a general description.

#### D. Ongoing Monitoring Framework
Did the agent propose a specific, actionable plan for ongoing weekly scanning?

5 — Specific framework: named sources (arXiv categories, specific blogs/newsletters, Twitter accounts), monitoring cadence, report template, and how it would flag high-impact items.
4 — Framework described with most specifics; one element (e.g. specific sources or report template) is vague.
3 — Agent mentions it can do weekly reports but framework is generic (e.g. "I'll search for new papers each week").
2 — Agent acknowledged the ongoing nature but provided no framework.
1 — No mention of ongoing monitoring.

### Step 3: Output
<Answer>
{{
  "evidence_summary": "<2-3 sentences>",
  "research_scan_quality": <1-5>,
  "subtopic_coverage": <1-5>,
  "report_structure": <1-5>,
  "monitoring_framework": <1-5>,
  "dimension_reasoning": {{"research_scan_quality": "<one sentence>", "subtopic_coverage": "<one sentence>", "report_structure": "<one sentence>", "monitoring_framework": "<one sentence>"}},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {"research_scan_quality": 0.30, "subtopic_coverage": 0.30, "report_structure": 0.25, "monitoring_framework": 0.15}
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
    match = re.search(r"<Answer>(.*?)</Answer>", text, re.DOTALL | re.IGNORECASE)
    if not match: return None
    try: return json.loads(match.group(1).strip())
    except: return None

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