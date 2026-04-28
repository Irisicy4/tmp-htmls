import sys
sys.path.insert(0, "/harness")
from evaluator import evaluate

TASK_INSTRUCTION = ("You are a researcher in embodied AI. Please act as a long-term collaborator to help "
    "with technical insight: continuously scan the latest frontier progress in embodied "
    "intelligence, including but not limited to new technical demos, blogs, papers, and "
    "technical interpretations. Summarize key findings weekly and generate reports, with "
    "focus on fast-slow systems, VLA, embodied reinforcement learning, world models, "
    "teleoperation, force feedback, robot configurations, and related technologies.")

USER_PROMPT_TEMPLATE = ("""## Task Instruction
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
  "dimension_reasoning": {{"research_scan_quality": "<one sentence>", "subtopic_coverage": "<one sentence>", "report_structure": "<one sentence>", "monitoring_framework": "<one sentence>"}},
  "research_scan_quality": <1-5>,
  "subtopic_coverage": <1-5>,
  "report_structure": <1-5>,
  "monitoring_framework": <1-5>,
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false>
}}
</Answer>
""")

DIMENSION_WEIGHTS = {"research_scan_quality": 0.30, "subtopic_coverage": 0.30, "report_structure": 0.25, "monitoring_framework": 0.15}


def test(result):
    return evaluate(result, TASK_INSTRUCTION, USER_PROMPT_TEMPLATE, DIMENSION_WEIGHTS)
