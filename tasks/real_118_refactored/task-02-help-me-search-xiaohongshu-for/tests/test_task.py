import sys
sys.path.insert(0, "/harness")
from evaluator import evaluate

TASK_INSTRUCTION = ("Help me search Xiaohongshu for all interview questions related to the National University "
    "of Singapore (NUS) and summarize them into a list")

USER_PROMPT_TEMPLATE = ("""## Task Instruction
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
  "dimension_reasoning": {{
    "platform_execution": "<one sentence citing specific evidence>",
    "source_coverage": "<one sentence citing specific evidence>",
    "content_relevance": "<one sentence citing specific evidence>",
    "synthesis_quality": "<one sentence citing specific evidence>"
  }},
  "platform_execution": <1-5>,
  "source_coverage": <1-5>,
  "content_relevance": <1-5>,
  "synthesis_quality": <1-5>,
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
""")

DIMENSION_WEIGHTS = {
    "platform_execution": 0.35,
    "source_coverage":    0.20,
    "content_relevance":  0.30,
    "synthesis_quality":  0.15,
}


def test(result):
    return evaluate(result, TASK_INSTRUCTION, USER_PROMPT_TEMPLATE, DIMENSION_WEIGHTS)
