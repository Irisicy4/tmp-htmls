import sys
sys.path.insert(0, "/harness")
from evaluator import evaluate

TASK_INSTRUCTION = ("""You are helping plan an upcoming school art festival. Research the Tokyo University of the Arts high school art festival and identify one featured or alumni artist from their recent exhibitions. Create a Google Document summarizing the festival and the artist. In a separate tab, draft a proposal for inviting that artist to host a workshop, including a proposed schedule and content outline.""")

USER_PROMPT_TEMPLATE = ("""## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Institution: Tokyo University of the Arts (東京藝術大学, TUA) affiliated high school
- Task 1: Research the TUA high school art festival (dates, theme, featured works/exhibitions)
- Task 2: Identify ONE specific artist — featured in or notable alumnus from recent TUA exhibitions (must be discovered through research, not assumed)
- Task 3: Create a Google Document with (a) festival summary and (b) identified artist profile
- Task 4: Add a separate tab in the same Google Document with a workshop proposal for the identified artist (must include a proposed schedule and content outline)
- The artist should be real and verifiable from recent TUA-related exhibition research

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent research the TUA affiliated high school art festival? Were specific details found (dates, theme, works)?
- Did the agent identify a specific named artist from recent TUA exhibitions? Is there evidence the artist was discovered through research?
- Was a Google Doc created? Is there a URL or confirmed creation in the trace?
- Does the Doc contain both a festival summary and an artist profile?
- Does the separate tab contain a concrete workshop proposal with schedule and content outline?

### Step 2: Dimension Scoring

#### A. Research Quality (0.25)
Did the agent find real information about the TUA high school art festival and identify a real artist?

5 — Specific findings on the TUA festival (dates, theme, or featured works) AND a named artist identified with evidence linking them to recent TUA exhibitions.
4 — Festival researched and artist named, but one element is thinly supported.
3 — One thoroughly researched; the other vague or inferred without clear sourcing.
2 — Only the festival or only an artist researched; no clear link to TUA exhibitions.
1 — No research; response based on prior knowledge or hallucination.

#### B. Google Doc Creation (0.2)
Was a Google Document actually created?

5 — Google Doc created with confirmed URL or sharing link visible in trace or response.
4 — Google Doc created but link not confirmed or sharing status unclear.
3 — Doc described or partially created; creation not clearly confirmed in trace.
2 — Content written in agent response instead of a Google Doc.
1 — No Google Doc created.

#### C. Summary Completeness (0.25)
Does the Google Doc contain both a festival summary and artist profile?

5 — Both present with specific facts: festival dates/theme/works and artist name/background/notable works.
4 — Both present but one section is thin or lacks specific details.
3 — One section complete; the other minimal or absent.
2 — Only one section present (festival summary or artist profile, not both).
1 — No summaries in the document.

#### D. Proposal Concreteness (0.3)
Does the separate tab contain a concrete workshop proposal with schedule and content outline?

5 — Specific workshop content tailored to the identified artist's practice, duration, proposed schedule (e.g. timed agenda), and content outline for student participants all defined.
4 — Good proposal with schedule and content but one element (e.g. timing detail) is vague.
3 — Proposal present but generic, lacking artist-specific content or a clear schedule.
2 — Vague proposal or only a brief mention without schedule or content outline.
1 — No proposal in the separate tab.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "research_quality": <1-5>,
  "google_doc_creation": <1-5>,
  "summary_completeness": <1-5>,
  "proposal_concreteness": <1-5>,
  "dimension_reasoning": {{
    "research_quality": "<one sentence citing specific evidence>",
    "google_doc_creation": "<one sentence citing specific evidence>",
    "summary_completeness": "<one sentence citing specific evidence>",
    "proposal_concreteness": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
""")

DIMENSION_WEIGHTS = {
    "research_quality": 0.25,
    "google_doc_creation": 0.2,
    "summary_completeness": 0.25,
    "proposal_concreteness": 0.3,
}


def test(result):
    return evaluate(result, TASK_INSTRUCTION, USER_PROMPT_TEMPLATE, DIMENSION_WEIGHTS)
