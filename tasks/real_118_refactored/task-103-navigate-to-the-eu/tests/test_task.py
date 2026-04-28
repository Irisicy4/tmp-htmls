import sys
sys.path.insert(0, "/harness")
from evaluator import evaluate

TASK_INSTRUCTION = ("Navigate to EUR-Lex and retrieve the full text of GDPR Article 28 (processor obligations) "
    "and Article 32 (security of processing). Then go to the UK ICO website and find their guidance "
    "on data processor contracts and security requirements. Finally, visit the CNIL guidance page "
    "and locate their published checklist for data processors. Using all three sources, compile a "
    "GDPR compliance checklist for a B2B SaaS company acting as a data processor, organized into "
    "sections: Contractual Requirements, Technical Security Measures, Organizational Measures, and "
    "Breach Notification Obligations. Each checklist item must cite its specific source.")

USER_PROMPT_TEMPLATE = ("""## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Sources: Must use EUR-Lex AND ICO (ico.org.uk) AND CNIL (cnil.fr) — all three required
- EUR-Lex content: Must reference GDPR Article 28 and Article 32 specifically
- ICO content: Must reference guidance on data processor contracts or security
- CNIL content: Must reference their processor checklist or guidance
- Output: A checklist organized into four named sections with source citations per item
- No fabrication: Checklist items must be traceable to real regulatory text

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate EUR-Lex and retrieve GDPR Article 28 and Article 32?
- Did the agent navigate the ICO website and find processor guidance?
- Did the agent navigate the CNIL website and find their processor checklist?
- Is the output organized into the four required sections?
- Does each checklist item include a source citation?

### Step 2: Dimension Scoring

#### A. Source Coverage
Did the agent access all three required sources (EUR-Lex, ICO, CNIL)?

5 — All three sources accessed; agent cites specific pages or articles from each.
4 — Two of three sources accessed with specific content; third attempted but limited.
3 — Two of three sources accessed; third source not used or not credibly cited.
2 — Only one source accessed; the other two are absent or fabricated.
1 — No credible source navigation; response is entirely from prior knowledge.

#### B. GDPR Article Accuracy
Are GDPR Article 28 and Article 32 correctly identified and accurately summarized?

5 — Both Article 28 (processor contract requirements) and Article 32 (security measures including encryption, pseudonymization, resilience) correctly summarized with specific clause-level detail.
4 — Both articles addressed but one is summarized with minor inaccuracies or missing key obligations.
3 — One article correctly addressed; the other is vague or confused with other articles.
2 — Articles referenced by number but content is generic or largely incorrect.
1 — Articles not identified or content is fabricated.

#### C. Checklist Structure & Completeness
Is the checklist organized into all four required sections with substantive items in each?

5 — All four sections present (Contractual Requirements, Technical Security, Organizational Measures, Breach Notification) with at least three actionable items each.
4 — All four sections present but one section has fewer than two items.
3 — Three of four sections present with adequate items; one section missing.
2 — Checklist exists but organized differently; fewer than two of the four required sections.
1 — No checklist structure; narrative response only.

#### D. Source Citations
Does each checklist item (or group) cite a specific source?

5 — Every checklist item (or section) cites a specific source with article number, guidance section, or page reference.
4 — Most items (>75%) cite a source; a few items lack citations.
3 — Some items (~50%) cite a source; others do not.
2 — A few items cite sources but most are uncited.
1 — No source citations anywhere in the checklist.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "dimension_reasoning": {{
    "source_coverage": "<one sentence citing specific evidence>",
    "gdpr_article_accuracy": "<one sentence citing specific evidence>",
    "checklist_structure": "<one sentence citing specific evidence>",
    "source_citations": "<one sentence citing specific evidence>"
  }},
  "source_coverage": <1-5>,
  "gdpr_article_accuracy": <1-5>,
  "checklist_structure": <1-5>,
  "source_citations": <1-5>,
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
""")

DIMENSION_WEIGHTS = {
    "source_coverage":       0.25,
    "gdpr_article_accuracy": 0.30,
    "checklist_structure":   0.25,
    "source_citations":      0.20,
}


def test(result):
    return evaluate(result, TASK_INSTRUCTION, USER_PROMPT_TEMPLATE, DIMENSION_WEIGHTS)
