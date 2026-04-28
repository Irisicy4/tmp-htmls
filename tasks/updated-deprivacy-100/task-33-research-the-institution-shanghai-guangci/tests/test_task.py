"""LLM-as-judge evaluator for task-33-research-the-institution-shanghai-guangci.

Category: Medical & Clinical & Bio
Task: Research the institution: Shanghai Guangci Rehabilitation Hospital. Include but not limited to: authenticity of the institution, quality of medical staff, patient feedback, and what products and servi
"""

import sys
sys.path.insert(0, "/harness")
from evaluator import run_judge

TASK_INSTRUCTION = 'Research the institution: Shanghai Guangci Rehabilitation Hospital. Include but not limited to: authenticity of the institution, quality of medical staff, patient feedback, and what products and services they offer with corresponding prices.'

SYSTEM_PROMPT_EXTRA = 'This is a due-diligence research task on a specific Chinese rehabilitation hospital. Authenticity verification is particularly important — the agent should assess whether Shanghai Guangci Rehabilitation Hospital (上海广慈康复医院) is a legitimate, licensed medical institution.'

CONSTRAINTS = """- Institution: Shanghai Guangci Rehabilitation Hospital (上海广慈康复医院)
- Required aspects: (1) authenticity/legitimacy, (2) medical staff quality, (3) patient feedback/reviews, (4) products/services with prices
- Sources: Chinese business registry, medical licensing databases, consumer review platforms (e.g. Dianping, Baidu Maps, 39.net, haodf.com)
- This may be a questionable or low-credibility institution — the agent should flag concerns if found"""

EVIDENCE_QUESTIONS = """- Did the agent search for Shanghai Guangci Rehabilitation Hospital specifically? What sources?
- Was the institution's registration/license status checked?
- Were patient reviews or complaints found?
- Were specific products/services and prices listed?
- Did the agent flag any red flags about legitimacy?"""

DIMENSION_RUBRICS = """#### A. Source Quality
Did the agent use credible, appropriate sources?

5 — Used 2+ of: Chinese business registry (企查查/天眼查), medical licensing database (卫生部/上海卫健委), consumer review platform (大众点评/好大夫), or official institution website.
4 — Used 1 credible source; others are secondary.
3 — Used general web search; some results from credible platforms but not specifically targeted.
2 — Used low-quality or unverifiable sources only.
1 — No sources; response from prior knowledge.

#### B. Authenticity Assessment
Did the agent investigate and report on Shanghai Guangci Rehabilitation Hospital's legitimacy?

5 — Agent verified (or attempted to verify) business registration, medical licensing, and noted any red flags (complaints, regulatory issues, unclear credentials).
4 — Registration or licensing checked; patient complaint history not investigated.
3 — Agent mentioned legitimacy concern but did not verify with primary sources.
2 — Authenticity not investigated; agent assumed institution is legitimate.
1 — No authenticity assessment.

#### C. Information Coverage
Were all 4 required aspects addressed?

5 — All 4 aspects: authenticity, staff quality, patient feedback, products/prices — each with specific findings for Shanghai Guangci Rehabilitation Hospital.
4 — 3 of 4 aspects covered with specific findings; one is briefly mentioned.
3 — 2–3 aspects covered; products/prices or patient feedback missing.
2 — Only 1–2 aspects covered.
1 — No substantive coverage of required aspects.

#### D. Output Organisation
Is the research presented in a clear, structured format?

5 — Sections for each required aspect; findings clearly attributed to sources; any red flags highlighted.
4 — Structured output; source attribution thin for some sections.
3 — Content present but mixed together without clear section structure.
2 — Narrative paragraph with relevant facts buried.
1 — No structured output."""

DIMENSION_WEIGHTS = {
    'source_quality': 0.2,
    'authenticity_assessment': 0.3,
    'information_coverage': 0.3,
    'output_organisation': 0.2,
}


def test(result):
    return run_judge(
        result,
        task_instruction=TASK_INSTRUCTION,
        system_prompt_extra=SYSTEM_PROMPT_EXTRA,
        constraints=CONSTRAINTS,
        evidence_questions=EVIDENCE_QUESTIONS,
        dimension_rubrics=DIMENSION_RUBRICS,
        dimension_weights=DIMENSION_WEIGHTS,
    )
