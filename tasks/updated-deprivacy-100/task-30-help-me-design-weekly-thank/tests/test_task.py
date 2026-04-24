"""LLM-as-judge evaluator for task-30-help-me-design-weekly-thank.

Category: Design
Task: Help me design a weekly thank-you card template to send to employees. The card should include: this week's completed tasks + a philosophical quote + an encouraging message. (I will fill in the content
"""

import sys
sys.path.insert(0, "/harness")
from evaluator import run_judge

TASK_INSTRUCTION = "Help me design a weekly thank-you card template to send to employees. The card should include: this week's completed tasks + a philosophical quote + an encouraging message. (I will fill in the content; mainly help me design the template.)"

SYSTEM_PROMPT_EXTRA = 'This is a design task — the agent must produce a visual or well-structured card template. The user explicitly said they will fill in content, so the agent should focus on structure, layout, and visual design rather than writing the actual quote or tasks.'

CONSTRAINTS = """- Output: a template (not a filled-in card) — placeholder text for all content sections
- Required sections: (1) completed tasks area, (2) philosophical quote area, (3) encouraging message area
- Format: visual design preferred — HTML/CSS card, image design, or printable template
- User fills content: agent should use placeholders like "[Employee Name]", "[Tasks this week]", "[Quote]"
- The card should look professionally designed, not just a text document"""

EVIDENCE_QUESTIONS = """- What format did the agent use for the template (HTML, image, text, Word doc)?
- Are all 3 required sections present (tasks, quote, encouraging message)?
- Are placeholder labels used rather than filled-in content?
- Is there visual design consideration (layout, colours, typography)?
- Was a file saved/exported?"""

DIMENSION_RUBRICS = """#### A. Template Structure
Are all 3 required sections present with appropriate placeholders?

5 — All 3 sections present with clear labels and placeholder text; additional elements like employee name, date, sender name also included as placeholders.
4 — All 3 sections present; 1–2 are labelled but missing placeholder text.
3 — 2 of 3 sections present; one section is missing entirely.
2 — Only 1 section present; card is mostly a generic template without required sections.
1 — No structured sections; just a text response.

#### B. Visual Design Quality
Does the template have professional visual design?

5 — Visual card design with: clear layout, typography hierarchy, colour scheme, borders/decorative elements; card looks like a real thank-you card.
4 — Good layout with some visual design; typography or colour is basic but intentional.
3 — Structured format (e.g. HTML table or numbered sections) but minimal visual design.
2 — Plain text with section headers; no visual design consideration.
1 — No design; just a description of what the template should contain.

#### C. Format Appropriateness
Is the output format practical for the use case (sending to employees)?

5 — Format is practical: HTML card (easily emailed), printable PDF template, or image design; reusable without modification.
4 — Format is usable but requires minor adjustment to be practical for sending (e.g. plain HTML without email-friendly styling).
3 — Format works as a reference but not directly usable (e.g. markdown that would need conversion).
2 — Format is inappropriate for the use case (e.g. a code snippet with no visual output).
1 — No usable output format.

#### D. File Saved
Was the template saved as a file or concrete output?

5 — File created and saved (HTML, image, PDF, or doc); file path or confirmation provided.
4 — File creation attempted; trace confirms write but no explicit path in response.
3 — Complete template provided in response that could be directly saved; no file created.
2 — Partial template in response; incomplete for direct use.
1 — No concrete template output; only a description."""

DIMENSION_WEIGHTS = {
    'template_structure': 0.3,
    'visual_design_quality': 0.3,
    'format_appropriateness': 0.2,
    'file_saved': 0.2,
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
