"""LLM-as-judge evaluator for task-42-create-complete-modern-website-for.

Category: Software Engineering
Task: Create a complete modern website for a French art museum with 3D. Here is the style of site I want: https://www.ciaokombucha.com
"""

import sys
sys.path.insert(0, "/harness")
from evaluator import run_judge

TASK_INSTRUCTION = 'Create a complete modern website for a French art museum with 3D. Here is the style of site I want: https://www.ciaokombucha.com'

SYSTEM_PROMPT_EXTRA = 'This is a creative web development task requiring 3D elements in the style of a specific reference site (ciaokombucha.com — known for bold typography, scroll animations, and immersive 3D/WebGL elements). The agent should have fetched the reference site to understand the style.'

CONSTRAINTS = """- Reference style: https://www.ciaokombucha.com — bold typography, scroll-triggered animations, immersive 3D/WebGL, dark aesthetic, cinematic feel
- Theme: French art museum — must include relevant museum content (collections, exhibitions, visitor info)
- 3D requirement: must use Three.js, WebGL, or equivalent for actual 3D elements (not just CSS transforms)
- Completeness: "complete" website with multiple sections (hero, collections, exhibitions, about, visit)"""

EVIDENCE_QUESTIONS = """- Did the agent visit the reference site? Cite evidence.
- What 3D technology was used (Three.js, WebGL, CSS 3D)?
- How many website sections are present?
- Does the visual style match the reference (bold type, dark, immersive)?
- Was a file created and saved?"""

DIMENSION_RUBRICS = """#### A. Reference Style Adoption
Does the website reflect the style of ciaokombucha.com?

5 — Clear influence: bold/large typography, dark or rich colour palette, scroll animations, cinematic/immersive feel — all present.
4 — 2–3 style elements present; overall feel is close but one key element missing.
3 — Modern design but generic; doesn't capture the specific reference aesthetic.
2 — Standard bootstrap/template design with no reference to the style guide.
1 — No website or completely wrong visual direction.

#### B. 3D Implementation
Are actual 3D elements implemented using an appropriate library?

5 — Three.js or WebGL used for real-time 3D elements (e.g. rotating sculpture, 3D gallery, particle system, 3D hero animation).
4 — 3D elements present using CSS 3D transforms or a simpler library; not full WebGL but visually 3D.
3 — Animated elements present but no true 3D depth (e.g. parallax only).
2 — Static layout with no 3D or animation of any kind.
1 — No 3D; agent described what 3D would look like without implementing it.

#### C. Museum Content
Does the website have appropriate French art museum content?

5 — Multiple sections with museum-appropriate content: collection highlights, current/upcoming exhibitions, visit information (hours, address), about the museum.
4 — 3–4 sections with relevant content; one area thin or placeholder-only.
3 — 2–3 sections present; content is very generic (not museum-specific).
2 — Single-page skeleton with minimal content.
1 — No museum content; purely generic template.

#### D. Code Completeness
Is the website code complete and runnable?

5 — Complete HTML/CSS/JS in one or few files; external libraries correctly imported; no obvious syntax errors; all sections linked together.
4 — Mostly complete; one section or feature is stubbed or broken.
3 — Significant code present but incomplete; would need fixes to run.
2 — Skeleton code only; most features not implemented.
1 — No code; only a description."""

DIMENSION_WEIGHTS = {
    'reference_style_adoption': 0.25,
    '3d_implementation': 0.3,
    'museum_content': 0.2,
    'code_completeness': 0.25,
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
