"""LLM-as-judge evaluator for task-10-you-are-a-graduate-student.

Category: Design (custom rubric — open-ended creative task)
Task: You are a graduate student in architecture. You want to build a lakeside park on the main campus of Tsinghua University. It is located on the central axis of the campus. It covers an area of about 4,0
"""

import sys
sys.path.insert(0, "/harness")
from evaluator import run_judge

TASK_INSTRUCTION = 'You are a graduate student in architecture. You want to build a lakeside park on the main campus of Tsinghua University. It is located on the central axis of the campus. It covers an area of about 4,000 square meters and has a height difference of half a meter. Find relevant cases that can be learned and generate a design sketch. It is required to reflect the genius loci of the place combined with northern Chinese garden characteristics.'

SYSTEM_PROMPT_EXTRA = "Your job is to assess whether an AI agent successfully completed an open-ended architectural design task.\n\nImportant context:\n- The site is on the main campus of Tsinghua University (清华大学) in Beijing, located on the central axis — a prestigious and historically significant location.\n- Genius loci (场所精神) refers to the distinctive atmosphere and spirit of a place — a standard architectural concept. The design should convey what makes Tsinghua's campus uniquely meaningful.\n- Northern Chinese garden characteristics (北方园林特色) include axial symmetry, imperial and literati garden motifs, stone rockeries, pavilions, planted pines and willows, moon gates, and integration with Beijing's cultural and historical landscape.\n- This task has two distinct sub-tasks: (1) research reference cases, and (2) generate a design sketch. Both must be attempted for a passing score."

CONSTRAINTS = """- Location: main campus of Tsinghua University (北京, Beijing), central axis, lakeside
- Area: approx. 4,000 square meters, trapezoidal site
- Topography: half-meter height difference across the site
- Design requirement: reflect genius loci of the place combined with northern Chinese garden characteristics
- Deliverables: (1) relevant precedent case studies AND (2) a design sketch/concept (visual output)"""

EVIDENCE_QUESTIONS = """- Did the agent research reference cases? What cases were found and are they relevant (campus parks, northern Chinese gardens, lakeside landscapes, Beijing heritage gardens)?
- Did the agent produce a design sketch? Is there an image, diagram, or visual output in the response or trace?
- Did the agent address the site constraints (4000sqm, 0.5m height difference, central axis on Tsinghua main campus)?
- Did the agent incorporate northern Chinese garden characteristics? What specific elements were mentioned (e.g. axial symmetry, pavilions, stone rockeries, pine/willow plantings, moon gates)?
- Did the agent address genius loci / sense of place for Tsinghua's main campus?"""

DIMENSION_RUBRICS = """#### A. Reference Case Research
Did the agent find and present relevant design reference cases?

5 — 3+ relevant reference cases found (campus parks, northern Chinese gardens, Beijing heritage gardens, or lakeside parks) with specific project names, locations, and design lessons drawn from each.
4 — 2–3 reference cases found with names and relevance explained; one case is weakly relevant.
3 — 1–2 reference cases found with names; connection to the design task is stated but thin.
2 — Reference cases mentioned vaguely (e.g. "traditional northern Chinese gardens") without specific projects.
1 — No reference case research performed; agent jumped directly to design without research.

#### B. Design Sketch Output
Did the agent actually generate a visual design sketch?

5 — A design sketch or diagram was produced (image file, rendered plan, or annotated diagram); it depicts the lakeside park layout.
4 — A sketch was produced but is rough or minimally annotated; clearly an attempt at visual output.
3 — Agent described a design in detail (zones, elements, layout) but produced no visual output.
2 — Agent mentioned it would generate a sketch but produced only text or failed to generate an image.
1 — No design output of any kind; agent refused or only asked clarifying questions.

#### C. Site & Program Responsiveness
Does the design respond to the specific site constraints?

5 — Design explicitly addresses: 4000sqm scale, 0.5m height change (e.g. terracing, ramps, level changes), lakeside edge treatment, and central axis relationship on Tsinghua main campus.
4 — Design addresses 3 of the 4 site constraints; one is missing or vague.
3 — Design addresses 2 site constraints; scale and height difference are either ignored or only briefly mentioned.
2 — Design is generic; site constraints are acknowledged but not reflected in the design decisions.
1 — Site constraints completely ignored; design could apply to any location.

#### D. Northern Chinese Garden & Genius Loci Integration
Does the design meaningfully incorporate northern Chinese garden characteristics and the genius loci of Tsinghua's campus?

5 — Design integrates 3+ specific northern Chinese garden elements (e.g. axial symmetry, pavilions, stone rockeries, pine/willow plantings, moon gates, imperial-style water features) AND articulates how the design reflects Tsinghua campus's genius loci (academic heritage, Beijing cultural landscape).
4 — 2 northern Chinese garden elements present with explanation; genius loci addressed but superficially.
3 — Northern Chinese garden style mentioned and 1 element described; sense of place for Tsinghua not explicitly addressed.
2 — Garden aesthetics referenced generally (e.g. "traditional Chinese garden") without specific northern elements or campus context.
1 — No northern Chinese garden characteristics or genius loci consideration; design is generic or lacks cultural grounding."""

DIMENSION_WEIGHTS = {
    'reference_case_research': 0.2,
    'design_sketch_output': 0.35,
    'site_program_responsiveness': 0.25,
    'lingnan_genius_loci': 0.2,
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
