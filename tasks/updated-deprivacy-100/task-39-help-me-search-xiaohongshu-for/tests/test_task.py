"""LLM-as-judge evaluator for task-39-help-me-search-xiaohongshu-for.

Category: Marketing & Analytics
Task: Help me search Xiaohongshu for highly-liked posts about AI + psychology and AI companionship products, and organize the main points of each post and the corresponding bloggers into a document.
"""

import sys
sys.path.insert(0, "/harness")
from evaluator import run_judge

TASK_INSTRUCTION = 'Help me search Xiaohongshu for highly-liked posts about AI + psychology and AI companionship products, and organize the main points of each post and the corresponding bloggers into a document.'

SYSTEM_PROMPT_EXTRA = 'You are an expert evaluator for AI agent benchmarks assessing a Xiaohongshu content research and document compilation task.'

CONSTRAINTS = """- Platform: Xiaohongshu (小红书) specifically
- Topics: AI + psychology AND AI companionship products (both topic areas required)
- Filter: highly-liked posts (sort by likes or engagement)
- Output: structured document with per-post main points AND blogger info
- Document must be saved (not just shown in response)"""

EVIDENCE_QUESTIONS = """- Did the agent navigate Xiaohongshu? Cite evidence.
- Were both topic areas (AI+psychology AND AI companionship) searched?
- How many posts were found and summarised?
- Are blogger names/handles included?
- Was a document saved?"""

DIMENSION_RUBRICS = """#### A. Platform Execution
Did the agent navigate Xiaohongshu as instructed?

5 — Agent navigated Xiaohongshu, searched for both topic areas, and retrieved posts sorted by likes.
4 — Agent accessed Xiaohongshu for one topic area; the other was supplemented from elsewhere.
3 — Agent referenced Xiaohongshu content without clearly navigating the platform.
2 — Agent used a different platform (e.g. WeChat, Weibo) without explanation.
1 — No platform navigation.

#### B. Topic Relevance
Do the posts cover both required topic areas?

5 — Posts from both "AI + psychology" and "AI companionship products" included; topics clearly distinguished.
4 — Both topics present but one area has fewer posts or less depth.
3 — Only one topic area represented; the other is missing or barely mentioned.
2 — Posts are broadly AI-related but not specifically psychology or companionship.
1 — Topics ignored or irrelevant posts included.

#### C. Content Extraction
Are the main points of each post extracted and blogger info included?

5 — 5+ posts summarised with: main point(s) per post AND blogger name/handle; summaries are substantive (not just post titles).
4 — 3–4 posts summarised with main points; blogger info present for most.
3 — 2–3 posts summarised; blogger info missing for some; summaries are thin.
2 — Posts listed by title only without main point extraction.
1 — No post content extracted.

#### D. Document Saved
Was the compiled content saved to a document?

5 — Document saved (file, Google Docs, etc.); path, URL, or trace confirmation; document is well-structured.
4 — Document creation attempted; trace confirms write; content present.
3 — Content structured in response that could be saved; no file created.
2 — Content present but unstructured; would require significant reformatting.
1 — No document output."""

DIMENSION_WEIGHTS = {
    'platform_execution': 0.25,
    'topic_relevance': 0.25,
    'content_extraction': 0.25,
    'document_saved': 0.25,
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
