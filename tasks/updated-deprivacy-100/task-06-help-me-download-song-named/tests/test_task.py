"""LLM-as-judge evaluator for task-06-help-me-download-song-named.

Category: (Self) Media
Task: help me download a song named happy birthday from youtube
"""

import sys
sys.path.insert(0, "/harness")
from evaluator import run_judge

TASK_INSTRUCTION = 'help me download a song named happy birthday from youtube'

SYSTEM_PROMPT_EXTRA = 'Your job is to assess whether an AI agent successfully completed a media download task from YouTube.'

CONSTRAINTS = """- Platform: YouTube specifically
- Content: a "Happy Birthday" song — any widely recognised version is acceptable
- Action required: an actual download must be attempted or completed, not just a link provided or instructions given
- File output: ideally an audio file (mp3, m4a, wav) or video file; the agent should confirm what was downloaded"""

EVIDENCE_QUESTIONS = """- Did the agent navigate to YouTube? Cite evidence.
- Did the agent identify a specific "Happy Birthday" video (title, channel, URL)?
- Did the agent attempt or complete a download? What method was used (yt-dlp, browser download, third-party tool)?
- Is there evidence the file was actually saved (file path, confirmation message, download completion)?
- Did the agent only provide instructions or a link instead of executing the download?"""

DIMENSION_RUBRICS = """#### A. Platform Execution
Did the agent actually navigate YouTube to find the content?

5 — Agent navigated YouTube, identified a specific Happy Birthday video with title/URL.
4 — Agent found a YouTube video but with limited detail (e.g. URL only, no title or channel).
3 — Agent searched YouTube but did not access a specific video page.
2 — Agent used a non-YouTube source or only searched generally without finding a specific video.
1 — No search performed; response is instructions only or from prior knowledge.

#### B. Download Execution
Did the agent actually attempt or complete the download?

5 — Download completed; file saved with confirmation (file path, size, or format mentioned).
4 — Download initiated and likely completed based on trace; no explicit confirmation in response.
3 — Download attempted but outcome uncertain (e.g. tool invoked but no success confirmation).
2 — Agent provided a download link or instructions but did not execute the download itself.
1 — No download attempted; agent only described how to download or refused.

#### C. Content Match
Is the downloaded or identified content a "Happy Birthday" song as requested?

5 — Content is clearly a Happy Birthday song (title confirms it; well-known version).
4 — Content is likely a Happy Birthday song but title is ambiguous or version is obscure.
3 — Agent found Happy Birthday content but it may be a cover, instrumental, or unrelated remix.
2 — Content is loosely related (birthday music) but not "Happy Birthday" specifically.
1 — Content is unrelated or agent could not find any matching content.

#### D. Response Clarity
Does the agent clearly communicate what was found and downloaded?

5 — Response states: video title, source URL, download method, and file format/location.
4 — Most details present; one piece missing (e.g. no file path or no format mentioned).
3 — Agent confirms a download but with minimal detail.
2 — Response is vague; unclear whether anything was actually downloaded.
1 — No useful output; response is a refusal, error, or generic instructions."""

DIMENSION_WEIGHTS = {
    'platform_execution': 0.2,
    'download_execution': 0.4,
    'content_match': 0.25,
    'response_clarity': 0.15,
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
